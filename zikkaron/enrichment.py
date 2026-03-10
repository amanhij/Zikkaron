"""Index-time enrichment pipeline for Zikkaron memories.

Generates implied facts, commonsense inferences, and synthetic queries
at storage time to bridge the cue-trigger semantic disconnect.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from zikkaron.config import Settings

logger = logging.getLogger(__name__)

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "of",
    "for", "and", "or", "but", "not", "with", "by", "from", "as",
    "be", "was", "were", "been", "are", "am", "do", "did", "does",
    "has", "had", "have", "will", "would", "could", "should", "may",
    "can", "this", "that", "these", "those", "what", "which", "who",
    "how", "when", "where", "why", "if", "then", "so", "no", "yes",
    "all", "any", "some", "my", "your", "its", "our", "their", "we",
    "he", "she", "they", "me", "him", "her", "us", "them", "i",
    "use", "using", "used", "like", "just", "get", "got", "set",
    "make", "made", "let", "try", "need", "want", "know", "think",
    "really", "very", "also", "about", "into", "over", "such",
    "been", "being", "going", "went", "come", "came", "said",
})

HARDCODED_EXPANSIONS = {
    "camping": ["outdoor_activity", "nature", "tent", "hiking", "national_park", "wilderness"],
    "hiking": ["outdoor_activity", "trail", "nature", "exercise", "mountain", "national_park"],
    "fishing": ["outdoor_activity", "water", "lake", "river", "nature", "patience"],
    "painting": ["art", "creative", "visual_art", "canvas", "artistic", "hobby"],
    "reading": ["book", "literature", "knowledge", "education", "hobby", "library"],
    "cooking": ["food", "kitchen", "recipe", "culinary", "meal", "hobby"],
    "gardening": ["plant", "nature", "outdoor_activity", "hobby", "flower", "garden"],
    "yoga": ["exercise", "meditation", "fitness", "health", "relaxation", "flexibility"],
    "running": ["exercise", "fitness", "marathon", "jogging", "sport", "cardio"],
    "swimming": ["water", "exercise", "pool", "sport", "fitness", "aquatic"],
    "piano": ["music", "instrument", "classical", "keyboard", "musical", "performance"],
    "violin": ["music", "instrument", "classical", "string", "orchestra", "musical"],
    "guitar": ["music", "instrument", "string", "acoustic", "musical", "performance"],
    "photography": ["camera", "art", "visual", "creative", "hobby", "image"],
    "travel": ["adventure", "tourism", "explore", "vacation", "journey", "destination"],
    "volunteering": ["charity", "community", "altruism", "helping", "service", "kindness"],
    "meditation": ["mindfulness", "relaxation", "mental_health", "calm", "peace", "spiritual"],
    "cycling": ["bicycle", "exercise", "sport", "fitness", "outdoor_activity", "commute"],
    "dancing": ["music", "art", "performance", "exercise", "rhythm", "movement"],
    "writing": ["literature", "creative", "author", "storytelling", "hobby", "expression"],
}

_HYPERNYM_MAP = {
    # Places
    "yellowstone": "national_park",
    "yosemite": "national_park",
    "grand canyon": "national_park",
    "zion": "national_park",
    "glacier": "national_park",
    "everglades": "national_park",
    "sequoia": "national_park",
    "acadia": "national_park",
    "denali": "national_park",
    "rocky mountain": "national_park",
    "paris": "city",
    "london": "city",
    "tokyo": "city",
    "new york": "city",
    # People / cultural
    "bach": "classical_music",
    "beethoven": "classical_music",
    "mozart": "classical_music",
    "chopin": "classical_music",
    "picasso": "visual_art",
    "monet": "visual_art",
    "van gogh": "visual_art",
    "shakespeare": "literature",
    "hemingway": "literature",
    "tolkien": "literature",
    # Languages
    "python": "programming_language",
    "javascript": "programming_language",
    "rust": "programming_language",
    "java": "programming_language",
    "typescript": "programming_language",
    # Animals
    "labrador": "dog",
    "golden retriever": "dog",
    "siamese": "cat",
    "persian": "cat",
}

_VERB_NOMINALIZATIONS = {
    "camping": "camping trip",
    "hiking": "hiking trip",
    "fishing": "fishing trip",
    "traveling": "travel journey",
    "travelling": "travel journey",
    "reading": "reading hobby",
    "cooking": "cooking activity",
    "painting": "painting activity",
    "gardening": "gardening hobby",
    "swimming": "swimming activity",
    "running": "running exercise",
    "cycling": "cycling activity",
    "dancing": "dancing activity",
    "writing": "writing activity",
    "singing": "singing performance",
    "climbing": "climbing adventure",
    "surfing": "surfing activity",
    "skiing": "skiing trip",
    "snowboarding": "snowboarding trip",
    "volunteering": "volunteer work",
    "meditating": "meditation practice",
    "studying": "study session",
    "practicing": "practice session",
    "teaching": "teaching session",
    "learning": "learning experience",
    "exploring": "exploration adventure",
}

# Patterns: ("verb/trigger word", "nominalization")
_VERB_PATTERN = re.compile(
    r'\b(?:went|goes|go|enjoys?|loves?|likes?|started|began|tries?|tried)\s+'
    r'(\w+ing)\b',
    re.IGNORECASE,
)


@dataclass
class EnrichmentResult:
    concepts: list[str] = field(default_factory=list)
    comet_inferences: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    logic_expansions: list[str] = field(default_factory=list)
    enriched_content: str = ""
    model_versions: dict = field(default_factory=dict)


class FPAFilter:
    """Cosine similarity noise filter for enrichment terms."""

    def __init__(self, embedding_engine) -> None:
        self._engine = embedding_engine

    def filter(
        self,
        original_embedding: bytes,
        enrichment_texts: list[str],
        threshold: float,
    ) -> list[str]:
        if not enrichment_texts:
            return []

        original_vec = np.frombuffer(original_embedding, dtype=np.float32)
        kept = []

        for text in enrichment_texts:
            encoded = self._engine.encode_query(text)
            if encoded is None:
                continue
            text_vec = np.frombuffer(encoded, dtype=np.float32)
            similarity = float(np.dot(original_vec, text_vec))
            if similarity >= threshold:
                kept.append(text)

        rejected = len(enrichment_texts) - len(kept)
        if rejected > 0:
            logger.info("FPA filter rejected %d/%d enrichment terms (threshold=%.2f)",
                        rejected, len(enrichment_texts), threshold)

        return kept


def _extract_terms(content: str) -> list[str]:
    """Extract content-bearing nouns/verbs via simple tokenization."""
    tokens = re.findall(r'[a-zA-Z]+', content.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]


class ConceptNetExpander:
    """Query ConceptNet for related concepts."""

    def __init__(self) -> None:
        self._conceptnet_lite = None
        self._lite_available: Optional[bool] = None
        self._http_available: Optional[bool] = None

    def _try_lite(self, term: str, relations: list[str], min_weight: float) -> list[str]:
        """Try conceptnet_lite local SQLite database."""
        if self._lite_available is False:
            return []
        try:
            if self._conceptnet_lite is None:
                import conceptnet_lite
                self._conceptnet_lite = conceptnet_lite
            self._lite_available = True

            results = []
            for rel in relations:
                edges = self._conceptnet_lite.query(
                    node=f"/c/en/{term}", rel=f"/r/{rel}", limit=10
                )
                for edge in edges:
                    if edge.weight >= min_weight:
                        # Extract the end node label
                        end = edge.end.label if hasattr(edge.end, 'label') else str(edge.end)
                        results.append(end.replace(" ", "_"))
            return results
        except (ImportError, Exception):
            self._lite_available = False
            return []

    def _try_http(self, term: str, relations: list[str], min_weight: float) -> list[str]:
        """Try ConceptNet HTTP API. Disabled by default — too slow for batch use."""
        # HTTP API is 5s/request — unusable for batch enrichment
        # Skip straight to hardcoded. Enable via self._http_available = None to test.
        if self._http_available is not True:
            return []
        try:
            import urllib.request
            import urllib.error

            results = []
            for rel in relations:
                url = f"http://api.conceptnet.io/query?node=/c/en/{term}&rel=/r/{rel}&limit=10"
                try:
                    req = urllib.request.Request(url, headers={"Accept": "application/json"})
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        data = json.loads(resp.read().decode())
                    for edge in data.get("edges", []):
                        if edge.get("weight", 0) >= min_weight:
                            end = edge.get("end", {}).get("label", "")
                            if end:
                                results.append(end.replace(" ", "_"))
                except (urllib.error.URLError, TimeoutError, OSError):
                    continue
            self._http_available = True if results else None
            return results
        except Exception:
            self._http_available = False
            return []

    def _try_hardcoded(self, term: str) -> list[str]:
        """Fall back to hardcoded expansions."""
        return list(HARDCODED_EXPANSIONS.get(term, []))

    def expand(self, content: str, settings: Settings) -> list[str]:
        relations = [r.strip() for r in settings.CONCEPTNET_RELATIONS.split(",")]
        min_weight = settings.CONCEPTNET_MIN_EDGE_WEIGHT
        max_terms = settings.CONCEPTNET_MAX_TERMS

        terms = _extract_terms(content)
        all_concepts: list[str] = []
        seen: set[str] = set()

        for term in terms:
            # Try sources in order: lite → HTTP → hardcoded
            concepts = self._try_lite(term, relations, min_weight)
            if not concepts:
                concepts = self._try_http(term, relations, min_weight)
            if not concepts:
                concepts = self._try_hardcoded(term)

            for c in concepts:
                if c not in seen:
                    seen.add(c)
                    all_concepts.append(c)

            if len(all_concepts) >= max_terms:
                break

        return all_concepts[:max_terms]


class CometInferencer:
    """COMET-BART commonsense inference engine."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = None
        self._unavailable = False

    def _ensure_model(self, model_name: str) -> bool:
        if self._model is not None:
            return True
        if self._unavailable:
            return False
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self._device)
            self._model.eval()
            return True
        except (ImportError, Exception) as e:
            logger.warning("COMET model unavailable: %s", e)
            self._unavailable = True
            return False

    def _extract_predicates(self, content: str) -> list[str]:
        """Extract sentences with named subjects and verbs."""
        sentences = re.split(r'[.!?]+', content)
        predicates = []
        # Match sentences that start with a capitalized word (potential named subject)
        # followed by a verb-like word
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            # Simple heuristic: sentence has a proper noun or pronoun + verb
            if re.match(r'^[A-Z][a-z]+\s+\w+', sent) or re.match(r'^(?:He|She|They|I|We)\s+', sent):
                predicates.append(sent)
        return predicates if predicates else [content.strip()]

    def infer(self, content: str, settings: Settings) -> list[str]:
        if not self._ensure_model(settings.COMET_MODEL):
            return []

        import torch

        relations = [r.strip() for r in settings.COMET_RELATIONS.split(",")]
        num_beams = settings.COMET_NUM_BEAMS
        top_k = settings.COMET_TOP_K_PER_RELATION
        min_confidence = settings.COMET_MIN_CONFIDENCE

        predicates = self._extract_predicates(content)
        all_inferences: list[str] = []
        seen: set[str] = set()

        for predicate in predicates:
            for relation in relations:
                prompt = f"{predicate} {relation} [GEN]"
                input_ids = self._tokenizer(
                    prompt, return_tensors="pt", padding=True, truncation=True
                ).input_ids.to(self._device)

                with torch.no_grad():
                    outputs = self._model.generate(
                        input_ids,
                        num_beams=num_beams,
                        num_return_sequences=min(top_k, num_beams),
                        max_length=64,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )

                # Compute per-sequence scores via softmax over sequence scores
                if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
                    scores = torch.softmax(outputs.sequences_scores, dim=0)
                else:
                    scores = torch.ones(len(outputs.sequences)) / len(outputs.sequences)

                for seq, score in zip(outputs.sequences, scores):
                    text = self._tokenizer.decode(seq, skip_special_tokens=True).strip()
                    if not text or text.lower() == "none":
                        continue
                    if float(score) >= min_confidence and text not in seen:
                        seen.add(text)
                        all_inferences.append(text)

                # Cap at 3 per relation
                if len(all_inferences) >= 9:
                    return all_inferences[:9]

        return all_inferences[:9]


class Doc2QueryExpander:
    """Synthetic query generation via doc2query."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = None
        self._unavailable = False

    def _ensure_model(self, model_name: str) -> bool:
        if self._model is not None:
            return True
        if self._unavailable:
            return False
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self._device)
            self._model.eval()
            return True
        except (ImportError, Exception) as e:
            logger.warning("Doc2Query model unavailable: %s", e)
            self._unavailable = True
            return False

    def _token_overlap(self, a: str, b: str) -> float:
        """Compute token overlap ratio between two strings."""
        tokens_a = set(a.lower().split())
        tokens_b = set(b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))

    def expand(self, content: str, settings: Settings) -> list[str]:
        if not self._ensure_model(settings.DOC2QUERY_MODEL):
            return []

        import torch

        num_queries = settings.DOC2QUERY_NUM_QUERIES

        input_ids = self._tokenizer(
            content, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).input_ids.to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                input_ids,
                num_beams=num_queries * 2,
                num_return_sequences=num_queries * 2,  # generate extra to filter
                max_length=64,
                do_sample=False,
            )

        queries: list[str] = []
        seen: set[str] = set()

        for seq in outputs:
            query = self._tokenizer.decode(seq, skip_special_tokens=True).strip()
            if not query:
                continue
            query_lower = query.lower()
            if query_lower in seen:
                continue
            if self._token_overlap(query, content) > 0.8:
                continue
            seen.add(query_lower)
            queries.append(query)
            if len(queries) >= num_queries:
                break

        return queries


class LogicExpander:
    """Rule-based natural logic expansion. No external models."""

    def expand(self, content: str) -> list[str]:
        expansions: list[str] = []
        content_lower = content.lower()

        # Hypernym lifting
        for term, hypernym in _HYPERNYM_MAP.items():
            if term in content_lower:
                if hypernym not in expansions:
                    expansions.append(hypernym)

        # Verb nominalization: "went camping" → "camping trip"
        for match in _VERB_PATTERN.finditer(content):
            gerund = match.group(1).lower()
            nominalization = _VERB_NOMINALIZATIONS.get(gerund)
            if nominalization and nominalization not in expansions:
                expansions.append(nominalization)

        # Also check for standalone gerunds in content
        tokens = set(re.findall(r'\b\w+ing\b', content_lower))
        for gerund in tokens:
            nominalization = _VERB_NOMINALIZATIONS.get(gerund)
            if nominalization and nominalization not in expansions:
                expansions.append(nominalization)

        return expansions


class EnrichmentPipeline:
    """Orchestrator for index-time enrichment techniques."""

    def __init__(self, settings: Settings, embedding_engine=None) -> None:
        self._settings = settings
        self._embedding_engine = embedding_engine
        self._fpa: Optional[FPAFilter] = None
        self._conceptnet: Optional[ConceptNetExpander] = None
        self._comet: Optional[CometInferencer] = None
        self._doc2query: Optional[Doc2QueryExpander] = None
        self._logic: Optional[LogicExpander] = None

    def _get_fpa(self) -> Optional[FPAFilter]:
        if self._fpa is None and self._embedding_engine is not None:
            self._fpa = FPAFilter(self._embedding_engine)
        return self._fpa

    def _get_conceptnet(self) -> ConceptNetExpander:
        if self._conceptnet is None:
            self._conceptnet = ConceptNetExpander()
        return self._conceptnet

    def _get_comet(self) -> CometInferencer:
        if self._comet is None:
            self._comet = CometInferencer()
        return self._comet

    def _get_doc2query(self) -> Doc2QueryExpander:
        if self._doc2query is None:
            self._doc2query = Doc2QueryExpander()
        return self._doc2query

    def _get_logic(self) -> LogicExpander:
        if self._logic is None:
            self._logic = LogicExpander()
        return self._logic

    def _apply_fpa(self, embedding: bytes, texts: list[str], threshold: float) -> list[str]:
        fpa = self._get_fpa()
        if fpa is None or embedding is None:
            return texts
        return fpa.filter(embedding, texts, threshold)

    def enrich(self, content: str, embedding: bytes, settings: Settings) -> EnrichmentResult:
        result = EnrichmentResult()

        if len(content) < settings.ENRICHMENT_MIN_CONTENT_LENGTH:
            result.enriched_content = content
            return result

        threshold = settings.FPA_SIMILARITY_THRESHOLD

        # ConceptNet expansion
        if settings.CONCEPTNET_ENRICHMENT_ENABLED:
            try:
                concepts = self._get_conceptnet().expand(content, settings)
                concepts = self._apply_fpa(embedding, concepts, threshold)
                result.concepts = concepts
            except Exception as e:
                logger.warning("ConceptNet enrichment failed: %s", e)

        # COMET commonsense inference
        if settings.COMET_ENRICHMENT_ENABLED:
            try:
                inferences = self._get_comet().infer(content, settings)
                inferences = self._apply_fpa(embedding, inferences, threshold)
                result.comet_inferences = inferences
                if self._get_comet()._model is not None:
                    result.model_versions["comet"] = settings.COMET_MODEL
            except Exception as e:
                logger.warning("COMET enrichment failed: %s", e)

        # Doc2Query synthetic queries
        if settings.DOC2QUERY_ENRICHMENT_ENABLED:
            try:
                queries = self._get_doc2query().expand(content, settings)
                queries = self._apply_fpa(embedding, queries, threshold)
                result.queries = queries
                if self._get_doc2query()._model is not None:
                    result.model_versions["doc2query"] = settings.DOC2QUERY_MODEL
            except Exception as e:
                logger.warning("Doc2Query enrichment failed: %s", e)

        # Logic expansion (no external deps, no FPA needed — these are structural)
        if settings.LOGIC_ENRICHMENT_ENABLED:
            try:
                expansions = self._get_logic().expand(content)
                result.logic_expansions = expansions
            except Exception as e:
                logger.warning("Logic enrichment failed: %s", e)

        # Build enriched content
        all_terms = (
            result.concepts
            + result.comet_inferences
            + result.queries
            + result.logic_expansions
        )

        if all_terms:
            result.enriched_content = content + "\n[enrichment] " + " | ".join(all_terms)
        else:
            result.enriched_content = content

        return result
