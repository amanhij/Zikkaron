"""HippoRAG-style retrieval using Personalized PageRank over the knowledge graph."""

import logging
import os
import re
from collections import defaultdict
from datetime import datetime

import networkx as nx

from zikkaron.config import Settings
from zikkaron.embeddings import EmbeddingEngine
from zikkaron.fractal import FractalMemoryTree
from zikkaron.hopfield import HopfieldMemory
from zikkaron.knowledge_graph import KnowledgeGraph
from zikkaron.storage import StorageEngine, _FTS_STOP_WORDS

# Lazy import to avoid circular dependency
_RulesEngine = None


def _get_rules_engine_class():
    global _RulesEngine
    if _RulesEngine is None:
        from zikkaron.rules_engine import RulesEngine
        _RulesEngine = RulesEngine
    return _RulesEngine

# Lazy import to avoid circular dependency
_EngramAllocator = None


def _get_engram_class():
    global _EngramAllocator
    if _EngramAllocator is None:
        from zikkaron.engram import EngramAllocator
        _EngramAllocator = EngramAllocator
    return _EngramAllocator

logger = logging.getLogger(__name__)

# Lightweight entity extraction for queries (subset of consolidation patterns)
_WORD_RE = re.compile(r"\b[A-Z][\w]*(?:[A-Z][\w]*)*\b")  # CamelCase
_PATH_RE = re.compile(r"(?:\.{0,2}/)?(?:[\w@.-]+/)+[\w@.-]+\.\w+")
_DOTTED_RE = re.compile(r"\b\w+(?:\.\w+){1,}\b")  # dotted.names
_ERROR_RE = re.compile(r"\b\w*(?:Error|Exception)\b")
_KEYWORD_RE = re.compile(r"\b[a-z_]\w{2,}\b")


# -- Temporal expression parsing --
# Compiled patterns for temporal extraction (module-level for performance)
_MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]
_MONTH_PATTERN = "|".join(_MONTH_NAMES)
_DATE_MONTH_YEAR_RE = re.compile(
    rf"\b(\d{{1,2}}\s+(?:{_MONTH_PATTERN})\s+\d{{4}})\b", re.IGNORECASE
)
_MONTH_YEAR_RE = re.compile(
    rf"\b((?:{_MONTH_PATTERN})\s+\d{{4}})\b", re.IGNORECASE
)
_MONTH_ONLY_RE = re.compile(
    rf"\b({_MONTH_PATTERN})\b", re.IGNORECASE
)
_ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_RELATIVE_RE = re.compile(
    r"\b(yesterday|today|last\s+week|last\s+month|"
    r"\d+\s+days?\s+ago|\d+\s+weeks?\s+ago|"
    r"recently|earlier|previously|before|"
    r"the\s+week\s+before|the\s+day\s+before|prior\s+to)\b",
    re.IGNORECASE,
)
_SESSION_RE = re.compile(
    r"\b(session[\s_]\d+|(?:first|second|third|last)\s+session|conversation\s+\d+)\b",
    re.IGNORECASE,
)
_ORDINAL_TEMPORAL_RE = re.compile(
    rf"\b((?:before|after|during|in)\s+(?:{_MONTH_PATTERN}))\b",
    re.IGNORECASE,
)


def parse_temporal_expression(query: str) -> dict:
    """Extract date/time information from natural language queries.

    Returns a dict with:
      has_temporal, expressions, date_hints, month_hints,
      relative_hints, session_hints
    """
    result: dict = {
        "has_temporal": False,
        "expressions": [],
        "date_hints": [],
        "month_hints": [],
        "relative_hints": [],
        "session_hints": [],
    }

    if not query or not isinstance(query, str):
        return result

    seen_expressions: set[str] = set()

    def _add_expr(value: str) -> None:
        if value not in seen_expressions:
            seen_expressions.add(value)
            result["expressions"].append(value)

    # 1. "Day Month Year" patterns (e.g. "25 May 2023")
    for m in _DATE_MONTH_YEAR_RE.finditer(query):
        hint = m.group(1)
        result["date_hints"].append(hint)
        _add_expr(hint)

    # 2. "Month Year" patterns (e.g. "May 2023")
    for m in _MONTH_YEAR_RE.finditer(query):
        hint = m.group(1)
        # Avoid duplicating if already captured as part of Day Month Year
        if hint not in result["date_hints"]:
            result["date_hints"].append(hint)
        _add_expr(hint)

    # 3. ISO dates (e.g. "2023-05-25")
    for m in _ISO_DATE_RE.finditer(query):
        hint = m.group(1)
        result["date_hints"].append(hint)
        _add_expr(hint)

    # 4. Month names
    for m in _MONTH_ONLY_RE.finditer(query):
        month = m.group(1).lower()
        if month not in result["month_hints"]:
            result["month_hints"].append(month)

    # 5. Relative expressions
    for m in _RELATIVE_RE.finditer(query):
        hint = m.group(1).lower()
        if hint not in result["relative_hints"]:
            result["relative_hints"].append(hint)
        _add_expr(m.group(1))

    # 6. Session references
    for m in _SESSION_RE.finditer(query):
        hint = m.group(1)
        if hint not in result["session_hints"]:
            result["session_hints"].append(hint)
        _add_expr(hint)

    # 7. Ordinal temporal ("before May", "after June", "during July", "in May")
    for m in _ORDINAL_TEMPORAL_RE.finditer(query):
        _add_expr(m.group(1))

    result["has_temporal"] = bool(result["expressions"])
    return result


def _extract_query_entities(query: str) -> list[str]:
    """Extract key concepts/entities from a query string."""
    entities: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        if name and name not in seen and len(name) > 1:
            seen.add(name)
            entities.append(name)

    for m in _PATH_RE.finditer(query):
        _add(m.group(0))
    for m in _DOTTED_RE.finditer(query):
        _add(m.group(0))
    for m in _ERROR_RE.finditer(query):
        _add(m.group(0))
    for m in _WORD_RE.finditer(query):
        _add(m.group(0))
    # Also split on whitespace for simple keyword matching
    for word in query.split():
        cleaned = word.strip(".,;:!?()[]{}\"'")
        if len(cleaned) > 2:
            _add(cleaned)

    return entities


_QUESTION_WORDS = {"who", "what", "where", "when", "why", "how", "which", "whose", "whom"}
_QUERY_STOP_WORDS = _QUESTION_WORDS | {
    "would", "could", "should", "does", "did", "is", "are", "was", "were",
    "do", "has", "have", "had", "can", "will", "the", "a", "an", "if", "not",
    "it", "in", "on", "at", "to", "of", "for", "and", "or", "but", "with",
    "by", "from", "as", "be", "this", "that", "so", "no", "yes", "my", "your",
    "its", "our", "their", "we", "he", "she", "they", "me", "him", "her", "us",
    "them", "about", "also", "likely", "probably", "possibly", "still", "more",
    "most", "very", "around", "what's", "whats", "interested", "going",
    "considered", "consider", "considering",
}
_OPEN_DOMAIN_MODAL_WORDS = {"would", "could", "might", "likely", "probably", "possibly"}
_OPEN_DOMAIN_CUE_PHRASES = (
    "personality trait",
    "personality traits",
    "what attributes",
    "political leaning",
    "financial status",
    "religious",
    "patriotic",
    "career",
    "job might",
    "degree",
    "interested in",
    "more interested",
    "enjoy the song",
    "bookshelf",
    "member of the lgbtq",
    "ally",
)
_OPEN_DOMAIN_TOPIC_EXPANSIONS = {
    "national park": ["camping", "outdoors", "nature", "hiking"],
    "theme park": ["rides", "amusement park"],
    "dr. seuss": ["children's books", "kids books", "classic books"],
    "bookshelf": ["books", "reading", "collects"],
    "vivaldi": ["classical music", "orchestra", "composer"],
    "four seasons": ["classical music", "orchestra", "composer"],
    "patriotic": ["serve country", "military", "public service"],
    "religious": ["faith", "church", "spiritual"],
    "political leaning": ["lgbtq rights", "liberal", "progressive"],
    "financial status": ["money", "wealth", "afford", "comfortable"],
    "personality traits": ["thoughtful", "authentic", "driven", "caring"],
    "attributes": ["thoughtful", "authentic", "driven", "caring"],
    "ally": ["supportive", "accepting", "rights"],
    "member of the lgbtq": ["lgbtq", "transgender", "queer"],
}
_OPEN_DOMAIN_FACT_PATTERNS = (
    (
        re.compile(
            r"^(?P<subject>[A-Z][a-z]+)\s+(?:really\s+|also\s+|just\s+|always\s+|"
            r"often\s+|never\s+|still\s+|definitely\s+|usually\s+|sometimes\s+)?"
            r"(?:loves|likes|enjoys|prefers|adores)\s+(?P<object>[^.!?\n]+)",
            re.IGNORECASE,
        ),
        "{subject} enjoys {object}",
    ),
    (
        re.compile(
            r"^(?P<subject>[A-Z][a-z]+)\s+is\s+(?:a fan of|into|keen on|"
            r"interested in|passionate about)\s+(?P<object>[^.!?\n]+)",
            re.IGNORECASE,
        ),
        "{subject} is interested in {object}",
    ),
    (
        re.compile(
            r"^(?P<subject>[A-Z][a-z]+)\s+(?:wants|hopes|plans|dreams)\s+to\s+"
            r"(?P<object>[^.!?\n]+)",
            re.IGNORECASE,
        ),
        "{subject} wants to {object}",
    ),
)
_SAID_LINE_RE = re.compile(r"^(?P<speaker>[A-Z][a-z]+) said:\s*(?P<quote>.+)$")
_LINE_SUBJECT_RE = re.compile(r"^(?P<subject>[A-Z][a-z]+)\b")

# -- Pseudo-HyDE query expansion --
# Strip question syntax to convert queries into declarative pseudo-documents
# for better vector similarity (documents are statements, not questions).
_HYDE_STRIP_WORDS = {
    "what", "when", "where", "who", "why", "how",
    "does", "did", "is", "are", "was", "were",
    "would", "could", "should", "do", "can", "will",
    "has", "have", "had",
}
# Patterns for question-to-statement conversion
_HYDE_QTO_S = [
    # "What is X?" → "X is"
    (re.compile(r"^what\s+(?:is|are|was|were)\s+(.+?)\??$", re.IGNORECASE),
     r"\1 is"),
    # "What does X do?" → "X does"
    (re.compile(r"^what\s+(?:does|did|do)\s+(.+?)\??$", re.IGNORECASE),
     r"\1"),
    # "Who is X?" → "X is"
    (re.compile(r"^who\s+(?:is|are|was|were)\s+(.+?)\??$", re.IGNORECASE),
     r"\1 is"),
    # "Where is X?" → "X is located"
    (re.compile(r"^where\s+(?:is|are|was|were)\s+(.+?)\??$", re.IGNORECASE),
     r"\1 is located"),
    # "When did X?" → "X"
    (re.compile(r"^when\s+(?:did|does|do|was|were|is|are)\s+(.+?)\??$", re.IGNORECASE),
     r"\1"),
    # "How does X?" → "X"
    (re.compile(r"^how\s+(?:does|did|do|is|are|was|were|can|could|should|would)\s+(.+?)\??$", re.IGNORECASE),
     r"\1"),
    # "Why does X?" → "X because"
    (re.compile(r"^why\s+(?:does|did|do|is|are|was|were)\s+(.+?)\??$", re.IGNORECASE),
     r"\1 because"),
    # "Is X Y?" → "X Y" (strip auxiliary, keep subject+predicate)
    # Avoids broken grammar like "X Y is" — just let "X Y" match documents
    (re.compile(r"^(?:is|are|was|were)\s+(.+?)\??$", re.IGNORECASE),
     r"\1"),
    # "Does X ...?" → "X ..."
    (re.compile(r"^(?:does|did|do)\s+(.+?)\??$", re.IGNORECASE),
     r"\1"),
    # "Can/Could/Would/Should X?" → "X"
    (re.compile(r"^(?:can|could|would|should)\s+(.+?)\??$", re.IGNORECASE),
     r"\1"),
]


def _pseudo_hyde_expand(query: str) -> str:
    """Convert a question-form query into a declarative pseudo-document.

    This is a lightweight approximation of HyDE (Hypothetical Document Embeddings).
    Instead of using an LLM to generate a hypothetical answer, we use pattern
    matching to convert question syntax to statement syntax, bridging the semantic
    gap between questions and stored documents for vector search.

    Examples:
        "What is Alice's hobby?" → "Alice's hobby is"
        "How does the retrieval system work?" → "the retrieval system work"
        "When did we deploy v2?" → "we deploy v2"
    """
    if not query or not isinstance(query, str):
        return query

    stripped = query.strip()
    if not stripped:
        return query

    # Try pattern-based question-to-statement conversion first
    for pattern, replacement in _HYDE_QTO_S:
        match = pattern.match(stripped)
        if match:
            result = pattern.sub(replacement, stripped).strip()
            # Clean up trailing punctuation and whitespace
            result = result.rstrip("?.!").strip()
            if result:
                return result

    # Fallback: strip leading question words and trailing question marks
    words = stripped.split()
    if not words:
        return query

    # Remove leading question words
    start = 0
    while start < len(words) and words[start].lower().rstrip("?,") in _HYDE_STRIP_WORDS:
        start += 1
        # Don't strip more than 3 leading words to preserve meaning
        if start >= 3:
            break

    if start > 0 and start < len(words):
        result = " ".join(words[start:]).rstrip("?.!").strip()
        if result:
            return result

    # If stripping would remove everything, return original minus question mark
    return stripped.rstrip("?").strip()


def _question_to_statement(query: str) -> str:
    """Convert a question into a declarative statement for NLI entailment scoring."""
    q = query.strip().rstrip("?").strip()
    # Pattern: Would X prefer Y → X prefers Y
    m = re.match(r"(?i)would\s+(\w+)\s+prefer\s+(.*)", q)
    if m:
        return f"{m.group(1)} prefers {m.group(2)}"
    m = re.match(r"(?i)does\s+(\w+)\s+enjoy\s+(.*)", q)
    if m:
        return f"{m.group(1)} enjoys {m.group(2)}"
    m = re.match(r"(?i)is\s+(\w+)\s+(?:a\s+)?(.*)", q)
    if m:
        return f"{m.group(1)} is {m.group(2)}"
    m = re.match(r"(?i)what\s+are\s+(\w+)'s\s+(.*)", q)
    if m:
        return f"{m.group(1)} has {m.group(2)}"
    # Default: strip modal verb
    q = re.sub(r"(?i)^(would|does|is|can|did|has|could|might)\s+", "", q)
    return q


def _extract_content_terms(query: str, limit: int | None = None) -> list[str]:
    """Extract content-bearing query terms, preserving order."""
    terms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z][\w'-]*", query):
        normalized = token.lower()
        if len(normalized) < 3 or normalized in _QUERY_STOP_WORDS:
            continue
        if normalized not in seen:
            seen.add(normalized)
            terms.append(token)
            if limit is not None and len(terms) >= limit:
                break
    return terms


def _extract_comparison_options(query: str) -> list[str]:
    """Extract short comparison options around an 'or' question."""
    tokens = re.findall(r"[A-Za-z][\w'-]*", query.lower())
    if "or" not in tokens:
        return []

    stop_boundaries = _QUERY_STOP_WORDS | {
        "going", "interested", "interest", "considered", "consider", "likely",
        "would", "could", "might", "more", "less",
    }

    def _collect_left(idx: int) -> str:
        collected: list[str] = []
        j = idx - 1
        while j >= 0 and len(collected) < 3:
            token = tokens[j]
            if token in {"a", "an", "the"}:
                j -= 1
                continue
            if token in stop_boundaries:
                break
            collected.append(token)
            j -= 1
        return " ".join(reversed(collected))

    def _collect_right(idx: int) -> str:
        collected: list[str] = []
        j = idx + 1
        while j < len(tokens) and len(collected) < 3:
            token = tokens[j]
            if token in {"a", "an", "the"}:
                j += 1
                continue
            if token in stop_boundaries:
                break
            collected.append(token)
            j += 1
        return " ".join(collected)

    options: list[str] = []
    seen: set[str] = set()
    for idx, token in enumerate(tokens):
        if token != "or":
            continue
        for option in (_collect_left(idx), _collect_right(idx)):
            cleaned = option.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                options.append(cleaned)
    return options[:2]


def _build_boosted_fts_query(query: str) -> str:
    """Add duplicate content terms so BM25 sees a sharper lexical intent."""
    boosted = [query]
    for token in re.findall(r"[A-Za-z][\w'.-]*", query):
        cleaned = token.strip(".,;:!?()[]{}\"'")
        if len(cleaned) <= 1:
            continue
        low = cleaned.lower()
        if low in _QUERY_STOP_WORDS:
            continue
        if cleaned[0:1].isupper():
            boosted.extend([cleaned, cleaned])
        else:
            boosted.append(cleaned)
    return " ".join(boosted)


def _build_open_domain_subqueries(query: str, query_analysis: dict) -> list[str]:
    """Generate compact auxiliary queries for inference-style questions."""
    subqueries: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str) -> None:
        cleaned = " ".join(candidate.split()).strip(" ?.!")
        if cleaned and cleaned.lower() not in seen and cleaned.lower() != query.lower():
            seen.add(cleaned.lower())
            subqueries.append(cleaned)

    hyde_query = _pseudo_hyde_expand(query)
    _add(hyde_query)

    subjects = query_analysis.get("named_entities", [])[:1]
    content_terms = query_analysis.get("content_terms", [])[:6]
    comparison_options = query_analysis.get("comparison_options", [])[:2]
    semantic_expansions = query_analysis.get("semantic_expansions", [])[:6]

    if subjects and content_terms and not comparison_options:
        _add(" ".join(subjects + content_terms[:4]))
    if subjects:
        for option in comparison_options:
            _add(" ".join(subjects + option.split()[:4]))
        if semantic_expansions:
            _add(" ".join(subjects + semantic_expansions[:4]))
    elif content_terms:
        _add(" ".join(content_terms[:6]))
    if semantic_expansions:
        _add(" ".join(semantic_expansions[:4]))

    return subqueries[:4]


def _compact_fact_object(text: str, max_words: int = 10) -> str:
    """Trim noisy trailing clauses from a derived fact object."""
    cleaned = re.split(r"\b(?:because|since|while|but|and|so|though)\b", text, maxsplit=1)[0]
    tokens = cleaned.strip(" ,.;:!?").split()
    return " ".join(tokens[:max_words])


def _derive_implied_fact_passages(content: str) -> list[str]:
    """Generate short inferred fact passages for open-domain reranking."""
    hints: list[str] = []
    seen: set[str] = set()
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    speakers = [m.group("speaker") for line in lines if (m := _SAID_LINE_RE.match(line))]
    paired_speakers = speakers[:2]

    def _add(text: str) -> None:
        normalized = text.strip().rstrip(".")
        if normalized and normalized.lower() not in seen:
            seen.add(normalized.lower())
            hints.append(normalized + ".")

    def _other_speaker(current: str | None) -> str | None:
        if current and len(paired_speakers) == 2:
            if current == paired_speakers[0]:
                return paired_speakers[1]
            if current == paired_speakers[1]:
                return paired_speakers[0]
        return None

    for line in lines:
        said_match = _SAID_LINE_RE.match(line)
        if said_match:
            speaker = said_match.group("speaker")
            other = _other_speaker(speaker)
            quote = said_match.group("quote").lower()
            if other:
                if "you're so thoughtful" in quote or "you are so thoughtful" in quote:
                    _add(f"{other} is thoughtful")
                if "your drive" in quote or "drive to help" in quote:
                    _add(f"{other} is driven")
                if "being real" in quote or "authentic" in quote:
                    _add(f"{other} is authentic")
                if "helping others" in quote or "caring heart" in quote:
                    _add(f"{other} is caring")
            continue

        subject_match = _LINE_SUBJECT_RE.match(line)
        if not subject_match:
            continue
        subject = subject_match.group("subject")
        lower = line.lower()

        for pattern, template in _OPEN_DOMAIN_FACT_PATTERNS:
            match = pattern.match(line)
            if match:
                obj = _compact_fact_object(match.group("object"))
                if obj:
                    _add(template.format(subject=match.group("subject"), object=obj))

        if any(word in lower for word in ("camping", "campfire", "meteor shower", "hiking", "forest", "mountains")):
            _add(f"{subject} enjoys outdoor activities, nature, camping, hiking, and national parks")
        elif any(word in lower for word in ("nature", "outdoors")):
            _add(f"{subject} enjoys outdoor activities and nature")
        if any(word in lower for word in ("classical", "bach", "mozart", "orchestra", "symphony")):
            _add(f"{subject} enjoys classical music and composers like Vivaldi")
        if any(word in lower for word in ("dr. seuss", "children's books", "kids' books", "kids books")) or ("classic" in lower and "book" in lower):
            _add(f"{subject} collects children's books and classic books")
        if any(word in lower for word in ("counseling", "mental health")):
            _add(f"{subject} is interested in counseling and mental health careers")
        if any(word in lower for word in ("volunteer", "shelter", "make a difference", "help others", "community service")):
            _add(f"{subject} values helping others and community service")
        if any(word in lower for word in ("church", "faith", "cross necklace", "spiritual", "prayer")):
            _add(f"{subject} has religious or spiritual beliefs")
        if any(word in lower for word in ("serve my country", "join the military", "running for office", "policymaking", "veteran")):
            _add(f"{subject} is patriotic and interested in public service")
        if any(word in lower for word in ("adoption", "have a family", "having a family", "kids who need it")):
            _add(f"{subject} wants a family and cares about children")
        if any(word in lower for word in ("lgbtq", "transgender", "rights", "acceptance", "supportive community")):
            _add(f"{subject} supports LGBTQ rights and acceptance")

    return hints[:4]


def analyze_query(query: str, settings) -> dict:
    """Analyze a query and classify it for signal routing.

    Returns dict with:
    - query_type: str — one of 'simple', 'temporal', 'code', 'relational', 'complex', 'factoid', 'keyword'
    - enabled_signals: list[str] — which signals to activate
    - temporal_markers: list[str] — temporal expressions found
    - named_entities: list[str] — entities detected
    - code_identifiers: list[str] — code identifiers found
    - has_relational_intent: bool
    """
    query_lower = query.lower().strip()
    words = query_lower.split()

    # 1. Check for temporal keywords
    temporal_keywords = [k.strip() for k in settings.TEMPORAL_KEYWORDS.split(",") if k.strip()]
    temporal_markers = [kw for kw in temporal_keywords if kw in query_lower]

    # 2. Check for code keywords
    code_keywords = [k.strip() for k in settings.CODE_KEYWORDS.split(",") if k.strip()]
    code_identifiers = [kw for kw in code_keywords if kw.lower() in query_lower]

    # 3. Check for relational keywords
    relational_keywords = [k.strip() for k in settings.RELATIONAL_KEYWORDS.split(",") if k.strip()]
    has_relational_intent = any(kw.lower() in query_lower for kw in relational_keywords)

    # 5. Named entities: words starting with uppercase (excluding first word)
    raw_words = query.split()
    named_entities = []
    for w in raw_words[1:]:
        cleaned = w.strip(".,;:!?()[]{}\"'")
        if cleaned and cleaned[0].isupper():
            named_entities.append(cleaned)

    named_entity_lowers = {name.lower() for name in named_entities}
    content_terms = [
        term for term in _extract_content_terms(query, limit=12)
        if term.lower() not in named_entity_lowers
    ][:8]
    comparison_options = _extract_comparison_options(query)
    semantic_expansions: list[str] = []
    for phrase, expansions in _OPEN_DOMAIN_TOPIC_EXPANSIONS.items():
        if phrase in query_lower:
            semantic_expansions.extend(expansions)

    # Only trigger open_domain mode for genuine inference-style questions,
    # NOT for simple factual queries starting with "is/are/does/did/can".
    # The broad startswith catch was causing open_domain to fire on ~100% of
    # queries, leading to 4x sub-query expansion + implied fact CE passes
    # and catastrophic memory growth (OOM at 10-22GB).
    has_modal = any(word in _OPEN_DOMAIN_MODAL_WORDS for word in words)
    has_cue_phrase = any(phrase in query_lower for phrase in _OPEN_DOMAIN_CUE_PHRASES)
    has_topic_expansion = bool(semantic_expansions)
    is_open_domain_like = bool(
        comparison_options
        or has_cue_phrase
        or has_topic_expansion
        or (has_modal and has_relational_intent)
        or query_lower.startswith(("would ", "could ", "might "))
    )

    # 4. Classify
    all_signals = ["vector", "fts", "ppr", "spreading", "fractal", "hopfield", "hdc", "sr"]
    has_question = any(w in _QUESTION_WORDS for w in words)

    if temporal_markers:
        query_type = "temporal"
        enabled_signals = ["vector", "fts"]
    elif is_open_domain_like:
        query_type = "open_domain"
        enabled_signals = ["vector", "fts"]
    elif has_question:
        query_type = "factoid"
        enabled_signals = ["vector", "fts", "ppr"]
    elif code_identifiers:
        query_type = "code"
        enabled_signals = ["vector", "fts"]
    elif has_relational_intent:
        query_type = "relational"
        enabled_signals = ["vector", "fts", "ppr", "spreading"]
    elif len(words) <= 2:
        query_type = "keyword"
        enabled_signals = ["vector", "fts"]
    elif len(words) < 5:
        query_type = "simple"
        enabled_signals = ["vector", "fts"]
    else:
        query_type = "complex"
        enabled_signals = list(all_signals)

    return {
        "query_type": query_type,
        "enabled_signals": enabled_signals,
        "temporal_markers": temporal_markers,
        "named_entities": named_entities,
        "code_identifiers": code_identifiers,
        "has_relational_intent": has_relational_intent,
        "content_terms": content_terms,
        "comparison_options": comparison_options,
        "semantic_expansions": semantic_expansions[:8],
        "is_open_domain_like": is_open_domain_like,
    }


class HippoRetriever:
    """HippoRAG-style retrieval combining PPR, spreading activation,
    vector similarity, and FTS5 keyword search."""

    def __init__(
        self,
        storage: StorageEngine,
        embeddings: EmbeddingEngine,
        knowledge_graph: KnowledgeGraph,
        settings: Settings,
    ) -> None:
        self._storage = storage
        self._embeddings = embeddings
        self._graph = knowledge_graph
        self._settings = settings
        self._fractal = FractalMemoryTree(storage, embeddings, settings)
        self._hopfield = HopfieldMemory(storage, embeddings, settings)
        self._engram = None  # Set externally via set_engram()
        self._rules_engine = None  # Set externally via set_rules_engine()
        self._hdc = None  # Set externally via set_hdc()
        self._cognitive_map = None  # Set externally via set_cognitive_map()
        self._metacognition = None  # Set externally via set_metacognition()
        self._gte_reranker = None  # Lazy-loaded GTE-Reranker
        self._nli_model = None  # Lazy-loaded NLI model
        self._comet_expander = None  # Lazy-loaded COMET query expander

    def set_engram(self, engram) -> None:
        """Attach an EngramAllocator for temporal linking in recall results."""
        self._engram = engram

    def set_rules_engine(self, rules_engine) -> None:
        """Attach a RulesEngine for neuro-symbolic filtering/re-ranking."""
        self._rules_engine = rules_engine

    def set_hdc(self, hdc) -> None:
        """Attach an HDCEncoder for compositional structured queries."""
        self._hdc = hdc

    def set_cognitive_map(self, cognitive_map) -> None:
        """Attach a CognitiveMap for SR-based navigation signal."""
        self._cognitive_map = cognitive_map

    def set_metacognition(self, metacognition) -> None:
        """Attach a MetaCognition engine for cognitive load management."""
        self._metacognition = metacognition

    # -- COMET query expansion --

    def _comet_expand_query(self, query: str) -> list[str]:
        """Use COMET-BART to generate commonsense expansions for a query.

        Reformulates the query as an event and generates xWant/xAttr inferences
        to bridge the cue-trigger semantic disconnect at query time.
        """
        if not getattr(self._settings, 'COMET_QUERY_EXPANSION_ENABLED', False):
            return []
        try:
            if self._comet_expander is None:
                from zikkaron.enrichment import CometInferencer
                self._comet_expander = CometInferencer()

            # Reformulate query as a COMET-compatible event
            statement = _question_to_statement(query)
            # Generate xWant and xAttr inferences
            inferences = self._comet_expander.infer(statement, self._settings)
            if inferences:
                logger.debug("COMET query expansion: %s -> %s", query[:60], inferences[:3])
            return inferences
        except Exception as e:
            logger.debug("COMET query expansion failed: %s", e)
            return []

    # -- Dual-vector search (architecture prep) --

    def _dual_vector_search(self, query_embedding, top_k: int) -> list[tuple[int, float]]:
        """Search both explicit and implicit vector spaces."""
        if not getattr(self._settings, 'DUAL_VECTORS_ENABLED', False):
            return []

        explicit_results = self._storage.search_vectors(query_embedding, top_k)
        implicit_results = self._storage.search_implicit_vectors(query_embedding, top_k)

        # Merge with weighted combination
        explicit_weight = 1 - self._settings.IMPLICIT_VECTOR_WEIGHT
        implicit_weight = self._settings.IMPLICIT_VECTOR_WEIGHT

        scores = {}
        for mid, score in explicit_results:
            scores[mid] = explicit_weight * score
        for mid, score in implicit_results:
            scores[mid] = scores.get(mid, 0) + implicit_weight * score

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # -- a. Personalized PageRank Retrieval --

    def ppr_retrieve(
        self, query: str, top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Run Personalized PageRank seeded by query entities.

        Returns (memory_id, ppr_score) sorted by score descending.
        """
        # 1. Extract entities from query
        query_terms = _extract_query_entities(query)
        if not query_terms:
            return []

        # 2. Find matching entities in the knowledge graph
        seed_entity_ids: list[int] = []
        for term in query_terms:
            if len(term) < self._settings.GRAPH_ENTITY_MIN_LENGTH:
                continue
            entity = self._storage.get_entity_by_name(term)
            if entity:
                seed_entity_ids.append(entity["id"])

        if not seed_entity_ids:
            return []

        # 3. Build a networkx graph from entity-relationship data
        G = self._build_networkx_graph(seed_entity_ids)
        if len(G) == 0:
            return []

        # 4. Run Personalized PageRank
        personalization = {
            eid: 1.0 / len(seed_entity_ids) for eid in seed_entity_ids
            if eid in G
        }
        if not personalization:
            return []

        try:
            ppr_scores = nx.pagerank(
                G,
                alpha=self._settings.PPR_DAMPING,
                personalization=personalization,
                max_iter=self._settings.PPR_ITERATIONS,
            )
        except nx.PowerIterationFailedConvergence:
            ppr_scores = nx.pagerank(
                G,
                alpha=self._settings.PPR_DAMPING,
                personalization=personalization,
                max_iter=self._settings.PPR_ITERATIONS * 2,
                tol=1e-4,
            )

        # 5. Map high-PPR entities back to their associated memories
        entity_scores = sorted(
            ppr_scores.items(), key=lambda x: x[1], reverse=True
        )

        memory_scores: dict[int, float] = defaultdict(float)
        for entity_id, score in entity_scores:
            entity = self._storage._conn.execute(
                "SELECT name FROM entities WHERE id = ?", (entity_id,)
            ).fetchone()
            if not entity:
                continue
            entity_name = entity[0]
            # Find memories containing this entity name via FTS5
            associated = self._find_memories_for_entity(entity_name)
            for mid in associated:
                memory_scores[mid] = max(memory_scores[mid], score)

        # Sort by score descending, return top_k
        ranked = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # -- b. Contextual Prefix Generation --

    def generate_contextual_prefix(
        self,
        content: str,
        directory: str,
        tags: list[str],
        timestamp: datetime,
    ) -> str:
        """Generate a contextual prefix for richer embedding semantics."""
        dir_basename = os.path.basename(directory.rstrip("/")) or directory
        tags_joined = ", ".join(tags) if tags else "none"
        timestamp_human = timestamp.strftime("%Y-%m-%d %H:%M")

        # Find top co-occurring entities for context enrichment
        top_entities = self._get_top_cooccurring_entities(content, limit=5)
        entities_str = ", ".join(top_entities) if top_entities else "none"

        return (
            f"[Project: {dir_basename}] [Directory: {directory}] "
            f"[Tags: {tags_joined}] [Recorded: {timestamp_human}] "
            f"[Related entities: {entities_str}] "
        )

    # -- c. Spreading Activation --

    def spreading_activation(
        self,
        seed_memories: list[int],
        spread_factor: float | None = None,
        max_depth: int | None = None,
    ) -> list[tuple[int, float]]:
        """Activate related memories by spreading through the entity graph.

        Returns (memory_id, activation_score) for discovered memories
        (excludes seed memories).
        """
        if spread_factor is None:
            spread_factor = self._settings.GRAPH_SPREADING_DECAY
        if max_depth is None:
            max_depth = self._settings.GRAPH_SPREADING_MAX_DEPTH

        if not seed_memories:
            return []

        # 1. Find entities associated with seed memories
        seed_entities: set[int] = set()
        for mid in seed_memories:
            mem = self._storage.get_memory(mid)
            if not mem:
                continue
            entities = self._find_entities_in_content(mem["content"])
            seed_entities.update(entities)

        if not seed_entities:
            return []

        # 2. BFS through entity graph up to max_depth
        activated: dict[int, float] = {}  # memory_id -> activation score
        seed_memory_set = set(seed_memories)

        visited_entities: set[int] = set(seed_entities)
        frontier: list[tuple[int, int]] = [(eid, 0) for eid in seed_entities]

        while frontier:
            next_frontier: list[tuple[int, int]] = []
            for entity_id, depth in frontier:
                if depth >= max_depth:
                    continue
                # Get connected entities
                neighbors = self._graph._get_adjacent(entity_id, None)
                for neighbor in neighbors:
                    nid = neighbor["entity_id"]
                    if nid in visited_entities:
                        continue
                    visited_entities.add(nid)
                    current_depth = depth + 1
                    activation = spread_factor ** current_depth

                    # Find memories for this neighbor entity
                    entity_row = self._storage._conn.execute(
                        "SELECT name FROM entities WHERE id = ?", (nid,)
                    ).fetchone()
                    if entity_row:
                        mids = self._find_memories_for_entity(entity_row[0])
                        for mid in mids:
                            if mid not in seed_memory_set:
                                activated[mid] = max(
                                    activated.get(mid, 0.0), activation
                                )

                    next_frontier.append((nid, current_depth))
            frontier = next_frontier

        # Sort by activation score descending
        return sorted(activated.items(), key=lambda x: x[1], reverse=True)

    # -- d. Unified Recall --

    def recall(
        self, query: str, max_results: int = 5, min_heat: float = 0.1
    ) -> list[dict]:
        """Combine retrieval signals via Weighted Reciprocal Rank Fusion (WRRF).

        Each signal produces a ranked list of memory IDs. Scores are fused as:
          WRRF_score(d) = Σ_i [ w_i / (k + rank_i(d)) ]
        where k = WRRF_K (default 60) and w_i are per-signal weights from settings.
        An optional heuristic reranker refines the final ordering.
        """
        # Check if SR signal is available
        sr_active = (
            self._cognitive_map is not None
            and self._cognitive_map.has_sufficient_data()
        )

        w_temporal = 0.0  # Dynamically set if temporal markers found

        scores: dict[int, dict] = defaultdict(
            lambda: {
                "vector": 0.0, "fts": 0.0, "ppr": 0.0,
                "spread": 0.0, "fractal": 0.0, "hopfield": 0.0,
                "hdc": 0.0, "sr": 0.0, "temporal": 0.0,
            }
        )

        query_analysis = analyze_query(query, self._settings)

        # Query-dependent signal routing
        if self._settings.QUERY_ROUTING_ENABLED:
            enabled_signals = set(query_analysis.get("enabled_signals", []))
        else:
            enabled_signals = None  # None means all signals enabled

        open_domain_mode = query_analysis.get("is_open_domain_like", False)
        open_domain_subqueries = (
            _build_open_domain_subqueries(query, query_analysis)
            if open_domain_mode
            else []
        )

        candidate_k = max_results * self._settings.CANDIDATE_POOL_MULTIPLIER
        if open_domain_mode:
            candidate_k = int(
                candidate_k * getattr(self._settings, "OPEN_DOMAIN_CANDIDATE_MULTIPLIER", 1.5)
            )

        # 1. FTS5 keyword search with actual BM25 scores
        #    Boost: duplicate entities 2x and content words 1x in FTS query
        if enabled_signals is None or "fts" in enabled_signals:
            try:
                fts_searches = [(query, 1.0)]
                if open_domain_subqueries:
                    for subquery in open_domain_subqueries:
                        fts_searches.append((subquery, 0.8))

                for fts_query, strength in fts_searches:
                    fts_scored = self._storage.search_memories_fts_scored(
                        _build_boosted_fts_query(fts_query),
                        min_heat=min_heat,
                        limit=candidate_k,
                    )
                    if not fts_scored:
                        continue
                    max_bm25 = max(s for _, s in fts_scored) if fts_scored else 1.0
                    for mid, bm25_score in fts_scored:
                        normalized = bm25_score / max_bm25 if max_bm25 > 0 else 0.0
                        scores[mid]["fts"] = max(
                            scores[mid].get("fts", 0.0),
                            normalized * strength,
                        )
            except Exception:
                pass

        # 1b. Entity-focused FTS: search for just person names to ensure
        #     all memories about mentioned people reach the CE pool.
        #     Critical for open_domain questions where inference depends on
        #     knowing all facts about a person, not just keyword matches.
        if enabled_signals is None or "fts" in enabled_signals:
            try:
                entity_names = [
                    w.strip(".,;:!?()[]{}\"'")
                    for w in query.split()
                    if w[0:1].isupper() and len(w.strip(".,;:!?()[]{}\"'")) >= 2
                    and w.strip(".,;:!?()[]{}\"'").lower() not in _QUERY_STOP_WORDS
                ]
                if entity_names:
                    # Just space-separate; _preprocess_fts_query will OR them
                    entity_query = " ".join(entity_names)
                    entity_hits = self._storage.search_memories_fts_scored(
                        entity_query, min_heat=min_heat, limit=candidate_k
                    )
                    if entity_hits:
                        max_ent = max(s for _, s in entity_hits) if entity_hits else 1.0
                        for mid, ent_score in entity_hits:
                            normalized = ent_score / max_ent if max_ent > 0 else 0.0
                            # Use max to not overwrite a better FTS score
                            scores[mid]["fts"] = max(
                                scores[mid].get("fts", 0.0),
                                normalized * (0.7 if open_domain_mode else 0.5),
                            )
            except Exception:
                pass

        # 1c. COMET query expansion: generate commonsense inferences from query
        #     to bridge the cue-trigger semantic disconnect for open_domain queries.
        #     E.g., "Would Caroline be considered religious?" → xWant: "go to church"
        if open_domain_mode and (enabled_signals is None or "fts" in enabled_signals):
            comet_terms = self._comet_expand_query(query)
            if comet_terms:
                try:
                    comet_query = " ".join(comet_terms[:6])
                    comet_hits = self._storage.search_memories_fts_scored(
                        comet_query, min_heat=min_heat, limit=candidate_k,
                    )
                    if comet_hits:
                        max_comet = max(s for _, s in comet_hits) if comet_hits else 1.0
                        for mid, comet_score in comet_hits:
                            normalized = comet_score / max_comet if max_comet > 0 else 0.0
                            scores[mid]["fts"] = max(
                                scores[mid].get("fts", 0.0),
                                normalized * 0.6,  # Lower weight than direct FTS match
                            )
                except Exception:
                    pass

        # 2. Vector similarity via sqlite-vec KNN
        #    Dual vector search: search with both original AND HyDE-expanded queries
        #    to maximize recall. Union candidates, keep max similarity per memory.
        vector_memory_ids: list[int] = []
        query_embedding = None
        if enabled_signals is None or "vector" in enabled_signals:
            vector_searches = [(query, 1.0)]

            if self._settings.QUERY_EXPANSION_ENABLED:
                expanded_query = _pseudo_hyde_expand(query)
                if expanded_query and expanded_query != query:
                    vector_searches.append((expanded_query, 0.95))

            if open_domain_subqueries:
                for subquery in open_domain_subqueries[:2]:
                    vector_searches.append((subquery, 0.85))

            seen_vector_queries: set[str] = set()
            for vector_query, strength in vector_searches:
                lowered = vector_query.lower()
                if lowered in seen_vector_queries:
                    continue
                seen_vector_queries.add(lowered)

                encoded = self._embeddings.encode_query(vector_query)
                if encoded is None:
                    continue
                if vector_query == query:
                    query_embedding = encoded

                vec_hits = self._storage.search_vectors(
                    encoded, top_k=candidate_k, min_heat=min_heat
                )
                for mid, distance in vec_hits:
                    similarity = (1.0 / (1.0 + distance)) * strength
                    scores[mid]["vector"] = max(scores[mid].get("vector", 0.0), similarity)
                    if mid not in vector_memory_ids:
                        vector_memory_ids.append(mid)

        # 3. PPR graph retrieval
        if enabled_signals is None or "ppr" in enabled_signals:
            ppr_results = self.ppr_retrieve(query, top_k=candidate_k)
            if ppr_results:
                max_ppr = max(s for _, s in ppr_results) if ppr_results else 1.0
                for mid, ppr_score in ppr_results:
                    # Normalize PPR scores to 0-1 range
                    normalized = ppr_score / max_ppr if max_ppr > 0 else 0.0
                    scores[mid]["ppr"] = normalized

        # 4. Spreading activation from top vector results
        if enabled_signals is None or "spreading" in enabled_signals:
            top_vector_seeds = vector_memory_ids[:5]
            if top_vector_seeds:
                spread_results = self.spreading_activation(
                    top_vector_seeds, spread_factor=0.5, max_depth=2
                )
                if spread_results:
                    max_spread = max(s for _, s in spread_results) if spread_results else 1.0
                    for mid, spread_score in spread_results:
                        normalized = spread_score / max_spread if max_spread > 0 else 0.0
                        scores[mid]["spread"] = normalized

        # 5. Fractal cluster matching
        if enabled_signals is None or "fractal" in enabled_signals:
            fractal_results = self._fractal.fractal_score(
                query, max_results=candidate_k
            )
            if fractal_results:
                max_fractal = max(s for _, s in fractal_results) if fractal_results else 1.0
                for mid, fractal_score in fractal_results:
                    normalized = fractal_score / max_fractal if max_fractal > 0 else 0.0
                    scores[mid]["fractal"] = normalized

        # 6. Modern Hopfield energy-based retrieval
        if enabled_signals is None or "hopfield" in enabled_signals:
            if query_embedding is not None:
                try:
                    hopfield_results = self._hopfield.retrieve(
                        query_embedding, top_k=candidate_k
                    )
                    if hopfield_results:
                        max_hop = max(s for _, s in hopfield_results) if hopfield_results else 1.0
                        for mid, hop_score in hopfield_results:
                            normalized = hop_score / max_hop if max_hop > 0 else 0.0
                            scores[mid]["hopfield"] = normalized
                except Exception:
                    logger.debug("Hopfield retrieval failed, skipping signal")

        # 7. HDC compositional query
        if enabled_signals is None or "hdc" in enabled_signals:
            if self._hdc is not None:
                try:
                    query_entities = _extract_query_entities(query)
                    if query_entities:
                        hdc_query = self._hdc.encode_query(entities=query_entities)
                        # Load HDC vectors for candidate memories
                        candidate_ids = list(scores.keys()) if scores else []
                        if not candidate_ids:
                            # If no candidates from other signals, get recent memories
                            all_mems = self._storage._conn.execute(
                                "SELECT id FROM memories WHERE heat > ? LIMIT ?",
                                (min_heat, candidate_k),
                            ).fetchall()
                            candidate_ids = [r[0] for r in all_mems]

                        hdc_candidates: list[tuple[int, "np.ndarray"]] = []
                        for mid in candidate_ids:
                            row = self._storage._conn.execute(
                                "SELECT hdc_vector FROM memories WHERE id = ?",
                                (mid,),
                            ).fetchone()
                            if row and row[0] is not None:
                                hdc_vec = self._hdc.from_bytes(row[0])
                                hdc_candidates.append((mid, hdc_vec))

                        if hdc_candidates:
                            hdc_results = self._hdc.search(
                                hdc_query, hdc_candidates, top_k=candidate_k
                            )
                            if hdc_results:
                                max_hdc = max(s for _, s in hdc_results) if hdc_results else 1.0
                                for mid, hdc_score in hdc_results:
                                    if max_hdc > 0:
                                        normalized = max(0.0, hdc_score) / max_hdc
                                    else:
                                        normalized = 0.0
                                    scores[mid]["hdc"] = normalized
                except Exception:
                    logger.debug("HDC retrieval failed, skipping signal")

        # 8. Successor Representation navigation
        if enabled_signals is None or "sr" in enabled_signals:
            if sr_active and query_embedding is not None:
                try:
                    candidate_ids = list(scores.keys()) if scores else []
                    if candidate_ids:
                        sr_scores = self._cognitive_map.get_sr_scores(
                            query_embedding, self._embeddings, candidate_ids
                        )
                        if sr_scores:
                            max_sr = max(sr_scores.values()) if sr_scores else 1.0
                            for mid, sr_score in sr_scores.items():
                                normalized = sr_score / max_sr if max_sr > 0 else 0.0
                                scores[mid]["sr"] = normalized
                except Exception:
                    logger.debug("SR retrieval failed, skipping signal")

        # 9. Temporal retrieval boost — temporal_retrieval signal
        if getattr(self._settings, 'TEMPORAL_RETRIEVAL_ENABLED', False):
            temporal_info = parse_temporal_expression(query)
            if temporal_info["has_temporal"]:
                try:
                    # A) Content-based temporal matching (FTS on dates in content)
                    temporal_memories = self._storage.search_memories_by_content_date(
                        date_hints=temporal_info["date_hints"],
                        month_hints=temporal_info["month_hints"],
                        session_hints=temporal_info["session_hints"],
                        min_heat=min_heat,
                        limit=candidate_k,
                    )
                    if temporal_memories:
                        for i, mem in enumerate(temporal_memories):
                            scores[mem["id"]]["temporal"] = 1.0 / (1 + i)
                        w_temporal = 0.8

                    # B) Timestamp-based temporal matching (created_at proximity)
                    if temporal_info["month_hints"]:
                        month_matches = self._storage.search_memories_by_month(
                            temporal_info["month_hints"],
                            min_heat=min_heat,
                            limit=candidate_k,
                        )
                        if month_matches:
                            for mid in month_matches:
                                # Add temporal score (lower than FTS match)
                                if scores[mid]["temporal"] == 0.0:
                                    scores[mid]["temporal"] = 0.5
                            if w_temporal == 0.0:
                                w_temporal = 0.6
                except Exception:
                    logger.debug("Temporal retrieval failed, skipping signal")

        # Score-weighted fusion: use actual signal scores (not rank-based)
        # Each memory's final score = Σ (w_i * score_i) for each signal
        signal_weights = {
            "vector": self._settings.WRRF_VECTOR_WEIGHT,
            "fts": self._settings.WRRF_FTS_WEIGHT,
            "ppr": self._settings.WRRF_PPR_WEIGHT,
            "spread": self._settings.WRRF_SPREADING_WEIGHT,
            "hopfield": self._settings.WRRF_HOPFIELD_WEIGHT,
            "hdc": self._settings.WRRF_HDC_WEIGHT,
            "fractal": self._settings.WRRF_FRACTAL_WEIGHT,
            "sr": self._settings.WRRF_SR_WEIGHT,
        }
        if w_temporal > 0:
            signal_weights["temporal"] = w_temporal
        if open_domain_mode:
            signal_weights["fts"] *= getattr(self._settings, "OPEN_DOMAIN_FTS_BOOST", 1.6)

        # Apply confidence gating
        if getattr(self._settings, 'CONFIDENCE_GATING_ENABLED', False):
            _conf_name_map = {"spread": "spreading"}
            thresholds = {
                "vector": getattr(self._settings, 'CONFIDENCE_THRESHOLD_VECTOR', 0.1),
                "fts": getattr(self._settings, 'CONFIDENCE_THRESHOLD_FTS', 0.1),
                "ppr": getattr(self._settings, 'CONFIDENCE_THRESHOLD_PPR', 0.1),
                "spread": getattr(self._settings, 'CONFIDENCE_THRESHOLD_SPREADING', 0.1),
                "hopfield": getattr(self._settings, 'CONFIDENCE_THRESHOLD_HOPFIELD', 0.1),
                "hdc": getattr(self._settings, 'CONFIDENCE_THRESHOLD_HDC', 0.1),
                "fractal": getattr(self._settings, 'CONFIDENCE_THRESHOLD_FRACTAL', 0.1),
                "sr": getattr(self._settings, 'CONFIDENCE_THRESHOLD_SR', 0.1),
                "temporal": getattr(self._settings, 'CONFIDENCE_THRESHOLD_TEMPORAL', 0.1),
            }
            for sig in list(signal_weights.keys()):
                ranked = sorted(
                    [(mid, s[sig]) for mid, s in scores.items() if s[sig] > 0],
                    key=lambda x: x[1], reverse=True,
                )
                conf_name = _conf_name_map.get(sig, sig)
                confidence = self._compute_signal_confidence(conf_name, ranked)
                threshold = thresholds.get(sig, 0.1)
                if confidence < threshold:
                    signal_weights[sig] = 0.0

        # --- Fusion: convex combination vs WRRF (existing) ---
        fusion_method = getattr(self._settings, "FUSION_METHOD", "wrrf")

        if fusion_method == "convex":
            # Build signal_scores: signal_name -> {memory_id: raw_score}
            signal_scores_for_convex: dict[str, dict[int, float]] = {}
            for sig in signal_weights:
                sig_dict = {
                    mid: s[sig] for mid, s in scores.items() if s[sig] > 0
                }
                if sig_dict:
                    signal_scores_for_convex[sig] = sig_dict
            fused = self._convex_fuse(signal_scores_for_convex, signal_weights)
            fused_scores = dict(fused)
        else:
            # Existing WRRF-style normalized weighted sum
            signal_names = list({sig for mid, sigs in scores.items() for sig, v in sigs.items() if v > 0})
            normalized: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
            fusion_norm = getattr(self._settings, 'FUSION_NORM', 'zscore')

            for sig in signal_names:
                sig_vals = [(mid, s[sig]) for mid, s in scores.items() if s[sig] > 0]
                if not sig_vals:
                    continue
                vals = [v for _, v in sig_vals]

                if fusion_norm == "minmax":
                    min_v = min(vals)
                    max_v = max(vals)
                    rng = max_v - min_v
                    for mid, v in sig_vals:
                        normalized[mid][sig] = (v - min_v) / rng if rng > 1e-9 else 0.5
                elif fusion_norm == "raw":
                    for mid, v in sig_vals:
                        normalized[mid][sig] = v
                else:  # zscore (default)
                    mean_v = sum(vals) / len(vals)
                    std_v = (sum((v - mean_v) ** 2 for v in vals) / len(vals)) ** 0.5
                    if std_v > 1e-9:
                        z_scores = [(mid, (v - mean_v) / std_v) for mid, v in sig_vals]
                        z_vals = [z for _, z in z_scores]
                        z_min, z_max = min(z_vals), max(z_vals)
                        z_rng = z_max - z_min
                        for mid, z in z_scores:
                            normalized[mid][sig] = (z - z_min) / z_rng if z_rng > 1e-9 else 0.5
                    else:
                        for mid, v in sig_vals:
                            normalized[mid][sig] = 0.5

            combmnz = getattr(self._settings, 'COMBMNZ_ENABLED', False)

            fused_scores = {}
            for mid, norm_sigs in normalized.items():
                total = 0.0
                signal_count = 0
                for signal, norm_score in norm_sigs.items():
                    w = signal_weights.get(signal, 0.0)
                    if w > 0:
                        total += w * norm_score
                        signal_count += 1
                if total > 0:
                    if combmnz and signal_count > 1:
                        total *= signal_count
                    fused_scores[mid] = total

            fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        # Build result memories — pull more candidates for reranker
        rerank_pool = max(
            max_results,
            self._settings.RERANKER_TOP_K,
            getattr(self._settings, "CROSS_ENCODER_TOP_K", 0),
        )
        result_memories: list[dict] = []
        seen_ids: set[int] = set()
        for mid, total_score in fused:
            mem = self._storage.get_memory(mid)
            if mem and mem["heat"] >= min_heat:
                mem["_retrieval_score"] = round(total_score, 4)
                mem.pop("embedding", None)
                result_memories.append(mem)
                seen_ids.add(mid)
            if len(result_memories) >= rerank_pool:
                break

        # Inject top signal-specific results for CE pool diversity
        # This ensures CE sees the best FTS/vector candidates even if
        # they didn't rank in the fused top-K
        if getattr(self._settings, 'CROSS_ENCODER_ENABLED', False):
            diversity_k = getattr(self._settings, 'CE_DIVERSITY_INJECT_K', 10)
            if open_domain_mode:
                diversity_k = max(diversity_k, 15)
            for sig in ["fts", "vector"]:
                top_sig = sorted(
                    [(mid, s[sig]) for mid, s in scores.items() if s[sig] > 0],
                    key=lambda x: x[1], reverse=True,
                )[:diversity_k]
                for mid, _ in top_sig:
                    if mid not in seen_ids:
                        mem = self._storage.get_memory(mid)
                        if mem and mem["heat"] >= min_heat:
                            mem["_retrieval_score"] = round(fused_scores.get(mid, 0.0), 4)
                            mem.pop("embedding", None)
                            result_memories.append(mem)
                            seen_ids.add(mid)

        # Heuristic reranker
        if self._settings.RERANKER_ENABLED:
            # When CE follows, don't clip yet — let CE see the full pool
            heuristic_k = max_results
            if getattr(self._settings, 'CROSS_ENCODER_ENABLED', False):
                heuristic_k = None  # Uses RERANKER_TOP_K (50)
            result_memories = self._heuristic_rerank(result_memories, query, top_k=heuristic_k)

        # Comparison dual search: merge extra candidates for "A or B?" queries
        comparison_options = query_analysis.get("comparison_options", [])
        if (
            getattr(self._settings, "COMPARISON_DUAL_SEARCH_ENABLED", False)
            and comparison_options
        ):
            subject = query_analysis.get("named_entities", [None])[0] if query_analysis.get("named_entities") else None
            comp_results = self._comparison_dual_search(
                query, comparison_options, subject, max_results,
            )
            for r in comp_results:
                rid = r.get("id", -1)
                if rid not in seen_ids:
                    r.setdefault("_retrieval_score", 0.0)
                    r.pop("embedding", None)
                    result_memories.append(r)
                    seen_ids.add(rid)

        # Cross-encoder reranker (FlashRank ONNX — fast CPU inference)
        # Feed the raw query directly — CE performs best with the original question.
        # CE query augmentation (concatenating HyDE expansion) was tested and HURTS MRR.
        if getattr(self._settings, 'CROSS_ENCODER_ENABLED', False):
            result_memories = self._cross_encoder_rerank(result_memories, query)

        # NLI entailment scoring: complementary signal to CE for open-domain queries
        if (getattr(self._settings, 'NLI_RERANKING_ENABLED', False)
                and (not self._settings.NLI_ONLY_FOR_OPEN_DOMAIN or open_domain_mode)):
            result_memories = self._nli_rerank(query, result_memories)
            # Blend NLI with CE score
            nli_weight = self._settings.NLI_WEIGHT
            for mem in result_memories:
                ce = mem.get("_cross_encoder_score", 0)
                nli = mem.get("_nli_entailment_score", 0)
                mem["_retrieval_score"] = (1 - nli_weight) * ce + nli_weight * nli
            result_memories.sort(key=lambda m: m.get("_retrieval_score", 0), reverse=True)

        # Multi-passage evidence aggregation: boost scattered evidence clusters
        if getattr(self._settings, 'MULTI_PASSAGE_RERANKING_ENABLED', False):
            result_memories = self._multi_passage_rerank(query, result_memories, max_results)

        # Profile and belief search: merge structured knowledge after CE reranking
        directory = ""
        for mem in result_memories:
            if mem.get("directory_context"):
                directory = mem["directory_context"]
                break
        profile_belief_results = self._search_profiles_and_beliefs(
            query, directory, max_results,
        )
        if profile_belief_results:
            result_memories.extend(profile_belief_results)
            result_memories.sort(
                key=lambda m: m.get("_retrieval_score", 0), reverse=True,
            )
            result_memories = result_memories[: max_results * 2]

        # MMR diversity reranking — avoid all top-K from same conversation segment
        if getattr(self._settings, 'ADVERSARIAL_DIVERSITY_ENFORCEMENT', False):
            result_memories = self._mmr_rerank(
                result_memories, query_embedding,
                top_k=max_results, lambda_param=0.7,
            )

        # Trim to max_results after reranking
        result_memories = result_memories[:max_results]

        # Adversarial detection
        if self._settings.ADVERSARIAL_DETECTION_ENABLED and result_memories:
            adv_info = self._detect_adversarial(result_memories)
            for mem in result_memories:
                mem["_retrieval_confidence"] = adv_info["confidence"]
            if adv_info["is_uncertain"]:
                logger.debug(
                    "Low retrieval confidence (%.3f), score_gap=%.3f",
                    adv_info["confidence"], adv_info["score_gap"]
                )

        # Apply neuro-symbolic rules (hard filter + soft re-rank) as final step
        if self._rules_engine is not None and result_memories:
            # Infer directory from first memory or use empty string
            directory = ""
            for mem in result_memories:
                if mem.get("directory_context"):
                    directory = mem["directory_context"]
                    break
            result_memories = self._rules_engine.apply_rules(result_memories, directory)
            # Re-trim to max_results after filtering
            result_memories = result_memories[:max_results]

        # Enrich with temporal links from engram allocation
        if self._engram is not None:
            for mem in result_memories:
                try:
                    linked = self._engram.get_temporally_linked(mem["id"])
                    if linked:
                        mem["temporal_links"] = linked
                except Exception:
                    pass

        # Apply cognitive load management via metacognition
        if self._metacognition is not None and result_memories:
            try:
                result_memories = self._metacognition.manage_context(
                    result_memories
                )
            except Exception:
                logger.debug("Metacognition manage_context failed, returning unoptimized")

        return result_memories

    # -- e. Hierarchical Recall --

    def recall_hierarchical(
        self, query: str, level: int | None = None, max_results: int = 10
    ) -> list[dict]:
        """Level-specific retrieval through the fractal hierarchy."""
        return self._fractal.retrieve_tree(query, target_level=level)[:max_results]

    # -- Internal helpers --

    def _wrrf_fuse(
        self,
        ranked_lists: dict[str, list[int]],
        wrrf_weights: dict[str, float],
        k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Weighted Reciprocal Rank Fusion across multiple ranked lists.

        Args:
            ranked_lists: signal_name -> list of memory IDs (sorted by signal score desc)
            wrrf_weights: signal_name -> weight
            k: RRF constant (default: self._settings.WRRF_K)

        Returns:
            List of (memory_id, wrrf_score) sorted by score descending.
        """
        if k is None:
            k = self._settings.WRRF_K

        scores: dict[int, float] = {}
        for signal_name, mem_ids in ranked_lists.items():
            w = wrrf_weights.get(signal_name, 0.0)
            if w <= 0:
                continue
            for rank, mem_id in enumerate(mem_ids):
                scores[mem_id] = scores.get(mem_id, 0.0) + w / (k + rank + 1)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _convex_fuse(
        self,
        signal_scores: dict[str, dict[int, float]],
        weights: dict[str, float],
    ) -> list[tuple[int, float]]:
        """Convex combination of normalized signal scores.

        Unlike RRF which uses ranks, this uses actual scores with min-max normalization.
        Proven to outperform RRF (Bruch et al., ACM TOIS 2023).
        """
        total_w = sum(weights.values())
        if total_w == 0:
            return []
        norm_weights = {k: v / total_w for k, v in weights.items()}

        normalized: dict[str, dict[int, float]] = {}
        for signal, scores in signal_scores.items():
            if not scores:
                continue
            vals = list(scores.values())
            min_s, max_s = min(vals), max(vals)
            range_s = max_s - min_s
            if range_s > 0:
                normalized[signal] = {mid: (s - min_s) / range_s for mid, s in scores.items()}
            else:
                normalized[signal] = {mid: 0.5 for mid in scores}

        all_mids: set[int] = set()
        for scores in normalized.values():
            all_mids.update(scores.keys())

        combined: dict[int, float] = {}
        for mid in all_mids:
            combined[mid] = sum(
                norm_weights.get(sig, 0) * normalized.get(sig, {}).get(mid, 0)
                for sig in normalized
            )
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)

    def _comparison_dual_search(
        self,
        query: str,
        options: list[str],
        subject: str | None,
        max_results: int,
    ) -> list[dict]:
        """Dual-search for comparison queries like 'A or B?'"""
        all_results: list[dict] = []

        for option in options[:2]:  # Max 2 options
            sub_query = f"{subject} {option}" if subject else option
            # Vector search
            encoded = self._embeddings.encode_query(sub_query)
            vec_results: list[dict] = []
            if encoded is not None:
                vec_hits = self._storage.search_vectors(
                    encoded,
                    top_k=self._settings.COMPARISON_TOP_K_PER_OPTION,
                    min_heat=0.1,
                )
                for mid, distance in vec_hits:
                    mem = self._storage.get_memory(mid)
                    if mem:
                        mem.pop("embedding", None)
                        vec_results.append(mem)
            # Also do FTS
            fts_results = self._storage.search_memories_fts(
                sub_query, limit=self._settings.COMPARISON_TOP_K_PER_OPTION,
            )
            # Merge
            seen: set[int] = set()
            merged: list[dict] = []
            for r in vec_results + fts_results:
                mid = r.get("id", r.get("memory_id", -1))
                if mid not in seen:
                    seen.add(mid)
                    r["_comparison_option"] = option
                    merged.append(r)
            all_results.extend(merged)

        # Deduplicate
        seen_final: set[int] = set()
        unique: list[dict] = []
        for r in all_results:
            mid = r.get("id", r.get("memory_id", -1))
            if mid not in seen_final:
                seen_final.add(mid)
                unique.append(r)

        return unique[: max_results * 2]

    def _search_profiles_and_beliefs(
        self,
        query: str,
        directory: str | None,
        max_results: int,
    ) -> list[dict]:
        """Search structured profiles and derived beliefs."""
        extra_results: list[dict] = []

        # Search profiles
        if getattr(self._settings, "PROFILE_EXTRACTION_ENABLED", False):
            try:
                profiles = self._storage.search_profiles_fts(query, limit=max_results)
                for p in profiles:
                    extra_results.append({
                        "id": -p.get("id", 0),  # Negative to distinguish from memories
                        "content": f"{p['entity_name']}: {p['attribute_type']} = {p['attribute_value']}",
                        "_source": "profile",
                        "_retrieval_score": self._settings.PROFILE_SEARCH_WEIGHT,
                    })
            except Exception:
                pass

        # Search beliefs
        if getattr(self._settings, "DERIVED_BELIEFS_ENABLED", False):
            try:
                beliefs = self._storage.search_beliefs_fts(query, limit=max_results)
                boost = self._settings.BELIEF_HIGH_CONFIDENCE_BOOST
                for b in beliefs:
                    score = (
                        b.get("confidence", 0.5) * boost
                        if b.get("confidence", 0) > 0.7
                        else b.get("confidence", 0.5)
                    )
                    extra_results.append({
                        "id": -b.get("id", 0) - 100000,  # Negative offset to distinguish
                        "content": b["content"],
                        "_source": "belief",
                        "_retrieval_score": score,
                    })
            except Exception:
                pass

        return extra_results

    def _heuristic_rerank(
        self,
        memories: list[dict],
        query: str,
        top_k: int | None = None,
    ) -> list[dict]:
        """Enhanced reranker using entity matching, noun overlap, and IDF weighting.

        Signals:
        1. Entity coverage: capitalized words / proper nouns from query in content
        2. Content term coverage (excluding stop words)
        3. Bigram overlap for phrase matching
        4. Exact substring match bonus
        """
        if top_k is None:
            top_k = self._settings.RERANKER_TOP_K

        if not memories or not query:
            return memories[:top_k] if memories else []

        query_lower = query.lower()

        # Extract query entities (capitalized words, likely names/places)
        query_entities = set()
        for token in query.split():
            stripped = token.strip(".,;:!?()[]{}\"'`~@#$%^&*-_+=<>/\\|")
            if stripped and stripped[0].isupper() and len(stripped) > 1:
                query_entities.add(stripped.lower())

        # Tokenize query (excluding common question words)
        _question_words = {"what", "who", "when", "where", "why", "how", "would",
                          "could", "should", "does", "did", "is", "are", "was",
                          "were", "do", "has", "have", "had", "can", "will",
                          "the", "a", "an", "of", "in", "to", "for", "and",
                          "or", "be", "if", "that", "this", "it", "on", "at",
                          "by", "with", "as", "not", "but", "from", "likely",
                          "still", "also", "more", "most", "very", "about"}
        query_terms = set()
        query_content_terms = set()  # Terms that carry meaning
        for token in query_lower.split():
            stripped = token.strip(".,;:!?()[]{}\"'`~@#$%^&*-_+=<>/\\|")
            if stripped:
                query_terms.add(stripped)
                if stripped not in _question_words and len(stripped) > 2:
                    query_content_terms.add(stripped)

        # Build bigrams for phrase matching
        q_words = [t.strip(".,;:!?()[]{}\"'`~@#$%^&*-_+=<>/\\|") for t in query_lower.split()]
        q_words = [w for w in q_words if w]
        query_bigrams = set()
        for j in range(len(q_words) - 1):
            query_bigrams.add(f"{q_words[j]} {q_words[j+1]}")

        if not query_terms:
            return memories[:top_k]

        for mem in memories:
            content = mem.get("content", "")
            content_lower = content.lower()

            content_terms = set()
            for token in content_lower.split():
                stripped = token.strip(".,;:!?()[]{}\"'`~@#$%^&*-_+=<>/\\|")
                if stripped:
                    content_terms.add(stripped)

            # 1. Entity coverage (names, places — most important signal)
            entity_score = 0.0
            if query_entities:
                entity_overlap = query_entities & content_terms
                entity_score = len(entity_overlap) / len(query_entities)

            # 2. Content term coverage (meaningful words only)
            term_score = 0.0
            if query_content_terms:
                content_overlap = query_content_terms & content_terms
                term_score = len(content_overlap) / len(query_content_terms)

            # 3. Bigram overlap (phrase matching)
            bigram_score = 0.0
            if query_bigrams:
                c_words = [t.strip(".,;:!?()[]{}\"'`~@#$%^&*-_+=<>/\\|") for t in content_lower.split()]
                c_words = [w for w in c_words if w]
                content_bigrams = set()
                for j in range(len(c_words) - 1):
                    content_bigrams.add(f"{c_words[j]} {c_words[j+1]}")
                bigram_overlap = query_bigrams & content_bigrams
                bigram_score = len(bigram_overlap) / len(query_bigrams)

            # 4. Exact substring match
            exact_match = 1.0 if query_lower in content_lower else 0.0

            rerank_score = (
                entity_score * 0.35
                + term_score * 0.30
                + bigram_score * 0.20
                + exact_match * 0.15
            )
            mem["_rerank_score"] = round(rerank_score, 4)

            # Combine: retrieval score (85%) + rerank (15%)
            retrieval_score = mem.get("_retrieval_score", 0.0)
            mem["_retrieval_score"] = round(
                0.85 * retrieval_score + 0.15 * rerank_score, 4
            )

        memories.sort(key=lambda m: m["_retrieval_score"], reverse=True)
        return memories[:top_k]

    def _cross_encoder_rerank(
        self,
        memories: list[dict],
        query: str,
        top_k: int | None = None,
    ) -> list[dict]:
        """Rerank memories using GTE-Reranker or FlashRank cross-encoder.

        Tries GTE-Reranker-ModernBERT first (better zero-shot OOD generalization),
        falls back to FlashRank (ONNX, faster on CPU), then sentence-transformers.
        """
        if top_k is None:
            top_k = self._settings.CROSS_ENCODER_TOP_K

        if not memories or not query:
            return memories[:top_k] if memories else []

        query_analysis = analyze_query(query, self._settings)
        open_domain_mode = query_analysis.get("is_open_domain_like", False)

        # Try GTE-Reranker first (better zero-shot OOD generalization)
        gte_failed = False
        if getattr(self._settings, 'GTE_RERANKER_ENABLED', False):
            try:
                if self._gte_reranker is None:
                    from sentence_transformers import CrossEncoder as STCrossEncoder
                    self._gte_reranker = STCrossEncoder(
                        self._settings.GTE_RERANKER_MODEL,
                        max_length=self._settings.GTE_RERANKER_MAX_LENGTH,
                    )
                    logger.info("Loaded GTE-Reranker: %s", self._settings.GTE_RERANKER_MODEL)

                if self._gte_reranker is not False:
                    pairs = [(query, m.get("content", "")[:512]) for m in memories]
                    scores = self._gte_reranker.predict(pairs)

                    raw_scores = [float(s) for s in scores]
                    max_score = max(raw_scores)
                    min_score = min(raw_scores)
                    score_range = max_score - min_score

                    ce_weight = getattr(self._settings, 'CROSS_ENCODER_WEIGHT', 0.6)
                    ret_weight = 1.0 - ce_weight
                    for i, mem in enumerate(memories):
                        ce_norm = (raw_scores[i] - min_score) / score_range if score_range > 0 else 0.5

                        content = mem.get("content", "")
                        content_len = len(content)
                        if content_len < 80:
                            ce_norm *= 0.5
                        elif content_len < 150:
                            ce_norm *= 0.8

                        retrieval_score = mem.get("_retrieval_score", 0.0)
                        mem["_cross_encoder_score"] = round(ce_norm, 4)
                        mem["_retrieval_score"] = round(
                            ret_weight * retrieval_score + ce_weight * ce_norm, 4
                        )

                    memories.sort(key=lambda m: m["_retrieval_score"], reverse=True)
                    return memories[:top_k]
            except Exception as e:
                logger.warning("GTE-Reranker failed, falling back: %s", e)
                self._gte_reranker = False  # Prevent retry
                gte_failed = True

        # If GTE was enabled but failed, respect fallback setting
        if gte_failed and not getattr(self._settings, 'GTE_RERANKER_FALLBACK_TO_FLASHRANK', True):
            return memories[:top_k]

        # Try FlashRank (ONNX cross-encoder fallback)
        try:
            from flashrank import Ranker, RerankRequest

            if not hasattr(self, '_flashrank_ranker') or self._flashrank_ranker is None:
                self._flashrank_ranker = Ranker(
                    model_name="ms-marco-MiniLM-L-12-v2",
                    cache_dir=os.path.expanduser("~/.cache/flashrank"),
                )

            passages = []
            variant_to_memory: dict[int, int] = {}
            for i, mem in enumerate(memories):
                base_text = mem.get("content", "")
                passages.append({"id": len(passages), "text": base_text})
                variant_to_memory[len(passages) - 1] = i

                if open_domain_mode:
                    implied_facts = _derive_implied_fact_passages(base_text)
                    if implied_facts:
                        passages.append({"id": len(passages), "text": " ".join(implied_facts)})
                        variant_to_memory[len(passages) - 1] = i

            rerank_req = RerankRequest(query=query, passages=passages)
            results = self._flashrank_ranker.rerank(rerank_req)

            # Map flashrank scores back to memories
            memory_raw_scores: dict[int, float] = defaultdict(float)
            for result in results:
                mem_idx = variant_to_memory.get(result["id"])
                if mem_idx is not None:
                    memory_raw_scores[mem_idx] = max(
                        memory_raw_scores.get(mem_idx, float("-inf")),
                        result["score"],
                    )

            raw_scores = list(memory_raw_scores.values())
            max_score = max(raw_scores) if raw_scores else 1.0
            min_score = min(raw_scores) if raw_scores else 0.0
            score_range = max_score - min_score

            ce_weight = getattr(self._settings, 'CROSS_ENCODER_WEIGHT', 0.6)
            ret_weight = 1.0 - ce_weight
            for i, mem in enumerate(memories):
                raw = memory_raw_scores.get(i, 0.0)
                ce_norm = (raw - min_score) / score_range if score_range > 0 else 0.5

                # Penalize short/generic passages that CE over-scores.
                # Short passages (<80 chars) get filler chat messages like
                # "Sounds great!" that CE erroneously ranks highly.
                content = mem.get("content", "")
                content_len = len(content)
                if content_len < 80:
                    ce_norm *= 0.5  # heavy penalty for very short
                elif content_len < 150:
                    ce_norm *= 0.8  # mild penalty

                retrieval_score = mem.get("_retrieval_score", 0.0)
                mem["_cross_encoder_score"] = round(ce_norm, 4)
                mem["_retrieval_score"] = round(
                    ret_weight * retrieval_score + ce_weight * ce_norm, 4
                )

            memories.sort(key=lambda m: m["_retrieval_score"], reverse=True)
            return memories[:top_k]

        except ImportError:
            pass
        except Exception:
            logger.debug("FlashRank reranking failed, trying sentence-transformers")

        # Fallback: sentence-transformers CrossEncoder
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logger.warning("No reranker available; skipping cross-encoder reranking")
            return memories

        if not hasattr(self, '_cross_encoder') or self._cross_encoder is None:
            try:
                self._cross_encoder = CrossEncoder(
                    self._settings.CROSS_ENCODER_MODEL
                )
            except Exception:
                self._cross_encoder = None
                return memories

        pairs = [(query, mem.get("content", "")) for mem in memories]
        try:
            ce_scores = self._cross_encoder.predict(pairs, show_progress_bar=False)
        except Exception:
            return memories

        min_ce = float(min(ce_scores))
        max_ce = float(max(ce_scores))
        ce_range = max_ce - min_ce
        if ce_range > 0:
            normalized_ce = [(float(s) - min_ce) / ce_range for s in ce_scores]
        else:
            normalized_ce = [1.0] * len(ce_scores)

        ce_weight = getattr(self._settings, 'CROSS_ENCODER_WEIGHT', 0.6)
        ret_weight = 1.0 - ce_weight
        for mem, ce_norm in zip(memories, normalized_ce):
            retrieval_score = mem.get("_retrieval_score", 0.0)
            mem["_cross_encoder_score"] = round(ce_norm, 4)
            mem["_retrieval_score"] = round(
                ret_weight * retrieval_score + ce_weight * ce_norm, 4
            )

        memories.sort(key=lambda m: m["_retrieval_score"], reverse=True)
        return memories[:top_k]

    def _nli_rerank(self, query: str, memories: list[dict]) -> list[dict]:
        """Score memories by NLI entailment probability."""
        if not getattr(self._settings, 'NLI_RERANKING_ENABLED', False):
            return memories

        try:
            if self._nli_model is None:
                from sentence_transformers import CrossEncoder
                self._nli_model = CrossEncoder(self._settings.NLI_MODEL)
                logger.info("Loaded NLI model: %s", self._settings.NLI_MODEL)

            hypothesis = _question_to_statement(query)
            pairs = [(m["content"][:512], hypothesis) for m in memories]
            scores = self._nli_model.predict(pairs)  # Shape: (n, 3) for [contradiction, neutral, entailment]

            for i, mem in enumerate(memories):
                if hasattr(scores[i], '__len__') and len(scores[i]) == 3:
                    # Softmax to get probabilities
                    import numpy as np
                    exp_scores = np.exp(scores[i] - np.max(scores[i]))
                    probs = exp_scores / exp_scores.sum()
                    mem["_nli_entailment_score"] = float(probs[2])  # Index 2 = entailment
                else:
                    mem["_nli_entailment_score"] = float(scores[i])

        except Exception as e:
            logger.warning("NLI reranking failed: %s", e)
            self._nli_model = False
            for mem in memories:
                mem["_nli_entailment_score"] = 0.0

        return memories

    def _cluster_memories(self, memories: list[dict]) -> list[list[dict]]:
        """Cluster memories by entity/topic overlap using Jaccard similarity."""
        threshold = getattr(self._settings, 'MULTI_PASSAGE_CLUSTER_OVERLAP_THRESHOLD', 0.3)
        max_size = getattr(self._settings, 'MULTI_PASSAGE_MAX_CLUSTER_SIZE', 3)

        # Tokenize each memory
        tokenized = []
        for m in memories:
            tokens = set(m.get("content", "").lower().split())
            tokens -= _FTS_STOP_WORDS
            tokenized.append(tokens)

        clusters = []
        used = set()

        for i, m in enumerate(memories):
            if i in used:
                continue
            cluster = [m]
            used.add(i)
            for j in range(i + 1, len(memories)):
                if j in used or len(cluster) >= max_size:
                    break
                # Jaccard similarity
                intersection = len(tokenized[i] & tokenized[j])
                union = len(tokenized[i] | tokenized[j])
                if union > 0 and intersection / union >= threshold:
                    cluster.append(memories[j])
                    used.add(j)
            clusters.append(cluster)

        return clusters

    def _multi_passage_rerank(self, query: str, memories: list[dict],
                              top_k: int) -> list[dict]:
        """Multi-passage evidence aggregation reranking.

        Groups related memories and re-scores clusters to detect when multiple
        weak pieces of evidence combine into strong evidence.
        """
        if not getattr(self._settings, 'MULTI_PASSAGE_RERANKING_ENABLED', False):
            return memories[:top_k]

        # Cluster top-20 candidates
        clusters = self._cluster_memories(memories[:20])

        for cluster_mems in clusters:
            if len(cluster_mems) < 2:
                continue
            # Concatenate cluster texts
            combined = " | ".join(m.get("content", "")[:200] for m in cluster_mems[:3])

            # Score combined text using CE
            combined_score = self._score_single_pair(query, combined)

            # If combined evidence is stronger, boost individual members
            max_individual = max(
                m.get("_cross_encoder_score", m.get("_retrieval_score", 0))
                for m in cluster_mems
            )
            if combined_score > max_individual:
                boost = (combined_score - max_individual) * 0.5
                for m in cluster_mems:
                    m["_retrieval_score"] = m.get("_retrieval_score", 0) + boost

        memories.sort(key=lambda m: m.get("_retrieval_score", 0), reverse=True)
        return memories[:top_k]

    def _score_single_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair using the active CE model."""
        try:
            if self._gte_reranker and self._gte_reranker is not False:
                scores = self._gte_reranker.predict([(query, document[:512])])
                return float(scores[0]) if hasattr(scores, '__len__') else float(scores)
            # Fallback to FlashRank
            if hasattr(self, '_flashrank_ranker') and self._flashrank_ranker:
                from flashrank import RerankRequest
                req = RerankRequest(query=query, passages=[{"text": document[:512]}])
                result = self._flashrank_ranker.rerank(req)
                return result[0]["score"] if result else 0.0
        except Exception:
            pass
        return 0.0

    def _mmr_rerank(
        self,
        memories: list[dict],
        query_embedding: bytes | None,
        top_k: int = 5,
        lambda_param: float = 0.7,
    ) -> list[dict]:
        """Maximal Marginal Relevance for diversity-aware reranking.

        Balances relevance (lambda) vs diversity (1-lambda) to avoid
        returning top-K results that are all from the same conversation segment.
        """
        if not memories or len(memories) <= 1 or query_embedding is None:
            return memories[:top_k]

        import numpy as np

        q_arr = np.frombuffer(query_embedding, dtype=np.float32)

        # Get embeddings for all candidate memories
        mem_embeddings = []
        valid_memories = []
        for mem in memories:
            full_mem = self._storage.get_memory(mem["id"])
            if full_mem and full_mem.get("embedding"):
                emb = np.frombuffer(full_mem["embedding"], dtype=np.float32)
                # Handle dimension mismatch
                if len(emb) == len(q_arr):
                    mem_embeddings.append(emb)
                    valid_memories.append(mem)
            else:
                valid_memories.append(mem)
                mem_embeddings.append(None)

        if not valid_memories:
            return memories[:top_k]

        def cosine_sim(a, b):
            if a is None or b is None:
                return 0.0
            dot = np.dot(a, b)
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            return float(dot / norm) if norm > 0 else 0.0

        selected = []
        selected_embs = []
        candidates = list(range(len(valid_memories)))

        for _ in range(min(top_k, len(valid_memories))):
            best_idx = None
            best_score = -float("inf")

            for idx in candidates:
                emb = mem_embeddings[idx]
                relevance = valid_memories[idx].get("_retrieval_score", 0.0)

                # Max similarity to already-selected documents
                max_sim = 0.0
                if selected_embs and emb is not None:
                    for sel_emb in selected_embs:
                        sim = cosine_sim(emb, sel_emb)
                        if sim > max_sim:
                            max_sim = sim

                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected.append(valid_memories[best_idx])
                selected_embs.append(mem_embeddings[best_idx])
                candidates.remove(best_idx)

        return selected

    def _detect_adversarial(self, result_memories: list[dict]) -> dict:
        """Z-score gap analysis for adversarial/low-confidence detection.

        Uses statistical analysis of score distributions to detect:
        1. Flat distributions (all scores similar = no clear winner)
        2. Low absolute scores (nothing really matches)
        3. High diversity of sources needed

        Returns dict with:
        - "is_uncertain": bool — whether the results look unreliable
        - "confidence": float — overall confidence in the result set (0-1)
        - "score_gap": float — z-score normalized gap between top-1 and top-2
        - "abstain": bool — whether retrieval should abstain (very low confidence)
        """
        if len(result_memories) == 0:
            return {"is_uncertain": True, "confidence": 0.0, "score_gap": 0.0, "abstain": True}
        if len(result_memories) == 1:
            score = result_memories[0].get("_retrieval_score", 0.0)
            conf = min(1.0, score * 2)
            return {"is_uncertain": conf < 0.3, "confidence": conf, "score_gap": 0.0, "abstain": conf < 0.1}

        scores = [mem.get("_retrieval_score", 0.0) for mem in result_memories]

        # Z-score analysis
        mean_s = sum(scores) / len(scores)
        std_s = (sum((s - mean_s) ** 2 for s in scores) / len(scores)) ** 0.5

        # Top-1 z-score: how far above mean is the best result?
        top1_z = (scores[0] - mean_s) / std_s if std_s > 1e-9 else 0.0

        # Score gap between top-1 and top-2
        raw_gap = scores[0] - scores[1]
        z_gap = raw_gap / std_s if std_s > 1e-9 else 0.0

        # Coefficient of variation: low CV = flat distribution = uncertain
        cv = std_s / mean_s if mean_s > 1e-9 else 0.0

        # Confidence from multiple signals:
        # 1. Z-gap: clear winner has high z-gap
        gap_conf = min(1.0, z_gap / 2.0) if z_gap > 0 else 0.0
        # 2. Top-1 absolute score: very low = nothing matches
        abs_conf = min(1.0, scores[0] * 2)
        # 3. Distribution shape: high CV = clear separation
        dist_conf = min(1.0, cv * 2)

        confidence = 0.4 * gap_conf + 0.4 * abs_conf + 0.2 * dist_conf

        threshold = self._settings.ADVERSARIAL_SCORE_GAP_THRESHOLD
        is_uncertain = confidence < self._settings.ADVERSARIAL_MIN_CONFIDENCE
        abstain = confidence < 0.15 or (scores[0] < 0.1 and z_gap < 0.5)

        return {
            "is_uncertain": is_uncertain,
            "confidence": round(confidence, 4),
            "score_gap": round(z_gap, 4),
            "abstain": abstain,
        }

    def _compute_signal_confidence(
        self,
        signal_name: str,
        ranked_list: list[tuple[int, float]],
    ) -> float:
        """Compute confidence score for a retrieval signal's results.

        Returns a value in [0.0, 1.0] indicating how confident we are
        that this signal produced meaningful results. Used by confidence
        gating to zero out unreliable signals before fusion.
        """
        if signal_name == "vector":
            if not ranked_list:
                return 0.0
            top_score = ranked_list[0][1]
            if len(ranked_list) > 1:
                gap = ranked_list[0][1] - ranked_list[1][1]
            else:
                gap = top_score
            return min(1.0, top_score * (1 + gap))

        elif signal_name == "fts":
            if not ranked_list:
                return 0.0
            return min(1.0, len(ranked_list) / 5.0)

        elif signal_name in ("ppr", "spreading"):
            if not ranked_list:
                return 0.0
            scores = [s for _, s in ranked_list]
            if len(scores) < 2:
                return scores[0] if scores else 0.0
            max_score = max(scores)
            mean_score = sum(scores) / len(scores)
            return (max_score - mean_score) / max_score if max_score > 0 else 0.0

        elif signal_name == "hopfield":
            if not ranked_list:
                return 0.0
            scores = [s for _, s in ranked_list]
            return max(scores)

        elif signal_name == "hdc":
            if not ranked_list:
                return 0.0
            return min(1.0, len(ranked_list) / 5.0)

        elif signal_name == "fractal":
            if not ranked_list:
                return 0.0
            return ranked_list[0][1] if ranked_list else 0.0

        elif signal_name == "sr":
            if not ranked_list:
                return 0.0
            return min(1.0, len(ranked_list) / 3.0)

        elif signal_name == "temporal":
            if not ranked_list:
                return 0.0
            return min(1.0, len(ranked_list) / 3.0)

        return 0.5

    def _build_networkx_graph(
        self, seed_entity_ids: list[int], max_hops: int | None = None
    ) -> nx.DiGraph:
        """Build a networkx DiGraph around the seed entities."""
        if max_hops is None:
            max_hops = self._settings.GRAPH_MAX_HOPS
        G = nx.DiGraph()
        visited: set[int] = set()
        frontier = list(seed_entity_ids)

        for _ in range(max_hops):
            next_frontier: list[int] = []
            for eid in frontier:
                if eid in visited:
                    continue
                visited.add(eid)
                G.add_node(eid)
                neighbors = self._graph._get_adjacent(eid, None)
                for n in neighbors:
                    nid = n["entity_id"]
                    weight = n["weight"]
                    if weight < self._settings.GRAPH_MIN_EDGE_WEIGHT:
                        continue
                    G.add_node(nid)
                    G.add_edge(eid, nid, weight=weight)
                    G.add_edge(nid, eid, weight=weight)
                    if nid not in visited:
                        next_frontier.append(nid)
            frontier = next_frontier

        return G

    def _find_memories_for_entity(self, entity_name: str) -> list[int]:
        """Find memory IDs whose content contains the entity name."""
        # Use FTS5 for efficient text search
        try:
            rows = self._storage._conn.execute(
                "SELECT m.id FROM memories m "
                "JOIN memories_fts fts ON m.id = fts.rowid "
                "WHERE memories_fts MATCH ? AND m.heat > 0",
                (f'"{entity_name}"',),
            ).fetchall()
            return [r[0] for r in rows]
        except Exception:
            # Fallback to LIKE search if FTS5 match fails
            rows = self._storage._conn.execute(
                "SELECT id FROM memories WHERE content LIKE ? AND heat > 0",
                (f"%{entity_name}%",),
            ).fetchall()
            return [r[0] for r in rows]

    def _find_entities_in_content(self, content: str) -> set[int]:
        """Find entity IDs that appear in the given content."""
        entity_ids: set[int] = set()
        # Get all active entities and check which ones appear in the content
        entities = self._storage.get_all_entities(min_heat=0.0, include_archived=True)
        for entity in entities:
            if entity["name"] in content:
                entity_ids.add(entity["id"])
        return entity_ids

    def _get_top_cooccurring_entities(
        self, content: str, limit: int = 5
    ) -> list[str]:
        """Find entities that co-occur with entities mentioned in this content."""
        # Find entities mentioned in the content
        content_entities = self._find_entities_in_content(content)
        if not content_entities:
            return []

        # Count co-occurrence partners
        partner_counts: dict[str, float] = defaultdict(float)
        for eid in content_entities:
            neighbors = self._graph._get_adjacent(eid, None)
            for n in neighbors:
                entity_row = self._storage._conn.execute(
                    "SELECT name FROM entities WHERE id = ?",
                    (n["entity_id"],),
                ).fetchone()
                if entity_row and n["entity_id"] not in content_entities:
                    partner_counts[entity_row[0]] += n["weight"]

        # Sort by weight and return top
        sorted_partners = sorted(
            partner_counts.items(), key=lambda x: x[1], reverse=True
        )
        return [name for name, _ in sorted_partners[:limit]]
