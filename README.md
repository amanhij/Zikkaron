# Zikkaron

<!-- mcp-name: io.github.amanhij/zikkaron -->

[![PyPI](https://img.shields.io/pypi/v/zikkaron)](https://pypi.org/project/zikkaron/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-718%20passed-brightgreen)](#testing)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

*Zikkaron (זיכרון) is Hebrew for "memory."*

Your AI forgets you every time you close the tab. Every architecture decision you explained, every debugging rabbit hole you went down together, every "remember, we're using Postgres not SQLite" correction. Gone. You start the next session a stranger to your own tools.

Zikkaron is a persistent memory engine for Claude Code built on computational neuroscience. It remembers what you worked on, how you think, what you decided and why. Not as a dumb text dump that gets shoved into context, but as a living memory system that consolidates, forgets intelligently, and reconstructs the right context at the right time.

26 subsystems. 23 MCP tools. Runs entirely on your machine. One SQLite file.

## Two minutes to never repeat yourself again

```bash
pip install zikkaron
```

Add to your Claude Code config:

```json
{
  "mcpServers": {
    "zikkaron": {
      "command": "zikkaron"
    }
  }
}
```

Tell Claude how to use it. Drop this in your global `~/CLAUDE.md` (your home directory, not per-project):

```markdown
## Memory
- On every new session, call `recall` with the current project name
- Before starting any task, call `get_project_context` for the current directory
- After completing significant work, call `remember` to store decisions and outcomes
```

Or just let Zikkaron handle it. On every startup, it automatically syncs `~/CLAUDE.md` with the latest instructions via `sync_instructions`. You set it up once and never think about it again.

## What this actually feels like

**Monday.** You spend an hour debugging a nasty auth token race condition. Claude helps you trace it to a TTL mismatch between Redis and your JWT config. You fix it. Claude stores the memory.

**Thursday.** A user reports intermittent logouts. You open Claude Code in the same project. Before you even describe the bug, Claude recalls the Redis TTL fix from Monday, checks if it's related, and asks whether the middleware you added is handling the edge case where Redis restarts mid-session.

That's the difference. Not "here's your conversation history." Real recall. The kind where your tools understand the shape of what you've been building, not just the words you typed last time.

## Retrieval that actually works

We tested Zikkaron against [LoCoMo](https://snap-research.github.io/locomo/) (Maharana et al., ACL 2024), the standard benchmark for long conversation memory. 10 conversations, 1,986 questions, everything from simple factual lookups to multi-hop reasoning to adversarial trick questions designed to trip you up.

| | Zikkaron | What it means |
|---|---|---|
| **Recall@10** | **86.8%** | The right memory shows up in the top 10 nearly 9 times out of 10 |
| **MRR** | **0.708** | The correct answer is usually the first or second result |
| **Single-hop MRR** | **0.757** | Factual questions, almost always nails it on the first try |
| **Temporal MRR** | **0.712** | "When did X happen?" queries, strong time awareness |

The thing is, there's no LLM running at query time. No API calls. No billion parameter models. Just a 22MB embedding model, a SQLite file, and a bunch of neuroscience algorithms doing the heavy lifting. Most systems that hit numbers like these need GPT-4 in the loop. Zikkaron gets there with Hopfield energy scoring, spreading activation, and a cross-encoder reranker.

## Hippocampal Replay: Context that survives compaction

Here's a problem nobody talks about. Claude Code has a 200K token context window. During long sessions, when that window fills up, it *compacts*: summarizes older messages, strips tool outputs, paraphrases your instructions. Important nuance evaporates. Decisions you anchored early in the conversation dissolve into vague summaries.

**Hippocampal Replay** fixes this. Named after the neuroscience phenomenon where your brain replays important experiences during sleep to consolidate them into long-term memory, it treats context compaction as the "sleep" and replays what matters when Claude "wakes up."

**How it works:**

Before compaction hits, a hook fires. Zikkaron drains your active context: what you were working on, which files were open, what decisions you'd made, what errors were unresolved. It stores all of this as a checkpoint.

After compaction, a second hook fires. Zikkaron reconstructs your context intelligently. Not by dumping everything back in, but by assembling the right pieces: your latest checkpoint, any facts you'd anchored as critical, the hottest project memories, and predictions about what you'll need next based on your usage patterns.

You can also be explicit about what matters:

```
Tool: anchor
  content: "We're using the event-sourcing pattern. All state changes go through the event bus."
  reason: "Architecture constraint"
```

Anchored memories get maximum protection. They always survive compaction, no matter what.

**One-time setup per project:**

```
Tool: install_hooks
  project_directory: "/path/to/your/project"
```

After that, everything is automatic. You don't think about it. You don't call anything manually. The hooks fire, the context drains, the context restores. Your long sessions just... work.

## The science under the hood

Zikkaron doesn't store memories the way a database stores rows. It treats them more like a brain treats experiences.

**Memories have temperature.** Every memory starts hot. If you keep accessing it, it stays hot. If you don't, it cools. Below a threshold, it compresses: first to a gist, then to tags, then eventually it fades entirely. This isn't a bug. It's rate-distortion optimal forgetting, the same mathematical framework your brain uses to decide what's worth keeping. Important memories resist compression. Surprising ones get a heat boost. Boring, redundant ones quietly disappear.

**Storage has a gatekeeper.** Not everything deserves to be remembered. Zikkaron maintains a predictive model of what it already knows, and only stores information that violates its expectations. Tell it the same thing twice and the write gate blocks the second attempt. This is predictive coding: the same mechanism your neocortex uses to filter sensory input. Only prediction errors get through.

**Retrieval changes the memory.** When you recall a memory in a new context, it doesn't just passively hand it back. It compares the retrieval context against the storage context, and if there's enough mismatch, it *reconsolidates*: updates the memory to reflect what's true now. Severe mismatch archives the old version and creates a new one. This is real neuroscience. Nader et al. showed in 2000 that retrieved memories become labile and can be rewritten. Your codebase evolves, and so do Zikkaron's memories of it.

**Memories compete for space.** A pool of engram slots, each with an excitability score that spikes on use and decays over time. When a new memory arrives, it goes to the most excitable slot. Memories in the same slot get temporally linked, creating chains of related experiences even when their content has nothing in common. This models how real neurons allocate engrams through CREB-dependent excitability.

**Background consolidation runs like sleep.** When you're idle, an astrocyte daemon wakes up and processes recent experiences. It extracts entities and relationships, builds the knowledge graph, merges near-duplicates, discovers causal chains, and runs "dream replay" where random memory pairs are compared and new connections emerge. Four domain-specialized processes handle different types of knowledge at different rates: code structure, architectural decisions, error patterns, and dependencies.

**A cognitive map organizes everything.** Successor representations build a 2D map of concept space where memories that get accessed in similar contexts cluster together, even if their content is completely different. Debugging memories cluster near other debugging memories. Architecture decisions cluster together. Navigate this map, and you find related knowledge that keyword search would never surface.

## All 23 tools

| Tool | Purpose |
|------|---------|
| `remember` | Store a memory through the predictive coding write gate |
| `recall` | Multi-signal retrieval with heat-weighted ranking |
| `forget` | Delete a memory |
| `validate_memory` | Check staleness against current file state |
| `get_project_context` | Get hot memories for a directory |
| `consolidate_now` | Force a consolidation cycle |
| `memory_stats` | System statistics across all subsystems |
| `rate_memory` | Usefulness feedback for metamemory tracking |
| `recall_hierarchical` | Query the fractal hierarchy at a specific abstraction level |
| `drill_down` | Navigate into a memory cluster |
| `create_trigger` | Set prospective triggers that fire on matching context |
| `get_project_story` | Autobiographical narrative of a project |
| `add_rule` | Neuro-symbolic constraints for filtering and re-ranking |
| `get_rules` | List active rules |
| `navigate_memory` | Traverse concept space via successor representations |
| `get_causal_chain` | Causal ancestors and descendants for an entity |
| `assess_coverage` | Evaluate knowledge coverage with gap identification |
| `detect_gaps` | Find isolated entities, stale regions, missing connections |
| `checkpoint` | Snapshot working state for compaction recovery |
| `restore` | Reconstruct context after compaction via Hippocampal Replay |
| `anchor` | Mark critical facts as compaction-resistant |
| `install_hooks` | Enable automatic drain/restore on context compaction |
| `sync_instructions` | Update CLAUDE.md with latest Zikkaron capabilities |

## Architecture

Everything runs locally. A single SQLite database with WAL mode, FTS5 full-text search, and `sqlite-vec` for approximate nearest neighbor vector search.

26 subsystems organized into five layers:

<details>
<summary><strong>Core Storage and Retrieval</strong></summary>

| Module | Role |
|--------|------|
| `storage.py` | SQLite WAL engine, 15 tables, FTS5 indexing, `sqlite-vec` ANN search |
| `embeddings.py` | Sentence-transformer encoding (`all-MiniLM-L6-v2`), batched operations |
| `retrieval.py` | Four-signal fusion: vector similarity, FTS5 BM25, knowledge graph PPR, spreading activation |
| `models.py` | Pydantic data models for the full type hierarchy |
| `config.py` | Environment-based configuration with `ZIKKARON_` prefix |

</details>

<details>
<summary><strong>Memory Dynamics</strong></summary>

| Module | Role |
|--------|------|
| `thermodynamics.py` | Heat, surprise, importance, emotional valence, temporal decay |
| `reconsolidation.py` | Labile retrieval with three outcomes per Nader et al. (2000) |
| `predictive_coding.py` | Write gate that filters redundancy via prediction error |
| `engram.py` | Competitive slot allocation with CREB-like excitability |
| `compression.py` | Rate-distortion optimal forgetting over three compression levels |
| `staleness.py` | File-change watchdog via SHA-256 hash comparison |

</details>

<details>
<summary><strong>Consolidation and Organization</strong></summary>

| Module | Role |
|--------|------|
| `consolidation.py` | Background astrocyte daemon for periodic consolidation |
| `astrocyte_pool.py` | Domain-specialized worker processes for code, decisions, errors, deps |
| `sleep_compute.py` | Dream replay, Louvain community detection, temporal compression |
| `fractal.py` | Multi-scale memory tree with drill-down navigation |
| `cls_store.py` | Complementary Learning Systems: fast episodic + slow semantic stores |

</details>

<details>
<summary><strong>Knowledge Structure</strong></summary>

| Module | Role |
|--------|------|
| `knowledge_graph.py` | Typed entity-relationship graph with Personalized PageRank |
| `causal_discovery.py` | PC algorithm for causal DAGs from coding session data |
| `cognitive_map.py` | Successor Representation for navigation-based retrieval |
| `narrative.py` | Autobiographical project story synthesis |
| `curation.py` | Duplicate merging, contradiction detection, cross-reference linking |

</details>

<details>
<summary><strong>Frontier Capabilities</strong></summary>

| Module | Role |
|--------|------|
| `hopfield.py` | Modern continuous Hopfield networks (Ramsauer et al., 2021) |
| `hdc_encoder.py` | Hyperdimensional Computing in 10,000-dimensional bipolar space |
| `metacognition.py` | Self-assessment of knowledge coverage and gap detection |
| `rules_engine.py` | Hard and soft neuro-symbolic constraints |
| `crdt_sync.py` | Multi-agent memory sharing via CRDTs |
| `prospective.py` | Future-oriented triggers on directory, keyword, entity, or time |
| `sensory_buffer.py` | Episodic capture buffer for raw session content |
| `restoration.py` | Hippocampal Replay engine for context compaction resilience |

</details>

## Advanced setup

### From source

```bash
git clone https://github.com/amanhij/Zikkaron.git
cd Zikkaron
pip install -e .
```

### SSE transport

Run as a persistent background server instead of stdio:

```bash
zikkaron --transport sse
```

Then point Claude Code at the URL:

```json
{
  "mcpServers": {
    "zikkaron": {
      "type": "sse",
      "url": "http://127.0.0.1:8742/sse"
    }
  }
}
```

Default port is `8742`. Override with `--port`. Database defaults to `~/.zikkaron/memory.db`, override with `--db-path`.

## Configuration

All settings use the `ZIKKARON_` environment variable prefix:

| Variable | Default | What it controls |
|----------|---------|-----------------|
| `ZIKKARON_PORT` | `8742` | Server port |
| `ZIKKARON_DB_PATH` | `~/.zikkaron/memory.db` | Database location |
| `ZIKKARON_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `ZIKKARON_DECAY_FACTOR` | `0.95` | Heat decay per consolidation cycle |
| `ZIKKARON_COLD_THRESHOLD` | `0.05` | Heat below which memories become archival candidates |
| `ZIKKARON_WRITE_GATE_THRESHOLD` | `0.4` | Minimum surprisal to pass the write gate |
| `ZIKKARON_HOPFIELD_BETA` | `8.0` | Hopfield network sharpness |
| `ZIKKARON_SR_DISCOUNT` | `0.9` | Successor representation discount factor |
| `ZIKKARON_COGNITIVE_LOAD_LIMIT` | `4` | Active context chunk limit (Cowan's 4 +/- 1) |

Full list in `zikkaron/config.py`.

## Testing

```bash
python -m pytest zikkaron/tests/ -x -q
```

718 tests across 34 test files covering every subsystem.

## References

<details>
<summary>The papers and books behind the implementation</summary>

Ramsauer et al. "Hopfield Networks is All You Need" (ICLR 2021, arXiv:2008.02217)

Nader, Schafe, LeDoux. "Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval" (Nature 406, 2000)

Osan, Tort, Bhatt, Amaral. "Three outcomes of reconsolidation" (PLoS ONE, 2011)

McClelland, McNaughton, O'Reilly. "Why there are complementary learning systems in the hippocampus and neocortex" (Psychological Review 102, 1995)

Sun et al. "Organizing memories for generalization in complementary learning systems" (Nature Neuroscience 26, 2023)

Stachenfeld, Botvinick, Gershman. "The hippocampus as a predictive map" (Nature Neuroscience 20, 2017)

Whittington et al. "The Tolman-Eichenbaum Machine" (Cell 183, 2020)

Spirtes, Glymour, Scheines. *Causation, Prediction, and Search* (MIT Press, 2000)

Kanerva. *Sparse Distributed Memory* (MIT Press, 1988)

Frady, Kleyko, Sommer. "Variable Binding for Sparse Distributed Representations" (IEEE TNNLS, 2022)

Toth et al. "Optimal forgetting via rate-distortion theory" (PLoS Computational Biology, 2020)

Josselyn, Frankland. "Memory allocation: mechanisms and function" (Annual Review Neuroscience 41, 2018)

Rashid et al. "Competition between engrams influences fear memory formation and recall" (Science 353, 2016)

Zhou et al. "MetaRAG: Metacognitive Retrieval-Augmented Generation" (ACM Web, 2024)

</details>

## License

MIT
