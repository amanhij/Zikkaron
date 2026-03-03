# Zikkaron

<!-- mcp-name: io.github.amanhij/zikkaron -->

[![PyPI](https://img.shields.io/pypi/v/zikkaron)](https://pypi.org/project/zikkaron/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-891%20passed-brightgreen)](#testing)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

Persistent memory for Claude Code. 26 cognitive subsystems, 18 MCP tools, runs locally on SQLite.

## The Problem

Every time you start a new Claude Code session, it forgets everything. Architecture decisions, debugging history, project conventions, file patterns you explained three times already — all gone. You end up re-explaining your entire codebase from scratch.

Zikkaron fixes this. It gives Claude Code a persistent memory that survives across sessions, consolidates over time, and surfaces the right context when you need it.

## Quick Start

```bash
pip install zikkaron
```

Add to your Claude Code MCP config (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "zikkaron": {
      "command": "zikkaron"
    }
  }
}
```

Done. Claude Code now has persistent memory.

### Make Claude use it automatically

Add this to your project's `CLAUDE.md`:

```markdown
## Memory — Zikkaron
- On every new session, call `recall` with the current project name to load prior context
- Before starting any task, call `get_project_context` for the current working directory
- After completing any significant task, call `remember` to store what was done, decisions made, and outcomes
```

## What It Looks Like

**Session 1** — You're debugging a tricky auth issue:
```
Tool: remember
  content: "Auth tokens expire silently when Redis cache is cold-started.
            Fix: added token refresh middleware in auth/middleware.py.
            Root cause was TTL mismatch between Redis and JWT expiry."
  context: "myapp backend debugging"
  tags: ["auth", "redis", "debugging"]
```

**Session 2** — Days later, a related bug appears. Claude automatically recalls:
```
Tool: recall
  query: "authentication token issues"

> Memory #42 (heat: 0.87): Auth tokens expire silently when Redis cache
> is cold-started. Fix: added token refresh middleware in auth/middleware.py.
> Root cause was TTL mismatch between Redis and JWT expiry.
```

No re-explaining. No digging through old conversations. It just remembers.

## How It Works

Zikkaron isn't a text file that gets loaded on startup. It's a memory engine built on computational neuroscience:

- **Predictive coding write gate** — Only stores what's actually new. Redundant information is filtered at ingest.
- **Heat-based salience** — Frequently accessed memories stay hot. Unused ones decay naturally, like biological memory.
- **Sleep consolidation** — Background process replays memories, discovers cross-project connections, and compresses old knowledge.
- **Reconsolidation** — Memories update when retrieved in new contexts, staying accurate as your codebase evolves.
- **Fractal hierarchy** — Memories cluster into summaries at multiple scales. Drill down from high-level architecture to specific implementation details.
- **Knowledge graph** — Entities and relationships are extracted and linked. Personalized PageRank surfaces contextually relevant memories.
- **Causal discovery** — Learns cause-effect relationships from your coding sessions using the PC algorithm.
- **Successor representations** — Memories that co-occur in similar contexts cluster together, even when their content differs.

All data stays on your machine in a single SQLite database. No cloud, no API calls, no telemetry.

## MCP Tools

Zikkaron exposes 18 tools over MCP:

| Tool | What it does |
|---|---|
| `remember` | Store a memory (passes through the predictive coding write gate) |
| `recall` | Semantic + keyword search with heat-weighted ranking |
| `forget` | Delete a memory |
| `validate_memory` | Check if a memory is still valid against current file state |
| `get_project_context` | Get all active memories for a directory |
| `consolidate_now` | Trigger a consolidation cycle |
| `memory_stats` | System statistics |
| `rate_memory` | Give usefulness feedback for metamemory tracking |
| `recall_hierarchical` | Query the fractal hierarchy at a specific level |
| `drill_down` | Navigate into a memory cluster |
| `create_trigger` | Set a prospective trigger that fires on matching context |
| `get_project_story` | Get the autobiographical narrative for a project |
| `add_rule` | Define neuro-symbolic rules for filtering/re-ranking |
| `get_rules` | List active rules |
| `navigate_memory` | Traverse concept space using successor representations |
| `get_causal_chain` | Get causal ancestors/descendants for an entity |
| `assess_coverage` | Evaluate knowledge coverage with gap identification |
| `detect_gaps` | Find knowledge gaps: isolated entities, stale regions, missing connections |

## Architecture

Zikkaron runs as a local MCP server. All data stays on your machine in a single SQLite database with WAL mode, FTS5 full-text search, and `sqlite-vec` for vector similarity.

26 subsystems organized into five tiers:

<details>
<summary><strong>Core Storage and Retrieval</strong></summary>

| Module | Role |
|---|---|
| `storage.py` | SQLite WAL engine with 15 tables, FTS5 indexing, `sqlite-vec` ANN search |
| `embeddings.py` | Sentence-transformer encoding (`all-MiniLM-L6-v2`) with batched operations |
| `retrieval.py` | Multi-signal fusion retriever combining vector similarity, FTS5 BM25, knowledge graph PPR, spreading activation, and fractal hierarchy traversal |
| `models.py` | Pydantic data models for memories, entities, relationships, clusters, rules, and causal edges |
| `config.py` | Environment-based configuration with `ZIKKARON_` prefix |

</details>

<details>
<summary><strong>Memory Dynamics</strong></summary>

| Module | Role |
|---|---|
| `thermodynamics.py` | Heat-based memory salience. Surprise scoring, importance heuristics, emotional valence, and temporal decay govern which memories stay accessible |
| `reconsolidation.py` | Memories become labile on retrieval and are rewritten based on context mismatch magnitude. Implements the Nader et al. (2000) reconsolidation model with three outcomes: reinforcement, modification, or archival |
| `predictive_coding.py` | Write gate that only stores prediction errors. Maintains a generative model per directory context and computes surprisal against existing knowledge — redundant information is filtered at ingest |
| `engram.py` | Competitive memory slot allocation based on CREB-like excitability (Josselyn & Frankland, 2007). High-excitability slots win allocation; temporally proximate memories share engram slots |
| `compression.py` | Rate-distortion optimal forgetting (Toth et al., 2020). Memories degrade progressively: full fidelity at 0-7 days, gist compression at 7-30 days, semantic tag extraction beyond 30 days |
| `staleness.py` | File-change watchdog using SHA-256 hashing to detect when source code has diverged from stored memories |

</details>

<details>
<summary><strong>Consolidation and Organization</strong></summary>

| Module | Role |
|---|---|
| `consolidation.py` | Background astrocyte daemon running periodic consolidation cycles: decay application, staleness checks, prospective trigger evaluation |
| `astrocyte_pool.py` | Domain-specialized consolidation processes (code structure, architectural decisions, error patterns, dependency tracking) running as a worker pool |
| `sleep_compute.py` | Offline "dream replay" that replays memory pairs to discover cross-project connections, runs Louvain community detection for clustering, and performs temporal compression |
| `fractal.py` | Hierarchical multi-scale memory tree. Memories cluster at leaf level; clusters merge into intermediate summaries; summaries merge into root abstractions. Supports drill-down navigation |
| `cls_store.py` | Complementary Learning Systems (McClelland et al., 1995). Dual-store architecture: fast episodic capture in a hippocampal buffer, slow semantic abstraction in a neocortical store with periodic interleaved replay |

</details>

<details>
<summary><strong>Knowledge Structure</strong></summary>

| Module | Role |
|---|---|
| `knowledge_graph.py` | Typed entity-relationship graph with co-occurrence, causal, and temporal edges. Supports Personalized PageRank for contextual retrieval |
| `causal_discovery.py` | PC algorithm (Spirtes, Glymour, Scheines, 2000) for discovering causal DAGs from coding session event logs. Conditional independence testing via partial correlation |
| `cognitive_map.py` | Successor Representation (Stachenfeld et al., 2017) for navigation-based retrieval. Memories that co-occur in similar contexts cluster in SR space, enabling associative traversal even when content differs |
| `narrative.py` | Autobiographical project stories synthesized from memory timelines, key decisions, and significant events |
| `curation.py` | Automated memory maintenance: duplicate merging, contradiction detection, cross-reference linking |

</details>

<details>
<summary><strong>Frontier Capabilities</strong></summary>

| Module | Role |
|---|---|
| `hopfield.py` | Modern continuous Hopfield networks (Ramsauer et al., 2021). Energy-based associative retrieval equivalent to transformer attention: `softmax(beta * X^T * query)` |
| `hdc_encoder.py` | Hyperdimensional Computing / Vector Symbolic Architecture (Kanerva, 1988). Encodes memories as role-filler bindings in 10,000-dimensional bipolar space for structured queries |
| `metacognition.py` | Self-assessment of knowledge coverage. Gap detection across five dimensions: isolated entities, stale regions, low-confidence zones, missing connections, one-sided knowledge |
| `rules_engine.py` | Neuro-symbolic constraints. Hard rules (must satisfy) and soft rules (preference boosts/penalties) scoped to global, directory, or file level |
| `crdt_sync.py` | Multi-agent memory sharing via CRDTs (OR-Set for collections, LWW-Register for content, G-Counter for access counts). Automatic conflict resolution across agent instances |
| `prospective.py` | Future-oriented triggers that fire when matching context is detected — directory, keyword, entity, or time-based conditions |
| `sensory_buffer.py` | Episodic capture buffer for raw session content with configurable token windows and overlap |

</details>

## Advanced Setup

### From source

```bash
git clone https://github.com/amanhij/Zikkaron.git
cd Zikkaron
pip install -e .
```

### SSE transport

For running as a persistent background server instead of stdio:

```bash
zikkaron --transport sse
```

Then configure Claude Code to connect via URL:

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

Default port: `8742`. Override with `--port`. Database defaults to `~/.zikkaron/memory.db`, override with `--db-path`.

## Configuration

All settings are configurable via environment variables with the `ZIKKARON_` prefix:

| Variable | Default | Description |
|---|---|---|
| `ZIKKARON_PORT` | `8742` | Server port |
| `ZIKKARON_DB_PATH` | `~/.zikkaron/memory.db` | Database location |
| `ZIKKARON_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `ZIKKARON_DECAY_FACTOR` | `0.95` | Base heat decay per consolidation cycle |
| `ZIKKARON_COLD_THRESHOLD` | `0.05` | Heat below which memories are candidates for archival |
| `ZIKKARON_WRITE_GATE_THRESHOLD` | `0.4` | Minimum surprisal to pass the predictive coding write gate |
| `ZIKKARON_HOPFIELD_BETA` | `8.0` | Hopfield network sharpness parameter |
| `ZIKKARON_SR_DISCOUNT` | `0.9` | Successor representation discount factor |
| `ZIKKARON_COGNITIVE_LOAD_LIMIT` | `4` | Maximum chunks in active context (Cowan's 4 +/- 1) |

See `zikkaron/config.py` for the full list.

## Testing

```bash
python -m pytest zikkaron/tests/ -x -q
```

891 tests across 33 test files covering all subsystems.

## References

<details>
<summary>Academic papers and books that informed the implementation</summary>

- Ramsauer et al. "Hopfield Networks is All You Need" (ICLR 2021, arXiv:2008.02217)
- Nader, Schafe, LeDoux. "Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval" (Nature 406, 2000)
- Osan, Tort, Bhatt, Bhatt, Bhatt, Amaral. "Three outcomes of reconsolidation" (PLoS ONE, 2011)
- McClelland, McNaughton, O'Reilly. "Why there are complementary learning systems in the hippocampus and neocortex" (Psych. Review 102, 1995)
- Sun et al. "Organizing memories for generalization in complementary learning systems" (Nature Neuroscience 26, 2023)
- Stachenfeld, Botvinick, Gershman. "The hippocampus as a predictive map" (Nature Neuroscience 20, 2017)
- Whittington et al. "The Tolman-Eichenbaum Machine" (Cell 183, 2020)
- Spirtes, Glymour, Scheines. *Causation, Prediction, and Search* (MIT Press, 2000)
- Kanerva. *Sparse Distributed Memory* (MIT Press, 1988)
- Frady, Kleyko, Sommer. "Variable Binding for Sparse Distributed Representations" (IEEE TNNLS, 2022)
- Toth et al. "Optimal forgetting via rate-distortion theory" (PLoS Computational Biology, 2020)
- Josselyn, Frankland. "Memory allocation: mechanisms and function" (Annual Review Neuroscience 41, 2018)
- Rashid et al. "Competition between engrams influences fear memory formation and recall" (Science 353, 2016)
- Zhou et al. "MetaRAG: Metacognitive Retrieval-Augmented Generation" (ACM Web, 2024)

</details>

## License

MIT
