# Zikkaron

Persistent memory engine for Claude Code, implemented as an MCP server.

Large language model agents lose all accumulated knowledge when a session ends. Context windows function as volatile working memory — once cleared, every architectural decision, debugging insight, and project-specific convention must be re-established from scratch. Zikkaron provides a durable memory substrate that persists across sessions, using mechanisms drawn from computational neuroscience to organize, consolidate, and retrieve knowledge over time.

## Architecture

Zikkaron runs as a local MCP (Model Context Protocol) server over SSE. All data stays on your machine in a single SQLite database with WAL mode, FTS5 full-text search, and `sqlite-vec` for vector similarity.

The system is composed of 26 subsystems organized into three tiers:

### Core Storage and Retrieval

| Module | Role |
|---|---|
| `storage.py` | SQLite WAL engine with 15 tables, FTS5 indexing, `sqlite-vec` ANN search |
| `embeddings.py` | Sentence-transformer encoding (`all-MiniLM-L6-v2`) with batched operations |
| `retrieval.py` | Multi-signal fusion retriever combining vector similarity, FTS5 BM25, knowledge graph PPR, spreading activation, and fractal hierarchy traversal |
| `models.py` | Pydantic data models for memories, entities, relationships, clusters, rules, and causal edges |
| `config.py` | Environment-based configuration with `ZIKKARON_` prefix |

### Memory Dynamics

| Module | Role |
|---|---|
| `thermodynamics.py` | Heat-based memory salience. Surprise scoring, importance heuristics, emotional valence, and temporal decay govern which memories stay accessible |
| `reconsolidation.py` | Memories become labile on retrieval and are rewritten based on context mismatch magnitude. Implements the Nader et al. (2000) reconsolidation model with three outcomes: reinforcement, modification, or archival |
| `predictive_coding.py` | Write gate that only stores prediction errors. Maintains a generative model per directory context and computes surprisal against existing knowledge — redundant information is filtered at ingest |
| `engram.py` | Competitive memory slot allocation based on CREB-like excitability (Josselyn & Frankland, 2007). High-excitability slots win allocation; temporally proximate memories share engram slots |
| `compression.py` | Rate-distortion optimal forgetting (Toth et al., 2020). Memories degrade progressively: full fidelity at 0-7 days, gist compression at 7-30 days, semantic tag extraction beyond 30 days |
| `staleness.py` | File-change watchdog using SHA-256 hashing to detect when source code has diverged from stored memories |

### Consolidation and Organization

| Module | Role |
|---|---|
| `consolidation.py` | Background astrocyte daemon running periodic consolidation cycles: decay application, staleness checks, prospective trigger evaluation |
| `astrocyte_pool.py` | Domain-specialized consolidation processes (code structure, architectural decisions, error patterns, dependency tracking) running as a worker pool |
| `sleep_compute.py` | Offline "dream replay" that replays memory pairs to discover cross-project connections, runs Louvain community detection for clustering, and performs temporal compression |
| `fractal.py` | Hierarchical multi-scale memory tree. Memories cluster at leaf level; clusters merge into intermediate summaries; summaries merge into root abstractions. Supports drill-down navigation |
| `cls_store.py` | Complementary Learning Systems (McClelland et al., 1995). Dual-store architecture: fast episodic capture in a hippocampal buffer, slow semantic abstraction in a neocortical store with periodic interleaved replay |

### Knowledge Structure

| Module | Role |
|---|---|
| `knowledge_graph.py` | Typed entity-relationship graph with co-occurrence, causal, and temporal edges. Supports Personalized PageRank for contextual retrieval |
| `causal_discovery.py` | PC algorithm (Spirtes, Glymour, Scheines, 2000) for discovering causal DAGs from coding session event logs. Conditional independence testing via partial correlation |
| `cognitive_map.py` | Successor Representation (Stachenfeld et al., 2017) for navigation-based retrieval. Memories that co-occur in similar contexts cluster in SR space, enabling associative traversal even when content differs |
| `narrative.py` | Autobiographical project stories synthesized from memory timelines, key decisions, and significant events |
| `curation.py` | Automated memory maintenance: duplicate merging, contradiction detection, cross-reference linking |

### Frontier Capabilities

| Module | Role |
|---|---|
| `hopfield.py` | Modern continuous Hopfield networks (Ramsauer et al., 2021). Energy-based associative retrieval equivalent to transformer attention: `softmax(beta * X^T * query)` |
| `hdc_encoder.py` | Hyperdimensional Computing / Vector Symbolic Architecture (Kanerva, 1988). Encodes memories as role-filler bindings in 10,000-dimensional bipolar space for structured queries |
| `metacognition.py` | Self-assessment of knowledge coverage. Gap detection across five dimensions: isolated entities, stale regions, low-confidence zones, missing connections, one-sided knowledge |
| `rules_engine.py` | Neuro-symbolic constraints. Hard rules (must satisfy) and soft rules (preference boosts/penalties) scoped to global, directory, or file level |
| `crdt_sync.py` | Multi-agent memory sharing via CRDTs (OR-Set for collections, LWW-Register for content, G-Counter for access counts). Automatic conflict resolution across agent instances |
| `prospective.py` | Future-oriented triggers that fire when matching context is detected — directory, keyword, entity, or time-based conditions |
| `sensory_buffer.py` | Episodic capture buffer for raw session content with configurable token windows and overlap |

## MCP Tools

Zikkaron exposes 18 tools over MCP:

| Tool | Description |
|---|---|
| `remember` | Store a new memory. Passes through the predictive coding write gate and engram allocation |
| `recall` | Semantic + keyword search with heat-weighted ranking. Automatically boosts accessed memories |
| `forget` | Mark a memory for deletion by zeroing heat, then remove |
| `validate_memory` | Check memory validity against current file state via SHA-256 |
| `get_project_context` | Return all active memories for a directory, sorted by heat |
| `consolidate_now` | Trigger an immediate consolidation cycle |
| `memory_stats` | System statistics: counts, averages, subsystem status |
| `rate_memory` | Provide usefulness feedback for metamemory tracking |
| `recall_hierarchical` | Retrieve from the fractal hierarchy at a specific level or adaptively |
| `drill_down` | Navigate into a cluster to see its constituent memories |
| `create_trigger` | Create a prospective memory trigger for future context matching |
| `get_project_story` | Get the autobiographical narrative for a project directory |
| `add_rule` | Define a neuro-symbolic rule for filtering or re-ranking retrieval |
| `get_rules` | List active rules, optionally filtered by directory |
| `navigate_memory` | Traverse concept space using successor representation cognitive maps |
| `get_causal_chain` | Get causal ancestors and descendants for an entity from the PC algorithm DAG |
| `assess_coverage` | Evaluate knowledge coverage for a topic with gap identification |
| `detect_gaps` | Find knowledge gaps in a project: isolated entities, stale regions, missing connections |

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/amanhij/Zikkaron.git
cd Zikkaron
pip install -r requirements.txt
```

### Running the server

```bash
python -m zikkaron
```

Default port: `8742`. Override with `--port`. Database defaults to `~/.zikkaron/memory.db`, override with `--db-path`.

### Connecting to Claude Code

Add to your Claude Code MCP configuration (`~/.claude/settings.json`):

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

## Dependencies

- **mcp** — Model Context Protocol SDK
- **sentence-transformers** — Local embedding generation
- **sqlite-vec** — Vector similarity search extension for SQLite
- **networkx** — Graph algorithms for knowledge graph and causal discovery
- **numpy** — Numerical operations for Hopfield networks, HDC, and SR
- **fastapi / uvicorn / sse-starlette** — HTTP server and SSE transport
- **pydantic** — Data validation and settings management
- **watchdog** — File system change detection

## References

The implementation draws on ideas from the following work:

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

## License

MIT
