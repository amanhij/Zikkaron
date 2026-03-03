"""Entry point for python -m zikkaron."""

import argparse

from zikkaron import __version__
from zikkaron.server import main

VALID_TRANSPORTS = ("sse", "streamable-http")

STARTUP_BANNER = f"""\
=== Zikkaron v{__version__} ===
Biologically-inspired persistent memory engine for Claude Code

Active modules:
  * StorageEngine         (SQLite WAL + FTS5 + sqlite-vec)
  * EmbeddingEngine       (sentence-transformers)
  * SensoryBuffer         (episode capture)
  * MemoryThermodynamics  (surprise, importance, valence, decay)
  * KnowledgeGraph        (typed relationships, causal detection)
  * HippoRetriever        (PPR + vector + FTS5 + spreading activation + fractal)
  * MemoryCurator         (merge/link/create, contradiction, memify)
  * AstrocyteEngine       (background consolidation daemon)
  * AstrocytePool         (domain-aware processes: code/decisions/errors/deps)
  * SleepComputeEngine    (dream replay, compression, community detection)
  * FractalMemoryTree     (hierarchical multi-scale retrieval)
  * ProspectiveMemory     (future-oriented triggers)
  * NarrativeEngine       (autobiographical project stories)
  * StalenessDetector     (file-change watchdog)

MCP Tools: remember, recall, forget, validate_memory, get_project_context,
           consolidate_now, memory_stats, rate_memory, recall_hierarchical,
           drill_down, create_trigger, get_project_story

MCP Resources: memory://stats, memory://hot, memory://stale,
               memory://processes, memory://narrative/{{directory}}
"""


def cli():
    parser = argparse.ArgumentParser(description="Zikkaron memory engine MCP server")
    parser.add_argument("--port", type=int, default=None, help="Server port (default: 8742)")
    parser.add_argument("--db-path", type=str, default=None, help="SQLite database path")
    parser.add_argument(
        "--transport",
        type=str,
        default="sse",
        choices=VALID_TRANSPORTS,
        help="MCP transport protocol (default: sse)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress startup banner",
    )
    args = parser.parse_args()

    if not args.quiet:
        print(STARTUP_BANNER)
        print(f"Transport: {args.transport}")
        if args.port:
            print(f"Port: {args.port}")
        if args.db_path:
            print(f"Database: {args.db_path}")
        print()

    main(port=args.port, db_path=args.db_path, transport=args.transport)


if __name__ == "__main__":
    cli()
