"""Entry point for python -m zikkaron."""

import argparse
import sys

from zikkaron import __version__
from zikkaron.server import main

VALID_TRANSPORTS = ("stdio", "sse", "streamable-http")

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


def _init_replay_lightweight(db_path=None):
    """Initialize only the engines needed for drain/restore (no daemons, no server)."""
    import logging
    # Suppress all library logging — hooks must only output data to stdout
    logging.disable(logging.CRITICAL)

    from zikkaron.config import Settings
    from zikkaron.storage import StorageEngine
    from zikkaron.embeddings import EmbeddingEngine
    from zikkaron.cognitive_map import CognitiveMap
    from zikkaron.metacognition import MetaCognition
    from zikkaron.knowledge_graph import KnowledgeGraph
    from zikkaron.retrieval import HippoRetriever
    from zikkaron.restoration import HippocampalReplay

    settings = Settings()
    storage = StorageEngine(db_path or settings.DB_PATH)
    embeddings = EmbeddingEngine(settings.EMBEDDING_MODEL)
    kg = KnowledgeGraph(storage, settings)
    cognitive_map = CognitiveMap(storage, settings)
    retriever = HippoRetriever(storage, embeddings, kg, settings)
    retriever.set_cognitive_map(cognitive_map)
    metacognition = MetaCognition(storage, embeddings, kg, settings)

    replay = HippocampalReplay(
        storage=storage,
        embeddings=embeddings,
        retriever=retriever,
        cognitive_map=cognitive_map,
        metacognition=metacognition,
        settings=settings,
    )
    return storage, replay


def cmd_drain(args):
    """Pre-compaction drain: save context to DB before Claude compacts."""
    import json
    directory = args.directory
    storage, replay = _init_replay_lightweight(args.db_path)
    try:
        result = replay.pre_compact_drain(directory)
        # Output JSON to stdout so hook can parse it if needed
        print(json.dumps(result))
    finally:
        storage.close()


def cmd_restore(args):
    """Post-compaction restore: reconstruct context and print markdown to stdout."""
    import json
    directory = args.directory
    storage, replay = _init_replay_lightweight(args.db_path)
    try:
        result = replay.restore(directory)
        formatted = result.get("formatted", "")
        if formatted:
            print(formatted)
    finally:
        storage.close()


def cli():
    parser = argparse.ArgumentParser(description="Zikkaron memory engine MCP server")
    subparsers = parser.add_subparsers(dest="command")

    # Default server mode (no subcommand)
    parser.add_argument("--port", type=int, default=None, help="Server port (default: 8742)")
    parser.add_argument("--db-path", type=str, default=None, help="SQLite database path")
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=VALID_TRANSPORTS,
        help="MCP transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress startup banner",
    )

    # drain subcommand
    drain_parser = subparsers.add_parser("drain", help="Pre-compaction context drain")
    drain_parser.add_argument("directory", help="Project directory")
    drain_parser.add_argument("--db-path", type=str, default=None, help="SQLite database path")

    # restore subcommand
    restore_parser = subparsers.add_parser("restore", help="Post-compaction context restore")
    restore_parser.add_argument("directory", help="Project directory")
    restore_parser.add_argument("--db-path", type=str, default=None, help="SQLite database path")

    args = parser.parse_args()

    if args.command == "drain":
        cmd_drain(args)
    elif args.command == "restore":
        cmd_restore(args)
    else:
        # Default: run MCP server
        if not args.quiet and args.transport != "stdio":
            print(STARTUP_BANNER, file=sys.stderr)
            print(f"Transport: {args.transport}", file=sys.stderr)
            if args.port:
                print(f"Port: {args.port}", file=sys.stderr)
            if args.db_path:
                print(f"Database: {args.db_path}", file=sys.stderr)
            print(file=sys.stderr)

        main(port=args.port, db_path=args.db_path, transport=args.transport)


if __name__ == "__main__":
    cli()
