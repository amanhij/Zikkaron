"""Project seeding — bootstrap Zikkaron memory for existing codebases.

Scans a project directory and creates foundational memories from:
- Project structure and layout
- Config files (package.json, pyproject.toml, Cargo.toml, etc.)
- Documentation (README, ARCHITECTURE, CONTRIBUTING, etc.)
- CI/CD configuration
- Entry points and key source files
"""

import fnmatch
import json
import logging
import os
import tomllib
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Directories to always skip
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", ".node_modules",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env", ".env",
    "dist", "build", "out", "target",
    ".next", ".nuxt", ".output",
    ".tox", ".nox",
    "vendor",
    ".terraform",
    ".idea", ".vscode",
    "coverage", ".coverage",
    ".lockstep",
    ".claude",
    "egg-info",
})

# Binary file extensions to skip reading
_BINARY_EXTENSIONS = frozenset({
    ".pyc", ".pyo", ".so", ".o", ".a", ".dll", ".exe", ".dylib",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac",
    ".wasm", ".class", ".jar",
    ".db", ".sqlite", ".sqlite3",
    ".lock",  # package lock files are huge and not useful
})

# Config files that reveal project structure and dependencies
# Keys are either exact filenames or glob patterns (prefixed with *)
_CONFIG_EXACT = {
    # Python
    "pyproject.toml": "python",
    "setup.py": "python",
    "setup.cfg": "python",
    "Pipfile": "python",
    "requirements.txt": "python",
    # JavaScript/TypeScript
    "package.json": "javascript",
    "tsconfig.json": "typescript",
    "tsconfig.base.json": "typescript",
    # Rust
    "Cargo.toml": "rust",
    # Go
    "go.mod": "go",
    "go.sum": "go",
    # Java/Kotlin
    "build.gradle": "java",
    "build.gradle.kts": "kotlin",
    "pom.xml": "java",
    # Ruby
    "Gemfile": "ruby",
    # PHP
    "composer.json": "php",
    # Docker
    "Dockerfile": "docker",
    "docker-compose.yml": "docker",
    "docker-compose.yaml": "docker",
    "compose.yml": "docker",
    "compose.yaml": "docker",
    # CI/CD
    "Makefile": "build",
    "Justfile": "build",
    "Taskfile.yml": "build",
    # Config
    ".env.example": "config",
    ".env.sample": "config",
}

# Glob patterns for config files (checked via fnmatch)
_CONFIG_GLOBS = [
    ("*.csproj", "csharp"),
    ("*.fsproj", "fsharp"),
]

# Documentation files to read
_DOC_FILES = frozenset({
    "README.md", "README.rst", "README.txt", "README",
    "ARCHITECTURE.md", "DESIGN.md",
    "CONTRIBUTING.md", "CONTRIBUTING.rst",
    "CHANGELOG.md", "CHANGES.md",
    "CLAUDE.md",
})

# Entry point patterns (checked with fnmatch)
_ENTRY_PATTERNS = [
    "main.*", "index.*", "app.*", "server.*",
    "cli.*", "cmd.*", "__main__.py",
    "src/main.*", "src/index.*", "src/app.*",
    "src/lib.*", "lib.rs",
]

# CI/CD files detected by name (outside of .github/.gitlab dirs)
_CI_FILES = frozenset({
    ".gitlab-ci.yml", "Jenkinsfile", ".travis.yml",
    "azure-pipelines.yml", ".drone.yml",
})

# Max file size to read (64KB)
_MAX_FILE_SIZE = 64 * 1024

# Max content length per memory
_MAX_MEMORY_CONTENT = 2000

# Heat values by memory type
_HEAT_BY_TYPE = {
    "overview": 0.9,
    "documentation": 0.85,
    "config": 0.7,
    "ci_cd": 0.6,
    "entry_point": 0.75,
    "component": 0.5,
}


def _match_config(fname: str) -> Optional[str]:
    """Match a filename against config files (exact + glob patterns)."""
    if fname in _CONFIG_EXACT:
        return _CONFIG_EXACT[fname]
    for pattern, language in _CONFIG_GLOBS:
        if fnmatch.fnmatch(fname, pattern):
            return language
    return None


def _should_skip_dir(name: str) -> bool:
    """Check if a directory should be skipped."""
    if name in _SKIP_DIRS:
        return True
    if name.startswith(".") and name not in (".github", ".gitlab"):
        return True
    if name.endswith(".egg-info"):
        return True
    return False


def _read_file_safe(path: Path) -> Optional[str]:
    """Read file content safely, respecting size limits and skipping binary files."""
    try:
        if path.suffix.lower() in _BINARY_EXTENSIONS:
            return None
        if path.stat().st_size > _MAX_FILE_SIZE:
            return None
        return path.read_text(errors="replace")
    except Exception:
        return None


def _truncate(text: str, max_len: int = _MAX_MEMORY_CONTENT) -> str:
    """Truncate text to max length, breaking at a line boundary."""
    if len(text) <= max_len:
        return text
    # Find last newline before limit, accounting for suffix
    suffix = "\n[... truncated]"
    effective_max = max_len - len(suffix)
    cut = text.rfind("\n", 0, effective_max)
    if cut < effective_max // 2:
        cut = effective_max
    return text[:cut] + suffix


def _on_walk_error(error: OSError) -> None:
    """Log permission errors during os.walk."""
    logger.warning("Skipped (permission denied): %s", error.filename or error)


def scan_project(directory: str) -> dict:
    """Scan a project directory and return structured data for seeding.

    Returns a dict with keys:
    - project_name: str
    - structure: dict (directory tree summary)
    - configs: list[dict] (config file contents and their relative paths)
    - docs: list[dict] (documentation file contents)
    - entry_points: list[dict] (main entry points found)
    - ci_cd: list[dict] (CI/CD config files)
    - stats: dict (file counts by extension, total dirs, etc.)
    """
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    project_name = root.name

    # Collect structure
    structure = {}  # rel_path -> list of files
    configs = []
    docs = []
    entry_points = []
    ci_cd = []
    ext_counts: dict[str, int] = {}
    total_files = 0
    total_dirs = 0

    for dirpath, dirnames, filenames in os.walk(
        str(root), followlinks=False, onerror=_on_walk_error
    ):
        # Filter out skip directories in-place
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
        dirnames.sort()

        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            rel_dir = ""

        total_dirs += 1

        dir_files = []
        for fname in sorted(filenames):
            filepath = Path(dirpath) / fname
            total_files += 1

            # Count extensions
            ext = filepath.suffix.lower()
            if ext:
                ext_counts[ext] = ext_counts.get(ext, 0) + 1

            dir_files.append(fname)

            # Check if it's a config file (exact match + glob patterns)
            config_lang = _match_config(fname)
            if config_lang is not None:
                content = _read_file_safe(filepath)
                if content:
                    configs.append({
                        "path": os.path.join(rel_dir, fname) if rel_dir else fname,
                        "language": config_lang,
                        "content": content,
                    })

            # Check if it's a doc file
            if fname in _DOC_FILES:
                content = _read_file_safe(filepath)
                if content:
                    docs.append({
                        "path": os.path.join(rel_dir, fname) if rel_dir else fname,
                        "content": content,
                    })

            # Check CI/CD
            if rel_dir in (".github/workflows", ".github", ".gitlab"):
                content = _read_file_safe(filepath)
                if content:
                    ci_cd.append({
                        "path": os.path.join(rel_dir, fname) if rel_dir else fname,
                        "content": content,
                    })
            elif fname in _CI_FILES:
                content = _read_file_safe(filepath)
                if content:
                    ci_cd.append({
                        "path": os.path.join(rel_dir, fname) if rel_dir else fname,
                        "content": content,
                    })

            # Check entry points (only in root or src/)
            if rel_dir in ("", "src"):
                for pattern in _ENTRY_PATTERNS:
                    rel_path = os.path.join(rel_dir, fname) if rel_dir else fname
                    if fnmatch.fnmatch(rel_path, pattern):
                        content = _read_file_safe(filepath)
                        if content:
                            entry_points.append({
                                "path": rel_path,
                                "content": content,
                            })
                        break

        if dir_files:
            structure[rel_dir or "."] = dir_files

    # Sort extensions by count
    top_extensions = sorted(ext_counts.items(), key=lambda x: -x[1])[:10]

    return {
        "project_name": project_name,
        "root": str(root),
        "structure": structure,
        "configs": configs,
        "docs": docs,
        "entry_points": entry_points,
        "ci_cd": ci_cd,
        "stats": {
            "total_files": total_files,
            "total_dirs": total_dirs,
            "top_extensions": top_extensions,
        },
    }


def _detect_stack(configs: list[dict], stats: dict) -> str:
    """Detect the primary tech stack from configs and file stats."""
    languages = set()
    for cfg in configs:
        languages.add(cfg["language"])

    ext_map = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
        ".rs": "Rust", ".go": "Go", ".java": "Java", ".kt": "Kotlin",
        ".rb": "Ruby", ".php": "PHP", ".cs": "C#", ".cpp": "C++",
        ".c": "C", ".swift": "Swift", ".ex": "Elixir", ".zig": "Zig",
        ".jsx": "React/JSX", ".tsx": "React/TSX", ".vue": "Vue",
        ".svelte": "Svelte",
    }

    for ext, count in stats.get("top_extensions", []):
        if ext in ext_map:
            languages.add(ext_map[ext])

    return ", ".join(sorted(languages)) if languages else "Unknown"


def _summarize_structure(structure: dict, max_depth: int = 3) -> str:
    """Create a concise directory tree summary."""
    lines = []
    dirs = sorted(structure.keys())

    for d in dirs:
        depth = d.count(os.sep) if d != "." else 0
        if depth >= max_depth:
            continue

        indent = "  " * depth
        dirname = os.path.basename(d) if d != "." else "."
        files = structure[d]
        file_count = len(files)

        # Show first few files and count
        if file_count <= 5:
            file_list = ", ".join(files)
        else:
            file_list = ", ".join(files[:4]) + f", ... (+{file_count - 4} more)"

        lines.append(f"{indent}{dirname}/ [{file_list}]")

    return "\n".join(lines[:50])  # Cap at 50 lines


def _summarize_package_json(content: str) -> str:
    """Extract key info from package.json."""
    try:
        pkg = json.loads(content)
    except Exception:
        return content[:500]

    parts = []
    if pkg.get("name"):
        parts.append(f"Name: {pkg['name']}")
    if pkg.get("description"):
        parts.append(f"Description: {pkg['description']}")
    if pkg.get("version"):
        parts.append(f"Version: {pkg['version']}")

    if pkg.get("scripts"):
        scripts = list(pkg["scripts"].keys())
        parts.append(f"Scripts: {', '.join(scripts[:10])}")

    deps = list(pkg.get("dependencies", {}).keys())
    if deps:
        parts.append(f"Dependencies ({len(deps)}): {', '.join(deps[:15])}")

    dev_deps = list(pkg.get("devDependencies", {}).keys())
    if dev_deps:
        parts.append(f"DevDependencies ({len(dev_deps)}): {', '.join(dev_deps[:10])}")

    if pkg.get("workspaces"):
        ws = pkg["workspaces"]
        if isinstance(ws, list):
            parts.append(f"Workspaces: {', '.join(ws)}")
        elif isinstance(ws, dict) and ws.get("packages"):
            parts.append(f"Workspaces: {', '.join(ws['packages'])}")

    return "\n".join(parts)


def _summarize_pyproject(content: str) -> str:
    """Extract key info from pyproject.toml using proper TOML parsing."""
    try:
        data = tomllib.loads(content)
    except Exception:
        # Fallback for malformed TOML
        return _truncate(content, 800)

    parts = []
    project = data.get("project", {})

    if project.get("name"):
        parts.append(f"Name: {project['name']}")
    if project.get("description"):
        parts.append(f"Description: {project['description']}")
    if project.get("version"):
        parts.append(f"Version: {project['version']}")
    if project.get("requires-python"):
        parts.append(f"Python: {project['requires-python']}")

    deps = project.get("dependencies", [])
    if deps:
        parts.append(f"Dependencies ({len(deps)}): {', '.join(str(d) for d in deps[:15])}")

    optional = project.get("optional-dependencies", {})
    for group, group_deps in list(optional.items())[:3]:
        parts.append(f"Optional [{group}] ({len(group_deps)}): {', '.join(str(d) for d in group_deps[:8])}")

    scripts = project.get("scripts", {})
    if scripts:
        script_strs = [f"{k}={v}" for k, v in list(scripts.items())[:5]]
        parts.append(f"Entry points: {', '.join(script_strs)}")

    # Build system
    build = data.get("build-system", {})
    if build.get("build-backend"):
        parts.append(f"Build backend: {build['build-backend']}")

    if not parts:
        return _truncate(content, 800)

    return "\n".join(parts)


def _summarize_cargo_toml(content: str) -> str:
    """Extract key info from Cargo.toml using proper TOML parsing."""
    try:
        data = tomllib.loads(content)
    except Exception:
        return _truncate(content, 800)

    parts = []
    pkg = data.get("package", {})

    if pkg.get("name"):
        parts.append(f"Name: {pkg['name']}")
    if pkg.get("description"):
        parts.append(f"Description: {pkg['description']}")
    if pkg.get("edition"):
        parts.append(f"Edition: {pkg['edition']}")

    deps = data.get("dependencies", {})
    if deps:
        dep_names = list(deps.keys())
        parts.append(f"Dependencies ({len(dep_names)}): {', '.join(dep_names[:15])}")

    dev_deps = data.get("dev-dependencies", {})
    if dev_deps:
        parts.append(f"Dev dependencies ({len(dev_deps)}): {', '.join(list(dev_deps.keys())[:10])}")

    # Workspace members
    workspace = data.get("workspace", {})
    members = workspace.get("members", [])
    if members:
        parts.append(f"Workspace members: {', '.join(members[:10])}")

    if not parts:
        return _truncate(content, 800)

    return "\n".join(parts)


def _summarize_go_mod(content: str) -> str:
    """Extract key info from go.mod."""
    import re
    parts = []
    mod_match = re.search(r'^module\s+(\S+)', content, re.MULTILINE)
    if mod_match:
        parts.append(f"Module: {mod_match.group(1)}")

    go_match = re.search(r'^go\s+(\S+)', content, re.MULTILINE)
    if go_match:
        parts.append(f"Go version: {go_match.group(1)}")

    requires = re.findall(r'^\s+(\S+)\s+v', content, re.MULTILINE)
    if requires:
        parts.append(f"Dependencies ({len(requires)}): {', '.join(requires[:15])}")

    if not parts:
        return _truncate(content, 800)

    return "\n".join(parts)


def _summarize_config(config: dict) -> str:
    """Summarize a config file based on its type."""
    path = config["path"]
    content = config["content"]
    fname = os.path.basename(path)

    if fname == "package.json":
        return _summarize_package_json(content)
    elif fname == "pyproject.toml":
        return _summarize_pyproject(content)
    elif fname == "Cargo.toml":
        return _summarize_cargo_toml(content)
    elif fname == "go.mod":
        return _summarize_go_mod(content)
    elif fname == "requirements.txt":
        lines = [l.strip() for l in content.splitlines() if l.strip() and not l.startswith("#")]
        return f"Python dependencies ({len(lines)}): {', '.join(lines[:20])}"
    elif fname in ("Dockerfile", "docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"):
        return _truncate(content, 1000)
    else:
        return _truncate(content, 800)


def _find_subproject_boundaries(structure: dict, configs: list[dict]) -> list[str]:
    """Detect sub-project boundaries in monorepos.

    A sub-project boundary is any directory containing a config file
    (package.json, Cargo.toml, pyproject.toml, etc.) that is not the root.
    """
    # Directories that contain their own config files
    config_dirs = set()
    for cfg in configs:
        cfg_dir = os.path.dirname(cfg["path"])
        if cfg_dir:  # Skip root-level configs
            config_dirs.add(cfg_dir)

    # Also check top-level dirs that don't have configs but have significant content
    top_dirs = set()
    for d in structure:
        if d != "." and "/" not in d and os.sep not in d:
            top_dirs.add(d)

    # Merge: config-bearing dirs + top-level dirs without config children
    boundaries = set(config_dirs)
    for d in top_dirs:
        # Only add if no config_dir is a child of this top-level dir
        has_config_child = any(cd.startswith(d + "/") or cd == d for cd in config_dirs)
        if not has_config_child:
            boundaries.add(d)

    return sorted(boundaries)


def generate_memories(scan_data: dict) -> list[dict]:
    """Generate memory entries from scan data.

    Returns a list of dicts with keys: content, context, tags, heat_type
    Each represents one memory to store.
    """
    memories = []
    directory = scan_data["root"]
    project_name = scan_data["project_name"]
    stats = scan_data["stats"]
    structure = scan_data["structure"]

    # 1. Project overview memory
    stack = _detect_stack(scan_data["configs"], stats)
    structure_summary = _summarize_structure(structure)
    overview = (
        f"Project: {project_name}\n"
        f"Stack: {stack}\n"
        f"Files: {stats['total_files']}, Directories: {stats['total_dirs']}\n"
        f"Top extensions: {', '.join(f'{ext}({n})' for ext, n in stats['top_extensions'][:8])}\n\n"
        f"Structure:\n{structure_summary}"
    )
    memories.append({
        "content": _truncate(overview),
        "context": directory,
        "tags": ["_seed", "overview", "structure"],
        "heat_type": "overview",
    })

    # 2. Config file memories
    for config in scan_data["configs"]:
        summary = _summarize_config(config)
        content = f"Config: {config['path']}\nLanguage: {config['language']}\n\n{summary}"
        memories.append({
            "content": _truncate(content),
            "context": directory,
            "tags": ["_seed", "config", config["language"]],
            "heat_type": "config",
        })

    # 3. Documentation memories
    for doc in scan_data["docs"]:
        # Include path-based context in tags for distinguishability
        doc_dir = os.path.dirname(doc["path"])
        extra_tags = [doc_dir] if doc_dir else []
        content = f"Documentation: {doc['path']}\n\n{doc['content']}"
        memories.append({
            "content": _truncate(content),
            "context": directory,
            "tags": ["_seed", "documentation"] + extra_tags,
            "heat_type": "documentation",
        })

    # 4. CI/CD memories
    if scan_data["ci_cd"]:
        ci_parts = []
        for ci in scan_data["ci_cd"]:
            ci_parts.append(f"--- {ci['path']} ---\n{_truncate(ci['content'], 600)}")
        ci_content = f"CI/CD configuration for {project_name}:\n\n" + "\n\n".join(ci_parts)
        memories.append({
            "content": _truncate(ci_content),
            "context": directory,
            "tags": ["_seed", "ci_cd", "devops"],
            "heat_type": "ci_cd",
        })

    # 5. Entry point memories
    for ep in scan_data["entry_points"]:
        content = f"Entry point: {ep['path']}\n\n{_truncate(ep['content'], 1500)}"
        memories.append({
            "content": _truncate(content),
            "context": directory,
            "tags": ["_seed", "entry_point"],
            "heat_type": "entry_point",
        })

    # 6. Per-component memories — use sub-project boundaries for monorepos
    boundaries = _find_subproject_boundaries(structure, scan_data["configs"])
    if not boundaries:
        # Fallback: top-level directories
        boundaries = sorted(
            d for d in structure if d != "." and "/" not in d and os.sep not in d
        )

    for d in boundaries:
        # Gather all files under this component
        sub_files = []
        for key, files in structure.items():
            if key == d or key.startswith(d + "/") or key.startswith(d + os.sep):
                for f in files:
                    rel = os.path.join(key, f)
                    sub_files.append(rel)

        if not sub_files:
            continue

        # Build component summary
        exts = {}
        for f in sub_files:
            ext = os.path.splitext(f)[1].lower()
            if ext:
                exts[ext] = exts.get(ext, 0) + 1

        ext_summary = ", ".join(f"{ext}({n})" for ext, n in sorted(exts.items(), key=lambda x: -x[1])[:5])

        # Get subdirectory structure
        subdirs = sorted(set(
            key for key in structure
            if (key.startswith(d + "/") or key.startswith(d + os.sep)) and key != d
        ))
        subdir_names = [os.path.relpath(sd, d) for sd in subdirs[:15]]

        # Check if this component has its own README
        component_readme = None
        for doc in scan_data["docs"]:
            doc_dir = os.path.dirname(doc["path"])
            if doc_dir == d:
                component_readme = os.path.basename(doc["path"])
                break

        content = f"Component: {d}/\n"
        content += f"Files: {len(sub_files)} ({ext_summary})\n"
        if subdir_names:
            content += f"Subdirectories: {', '.join(subdir_names)}\n"
        if component_readme:
            content += f"Has own documentation: {component_readme}\n"
        if len(sub_files) <= 20:
            content += f"All files: {', '.join(os.path.basename(f) for f in sub_files)}\n"
        else:
            content += f"Sample files: {', '.join(os.path.basename(f) for f in sub_files[:15])}, ... (+{len(sub_files)-15} more)\n"

        memories.append({
            "content": _truncate(content),
            "context": directory,
            "tags": ["_seed", "component", d],
            "heat_type": "component",
        })

    return memories


def _delete_existing_seed_memories(storage, directory: str) -> int:
    """Delete existing _seed tagged memories for this directory before re-seeding.

    Returns count of deleted memories.
    """
    rows = storage._conn.execute(
        "SELECT id FROM memories WHERE directory_context = ? AND tags LIKE '%\"_seed\"%'",
        (directory,),
    ).fetchall()
    if not rows:
        return 0

    ids = [r[0] for r in rows]
    for mid in ids:
        storage._conn.execute("DELETE FROM memories WHERE id = ?", (mid,))
        # Also clean up vec0
        try:
            storage._conn.execute(
                "DELETE FROM memory_vectors WHERE rowid = ?", (mid,)
            )
        except Exception:
            pass
    storage._conn.commit()
    return len(ids)


def seed_project(
    directory: str,
    db_path: Optional[str] = None,
    dry_run: bool = False,
    storage=None,
    embeddings=None,
    thermo=None,
    curator=None,
) -> dict:
    """Scan a project and store foundational memories.

    Args:
        directory: Project root directory to scan.
        db_path: Optional SQLite database path override.
        dry_run: If True, scan and generate but don't store.
        storage: Optional pre-initialized StorageEngine (to reuse server's).
        embeddings: Optional pre-initialized EmbeddingEngine.
        thermo: Optional pre-initialized MemoryThermodynamics.
        curator: Optional pre-initialized MemoryCurator.

    Returns:
        Dict with scan stats and memories created/skipped.
    """
    # Scan
    scan_data = scan_project(directory)
    memories = generate_memories(scan_data)

    if dry_run:
        return {
            "project": scan_data["project_name"],
            "directory": scan_data["root"],
            "stats": scan_data["stats"],
            "memories_generated": len(memories),
            "memories": [{"content": m["content"][:200], "tags": m["tags"]} for m in memories],
            "stored": False,
        }

    # Use provided engines or initialize our own
    own_storage = storage is None
    if own_storage:
        from zikkaron.config import Settings
        from zikkaron.storage import StorageEngine
        from zikkaron.embeddings import EmbeddingEngine
        from zikkaron.thermodynamics import MemoryThermodynamics
        from zikkaron.curation import MemoryCurator
        from zikkaron.knowledge_graph import KnowledgeGraph

        settings = Settings()
        storage = StorageEngine(db_path or settings.DB_PATH)
        embeddings = EmbeddingEngine(settings.EMBEDDING_MODEL)
        kg = KnowledgeGraph(storage, settings)
        thermo = MemoryThermodynamics(storage, embeddings, settings)
        curator = MemoryCurator(storage, embeddings, thermo, settings)

    created = 0
    replaced = 0

    try:
        # Delete existing seed memories for this directory (replace, don't append)
        deleted = _delete_existing_seed_memories(storage, scan_data["root"])
        if deleted:
            logger.info("Cleared %d existing seed memories for %s", deleted, scan_data["root"])
            replaced = deleted

        for mem in memories:
            content = mem["content"]
            context = mem["context"]
            tags = mem["tags"]
            heat_type = mem.get("heat_type", "component")

            # Generate embedding
            embedding = embeddings.encode(content)

            # Base heat from memory type
            base_heat = _HEAT_BY_TYPE.get(heat_type, 0.6)

            # Compute thermodynamic scores
            surprise = thermo.compute_surprise(content, context)
            importance = thermo.compute_importance(content, tags)
            valence = thermo.compute_valence(content)
            # Use modest surprise boost so seeded memories don't all max out
            initial_heat = min(base_heat + surprise * 0.1, 1.0)

            # Insert directly (no curator dedup since we already cleared old seeds)
            memory_id = storage.insert_memory({
                "content": content,
                "embedding": embedding,
                "tags": tags,
                "directory_context": context,
                "heat": initial_heat,
                "is_stale": False,
                "file_hash": None,
                "embedding_model": embeddings.get_model_name(),
            })

            # Set thermodynamic scores
            storage.update_memory_scores(
                memory_id,
                surprise_score=surprise,
                importance=importance,
                emotional_valence=valence,
            )

            created += 1
            logger.info("Seed memory [created]: %s", content[:80])

    finally:
        if own_storage:
            storage.close()

    return {
        "project": scan_data["project_name"],
        "directory": scan_data["root"],
        "stats": scan_data["stats"],
        "memories_generated": len(memories),
        "created": created,
        "replaced": replaced,
        "stored": True,
    }
