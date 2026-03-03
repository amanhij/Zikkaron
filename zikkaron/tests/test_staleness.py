import pytest

from zikkaron.config import Settings
from zikkaron.staleness import StalenessDetector
from zikkaron.storage import StorageEngine


@pytest.fixture
def storage(tmp_path):
    db_path = str(tmp_path / "test_memory.db")
    engine = StorageEngine(db_path)
    yield engine
    engine.close()


@pytest.fixture
def settings():
    return Settings(DB_PATH=":memory:")


@pytest.fixture
def detector(storage, settings):
    return StalenessDetector(storage, settings)


def _make_memory(content="test memory", directory="/tmp/project", **kwargs):
    base = {
        "content": content,
        "directory_context": directory,
        "tags": ["test"],
    }
    base.update(kwargs)
    return base


class TestComputeFileHash:
    def test_compute_file_hash(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")

        hash1 = StalenessDetector._compute_file_hash(str(f))
        hash2 = StalenessDetector._compute_file_hash(str(f))

        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA-256 hex digest
        assert hash1 == hash2

    def test_hash_changes_on_modification(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("original content")
        hash1 = StalenessDetector._compute_file_hash(str(f))

        f.write_text("modified content")
        hash2 = StalenessDetector._compute_file_hash(str(f))

        assert hash1 != hash2

    def test_hash_returns_empty_for_missing_file(self, tmp_path):
        result = StalenessDetector._compute_file_hash(str(tmp_path / "nope.txt"))
        assert result == ""


class TestValidateMemory:
    def test_validate_memory_fresh(self, storage, detector, tmp_path):
        f = tmp_path / "source.py"
        f.write_text("def main(): pass")

        file_hash = StalenessDetector._compute_file_hash(str(f))
        storage.upsert_file_hash(str(f), file_hash)

        mem_id = storage.insert_memory(
            _make_memory(
                content="main function",
                directory=str(tmp_path),
                file_hash=file_hash,
            )
        )

        result = detector.validate_memory(mem_id)
        assert result["valid"] is True
        assert result["reason"] == "file unchanged"

    def test_validate_memory_stale(self, storage, detector, tmp_path):
        f = tmp_path / "source.py"
        f.write_text("def main(): pass")

        file_hash = StalenessDetector._compute_file_hash(str(f))
        storage.upsert_file_hash(str(f), file_hash)

        mem_id = storage.insert_memory(
            _make_memory(
                content="main function",
                directory=str(tmp_path),
                file_hash=file_hash,
            )
        )

        f.write_text("def main(): return 42")

        result = detector.validate_memory(mem_id)
        assert result["valid"] is False
        assert result["reason"] == "file changed"

        memory = storage.get_memory(mem_id)
        assert memory["is_stale"] is True

    def test_validate_memory_no_file_hash(self, storage, detector):
        mem_id = storage.insert_memory(_make_memory(content="no file ref"))

        result = detector.validate_memory(mem_id)
        assert result["valid"] is True
        assert result["reason"] == "no file reference"

    def test_validate_memory_not_found(self, detector):
        result = detector.validate_memory(9999)
        assert result["valid"] is False
        assert result["reason"] == "memory not found"


class TestScanDirectory:
    def test_scan_directory(self, storage, detector, tmp_path):
        a = tmp_path / "a.py"
        b = tmp_path / "b.py"
        a.write_text("content_a")
        b.write_text("content_b")

        hash_a = StalenessDetector._compute_file_hash(str(a))
        hash_b = StalenessDetector._compute_file_hash(str(b))
        storage.upsert_file_hash(str(a), hash_a)
        storage.upsert_file_hash(str(b), hash_b)

        storage.insert_memory(
            _make_memory(
                content="about a",
                directory=str(tmp_path),
                file_hash=hash_a,
            )
        )

        a.write_text("modified_a")

        result = detector.scan_directory(str(tmp_path))
        assert result["files_scanned"] == 2
        assert result["files_changed"] == 1
        assert result["memories_flagged"] >= 1

    def test_scan_skips_ignored_dirs(self, storage, detector, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("git config")

        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "mod.cpython-311.pyc").write_text("bytecode")

        normal = tmp_path / "main.py"
        normal.write_text("print('hi')")

        result = detector.scan_directory(str(tmp_path))
        assert result["files_scanned"] == 1

    def test_scan_skips_binary_files(self, storage, detector, tmp_path):
        text_file = tmp_path / "readme.txt"
        text_file.write_text("hello")

        binary_file = tmp_path / "image.dat"
        binary_file.write_bytes(b"\x00\x01\x02\x03")

        result = detector.scan_directory(str(tmp_path))
        assert result["files_scanned"] == 1


class TestStaleMemoryHeat:
    def test_stale_memory_heat_halved(self, storage, detector, tmp_path):
        f = tmp_path / "module.py"
        f.write_text("original code")

        file_hash = StalenessDetector._compute_file_hash(str(f))
        storage.upsert_file_hash(str(f), file_hash)

        mem_id = storage.insert_memory(
            _make_memory(
                content="module docs",
                directory=str(tmp_path),
                file_hash=file_hash,
                heat=0.8,
            )
        )

        f.write_text("changed code")

        detector.validate_memory(mem_id)

        memory = storage.get_memory(mem_id)
        assert memory["heat"] == pytest.approx(0.4)
        assert memory["is_stale"] is True

    def test_scan_halves_heat(self, storage, detector, tmp_path):
        f = tmp_path / "app.py"
        f.write_text("v1")

        file_hash = StalenessDetector._compute_file_hash(str(f))
        storage.upsert_file_hash(str(f), file_hash)

        mem_id = storage.insert_memory(
            _make_memory(
                content="app logic",
                directory=str(tmp_path),
                file_hash=file_hash,
                heat=1.0,
            )
        )

        f.write_text("v2")

        detector.scan_directory(str(tmp_path))

        memory = storage.get_memory(mem_id)
        assert memory["heat"] == pytest.approx(0.5)
        assert memory["is_stale"] is True


class TestFileDeletion:
    def test_file_deletion_detection(self, storage, detector, tmp_path):
        f = tmp_path / "ephemeral.py"
        f.write_text("temporary code")

        file_hash = StalenessDetector._compute_file_hash(str(f))
        storage.upsert_file_hash(str(f), file_hash)

        mem_id = storage.insert_memory(
            _make_memory(
                content="about ephemeral",
                directory=str(tmp_path),
                file_hash=file_hash,
            )
        )

        f.unlink()

        result = detector.validate_memory(mem_id)
        assert result["valid"] is False

        memory = storage.get_memory(mem_id)
        assert memory["is_stale"] is True
