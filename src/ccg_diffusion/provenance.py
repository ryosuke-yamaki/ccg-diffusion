"""Deterministic serialization, atomic artifacts and Git source sealing.

* ``canonical_json`` and the ``atomic_write_*`` writers define the bytes of
  every record CCG Diffusion writes (sorted keys, compact separators,
  ``ensure_ascii=False``, ``allow_nan=False``).
* ``repository_root`` and ``repo_relative`` locate repository files; new
  records store repository-relative paths.
* ``seal_components``, ``implementation_record``,
  ``verify_implementation_record``, ``git_head_components_record`` and
  ``source_tree_snapshot`` bind a run to the exact committed source bytes.

The module uses only the standard library.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


DEFAULT_CHUNK_SIZE = 8 << 20


@dataclass(frozen=True)
class ArtifactRecord:
    """Content-addressed description of one artifact."""

    path: Path
    sha256: str
    size_bytes: int

    def to_dict(self, *, relative_to: Path | None = None) -> dict[str, Any]:
        """Return a JSON-compatible record with an optional relative path."""

        path = self.path
        if relative_to is not None:
            try:
                path = path.resolve().relative_to(relative_to.resolve())
            except ValueError:
                path = self.path.resolve()
        return {
            "path": str(path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def canonical_json(
    value: Any,
    *,
    pretty: bool = False,
    compact: bool = True,
    ensure_ascii: bool = False,
    allow_nan: bool = False,
) -> str:
    """Serialize JSON with stable key ordering and whitespace.

    Compact output has no trailing newline. Pretty output uses two-space
    indentation and exactly one trailing newline, matching repository reports.
    """

    if pretty:
        return json.dumps(
            value,
            ensure_ascii=ensure_ascii,
            indent=2,
            sort_keys=True,
            allow_nan=allow_nan,
        ) + "\n"
    options: dict[str, Any] = {
        "ensure_ascii": ensure_ascii,
        "sort_keys": True,
        "allow_nan": allow_nan,
    }
    if compact:
        options["separators"] = (",", ":")
    return json.dumps(value, **options)


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest of ``value``."""

    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str, *, encoding: str = "utf-8") -> str:
    """Return the SHA-256 digest of encoded text."""

    return sha256_bytes(value.encode(encoding))


def sha256_file(path: str | Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Hash a file without loading it entirely into memory."""

    if isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: str | Path) -> ArtifactRecord:
    """Create a content-addressed record for an existing file."""

    resolved = Path(path)
    return ArtifactRecord(
        path=resolved,
        sha256=sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _atomic_write_chunks(
    path: str | Path,
    chunks: Iterable[bytes],
    *,
    overwrite: bool,
) -> ArtifactRecord:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and destination.exists():
        raise FileExistsError(destination)

    previous_mode = (
        stat.S_IMODE(destination.stat().st_mode) if destination.exists() else 0o644
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise TypeError("atomic write chunks must be bytes")
                handle.write(chunk)
                digest.update(chunk)
                size_bytes += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(previous_mode)
        if not overwrite and destination.exists():
            raise FileExistsError(destination)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return ArtifactRecord(
        path=destination,
        sha256=digest.hexdigest(),
        size_bytes=size_bytes,
    )


def atomic_write_bytes(
    path: str | Path,
    value: bytes,
    *,
    overwrite: bool = True,
) -> ArtifactRecord:
    """Atomically replace ``path`` with bytes written in the same directory."""

    if not isinstance(value, bytes):
        raise TypeError("value must be bytes")
    return _atomic_write_chunks(path, (value,), overwrite=overwrite)


def atomic_write_text(
    path: str | Path,
    value: str,
    *,
    encoding: str = "utf-8",
    overwrite: bool = True,
) -> ArtifactRecord:
    """Atomically write text with explicit encoding."""

    return atomic_write_bytes(path, value.encode(encoding), overwrite=overwrite)


def atomic_write_json(
    path: str | Path,
    value: Any,
    *,
    pretty: bool = True,
    ensure_ascii: bool = False,
    allow_nan: bool = False,
    overwrite: bool = True,
) -> ArtifactRecord:
    """Serialize and atomically write one deterministic JSON document."""

    content = canonical_json(
        value,
        pretty=pretty,
        ensure_ascii=ensure_ascii,
        allow_nan=allow_nan,
    )
    return atomic_write_text(path, content, overwrite=overwrite)


def _jsonl_chunks(
    rows: Iterable[Mapping[str, Any]],
    *,
    ensure_ascii: bool,
    allow_nan: bool,
    compact: bool,
) -> Iterator[bytes]:
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("JSONL rows must be mappings")
        yield (
            canonical_json(
                dict(row),
                compact=compact,
                ensure_ascii=ensure_ascii,
                allow_nan=allow_nan,
            )
            + "\n"
        ).encode("utf-8")


def atomic_write_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    ensure_ascii: bool = False,
    allow_nan: bool = False,
    compact: bool = True,
    overwrite: bool = True,
) -> ArtifactRecord:
    """Stream deterministic JSONL into a same-directory temporary file."""

    return _atomic_write_chunks(
        path,
        _jsonl_chunks(
            rows,
            ensure_ascii=ensure_ascii,
            allow_nan=allow_nan,
            compact=compact,
        ),
        overwrite=overwrite,
    )


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_jsonl(
    path: str | Path,
    *,
    reject_duplicate_keys: bool = False,
    require_objects: bool = True,
) -> list[Any]:
    """Read non-empty JSONL lines with path and line-number diagnostics."""

    source = Path(path)
    rows: list[Any] = []
    object_pairs_hook = _reject_duplicate_keys if reject_duplicate_keys else None
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line, object_pairs_hook=object_pairs_hook)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid JSONL at {source}:{line_number}: {exc}") from exc
            if require_objects and not isinstance(value, dict):
                raise ValueError(
                    f"invalid JSONL at {source}:{line_number}: expected an object"
                )
            rows.append(value)
    return rows


def digest(value: Any) -> str:
    """SHA-256 of the canonical JSON of ``value``."""

    return sha256_text(canonical_json(value))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("expected an object")
    return value


def immutable_json(path: Path, value: Any) -> None:
    """Permit identical preparation retries; reject replacement of a sealed artifact."""
    payload = canonical_json(value, pretty=True).encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Immutable artifact differs: {path}")
        return
    atomic_write_bytes(path, payload, overwrite=False)


def immutable_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(canonical_json(row) + "\n" for row in rows).encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Immutable rows differ: {path}")
        return
    atomic_write_bytes(path, payload, overwrite=False)


# --- repository paths ---------------------------------------------------------------------------------

# Historical records store absolute paths under the container checkout; readers
# re-root such a path against the repository root when it does not exist.
HISTORICAL_REPOSITORY_ROOT = "/workspace/holsd"


def repository_root() -> Path:
    """The source checkout that contains this package (editable install only)."""

    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file() or not (root / "configs" / "parsing").is_dir():
        raise RuntimeError("ccg_diffusion must be used from a source checkout; install it with: pip install -e .")
    return root


def _root(value: Path | None) -> Path:
    return (value if value is not None else repository_root()).resolve()


def repo_relative(path: Path | str, *, repository_root: Path | None = None) -> str:
    """The repository-relative POSIX path that new records store."""

    root = _root(repository_root)
    path = Path(path)
    return (path if path.is_absolute() else root / path).resolve().relative_to(root).as_posix()


def portable_path(path: Path) -> str:
    """Repository-relative path inside the checkout, absolute path elsewhere."""

    root = repository_root()
    resolved = path.resolve()
    return resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else str(resolved)


def resolve_recorded_path(value: str, *, repository_root: Path | None = None) -> Path:
    """Resolve a path read from a record.

    Relative paths resolve against the repository root; a historical
    ``/workspace/holsd/...`` path that does not exist is re-rooted there too.
    """

    root = _root(repository_root)
    path = Path(value)
    if not path.is_absolute():
        return (root / path).resolve()
    if not path.exists() and path.is_relative_to(HISTORICAL_REPOSITORY_ROOT):
        return (root / path.relative_to(HISTORICAL_REPOSITORY_ROOT)).resolve()
    return path.resolve()


def verify_artifact_record(record: Mapping[str, Any], *, repository_root: Path | None = None) -> Path:
    """Check that ``{"path", "sha256"[, "size_bytes"]}`` still describes the file; return its path."""

    if not isinstance(record, Mapping) or "path" not in record or "sha256" not in record:
        raise ValueError("artifact must contain path and SHA-256")
    if not isinstance(record["path"], str) or not record["path"]:
        raise ValueError("artifact path must be a nonempty string")
    path = resolve_recorded_path(record["path"], repository_root=repository_root)
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise ValueError(f"artifact bytes do not match their source seal: {path}")
    if "size_bytes" in record and path.stat().st_size != record["size_bytes"]:
        raise ValueError(f"artifact size differs from source seal: {path}")
    return path


# --- Git provenance and source sealing --------------------------------------------------------------


def _git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip()


def git_head(root: Path | None = None) -> str | None:
    """The checked-out commit, or ``None`` outside a readable repository."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=root,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


class SourceSealError(ValueError):
    """Source files are not the committed bytes a record requires."""


def _is_commit(value: Any, lengths: tuple[int, ...] = (40, 64)) -> bool:
    return isinstance(value, str) and len(value) in lengths and all(char in "0123456789abcdef" for char in value)


def _component_paths(component_files: Sequence[str]) -> tuple[str, ...]:
    paths = tuple(Path(value).as_posix() for value in component_files)
    if (
        not paths
        or len(set(paths)) != len(paths)
        or any(Path(value).is_absolute() or ".." in Path(value).parts for value in paths)
    ):
        raise SourceSealError("source component paths are invalid")
    return paths


def _git_blob(root: Path, commit: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=root, check=True, capture_output=True).stdout


def seal_components(
    component_files: Sequence[str], *, repository_root: Path | None = None,
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Bind clean runtime bytes to their exact Git HEAD blobs.

    Every listed file must be committed and unmodified (index and working
    tree); unrelated changes elsewhere in the repository do not matter.
    Returns ``(commit, {path: {"sha256", "size_bytes"}})``.
    """

    root = _root(repository_root)
    paths = _component_paths(component_files)
    try:
        commit = _git_output(root, "rev-parse", "HEAD")
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *paths],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise SourceSealError("failed to inspect source component Git provenance") from exc
    if not _is_commit(commit):
        raise SourceSealError("source Git HEAD is invalid")
    dirty = [line for line in status.splitlines() if line.strip()]
    if dirty:
        raise SourceSealError(
            f"source components contain uncommitted changes: {[line[3:] if len(line) > 3 else line for line in dirty]}"
        )
    components: dict[str, dict[str, Any]] = {}
    for relative_path in paths:
        runtime_path = root / relative_path
        if not runtime_path.is_file():
            raise SourceSealError(f"source component is missing: {relative_path}")
        runtime_bytes = runtime_path.read_bytes()
        try:
            git_bytes = _git_blob(root, commit, relative_path)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SourceSealError(f"source component is not tracked at HEAD: {relative_path}") from exc
        if runtime_bytes != git_bytes:
            raise SourceSealError(f"source component bytes differ from HEAD: {relative_path}")
        components[relative_path] = {"sha256": sha256_bytes(runtime_bytes), "size_bytes": len(runtime_bytes)}
    return commit, components


def require_committed(component_files: Sequence[str], *, repository_root: Path | None = None) -> None:
    """Fail unless every listed file is committed and unmodified."""

    seal_components(component_files, repository_root=repository_root)


def implementation_record(
    component_files: Sequence[str], *, schema: str, repository_root: Path | None = None,
) -> dict[str, Any]:
    """Seal ``component_files`` to the current commit under ``schema``."""

    commit, components = seal_components(component_files, repository_root=repository_root)
    return {
        "schema": schema,
        "repo_git_commit": commit,
        "components": components,
        "components_match_repo_git_commit": True,
    }


def verify_implementation_record(
    record: Any,
    component_files: Sequence[str],
    *,
    schema: str,
    at_head: bool = False,
    repository_root: Path | None = None,
) -> None:
    """Check that the current committed sources are the ones ``record`` sealed.

    With ``at_head=True`` the recorded commit must also be the checked-out
    commit, so a record becomes unusable after any later commit.
    """

    root = _root(repository_root)
    if not isinstance(record, dict) or set(record) != {
        "schema", "repo_git_commit", "components", "components_match_repo_git_commit",
    }:
        raise SourceSealError("implementation record has an unexpected field set")
    if record["schema"] != schema:
        raise SourceSealError(f"implementation record schema {record['schema']!r} differs from {schema!r}")
    if record["components_match_repo_git_commit"] is not True:
        raise SourceSealError("implementation record does not attest a clean commit")
    _, current = seal_components(component_files, repository_root=root)
    if record["components"] != current:
        raise SourceSealError("runtime component bytes differ from the implementation record")
    commit = record["repo_git_commit"]
    if not _is_commit(commit, (40,)):
        raise SourceSealError("implementation record needs a full Git commit")
    for path, expected in current.items():
        try:
            blob = _git_blob(root, commit, path)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SourceSealError(f"cannot inspect recorded Git blob: {path}") from exc
        if sha256_bytes(blob) != expected["sha256"] or len(blob) != expected["size_bytes"]:
            raise SourceSealError(f"source component differs from its recorded Git blob: {path}")
    if at_head and commit != git_head(root):
        raise SourceSealError("implementation record commit differs from HEAD; prepare the run again")


def git_head_components_record(
    component_files: Sequence[str], *, repository_root: Path | None = None,
) -> dict[str, Any]:
    """Run-lock record of the committed component bytes at HEAD."""

    commit, components = seal_components(component_files, repository_root=repository_root)
    return {
        "schema": "ccg_diffusion.git_head_components.v1",
        "repo_git_commit": commit,
        "component_sha256": {path: value["sha256"] for path, value in components.items()},
        "components_match_repo_git_commit": True,
    }


def source_tree_snapshot(
    roots: Sequence[str] = ("src/ccg_diffusion", "scripts"),
    *,
    extra_files: Sequence[str] = (),
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Commit, dirty flag and sha256 of every tracked or untracked-not-ignored ``*.py`` under ``roots``."""

    root = _root(repository_root)
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", *roots], cwd=root, text=True)
    hashes = {name: sha256_file(root / name) for name in sorted(set(paths.splitlines())) if name.endswith(".py")}
    for name in extra_files:
        hashes[name] = sha256_file(root / name)
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *roots], cwd=root, text=True)
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "dirty": bool(dirty.strip()), "files": hashes}
