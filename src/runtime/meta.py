"""Collect and write ``run_meta.json`` to record environment information for each pipeline run.

Usage::

    from src.runtime.meta import write_run_meta
    started_at = datetime.now().astimezone()
    try:
        ...  # run pipeline
    finally:
        write_run_meta(cfg.output_dir, cfg, started_at=started_at)
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

# Packages to track; used in scenarios such as PipelineConfig.embedding_model; not hard dependencies
_PACKAGES_OF_INTEREST = (
    "torch",
    "transformers",
    "sentence-transformers",
    "sacrebleu",
    "spacy",
    "qwen-asr",
    "vllm",
    "matplotlib",
    "numpy",
    "tqdm",
    "PyYAML",
    "Cython",
)


def _safe_run(cmd: list[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def _git_info(repo_dir: Path) -> Dict[str, Any]:
    cwd = str(repo_dir)
    commit = _safe_run(["git", "-C", cwd, "rev-parse", "HEAD"])
    if commit is None:
        return {"available": False}
    branch = _safe_run(["git", "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"])
    status = _safe_run(["git", "-C", cwd, "status", "--porcelain"])
    return {
        "available": True,
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "dirty_files_count": len(status.splitlines()) if status else 0,
    }


def _package_versions(names: Iterable[str]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            out[name] = None
        except Exception:
            out[name] = None
    return out


def _cuda_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": False}
    try:
        import torch
    except Exception:
        return info
    info["torch_version"] = torch.__version__
    info["available"] = bool(torch.cuda.is_available())
    if not info["available"]:
        return info
    info["cuda_version"] = torch.version.cuda
    info["device_count"] = torch.cuda.device_count()
    try:
        info["device_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    except Exception:
        info["device_names"] = None
    return info


def _config_to_dict(cfg: Any) -> Dict[str, Any]:
    if is_dataclass(cfg):
        d = asdict(cfg)
    elif isinstance(cfg, dict):
        d = dict(cfg)
    else:
        d = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}

    def _coerce(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, (list, tuple)):
            return [_coerce(v) for v in value]
        if isinstance(value, dict):
            return {str(k): _coerce(v) for k, v in value.items()}
        return repr(value)

    return {k: _coerce(v) for k, v in d.items()}


def collect_run_meta(
    config: Any,
    *,
    started_at: datetime,
    finished_at: Optional[datetime] = None,
    status: str = "success",
    error: Optional[BaseException] = None,
    repo_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Assemble a full run-meta dictionary in memory (no disk I/O).

    Args:
        config: A :class:`PipelineConfig` (or any dataclass/dict/object) that
            will be serialised under the ``config`` key.
        started_at: When the pipeline started (timezone-aware preferred).
        finished_at: When the pipeline finished; defaults to "now".
        status: ``"success"`` or ``"failed"`` (free-form string).
        error: Exception raised by the pipeline, if any; recorded under the
            ``error`` key.
        repo_dir: Repo root used to query git info; defaults to two levels
            above this file.

    Returns:
        A JSON-serialisable dict with keys ``status``, ``started_at``,
        ``finished_at``, ``elapsed_sec``, ``host``, ``git``, ``packages``,
        ``cuda``, ``config``, ``argv`` (and ``error`` when ``error`` is set).
    """
    if finished_at is None:
        finished_at = datetime.now().astimezone()
    if repo_dir is None:
        repo_dir = Path(__file__).resolve().parents[2]

    meta: Dict[str, Any] = {
        "status": status,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_sec": round((finished_at - started_at).total_seconds(), 3),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "python_full": sys.version,
            "cwd": os.getcwd(),
        },
        "git": _git_info(repo_dir),
        "packages": _package_versions(_PACKAGES_OF_INTEREST),
        "cuda": _cuda_info(),
        "config": _config_to_dict(config),
        "argv": list(sys.argv),
    }
    if error is not None:
        meta["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    return meta


def write_run_meta(
    output_dir: str | os.PathLike,
    config: Any,
    *,
    started_at: datetime,
    finished_at: Optional[datetime] = None,
    status: str = "success",
    error: Optional[BaseException] = None,
    filename: str = "run_meta.json",
) -> Path:
    """Collect run-meta via :func:`collect_run_meta` and write it as JSON.

    Args:
        output_dir: Directory the meta file will be written to; created if missing.
        config: Passed through to :func:`collect_run_meta`.
        started_at: Passed through to :func:`collect_run_meta`.
        finished_at: Passed through to :func:`collect_run_meta`.
        status: Passed through to :func:`collect_run_meta`.
        error: Passed through to :func:`collect_run_meta`.
        filename: File name within ``output_dir`` (default ``run_meta.json``).

    Returns:
        Absolute :class:`pathlib.Path` of the written file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    meta = collect_run_meta(
        config,
        started_at=started_at,
        finished_at=finished_at,
        status=status,
        error=error,
    )
    out_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path
