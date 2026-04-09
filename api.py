"""
Speech-to-Speech Latency Evaluation Pipeline — FastAPI Service

Start:
    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1

Swagger UI:  http://localhost:8000/docs
"""

import ctypes
import gc
import json
import logging
import os
import signal
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from config import PipelineConfig
from src.alignment import step2_segale
from src.asr import step1_asr, step1_asr_vllm
from src.asr.qwen_char_spans import add_char_spans_for_dir
from src.evaluation import step3_longyaal
from src.intermediate.prepare_artifacts import asr_to_instances, instances_to_segale

logger = logging.getLogger("s2s-api")
logging.basicConfig(level=logging.INFO)

# ── FastAPI app ──────────────────────────────────────────────
app = FastAPI(
    title="S2S Latency Evaluation API",
    description="Speech-to-Speech Translation Latency Evaluation Pipeline API",
    version="1.0.0",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    docs_url=None,
)

_SWAGGER_CUSTOM_HTML = """
<script>
(function() {
    function patch() {
        // Hide: curl, request-url, static responses docs, Responses header, Code/Details header
        var hideSelectors = [
            '.curl-command',
            '.request-url',
            '.responses-wrapper > .opblock-section-header',
            '.live-responses-table thead'
        ];
        hideSelectors.forEach(function(sel) {
            document.querySelectorAll(sel).forEach(function(el) {
                el.style.display = 'none';
            });
        });

        // Hide static doc tables (keep live ones)
        document.querySelectorAll('.responses-inner > table').forEach(function(t) {
            if (!t.classList.contains('live-responses-table')) {
                t.style.display = 'none';
            }
        });

        // Hide all "Responses" headings and Response headers
        document.querySelectorAll('.responses-wrapper h4, .responses-wrapper h5').forEach(function(el) {
            var t = el.textContent.trim();
            if (t === 'Responses' || t === 'Response headers') {
                el.style.display = 'none';
            }
        });

        // Add success/failure banner
        document.querySelectorAll('table.live-responses-table').forEach(function(table) {
            var parent = table.parentElement;
            if (!parent || parent.querySelector('.s2s-banner')) return;
            var codeEls = table.querySelectorAll('.response-col_status');
            var code = '';
            codeEls.forEach(function(el) {
                var t = el.textContent.trim();
                if (/^\d+$/.test(t)) code = t;
            });
            if (!code) return;
            var banner = document.createElement('div');
            banner.className = 's2s-banner';
            if (code.startsWith('2')) {
                banner.style.cssText = 'background:#f6ffed;border:1px solid #b7eb8f;border-radius:6px;padding:8px 16px;margin:10px 0;';
                banner.innerHTML = '<span style="color:#389e0d;font-size:16px;font-weight:600;">&#10004; Successful</span>';
            } else {
                banner.style.cssText = 'background:#fff2f0;border:1px solid #ffccc7;border-radius:6px;padding:8px 16px;margin:10px 0;';
                banner.innerHTML = '<span style="color:#cf1322;font-size:16px;font-weight:600;">&#10008; Failed (code: ' + code + ')</span>';
            }
            parent.insertBefore(banner, table);
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            new MutationObserver(patch).observe(document.body, {childList: true, subtree: true});
        });
    } else {
        new MutationObserver(patch).observe(document.body, {childList: true, subtree: true});
    }
})();
</script>
"""

@app.get("/docs", include_in_schema=False)
async def custom_swagger_docs():
    html = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title,
        swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    )
    body = html.body.decode()
    body = body.replace("</body>", _SWAGGER_CUSTOM_HTML + "</body>")
    return HTMLResponse(content=body)

# ── Task cancellation ────────────────────────────────────────
_cancel_requested: set = set()
_task_thread_ids: Dict[str, int] = {}  # task_id -> thread ident

_executor = ThreadPoolExecutor(max_workers=1)


def _check_cancelled(task_id: str):
    """Called between pipeline steps; raises if the task has been cancelled."""
    if task_id in _cancel_requested:
        _cancel_requested.discard(task_id)
        raise InterruptedError("task cancelled by user")


def _register_thread(task_id: str):
    """Record thread ID when a task starts, used for force kill."""
    _task_thread_ids[task_id] = threading.current_thread().ident


def _force_kill_thread(task_id: str) -> bool:
    """Inject KeyboardInterrupt into the target thread to force-terminate it."""
    tid = _task_thread_ids.get(task_id)
    if tid is None:
        return False
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid),
        ctypes.py_object(KeyboardInterrupt),
    )
    return res == 1


# ── Enums / Models ──────────────────────────────────────────
class ASRBackend(str, Enum):
    transformers = "transformers"
    vllm = "vllm"


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class PipelineRequest(BaseModel):
    """Full pipeline request parameters. Fields correspond to PipelineConfig in config.py."""
    model_name: str = "seed"
    src_lang: str = "en"
    tgt_lang: str = "ja"
    output_version: str = ""

    src_dir: str = "input/acl_6060_dev/full_wavs"
    tgt_dir: str = "input/acl_6060_dev_tgt_seed/en_ja"
    src_segments_yaml: str = "input/ACL.ACLdev2023.en-xx.gold_segments.yaml"
    src_txt: str = "input/acl_6060_dev/text/txt/ACL.6060.dev.en-xx.en.txt"
    tgt_ref_txt: str = "input/acl_6060_dev/text/txt/ACL.6060.dev.en-xx.ja.txt"
    output_dir: str = "output"

    asr_backend: ASRBackend = ASRBackend.transformers
    batch_size: int = 2
    max_new_tokens: int = 1024
    embedding_model: str = "sentence-transformers/LaBSE"
    proc_device: str = "cuda"


class ASRRequest(BaseModel):
    """Run the ASR step only."""
    model_name: str = "seed"
    src_lang: str = "en"
    tgt_lang: str = "ja"
    output_version: str = ""
    src_dir: str = "input/acl_6060_dev/full_wavs"
    tgt_dir: str = "input/acl_6060_dev_tgt_seed/en_ja"
    output_dir: str = "output"
    asr_backend: ASRBackend = ASRBackend.transformers
    batch_size: int = 2
    max_new_tokens: int = 1024


class AlignRequest(BaseModel):
    """Run the SEGALE alignment step only (requires ASR output to exist)."""
    model_name: str = "seed"
    src_lang: str = "en"
    tgt_lang: str = "ja"
    output_version: str = ""
    src_dir: str = "input/acl_6060_dev/full_wavs"
    tgt_dir: str = "input/acl_6060_dev_tgt_seed/en_ja"
    src_segments_yaml: str = "input/ACL.ACLdev2023.en-xx.gold_segments.yaml"
    src_txt: str = "input/acl_6060_dev/text/txt/ACL.6060.dev.en-xx.en.txt"
    tgt_ref_txt: str = "input/acl_6060_dev/text/txt/ACL.6060.dev.en-xx.ja.txt"
    output_dir: str = "output"
    embedding_model: str = "sentence-transformers/LaBSE"
    proc_device: str = "cuda"


class EvalRequest(BaseModel):
    """Run the evaluation step only (requires alignment output to exist)."""
    model_name: str = "seed"
    src_lang: str = "en"
    tgt_lang: str = "ja"
    output_version: str = ""
    output_dir: str = "output"
    src_segments_yaml: str = "input/ACL.ACLdev2023.en-xx.gold_segments.yaml"
    src_txt: str = "input/acl_6060_dev/text/txt/ACL.6060.dev.en-xx.en.txt"


class TaskInfo(BaseModel):
    task_id: str
    status: TaskStatus
    step: str = ""
    created_at: str
    finished_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


# ── In-memory task store ─────────────────────────────────────
_tasks: Dict[str, TaskInfo] = {}


# ── Helper: build PipelineConfig from request fields ─────────
def _cfg_from_dict(d: dict) -> PipelineConfig:
    valid_fields = {f.name for f in PipelineConfig.__dataclass_fields__.values()}
    return PipelineConfig(**{k: v for k, v in d.items() if k in valid_fields})


def _get_tgt_language(tgt_lang: str) -> str:
    return PipelineConfig.LANGUAGE_NAME_MAP.get(tgt_lang, tgt_lang)


def _get_bleu_tokenizer(tgt_lang: str) -> str:
    return PipelineConfig.BLEU_MAP.get(tgt_lang, "13a")


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _mark_cancelled(task_id: str, t: TaskInfo):
    t.status = TaskStatus.cancelled
    t.step = "cancelled"
    t.error = "cancelled by user"
    t.finished_at = datetime.now().isoformat()
    _cancel_requested.discard(task_id)


# ── Pipeline runners (executed in thread pool) ───────────────
def _run_full_pipeline(task_id: str, cfg: PipelineConfig):
    t = _tasks[task_id]
    _register_thread(task_id)
    try:
        t.status = TaskStatus.running
        t.step = "step1_asr"
        _check_cancelled(task_id)

        cfg.build_manifest()
        if cfg.asr_backend == "vllm":
            step1_asr_vllm(
                manifest=cfg.manifest,
                tgt_language=_get_tgt_language(cfg.tgt_lang),
                out_dir=cfg.output_dir_asr,
                batch_size=cfg.batch_size,
                max_new_tokens=cfg.max_new_tokens,
                gpu_memory_utilization=0.7,
            )
        else:
            step1_asr(
                manifest=cfg.manifest,
                tgt_language=_get_tgt_language(cfg.tgt_lang),
                out_dir=cfg.output_dir_asr,
                batch_size=cfg.batch_size,
                max_new_tokens=cfg.max_new_tokens,
            )
        _free_gpu()

        _check_cancelled(task_id)
        t.step = "intermediate"
        add_char_spans_for_dir(cfg.output_dir_asr, cfg.output_dir_asr_enriched)
        asr_to_instances(
            s2s=True,
            yaml_file=cfg.src_segments_yaml,
            asr_dir=cfg.output_dir_asr_enriched,
            output_file=cfg.output_path_instances,
        )
        instances_to_segale(
            src_txt=cfg.src_txt,
            tgt_ref_txt=cfg.tgt_ref_txt,
            src_segments_yaml=cfg.src_segments_yaml,
            instances=cfg.output_path_instances,
            out_dir=cfg.output_dir_segale,
        )

        _check_cancelled(task_id)
        t.step = "step2_segale"
        step2_segale(
            system_file=os.path.join(cfg.output_dir_segale, "hyp.jsonl"),
            ref_file=os.path.join(cfg.output_dir_segale, "ref.jsonl"),
            segmenter="spacy",
            task_lang=cfg.tgt_lang,
            proc_device=cfg.proc_device,
            embedding_model=cfg.embedding_model,
        )

        _check_cancelled(task_id)
        t.step = "step3_evaluation"
        step3_longyaal(
            yaml_file=cfg.src_segments_yaml,
            source_sentences_file=cfg.src_txt,
            instances_log=cfg.output_path_instances,
            segale_file=cfg.segale_file,
            output_folder=cfg.output_dir_evaluation,
            bleu_tokenizer=_get_bleu_tokenizer(cfg.tgt_lang),
        )

        scores = _read_scores(cfg.output_dir_evaluation)
        t.status = TaskStatus.completed
        t.step = "done"
        t.result = scores
        t.finished_at = datetime.now().isoformat()

    except (InterruptedError, KeyboardInterrupt):
        _mark_cancelled(task_id, t)
        _free_gpu()
    except Exception as e:
        logger.error("Pipeline failed: %s\n%s", e, traceback.format_exc())
        t.status = TaskStatus.failed
        t.error = str(e)
        t.finished_at = datetime.now().isoformat()
    finally:
        _task_thread_ids.pop(task_id, None)


def _run_asr_only(task_id: str, cfg: PipelineConfig):
    t = _tasks[task_id]
    _register_thread(task_id)
    try:
        t.status = TaskStatus.running
        t.step = "step1_asr"
        _check_cancelled(task_id)

        cfg.build_manifest()
        if cfg.asr_backend == "vllm":
            step1_asr_vllm(
                manifest=cfg.manifest,
                tgt_language=_get_tgt_language(cfg.tgt_lang),
                out_dir=cfg.output_dir_asr,
                batch_size=cfg.batch_size,
                max_new_tokens=cfg.max_new_tokens,
                gpu_memory_utilization=0.7,
            )
        else:
            step1_asr(
                manifest=cfg.manifest,
                tgt_language=_get_tgt_language(cfg.tgt_lang),
                out_dir=cfg.output_dir_asr,
                batch_size=cfg.batch_size,
                max_new_tokens=cfg.max_new_tokens,
            )
        _free_gpu()
        t.status = TaskStatus.completed
        t.step = "done"
        t.result = {"output_dir": cfg.output_dir_asr}
        t.finished_at = datetime.now().isoformat()
    except (InterruptedError, KeyboardInterrupt):
        _mark_cancelled(task_id, t)
        _free_gpu()
    except Exception as e:
        logger.error("ASR failed: %s\n%s", e, traceback.format_exc())
        t.status = TaskStatus.failed
        t.error = str(e)
        t.finished_at = datetime.now().isoformat()
    finally:
        _task_thread_ids.pop(task_id, None)


def _run_align_only(task_id: str, cfg: PipelineConfig):
    t = _tasks[task_id]
    _register_thread(task_id)
    try:
        t.status = TaskStatus.running

        _check_cancelled(task_id)
        t.step = "intermediate"
        add_char_spans_for_dir(cfg.output_dir_asr, cfg.output_dir_asr_enriched)
        asr_to_instances(
            s2s=True,
            yaml_file=cfg.src_segments_yaml,
            asr_dir=cfg.output_dir_asr_enriched,
            output_file=cfg.output_path_instances,
        )
        instances_to_segale(
            src_txt=cfg.src_txt,
            tgt_ref_txt=cfg.tgt_ref_txt,
            src_segments_yaml=cfg.src_segments_yaml,
            instances=cfg.output_path_instances,
            out_dir=cfg.output_dir_segale,
        )

        _check_cancelled(task_id)
        t.step = "step2_segale"
        step2_segale(
            system_file=os.path.join(cfg.output_dir_segale, "hyp.jsonl"),
            ref_file=os.path.join(cfg.output_dir_segale, "ref.jsonl"),
            segmenter="spacy",
            task_lang=cfg.tgt_lang,
            proc_device=cfg.proc_device,
            embedding_model=cfg.embedding_model,
        )

        t.status = TaskStatus.completed
        t.step = "done"
        t.result = {"segale_file": cfg.segale_file}
        t.finished_at = datetime.now().isoformat()
    except (InterruptedError, KeyboardInterrupt):
        _mark_cancelled(task_id, t)
        _free_gpu()
    except Exception as e:
        logger.error("Alignment failed: %s\n%s", e, traceback.format_exc())
        t.status = TaskStatus.failed
        t.error = str(e)
        t.finished_at = datetime.now().isoformat()
    finally:
        _task_thread_ids.pop(task_id, None)


def _run_eval_only(task_id: str, cfg: PipelineConfig):
    t = _tasks[task_id]
    _register_thread(task_id)
    try:
        t.status = TaskStatus.running
        t.step = "step3_evaluation"
        _check_cancelled(task_id)

        step3_longyaal(
            yaml_file=cfg.src_segments_yaml,
            source_sentences_file=cfg.src_txt,
            instances_log=cfg.output_path_instances,
            segale_file=cfg.segale_file,
            output_folder=cfg.output_dir_evaluation,
            bleu_tokenizer=_get_bleu_tokenizer(cfg.tgt_lang),
        )
        scores = _read_scores(cfg.output_dir_evaluation)
        t.status = TaskStatus.completed
        t.step = "done"
        t.result = scores
        t.finished_at = datetime.now().isoformat()
    except (InterruptedError, KeyboardInterrupt):
        _mark_cancelled(task_id, t)
    except Exception as e:
        logger.error("Evaluation failed: %s\n%s", e, traceback.format_exc())
        t.status = TaskStatus.failed
        t.error = str(e)
        t.finished_at = datetime.now().isoformat()
    finally:
        _task_thread_ids.pop(task_id, None)


def _read_scores(eval_dir: str) -> Dict[str, Any]:
    scores_path = os.path.join(eval_dir, "scores.resegmented.csv")
    instances_path = os.path.join(eval_dir, "instances.resegmented.json")
    result: Dict[str, Any] = {}

    if os.path.isfile(scores_path):
        with open(scores_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().splitlines()
        if len(lines) >= 2:
            keys = lines[0].split("\t")
            vals = lines[1].split("\t")
            result["scores"] = {k: float(v) for k, v in zip(keys, vals)}

    if os.path.isfile(instances_path):
        with open(instances_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        result["num_segments"] = len(data)
        result["instances_file"] = instances_path

    result["output_dir"] = eval_dir
    return result


def _create_task(step_label: str) -> str:
    task_id = uuid.uuid4().hex[:12]
    _tasks[task_id] = TaskInfo(
        task_id=task_id,
        status=TaskStatus.pending,
        step=step_label,
        created_at=datetime.now().isoformat(),
    )
    return task_id


# ── Graceful shutdown: force exit on Ctrl+C ──────────────────
def _force_exit(*args):
    logger.info("Received shutdown signal, force exiting...")
    os._exit(0)

signal.signal(signal.SIGINT, _force_exit)
signal.signal(signal.SIGTERM, _force_exit)


# ── API Endpoints ────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


@app.post("/pipeline/run", response_model=TaskInfo, summary="Run full pipeline")
def run_pipeline(req: PipelineRequest):
    """
    Submit a full pipeline task (ASR → SEGALE → Evaluation).
    Returns a task_id; poll progress via /pipeline/{task_id}/status.
    """
    cfg = _cfg_from_dict(req.model_dump())
    task_id = _create_task("queued")
    _executor.submit(_run_full_pipeline, task_id, cfg)
    return _tasks[task_id]


@app.post("/asr/run", response_model=TaskInfo, summary="Run ASR only")
def run_asr(req: ASRRequest):
    cfg = _cfg_from_dict(req.model_dump())
    task_id = _create_task("queued_asr")
    _executor.submit(_run_asr_only, task_id, cfg)
    return _tasks[task_id]


@app.post("/align/run", response_model=TaskInfo, summary="Run SEGALE alignment only")
def run_align(req: AlignRequest):
    """Requires ASR output to already exist in the output directory."""
    cfg = _cfg_from_dict(req.model_dump())
    task_id = _create_task("queued_align")
    _executor.submit(_run_align_only, task_id, cfg)
    return _tasks[task_id]


@app.post("/eval/run", response_model=TaskInfo, summary="Run evaluation only")
def run_eval(req: EvalRequest):
    """Requires SEGALE alignment output to already exist in the output directory."""
    cfg = _cfg_from_dict(req.model_dump())
    task_id = _create_task("queued_eval")
    _executor.submit(_run_eval_only, task_id, cfg)
    return _tasks[task_id]


@app.post("/pipeline/{task_id}/cancel", summary="Cancel / terminate a task")
def cancel_task(task_id: str):
    """Queued tasks are cancelled immediately; running tasks are force-terminated."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="task not found")
    t = _tasks[task_id]
    if t.status in (TaskStatus.completed, TaskStatus.failed, TaskStatus.cancelled):
        return {"message": f"task already {t.status.value}", "task": t}
    if t.status == TaskStatus.pending:
        _mark_cancelled(task_id, t)
        return {"message": "cancelled", "task": t}
    _cancel_requested.add(task_id)
    _force_kill_thread(task_id)
    return {"message": "terminating", "task": t}


@app.get("/pipeline/{task_id}/status", response_model=TaskInfo, summary="Query task status")
def get_task_status(task_id: str):
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="task not found")
    return _tasks[task_id]


@app.get("/pipeline/{task_id}/result", summary="Get task result")
def get_task_result(task_id: str):
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="task not found")
    t = _tasks[task_id]
    if t.status == TaskStatus.running:
        raise HTTPException(status_code=202, detail=f"still running (step: {t.step})")
    if t.status == TaskStatus.failed:
        raise HTTPException(status_code=500, detail=t.error)
    if t.status == TaskStatus.cancelled:
        raise HTTPException(status_code=499, detail="task was cancelled")
    if t.status == TaskStatus.pending:
        raise HTTPException(status_code=202, detail="pending")
    return t.result


@app.get(
    "/pipeline/{task_id}/download/{filename}",
    summary="Download evaluation artifact",
)
def download_artifact(task_id: str, filename: str):
    """
    Download an evaluation output file, e.g.:
    - instances.resegmented.json
    - scores.resegmented.csv
    """
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="task not found")
    t = _tasks[task_id]
    if t.status != TaskStatus.completed or not t.result:
        raise HTTPException(status_code=400, detail="task not completed yet")

    output_dir = t.result.get("output_dir", "")
    filepath = os.path.join(output_dir, filename)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail=f"file not found: {filename}")
    return FileResponse(filepath, filename=filename)


@app.get("/tasks", response_model=List[TaskInfo], summary="List all tasks")
def list_tasks():
    return list(_tasks.values())
