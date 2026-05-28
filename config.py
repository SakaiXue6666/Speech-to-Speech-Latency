import os
from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Tuple


@dataclass
class PipelineConfig:
    """Configuration for the end-to-end Speech-to-Speech Latency pipeline.

    Holds all input paths, runtime knobs, and derived output paths consumed by
    :func:`main.main`. The class is a :func:`dataclasses.dataclass` so it can be
    serialised into ``run_meta.json`` for reproducibility.

    Attributes:
        LANGUAGE_NAME_MAP: Mapping from ISO-style language code (e.g. ``"ja"``)
            to the full language name (e.g. ``"Japanese"``) consumed by the ASR
            model.
        BLEU_MAP: Mapping from language code to the sacreBLEU tokenizer name
            used during evaluation.
        src_lang: Source-side language code.
        tgt_lang: Target-side language code; must be a key of
            :attr:`LANGUAGE_NAME_MAP` and :attr:`BLEU_MAP`.
        src_audio_dir: Directory of source-language WAV files.
        tgt_audio_dir: Directory of target-language (translated) WAV files,
            paired with ``src_audio_dir`` by basename.
        src_segments_yaml: Gold segmentation YAML for the source audio.
        src_txt: Source-language transcript text file.
        tgt_ref_txt: Target-language reference translation text file.
        output_dir: Root directory for all generated artifacts.
        asr_backend: ASR backend, either ``"transformers"`` or ``"vllm"``.
        batch_size: ASR inference batch size.
        max_new_tokens: ASR ``max_new_tokens`` cap per sample.
        embedding_model: HuggingFace model id used by the SEGALE embedding
            step (e.g. ``"sentence-transformers/LaBSE"``).
        proc_device: Torch device string for the SEGALE embedding step.
        only_doc_ids: Optional whitelist of doc IDs to keep; ``None`` means
            keep all.
        seed: Random seed passed to ``step2_segale`` and propagated to
            ``torch`` / ``numpy`` / ``random`` / cuDNN for reproducibility.
    """

    LANGUAGE_NAME_MAP: ClassVar[Dict[str, str]] = {
        "en": "English",
        "zh": "Chinese",
        "de": "German",
        "ja": "Japanese",
    }
    BLEU_MAP: ClassVar[Dict[str, str]] = {
        "en": "13a",
        "zh": "zh",
        "de": "13a",
        "ja": "ja-mecab",
    }

    # ── Language ────────────────────────────────────────────────────
    src_lang: str = "en"
    tgt_lang: str = "ja"

    # ── Input paths ─────────────────────────────────────────────────
    src_audio_dir: str = "input/acl_6060_dev/full_wavs"
    tgt_audio_dir: str = "input/acl_6060_dev_tgt_seed/en_ja"
    src_segments_yaml: str = "input/ACL.ACLdev2023.en-xx.gold_segments.yaml"
    src_txt: str = "input/acl_6060_dev/text/txt/ACL.6060.dev.en-xx.en.txt"
    tgt_ref_txt: str = "input/acl_6060_dev/text/txt/ACL.6060.dev.en-xx.ja.txt"
    output_dir: str = "output"

    # ── Runtime parameters ──────────────────────────────────────────
    asr_backend: str = "transformers"
    batch_size: int = 5
    max_new_tokens: int = 1024
    embedding_model: str = "sentence-transformers/LaBSE"
    proc_device: str = "cuda"
    only_doc_ids: Optional[Tuple[str, ...]] = None
    # Reproducibility seed passed to step2_segale (torch / numpy / random / cudnn)
    seed: int = 42

    # ── Derived output paths ────────────────────────────────────────
    @property
    def manifest(self) -> str:
        return os.path.join(self.output_dir, "output_asr", "manifest.jsonl")

    @property
    def output_dir_asr(self) -> str:
        return os.path.join(self.output_dir, "output_asr")

    @property
    def output_path_instances(self) -> str:
        # char_spans are added in-place to *_asr.json files; no separate enriched directory
        return os.path.join(self.output_dir_asr, "instances.log")

    @property
    def output_dir_segale(self) -> str:
        return os.path.join(self.output_dir, "output_segmentation")

    @property
    def segale_file(self) -> str:
        return os.path.join(self.output_dir_segale, "hyp/aligned_spacy_hyp.jsonl")

    @property
    def output_dir_evaluation(self) -> str:
        return os.path.join(self.output_dir, "output_evaluation")

    def build_manifest(self, manifest_path: str = None) -> str:
        """Scan src/tgt audio dirs and write a pairing manifest as JSONL.

        Files are paired by basename (without extension); src WAVs with no
        matching tgt are silently skipped. Each output line is
        ``{"src": <src_wav>, "tgt": <tgt_wav>}``.

        Args:
            manifest_path: Output path of the manifest. ``None`` falls back to
                :attr:`manifest` (i.e. ``output_dir/output_asr/manifest.jsonl``).

        Returns:
            The path that the manifest was written to.
        """
        import json

        if manifest_path is None:
            manifest_path = self.manifest

        tgt_wavs = {
            os.path.splitext(f)[0]: os.path.join(self.tgt_audio_dir, f)
            for f in os.listdir(self.tgt_audio_dir)
            if f.endswith(".wav")
        }

        records = []
        for src_file in sorted(os.listdir(self.src_audio_dir)):
            if not src_file.endswith(".wav"):
                continue
            stem = os.path.splitext(src_file)[0]
            if stem not in tgt_wavs:
                continue
            records.append({
                "src": os.path.join(self.src_audio_dir, src_file).replace("\\", "/"),
                "tgt": tgt_wavs[stem].replace("\\", "/"),
            })

        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"Manifest written: {manifest_path} ({len(records)} records)")
        return manifest_path
