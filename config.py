import os
from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Tuple


@dataclass
class PipelineConfig:
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

    # ── 语言 ───────────────────────────────────────────────────────
    src_lang: str = "en"
    tgt_lang: str = "ja"

    # ── 输入路径 ───────────────────────────────────────────────────
    src_audio_dir: str = "input/acl_6060_dev/full_wavs"
    tgt_audio_dir: str = "input/acl_6060_dev_tgt_seed/en_ja"
    src_segments_yaml: str = "input/ACL.ACLdev2023.en-xx.gold_segments.yaml"
    src_txt: str = "input/acl_6060_dev/text/txt/ACL.6060.dev.en-xx.en.txt"
    tgt_ref_txt: str = "input/acl_6060_dev/text/txt/ACL.6060.dev.en-xx.ja.txt"
    output_dir: str = "output"

    # ── 运行参数 ───────────────────────────────────────────────────
    asr_backend: str = "transformers"
    batch_size: int = 2
    max_new_tokens: int = 1024
    embedding_model: str = "sentence-transformers/LaBSE"
    proc_device: str = "cuda"
    only_doc_ids: Optional[Tuple[str, ...]] = None
    # 复现性种子，传给 step2_segale 的 torch / numpy / random / cudnn
    seed: int = 42

    # ── 派生输出路径 ───────────────────────────────────────────────
    @property
    def manifest(self) -> str:
        return os.path.join(self.output_dir, "output_asr", "manifest.jsonl")

    @property
    def output_dir_asr(self) -> str:
        return os.path.join(self.output_dir, "output_asr")

    @property
    def output_path_instances(self) -> str:
        # char_span 是 in-place 加到 output_asr 里的 *_asr.json，无单独 enriched 目录
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
        """
        从 src_audio_dir / tgt_audio_dir 自动配对 src/tgt wav，生成 manifest.jsonl。
        按 basename 配对，找不到对应 tgt 的 src 跳过。
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

        print(f"manifest 已生成：{manifest_path}（{len(records)} 条）")
        return manifest_path
