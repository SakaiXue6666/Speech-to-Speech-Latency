import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.metrics import Instance, evaluate_instances
from src.evaluation.report import write_evaluation_report


def _instance(**overrides):
    values = {
        "index": 0,
        "doc_id": "talk.wav",
        "seg_id": 0,
        "source": "source sentence",
        "reference": "eins zwei drei vier",
        "prediction": "eins zwei drei vier",
        "source_length": 200.0,
        "delays": [100.0],
        "elapsed": [120.0],
        "recording_end": 1000.0,
        "null_alignment_type": "",
        "_match_method": "matched",
    }
    values.update(overrides)
    return values


class NullAwareMetricsTest(unittest.TestCase):
    def test_quality_keeps_nulls_while_latency_uses_timed_instances(self):
        rows = [
            Instance(_instance()),
            Instance(
                _instance(
                    index=1,
                    seg_id=1,
                    source="quality only",
                    reference="fuenf sechs sieben acht",
                    prediction="fuenf sechs sieben acht",
                    delays=[],
                    elapsed=[],
                    _match_method="quality_only_not_found",
                )
            ),
            Instance(
                _instance(
                    index=2,
                    seg_id=2,
                    reference="neun zehn elf zwoelf",
                    prediction="",
                    source_length=None,
                    delays=[],
                    elapsed=[],
                    recording_end=None,
                    null_alignment_type="under_translation",
                )
            ),
            Instance(
                _instance(
                    index=3,
                    seg_id=3,
                    source="",
                    reference="",
                    prediction="zusaetzlicher text",
                    source_length=None,
                    delays=[],
                    elapsed=[],
                    recording_end=None,
                    null_alignment_type="over_translation",
                )
            ),
        ]

        scores = evaluate_instances(rows, "13a")

        self.assertEqual(scores["segments"], 4)
        self.assertEqual(scores["valid_segments"], 2)
        self.assertEqual(scores["latency_segments"], 1)
        self.assertEqual(scores["under_translation_alignments"], 1)
        self.assertEqual(scores["over_translation_alignments"], 1)
        self.assertEqual(scores["null_alignments"], 2)
        self.assertLess(scores["bleu"], 100.0)
        self.assertEqual(scores["ending_offset"], -100.0)
        self.assertEqual(scores["ca_ending_offset"], -80.0)

    def test_report_marks_null_quality_overrides_and_renders_filters(self):
        rows = [
            _instance(),
            _instance(
                index=1,
                seg_id=1,
                prediction="",
                source_length=None,
                delays=[],
                elapsed=[],
                recording_end=None,
                null_alignment_type="under_translation",
            ),
            _instance(
                index=2,
                seg_id=2,
                source="",
                reference="",
                prediction="extra",
                source_length=None,
                delays=[],
                elapsed=[],
                recording_end=None,
                null_alignment_type="over_translation",
            ),
        ]

        with tempfile.TemporaryDirectory() as directory:
            summary = write_evaluation_report(rows, directory)
            quality_rows = [
                json.loads(line)
                for line in (Path(directory) / "quality_inputs.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            report = (Path(directory) / "sentence_details.html").read_text(
                encoding="utf-8"
            )

        self.assertEqual(summary["segments"], 3)
        self.assertEqual(summary["null_alignments"], 2)
        self.assertIsNone(quality_rows[0]["score_override"])
        self.assertEqual(quality_rows[1]["score_override"], 0.0)
        self.assertEqual(quality_rows[2]["score_override"], 0.0)
        self.assertIn("data-filter='normal'", report)
        self.assertIn("data-filter='under_translation'", report)
        self.assertIn("data-filter='over_translation'", report)


if __name__ == "__main__":
    unittest.main()
