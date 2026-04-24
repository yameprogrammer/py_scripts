from __future__ import annotations

import sys
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent / "scripts" / "webnovel_autowrite"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SPEC = spec_from_file_location("webnovel_main", SCRIPT_DIR / "main.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("main.py 모듈을 불러올 수 없습니다.")

webnovel_main = module_from_spec(SPEC)
sys.modules[SPEC.name] = webnovel_main
SPEC.loader.exec_module(webnovel_main)

run_revision_loop = webnovel_main.run_revision_loop


class FakeWriter:
    def __init__(self, drafts: list[str]) -> None:
        self.drafts = drafts
        self.generate_calls = 0
        self.revise_calls = 0

    def generate_draft(self, **_: Any) -> str:
        self.generate_calls += 1
        return self.drafts.pop(0)

    def revise_draft(self, **_: Any) -> str:
        self.revise_calls += 1
        return self.drafts.pop(0)


class FakeEditor:
    def __init__(self, reviews: list[dict[str, Any]]) -> None:
        self.reviews = reviews
        self.calls = 0

    def review_draft(self, **_: Any) -> dict[str, Any]:
        self.calls += 1
        return self.reviews.pop(0)


class RevisionLoopTests(unittest.TestCase):
    def test_run_revision_loop_retries_until_pass(self) -> None:
        writer = FakeWriter(["초안 1", "수정본 2"])
        editor = FakeEditor(
            [
                {
                    "scores": {"overall": 62.0, "length_adherence": 45.0},
                    "quality_gate": {"passed": False, "reasons": ["분량 미달", "점수 미달"]},
                    "manuscript_stats": {"char_count_no_space": 900},
                },
                {
                    "scores": {"overall": 87.5, "length_adherence": 100.0},
                    "quality_gate": {"passed": True, "reasons": []},
                    "manuscript_stats": {"char_count_no_space": 5200},
                },
            ]
        )

        manuscript, review, passed = run_revision_loop(
            writer,
            editor,
            world_setting={"series": {"target_length": {"chars_per_episode": 5000}}},
            state={"runtime": {"current_episode": 1}},
            episode_number=1,
            quality_goals={
                "target_score": 80.0,
                "max_revision_rounds": 3,
                "target_length": {"target_chars": 5000, "min_chars": 4500, "soft_max_chars": 5750},
            },
        )

        self.assertTrue(passed)
        self.assertEqual(manuscript, "수정본 2")
        self.assertEqual(review["final_attempt"], 2)
        self.assertEqual(len(review["attempt_history"]), 2)
        self.assertEqual(writer.generate_calls, 1)
        self.assertEqual(writer.revise_calls, 1)

    def test_run_revision_loop_stops_after_max_rounds(self) -> None:
        writer = FakeWriter(["초안 1", "수정본 2", "수정본 3"])
        editor = FakeEditor(
            [
                {
                    "scores": {"overall": 60.0, "length_adherence": 50.0},
                    "quality_gate": {"passed": False, "reasons": ["점수 미달"]},
                    "manuscript_stats": {"char_count_no_space": 1000},
                },
                {
                    "scores": {"overall": 70.0, "length_adherence": 75.0},
                    "quality_gate": {"passed": False, "reasons": ["점수 미달"]},
                    "manuscript_stats": {"char_count_no_space": 3500},
                },
            ]
        )

        manuscript, review, passed = run_revision_loop(
            writer,
            editor,
            world_setting={"series": {"target_length": {"chars_per_episode": 5000}}},
            state={"runtime": {"current_episode": 1}},
            episode_number=1,
            quality_goals={
                "target_score": 80.0,
                "max_revision_rounds": 2,
                "target_length": {"target_chars": 5000, "min_chars": 4500, "soft_max_chars": 5750},
            },
        )

        self.assertFalse(passed)
        self.assertEqual(manuscript, "수정본 2")
        self.assertEqual(review["final_attempt"], 2)
        self.assertFalse(review["final_passed"])
        self.assertEqual(writer.revise_calls, 1)


if __name__ == "__main__":
    unittest.main()

