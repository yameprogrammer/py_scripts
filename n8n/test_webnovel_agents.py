from __future__ import annotations

import sys
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent / "scripts" / "webnovel_autowrite"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SPEC = spec_from_file_location("webnovel_agents", SCRIPT_DIR / "agents.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("agents.py 모듈을 불러올 수 없습니다.")

webnovel_agents = module_from_spec(SPEC)
sys.modules[SPEC.name] = webnovel_agents
SPEC.loader.exec_module(webnovel_agents)

EditorAgent = webnovel_agents.EditorAgent
calculate_manuscript_stats = webnovel_agents.calculate_manuscript_stats
extract_target_length = webnovel_agents.extract_target_length
parse_editor_response = webnovel_agents.parse_editor_response


def make_world_setting(target_chars: int = 100) -> dict:
    return {"series": {"target_length": {"chars_per_episode": target_chars}}}


def make_target_length(target_chars: int = 100) -> dict:
    return extract_target_length(make_world_setting(target_chars))


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, str | None]] = []

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        if not self.responses:
            raise AssertionError("응답이 더 이상 없습니다.")
        return self.responses.pop(0)


class ParseEditorResponseTests(unittest.TestCase):
    def test_parse_plain_json(self) -> None:
        manuscript = "가" * 120
        payload = parse_editor_response(
            '{"episode":1,"status":"reviewed","scores":{"plot_coherence":8,"character_consistency":9,"immersion":8,"style":7,"overall":8},"strengths":["전개가 안정적"],"issues":["후반 긴장감 약함"],"revision_requests":["결말 훅 강화"],"reviewed_at":null}',
            1,
            calculate_manuscript_stats(manuscript),
            make_target_length(100),
            80,
        )
        self.assertEqual(payload["episode"], 1)
        self.assertEqual(payload["scores"]["length_adherence"], 100.0)
        self.assertEqual(payload["scores"]["overall"], 84.0)

    def test_parse_fenced_json_with_trailing_text(self) -> None:
        response = "검수 결과입니다.\n```json\n{\n  \"episode\": 2,\n  \"status\": \"reviewed\",\n  \"scores\": {\n    \"plot_coherence\": 8,\n    \"character_consistency\": 8,\n    \"immersion\": 7,\n    \"style\": 7,\n    \"overall\": 8\n  },\n  \"strengths\": [\"세계관 활용이 좋음\"],\n  \"issues\": [\"도입이 조금 김\"],\n  \"revision_requests\": [\"초반 압축\"],\n}\n```\n추가 설명 끝"
        payload = parse_editor_response(
            response,
            2,
            calculate_manuscript_stats("나" * 120),
            make_target_length(100),
            80,
        )
        self.assertEqual(payload["episode"], 2)
        self.assertEqual(payload["issues"], ["도입이 조금 김"])

    def test_coerce_missing_fields(self) -> None:
        payload = parse_editor_response(
            '{"scores":{"overall":"9"},"strengths":"좋은 문체"}',
            3,
            calculate_manuscript_stats("다" * 95),
            make_target_length(100),
            80,
        )
        self.assertEqual(payload["episode"], 3)
        self.assertEqual(payload["status"], "reviewed")
        self.assertEqual(payload["scores"]["length_adherence"], 95.0)
        self.assertEqual(payload["strengths"], ["좋은 문체"])
        self.assertEqual(payload["issues"], [])

    def test_short_manuscript_fails_quality_gate(self) -> None:
        payload = parse_editor_response(
            '{"scores":{"plot_coherence":9,"character_consistency":9,"immersion":9,"style":9},"issues":[],"revision_requests":[]}',
            4,
            calculate_manuscript_stats("짧은원고" * 5),
            make_target_length(200),
            80,
        )
        self.assertFalse(payload["quality_gate"]["passed"])
        self.assertTrue(any("분량 미달" in issue for issue in payload["issues"]))


class EditorAgentTests(unittest.TestCase):
    def test_review_draft_repairs_invalid_json_on_retry(self) -> None:
        broken_response = '{"episode": 1, "status": "reviewed", "scores": {"plot_coherence": 8, "character_consistency": 8, "immersion": 7, "style": 7, "overall": 8}, "strengths": ["문장에 "리듬"이 있음"], "issues": [], "revision_requests": ["결말 강화"]}'
        repaired_response = '{"episode": 1, "status": "reviewed", "scores": {"plot_coherence": 8, "character_consistency": 8, "immersion": 7, "style": 7, "overall": 8}, "strengths": ["문장에 \\\"리듬\\\"이 있음"], "issues": [], "revision_requests": ["결말 강화"], "reviewed_at": null}'
        agent = EditorAgent(FakeClient([broken_response, repaired_response]))

        review = agent.review_draft(
            world_setting=make_world_setting(10),
            episode_number=1,
            manuscript="충분한 길이의 원고입니다.",
            target_length=make_target_length(10),
            target_score=80,
        )

        self.assertGreaterEqual(review["scores"]["overall"], 80)
        self.assertEqual(review["strengths"], ['문장에 "리듬"이 있음'])

    def test_review_draft_raises_after_failed_retry(self) -> None:
        agent = EditorAgent(FakeClient(["not json", "still not json"]))

        with self.assertRaises(RuntimeError):
            agent.review_draft(world_setting=make_world_setting(10), episode_number=1, manuscript="초안")


if __name__ == "__main__":
    unittest.main()




