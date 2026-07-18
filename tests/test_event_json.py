import json
from copy import deepcopy
import unittest
from pathlib import Path

from tools.event_schema import FORBIDDEN_PHRASES, load_json, validate_event_payload


ROOT = Path(__file__).resolve().parents[1]


class EventJsonTest(unittest.TestCase):
    def test_kirc_raw_schema(self):
        payload = load_json(ROOT / "assets/event_bank/tcga_kirc/events_raw.json")
        self.assertEqual(
            validate_event_payload(
                payload,
                expected_count=30,
                expected_status="unreviewed_llm_candidates",
            ),
            [],
        )
        self.assertEqual(len({event["event_id"] for event in payload["events"]}), 30)
        self.assertTrue(all(len(event["prompt_templates"]) == 4 for event in payload["events"]))
        serialized = json.dumps(payload).casefold()
        self.assertTrue(all(phrase not in serialized for phrase in FORBIDDEN_PHRASES))

    def test_reviewed_status_is_not_expert_claim(self):
        payload = load_json(ROOT / "assets/event_bank/tcga_kirc/events_reviewed.json")
        self.assertEqual(payload["status"], "machine_checked_pending_expert_review")
        self.assertEqual(validate_event_payload(payload, expected_count=30), [])

    def test_kirc_v2_candidates_are_scale_and_evidence_aware(self):
        payload = load_json(
            ROOT / "assets/event_bank/tcga_kirc/v2/events_candidates.json"
        )
        self.assertEqual(
            validate_event_payload(
                payload,
                expected_count=38,
                expected_status="unreviewed_llm_candidates",
            ),
            [],
        )
        self.assertEqual(payload["schema_version"], "2.0")
        self.assertTrue(
            all(
                event["patch_observable"] == "no"
                for event in payload["events"]
                if event["observation_scale"] == "wsi_anatomic_context"
            )
        )

    def test_kirc_v2_core_is_patch_local_and_context_is_separate(self):
        core = load_json(ROOT / "assets/event_bank/tcga_kirc/v2/events_patch_core.json")
        context = load_json(ROOT / "assets/event_bank/tcga_kirc/v2/events_context.json")
        reserve = load_json(ROOT / "assets/event_bank/tcga_kirc/v2/events_reserve.json")
        self.assertEqual(validate_event_payload(core, expected_count=22), [])
        self.assertEqual(validate_event_payload(context, expected_count=8), [])
        self.assertEqual(validate_event_payload(reserve, expected_count=8), [])
        self.assertEqual(core["status"], "machine_curated_pending_pathologist_review")
        self.assertTrue(
            all(
                event["observation_scale"] == "patch_local"
                and event["patch_observable"] == "yes"
                for event in core["events"]
            )
        )
        core_ids = {event["event_id"] for event in core["events"]}
        context_ids = {event["event_id"] for event in context["events"]}
        reserve_ids = {event["event_id"] for event in reserve["events"]}
        self.assertFalse(core_ids & context_ids)
        self.assertFalse(core_ids & reserve_ids)
        self.assertFalse(context_ids & reserve_ids)
        self.assertEqual(len(core_ids | context_ids | reserve_ids), 38)

    def test_v2_schema_rejects_wsi_event_claimed_as_patch_observable(self):
        payload = load_json(
            ROOT / "assets/event_bank/tcga_kirc/v2/events_candidates.json"
        )
        invalid = deepcopy(payload)
        event = next(
            item
            for item in invalid["events"]
            if item["observation_scale"] == "wsi_anatomic_context"
        )
        event["patch_observable"] = "conditional"
        errors = validate_event_payload(invalid)
        self.assertTrue(any("WSI/anatomic events" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
