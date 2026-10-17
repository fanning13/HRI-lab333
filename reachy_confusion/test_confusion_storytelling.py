import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from confusion_storytelling import (STORIES, assistance_decision, classify,
    explicit_request, interaction_stall, parse_label, rating)


class StudyLogicTests(unittest.TestCase):
    def test_story_shape_and_checkpoints(self):
        for story in STORIES.values():
            self.assertEqual(len(story), 6)
            self.assertEqual(sum(t["formal"] for t in story), 3)

    def test_locked_story_details(self):
        self.assertIn("Emma", STORIES["A"][0]["narration"])
        self.assertIn("key", " ".join(t["narration"] for t in STORIES["A"]).lower())
        text = " ".join(t["narration"] for t in STORIES["B"])
        self.assertIn("New York City", text); self.assertIn("sketchbook", text.lower())

    def test_parse_labels_and_fail_safe(self):
        for label in ("CONFUSED", "NOT_CONFUSED", "UNCERTAIN"):
            self.assertEqual(parse_label('{"label":"' + label + '"}'), label)
        self.assertEqual(parse_label("confused"), "UNCERTAIN")

    def test_assistance_policy(self):
        self.assertEqual(assistance_decision("UNCERTAIN", True, False, "assistance"), "soft_check_in")
        self.assertEqual(assistance_decision("UNCERTAIN", False, False, "assistance"), "neutral")
        self.assertEqual(assistance_decision("CONFUSED", False, False, "baseline"), "neutral")
        self.assertEqual(assistance_decision("NOT_CONFUSED", False, True, "baseline"), "clarification")

    def test_stall_and_help_request(self):
        self.assertTrue(interaction_stall("", None, None, 2500))
        self.assertTrue(interaction_stall("maybe", 3000, None, 2500))
        self.assertFalse(interaction_stall("They look at the bench", 100, None, 2500))
        self.assertTrue(explicit_request("Could I have a hint?"))

    def test_rating_ranges(self):
        answers = iter(["7", "5"])
        self.assertEqual(rating("", 5, lambda _: next(answers)), "5")

    def test_classifier_sends_context_once(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"label":"NOT_CONFUSED"}'))])
        self.assertEqual(classify(client, "model", "idea", "LOCKED FACT").label, "NOT_CONFUSED")
        sent = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertEqual(sent.count("LOCKED FACT"), 1)


if __name__ == "__main__": unittest.main()
