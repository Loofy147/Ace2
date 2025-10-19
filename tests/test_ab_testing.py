import unittest
import uuid
from src.ace.core.implementation import PromptEngineeringLaboratory, Context

class TestPromptEngineeringLaboratory(unittest.TestCase):

    def setUp(self):
        """Set up a new PromptEngineeringLaboratory for each test."""
        self.lab = PromptEngineeringLaboratory()
        self.context = Context(content={"topic": "testing", "data": "sample data"})
        self.variants = {
            "variant_a": "Prompt A for {context}",
            "variant_b": "Prompt B for {context}"
        }
        self.lab.create_ab_test("feedback_test", self.variants)

    def test_get_prompt_variant_returns_trace_id(self):
        """Test that getting a variant returns a valid trace_id."""
        result = self.lab.get_prompt_variant("feedback_test", self.context)
        self.assertIn("trace_id", result)
        self.assertIsInstance(uuid.UUID(result["trace_id"]), uuid.UUID)

    def test_record_feedback_success(self):
        """Test that feedback is recorded successfully for a valid trace_id."""
        result = self.lab.get_prompt_variant("feedback_test", self.context)
        trace_id = result["trace_id"]

        success = self.lab.record_feedback(trace_id, 0.9)
        self.assertTrue(success)
        self.assertIn(trace_id, self.lab.feedback)
        self.assertEqual(self.lab.feedback[trace_id], 0.9)

    def test_record_feedback_invalid_trace_id(self):
        """Test that recording feedback for an invalid trace_id fails."""
        invalid_trace_id = str(uuid.uuid4())
        success = self.lab.record_feedback(invalid_trace_id, 0.9)
        self.assertFalse(success)
        self.assertNotIn(invalid_trace_id, self.lab.feedback)

    def test_statistics_with_feedback(self):
        """Test that statistics correctly aggregate usage and feedback."""
        # Variant A, score 0.8
        res1 = self.lab.get_prompt_variant("feedback_test", self.context)
        self.lab.record_feedback(res1["trace_id"], 0.8)

        # Variant B, score 0.9
        res2 = self.lab.get_prompt_variant("feedback_test", self.context)
        self.lab.record_feedback(res2["trace_id"], 0.9)

        # Variant A, score 0.6
        res3 = self.lab.get_prompt_variant("feedback_test", self.context)
        self.lab.record_feedback(res3["trace_id"], 0.6)

        # Variant B, no feedback
        self.lab.get_prompt_variant("feedback_test", self.context)

        stats = self.lab.get_test_statistics("feedback_test")

        self.assertIn("variants", stats)
        variants_stats = stats["variants"]

        # Check Variant A stats
        self.assertEqual(variants_stats["variant_a"]["usage_count"], 2)
        self.assertAlmostEqual(variants_stats["variant_a"]["total_score"], 1.4)
        self.assertAlmostEqual(variants_stats["variant_a"]["average_score"], 0.7)

        # Check Variant B stats
        self.assertEqual(variants_stats["variant_b"]["usage_count"], 2)
        self.assertAlmostEqual(variants_stats["variant_b"]["total_score"], 0.9)
        self.assertAlmostEqual(variants_stats["variant_b"]["average_score"], 0.45)

    def test_statistics_no_feedback(self):
        """Test that statistics are correct when no feedback is provided."""
        self.lab.get_prompt_variant("feedback_test", self.context)
        self.lab.get_prompt_variant("feedback_test", self.context)

        stats = self.lab.get_test_statistics("feedback_test")
        variants_stats = stats["variants"]

        self.assertEqual(variants_stats["variant_a"]["usage_count"], 1)
        self.assertEqual(variants_stats["variant_a"]["total_score"], 0.0)
        self.assertEqual(variants_stats["variant_a"]["average_score"], 0.0)

        self.assertEqual(variants_stats["variant_b"]["usage_count"], 1)
        self.assertEqual(variants_stats["variant_b"]["total_score"], 0.0)
        self.assertEqual(variants_stats["variant_b"]["average_score"], 0.0)

    def test_get_statistics_for_nonexistent_test(self):
        """Test that getting stats for a nonexistent test returns None."""
        stats = self.lab.get_test_statistics("nonexistent")
        self.assertIsNone(stats)

if __name__ == '__main__':
    unittest.main()
