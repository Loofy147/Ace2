import unittest
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

    def test_create_ab_test_success(self):
        """Test successful creation of an A/B test."""
        self.lab.create_ab_test("test1", self.variants)
        self.assertIn("test1", self.lab.ab_tests)
        self.assertEqual(self.lab.ab_tests["test1"], self.variants)

    def test_create_ab_test_already_exists(self):
        """Test that creating a test with a duplicate name raises a ValueError."""
        self.lab.create_ab_test("test1", self.variants)
        with self.assertRaises(ValueError):
            self.lab.create_ab_test("test1", self.variants)

    def test_create_ab_test_insufficient_variants(self):
        """Test that creating a test with fewer than two variants raises a ValueError."""
        with self.assertRaises(ValueError):
            self.lab.create_ab_test("test_single", {"variant_a": "prompt"})

    def test_get_prompt_variant_round_robin(self):
        """Test that variants are selected in a round-robin fashion."""
        self.lab.create_ab_test("test_rr", self.variants)

        # First call should get variant_a
        result1 = self.lab.get_prompt_variant("test_rr", self.context)
        self.assertEqual(result1['variant_name'], "variant_a")

        # Second call should get variant_b
        result2 = self.lab.get_prompt_variant("test_rr", self.context)
        self.assertEqual(result2['variant_name'], "variant_b")

        # Third call should loop back to variant_a
        result3 = self.lab.get_prompt_variant("test_rr", self.context)
        self.assertEqual(result3['variant_name'], "variant_a")

    def test_get_prompt_variant_nonexistent_test(self):
        """Test that getting a variant for a nonexistent test returns None."""
        result = self.lab.get_prompt_variant("nonexistent", self.context)
        self.assertIsNone(result)

    def test_statistics_tracking(self):
        """Test that usage statistics are tracked correctly."""
        self.lab.create_ab_test("test_stats", self.variants)

        self.lab.get_prompt_variant("test_stats", self.context)
        self.lab.get_prompt_variant("test_stats", self.context)
        self.lab.get_prompt_variant("test_stats", self.context)

        stats = self.lab.get_test_statistics("test_stats")
        self.assertEqual(stats["variant_a"], 2)
        self.assertEqual(stats["variant_b"], 1)

    def test_get_statistics_for_nonexistent_test(self):
        """Test that getting stats for a nonexistent test returns None."""
        stats = self.lab.get_test_statistics("nonexistent")
        self.assertIsNone(stats)

if __name__ == '__main__':
    unittest.main()
