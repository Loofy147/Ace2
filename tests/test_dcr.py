import unittest
import os
import shutil
import json
from src.ace.core.implementation import DynamicContextRepository, Context

class TestDynamicContextRepository(unittest.TestCase):

    def setUp(self):
        """Set up a temporary environment for each test."""
        self.test_storage_path = "test_dcr_storage"
        # Use small cache sizes for easier testing
        self.dcr = DynamicContextRepository(
            base_storage_path=self.test_storage_path,
            l1_max_size=3,
            l2_max_size=3
        )

    def tearDown(self):
        """Clean up the temporary environment after each test."""
        if os.path.exists(self.test_storage_path):
            shutil.rmtree(self.test_storage_path)

    def test_store_and_retrieve_l1(self):
        """Test basic storage and retrieval from the L1 cache."""
        context = Context(content={"data": "test1"})
        self.dcr.store(context)
        retrieved_context = self.dcr.retrieve(context.id)

        self.assertIsNotNone(retrieved_context)
        self.assertEqual(retrieved_context.id, context.id)
        self.assertIn(context.id, self.dcr.l1_cache)

    def test_l1_eviction_to_l2(self):
        """Test that when L1 is full, the LRU item moves to L2."""
        contexts = [Context(id=f"c{i}") for i in range(4)]

        # Fill L1 (size 3)
        for i in range(3):
            self.dcr.store(contexts[i])

        self.assertEqual(len(self.dcr.l1_cache), 3)
        self.assertEqual(len(self.dcr.l2_cache), 0)

        # This store should cause an eviction from L1 to L2
        self.dcr.store(contexts[3])

        self.assertEqual(len(self.dcr.l1_cache), 3)
        self.assertEqual(len(self.dcr.l2_cache), 1)

        # The first context (c0) should be the one evicted to L2
        self.assertNotIn("c0", self.dcr.l1_cache)
        self.assertIn("c0", self.dcr.l2_cache)
        self.assertIn("c3", self.dcr.l1_cache)

    def test_l2_eviction_to_l3(self):
        """Test that when L2 is full, the LRU item is written to L3 disk."""
        # Fill L1 and L2 with 6 contexts (c0-c5)
        contexts = [Context(id=f"c{i}") for i in range(6)]
        for ctx in contexts:
            self.dcr.store(ctx)

        # At this point, L1={c3,c4,c5} and L2={c0,c1,c2}
        self.assertEqual(len(self.dcr.l1_cache), 3)
        self.assertEqual(len(self.dcr.l2_cache), 3)

        # This store should evict c3 from L1->L2, and c0 from L2->L3
        new_context = Context(id="c6")
        self.dcr.store(new_context)

        # L2 should now contain {c1, c2, c3}
        self.assertEqual(len(self.dcr.l2_cache), 3)
        self.assertIn("c3", self.dcr.l2_cache)
        self.assertNotIn("c0", self.dcr.l2_cache)

        # Check that c0 is now in L3 storage
        l3_path = os.path.join(self.test_storage_path, "l3_cold")
        expected_file = os.path.join(l3_path, "c0.json")
        self.assertTrue(os.path.exists(expected_file))

    def test_promotion_from_l2_to_l1(self):
        """Test that retrieving an item from L2 promotes it to L1."""
        contexts = [Context(id=f"c{i}") for i in range(4)]
        for ctx in contexts:
            self.dcr.store(ctx) # c0 is now in L2

        self.assertIn("c0", self.dcr.l2_cache)
        self.assertNotIn("c0", self.dcr.l1_cache)

        # Retrieve c0
        retrieved_context = self.dcr.retrieve("c0")

        self.assertIsNotNone(retrieved_context)
        self.assertIn("c0", self.dcr.l1_cache)
        self.assertNotIn("c0", self.dcr.l2_cache)

    def test_promotion_from_l3_to_l1(self):
        """Test retrieving an item from L3 promotes it to L1."""
        # Push 7 contexts through to get c0 into L3
        contexts = [Context(id=f"c{i}") for i in range(7)]
        for ctx in contexts:
            self.dcr.store(ctx)

        l3_file = os.path.join(self.dcr.l3_path, "c0.json")
        self.assertTrue(os.path.exists(l3_file))

        # Retrieve from L3
        retrieved = self.dcr.retrieve("c0")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, "c0")

        # Check it was promoted to L1 and removed from L3 disk
        self.assertIn("c0", self.dcr.l1_cache)
        self.assertFalse(os.path.exists(l3_file))

    def test_lru_behavior_in_l1(self):
        """Test that accessing an L1 item marks it as most recently used."""
        contexts = [Context(id=f"c{i}") for i in range(3)]
        for ctx in contexts:
            self.dcr.store(ctx)

        # Access the LRU item 'c0'
        self.dcr.retrieve("c0")

        # Now, add a new item to trigger eviction
        self.dcr.store(Context(id="c3"))

        # The evicted item should now be 'c1', not 'c0'
        self.assertIn("c1", self.dcr.l2_cache)
        self.assertNotIn("c0", self.dcr.l2_cache)
        self.assertIn("c0", self.dcr.l1_cache)

    def test_retrieve_from_l4_archive(self):
        """Test that an item can be retrieved from L4 (archive)."""
        context = Context(id="archived_context")
        # Manually place a context in the L4 archive
        self.dcr._write_to_disk(context, self.dcr.l4_path)

        retrieved = self.dcr.retrieve(context.id)

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, context.id)

        # Item should NOT be promoted from L4
        self.assertNotIn(context.id, self.dcr.l1_cache)

        # Item should still exist in L4
        l4_file = os.path.join(self.dcr.l4_path, f"{context.id}.json")
        self.assertTrue(os.path.exists(l4_file))

if __name__ == '__main__':
    unittest.main()
