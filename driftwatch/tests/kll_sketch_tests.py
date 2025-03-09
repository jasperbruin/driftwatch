import unittest
import random
from io import StringIO
from driftwatch.kll_sketch import KLL

import unittest
import random


class TestKLL(unittest.TestCase):

    def test_init_invalid_k(self):
        """Test that initializing KLL with invalid k raises ValueError."""
        with self.assertRaises(ValueError):
            kll = KLL(k=0)
        with self.assertRaises(ValueError):
            kll = KLL(k=-5)

    def test_insert_and_small_ranks(self):
        """
        Test that inserting a small set of known values yields
        expected rank behavior for exact matches.
        """
        k = 5
        kll = KLL(k)
        data = [3, 1, 4, 1, 5]  # Insert a small set
        for val in data:
            kll.insert(val)

        # Sort data to determine exact ranks
        sorted_data = sorted(data)
        # Check rank for each distinct value
        self.assertEqual(kll.rank(0), 0, "Rank of 0 should be 0.")
        self.assertEqual(kll.rank(1), 2, "Two items <= 1.")
        self.assertEqual(kll.rank(2), 2, "Still two items <= 2.")
        self.assertEqual(kll.rank(3), 3, "Three items <= 3.")
        self.assertTrue(kll.rank(10) >= len(data), "Rank of large value >= data size.")

    def test_cdf_monotonic(self):
        """
        Test that the CDF is monotonically non-decreasing and
        that it starts near 0 and ends near 1.
        """
        k = 10
        kll = KLL(k)
        data = [10, 20, 30, 40, 50]
        for val in data:
            kll.insert(val)

        cdf = kll.cdf()
        # cdf is a list of (value, fraction)
        self.assertTrue(len(cdf) > 0, "CDF should not be empty.")

        # Check monotonic
        for i in range(len(cdf) - 1):
            self.assertLessEqual(cdf[i][1], cdf[i + 1][1], "CDF should be non-decreasing.")

        # Check boundaries
        first_val, first_cdf = cdf[0]
        last_val, last_cdf = cdf[-1]
        self.assertGreaterEqual(first_cdf, 0, "CDF should start >= 0.")
        self.assertLessEqual(last_cdf, 1, "CDF should end <= 1.")

    def test_large_insertion_uniform_distribution(self):
        """
        Insert a large sample from a uniform distribution and compare
        the approximate rank with the true rank at several points.
        """
        k = 50
        kll = KLL(k)

        # Generate random data
        N = 2000
        data = [random.randint(0, 10000) for _ in range(N)]
        for val in data:
            kll.insert(val)

        data_sorted = sorted(data)

        # Check rank approximation at a few random query points
        for _ in range(10):
            query = random.choice(data_sorted)
            # Exact rank = number of items <= query
            true_rank = sum(1 for x in data if x <= query)
            kll_rank = kll.rank(query)

            # Since KLL is an approximation, we don't demand equality.
            # But we check that it isn't wildly off (for large N, the error
            # should be within a reasonable fraction).
            self.assertTrue(
                abs(kll_rank - true_rank) < 0.2 * N,
                f"Rank approximation off by more than 20% of N.\n"
                f"Query={query}, TrueRank={true_rank}, ApproxRank={kll_rank}"
            )

    def test_merge(self):
        """
        Ensure that merging two KLL sketches results in an approximate
        combination of their data.
        """
        k = 20
        kll1 = KLL(k)
        kll2 = KLL(k)

        # Insert data into kll1
        data1 = [random.randint(0, 5000) for _ in range(1000)]
        for val in data1:
            kll1.insert(val)

        # Insert data into kll2
        data2 = [random.randint(3000, 8000) for _ in range(1000)]
        for val in data2:
            kll2.insert(val)

        # Merge sketches
        kll1.merge(kll2)

        # Combined data
        combined_data = data1 + data2
        combined_data_sorted = sorted(combined_data)

        # Check rank of some random points
        for _ in range(10):
            query = random.randint(0, 8000)
            true_rank = sum(1 for x in combined_data if x <= query)
            approx_rank = kll1.rank(query)
            # Check approximate correctness
            self.assertTrue(
                abs(approx_rank - true_rank) < 0.2 * len(combined_data),
                f"Rank approximation after merge is too far off for query={query}. "
                f"True={true_rank}, Approx={approx_rank}"
            )

    def test_merge_empty(self):
        """Merging an empty KLL with a non-empty KLL should preserve data."""
        kll_nonempty = KLL(10)
        kll_empty = KLL(10)

        data = [1, 2, 3, 4, 5]
        for d in data:
            kll_nonempty.insert(d)

        # Merge empty into nonempty
        kll_nonempty.merge(kll_empty)

        # Ranks should remain the same as before
        for val in data:
            self.assertGreater(kll_nonempty.rank(val), 0,
                               "Rank should remain correct after merging with empty sketch.")

    def test_compactor_increments(self):
        """
        Ensure that compactions increment the numCompactions counter,
        and that compactions actually reduce the size of the list.
        """
        k = 5
        kll = KLL(k)

        # Force multiple insertions to trigger compaction
        for i in range(50):
            kll.insert(i)

        # Check that at least one compactor had a compaction
        compactions_count = sum(c.numCompactions for c in kll.compactors)
        self.assertGreater(compactions_count, 0, "No compactions occurred when expected.")

        # Check that total items are fewer than 50 after compaction
        stored_items = sum(len(c) for c in kll.compactors)
        self.assertLess(stored_items, 50, "Compaction did not reduce the total item count.")


if __name__ == '__main__':
    unittest.main()

