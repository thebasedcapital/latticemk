"""Consumer-visible lookup boundaries and suffix-precedence regressions."""
import unittest
from runtime import Lookup


class LookupTests(unittest.TestCase):
    def test_longest_suffix_wins_over_latest_short_match(self):
        lookup = Lookup([1, 2, 3, 8, 2, 9, 1, 2])
        self.assertEqual(lookup.draft(2), ([3, 8], 2))

    def test_latest_equal_length_continuation_wins(self):
        lookup = Lookup([1, 2, 7, 1, 2, 8, 1, 2])
        self.assertEqual(lookup.draft(1), ([8], 2))

    def test_unseen_suffix_has_no_drafts(self):
        self.assertEqual(Lookup([4, 5, 6]).draft(4), ([], 0))
        self.assertEqual(Lookup([]).draft(4), ([], 0))

    def test_accepted_continuation_enters_history(self):
        lookup = Lookup([1, 2, 3, 1, 2])
        self.assertEqual(lookup.draft(4), ([3, 1, 2], 2))
        lookup.extend([9, 1, 2])
        self.assertEqual(lookup.draft(1), ([9], 2))


if __name__ == '__main__':
    unittest.main()
