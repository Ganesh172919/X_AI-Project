from __future__ import annotations

import unittest
from unittest.mock import patch

import main


class CliTests(unittest.TestCase):
    def test_default_variant_is_compare(self) -> None:
        with patch("sys.argv", ["main.py", "--dataset", "covertype"]):
            args = main.parse_args()
        self.assertEqual(args.variant, "compare")
        self.assertEqual(args.dataset, "covertype")

    def test_report_only_flag_parses(self) -> None:
        with patch("sys.argv", ["main.py", "--report-only"]):
            args = main.parse_args()
        self.assertTrue(args.report_only)
        self.assertEqual(args.dataset, "covertype")


if __name__ == "__main__":
    unittest.main()
