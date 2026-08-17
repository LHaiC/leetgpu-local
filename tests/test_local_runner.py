import tempfile
import unittest
from pathlib import Path

from scripts import local_runner


class LocalRunnerHelpersTest(unittest.TestCase):
    def test_cuda_staging_directory_is_ascii_for_unicode_source_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "中文提交" / "solution.cu"
            source.parent.mkdir()
            source.touch()

            with local_runner.ascii_temporary_directory("leetgpu-test-", source) as staging:
                self.assertTrue(str(staging).isascii())

    def test_parser_accepts_correctness_suite(self):
        args = local_runner.build_parser().parse_args(
            ["path/to/challenge", "--tests", "correctness"]
        )

        self.assertEqual(args.tests, "correctness")

    def test_resolve_solution_uses_language_specific_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            challenge = Path(temporary)
            solution_dir = challenge / "solution"
            solution_dir.mkdir()
            expected = solution_dir / "solution.triton.py"
            expected.write_text("def solve(): pass\n", encoding="utf-8")

            actual = local_runner.resolve_solution(challenge, "triton", None)

            self.assertEqual(actual, expected.resolve())

    def test_normalize_generated_cases_accepts_dict_and_list(self):
        case = {"N": 1}

        self.assertEqual(local_runner.normalize_generated_cases(case, "example"), [case])
        self.assertEqual(local_runner.normalize_generated_cases([case], "functional"), [case])

    def test_returned_outputs_maps_single_and_multiple_values(self):
        self.assertEqual(local_runner.returned_outputs(4, ["out"]), {"out": 4})
        self.assertEqual(
            local_runner.returned_outputs((1, 2), ["left", "right"]),
            {"left": 1, "right": 2},
        )

    def test_percentile_uses_nearest_rank(self):
        self.assertEqual(local_runner.percentile([4.0, 1.0, 3.0, 2.0], 0.95), 4.0)


if __name__ == "__main__":
    unittest.main()
