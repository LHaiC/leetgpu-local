import tempfile
import unittest
from pathlib import Path

from scripts.local_web import ChallengeCatalog, ScoreStore, resolve_challenges_dir
from support import create_challenge_tree


class ChallengeCatalogTest(unittest.TestCase):
    def test_discovers_external_challenge_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            challenges_dir = create_challenge_tree(Path(temporary))
            catalog = ChallengeCatalog(challenges_dir)

            challenges = {item["key"]: item for item in catalog.list_public()}

            key = "easy/1_synthetic_addition"
            self.assertIn(key, challenges)
            self.assertEqual(challenges[key]["title"], "Synthetic Addition")
            detail = catalog.detail(key)
            self.assertIn("cuda", detail["starters"])
            self.assertIn("Synthetic test challenge", detail["problem_html"])

    def test_rejects_missing_challenge_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing"

            with self.assertRaises(FileNotFoundError):
                ChallengeCatalog(missing)

    def test_resolves_an_upstream_repository_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            challenges_dir = create_challenge_tree(root / "upstream")

            resolved = resolve_challenges_dir(root, challenges_dir.parent)

            self.assertEqual(resolved, challenges_dir.resolve())


class ScoreStoreTest(unittest.TestCase):
    def test_leaderboard_keeps_each_runner_language_best(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ScoreStore(Path(temporary) / "scores.sqlite3")
            self.record(store, "one", "alice", "cuda", 2.0)
            self.record(store, "two", "alice", "cuda", 1.0)
            self.record(store, "three", "alice", "triton", 1.5)
            self.record(store, "four", "bob", "cuda", 0.8)

            scores = store.leaderboard("easy/1_vector_add")

            self.assertEqual([score["job_id"] for score in scores], ["four", "two", "three"])
            self.assertEqual([score["rank"] for score in scores], [1, 2, 3])
            submission = store.submission("two")
            self.assertEqual(submission["source_code"], "// two")
            self.assertTrue(store.delete("two"))
            self.assertEqual(store.leaderboard("easy/1_vector_add")[1]["job_id"], "three")

    def test_submission_progress_counts_attempts_and_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ScoreStore(Path(temporary) / "scores.sqlite3")
            failed_job = self.job("failed", "alice", "cuda")
            passed_job = self.job("passed", "bob", "triton")

            store.begin_submission(failed_job)
            failed_job.update(status="completed", finished_at="2026-01-01T00:01:00+00:00")
            store.finish_submission(failed_job, {"status": "failed"})
            progress = store.challenge_progress()["easy/1_vector_add"]
            self.assertEqual(progress, {"submission_count": 1, "passed": False})

            store.begin_submission(passed_job)
            passed_job.update(status="completed", finished_at="2026-01-01T00:02:00+00:00")
            store.finish_submission(passed_job, {"status": "passed"})
            progress = store.challenge_progress()["easy/1_vector_add"]
            self.assertEqual(progress, {"submission_count": 2, "passed": True})

    @staticmethod
    def record(store, job_id, runner, language, median):
        job = ScoreStoreTest.job(job_id, runner, language)
        job["result_path"] = "result.json"
        result = {
            "status": "passed",
            "challenge": "Vector Addition",
            "environment": {"gpu": "Test GPU"},
            "benchmark": {"gpu_ms": {"median": median, "min": median * 0.9, "p95": median * 1.1}},
        }
        store.record(job, result)

    @staticmethod
    def job(job_id, runner, language):
        return {
            "id": job_id,
            "challenge": "easy/1_vector_add",
            "runner_name": runner,
            "language": language,
            "mode": "submit",
            "source_hash": job_id * 8,
            "source_code": f"// {job_id}",
            "solution_path": f"solution.{language}",
            "created_at": f"2026-01-01T00:00:0{len(job_id)}+00:00",
        }


if __name__ == "__main__":
    unittest.main()
