import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from scripts.local_web import ApplicationServer, RequestHandler, ScoreStore, WebApplication
from support import create_challenge_tree


class LocalWebApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(__file__).resolve().parent.parent
        challenges_dir = create_challenge_tree(Path(cls.temporary.name))
        cls.app = WebApplication(
            root,
            Path(cls.temporary.name) / "data",
            challenges_dir=challenges_dir,
        )
        cls.server = ApplicationServer(("127.0.0.1", 0), RequestHandler, cls.app)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.app.jobs.stop()
        cls.temporary.cleanup()

    def test_challenge_and_system_endpoints(self):
        challenges = self.get_json("/api/challenges")
        system = self.get_json("/api/system")

        self.assertTrue(
            any(item["key"] == "easy/1_synthetic_addition" for item in challenges["challenges"])
        )
        self.assertIn("available", system)

    def test_submission_rejects_object_source(self):
        body = json.dumps(
            {
                "challenge": "easy/1_synthetic_addition",
                "language": "cuda",
                "source": {"value": "not source code"},
                "runner_name": "test",
            }
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/jobs",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        with self.assertRaises(HTTPError) as context:
            urlopen(request, timeout=5)

        self.assertEqual(context.exception.code, 400)

    def test_submission_is_accepted(self):
        body = json.dumps(
            {
                "challenge": "easy/1_synthetic_addition",
                "language": "cuda",
                "source": "extern \"C\" void solve() {}",
                "runner_name": "tester",
                "mode": "test",
            }
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/jobs",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        with urlopen(request, timeout=5) as response:
            job = json.load(response)

        self.assertEqual(response.status, 202)
        self.assertTrue(job["id"])
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["mode"], "test")

    def test_serves_nested_static_files(self):
        with urlopen(f"{self.base_url}/static/vendor/codemirror/lib/codemirror.js", timeout=5) as r:
            body = r.read()
        self.assertEqual(r.status, 200)
        self.assertIn(b"CodeMirror", body)

    def test_rejects_static_path_traversal(self):
        for path in ("/static/../scripts/local_web.py", "/static/vendor/../../local_web.py"):
            with self.assertRaises(HTTPError) as context:
                urlopen(f"{self.base_url}{path}", timeout=5)
            self.assertIn(context.exception.code, (403, 404))

    def test_static_assets_are_gzipped_and_revalidate(self):
        import gzip as gzip_module

        url = f"{self.base_url}/static/vendor/codemirror/lib/codemirror.js"
        request = Request(url, headers={"Accept-Encoding": "gzip"})
        with urlopen(request, timeout=5) as response:
            encoding = response.headers.get("Content-Encoding")
            etag = response.headers.get("ETag")
            payload = response.read()

        self.assertEqual(encoding, "gzip")
        self.assertTrue(etag)
        self.assertIn(b"CodeMirror", gzip_module.decompress(payload))

        revalidate = Request(url, headers={"If-None-Match": etag})
        with self.assertRaises(HTTPError) as context:
            urlopen(revalidate, timeout=5)
        self.assertEqual(context.exception.code, 304)

    def test_overall_leaderboard_and_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ScoreStore(Path(temporary) / "scores.sqlite3")
            # alice wins challenge A, bob wins challenge B → both percentiles 1.0
            self.record(store, "a1", "alice", 1.0, "easy/1_vector_add")
            self.record(store, "a2", "bob", 2.0, "easy/1_vector_add")
            self.record(store, "b1", "bob", 1.0, "easy/2_matmul")
            self.record(store, "b2", "alice", 3.0, "easy/2_matmul")

            overall = store.overall_leaderboard()
            by_user = {row["username"]: row for row in overall}
            self.assertEqual(set(by_user), {"alice", "bob"})
            for row in overall:
                self.assertEqual(row["challenges"], 2)
                self.assertEqual(row["wins"], 1)
                # Each user wins one challenge and is last on the other:
                # (1.0 + 0.0) / 2
                self.assertAlmostEqual(row["avg_percentile"], 0.5)
            self.assertEqual(by_user["alice"]["avg_place"], 1.5)

            profile = store.user_profile("alice")
            self.assertEqual(profile["username"], "alice")
            self.assertEqual(len(profile["bests"]), 2)
            self.assertIsNotNone(profile["summary"])
            self.assertEqual(profile["summary"]["wins"], 1)

    @staticmethod
    def record(store, job_id, runner, median, challenge):
        job = {
            "id": job_id,
            "challenge": challenge,
            "runner_name": runner,
            "language": "cuda",
            "mode": "submit",
            "source_hash": job_id * 8,
            "source_code": f"// {job_id}",
            "solution_path": "solution.cu",
            "created_at": "2026-01-01T00:00:00+00:00",
            "result_path": "result.json",
        }
        store.record(job, {
            "status": "passed",
            "challenge": challenge,
            "environment": {"gpu": "Test GPU"},
            "benchmark": {"gpu_ms": {"median": median, "min": median * 0.9, "p95": median * 1.1}},
        })

    def get_json(self, path):
        with urlopen(f"{self.base_url}{path}", timeout=5) as response:
            return json.load(response)


if __name__ == "__main__":
    unittest.main()
