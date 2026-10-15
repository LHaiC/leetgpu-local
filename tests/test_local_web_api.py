import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from scripts.local_web import ApplicationServer, RequestHandler, WebApplication
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

    def get_json(self, path):
        with urlopen(f"{self.base_url}{path}", timeout=5) as response:
            return json.load(response)


if __name__ == "__main__":
    unittest.main()
