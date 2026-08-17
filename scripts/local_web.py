#!/usr/bin/env python3
"""Local web UI, submission queue, and per-challenge leaderboard."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import mimetypes
import os
import queue
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from contextlib import closing
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import parse_qs, urlsplit

LOGGER = logging.getLogger("leetgpu.web")
LANGUAGES = ("cuda", "triton", "pytorch")
SOLUTION_SUFFIXES = {
    "cuda": ".cu",
    "triton": ".triton.py",
    "pytorch": ".pytorch.py",
}
STARTER_FILES = {
    "cuda": "starter.cu",
    "triton": "starter.triton.py",
    "pytorch": "starter.pytorch.py",
}
MAX_SOURCE_BYTES = 2 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def filesystem_name(value: str, fallback: str = "item") -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:100] or fallback


def constant_assignments(class_node: ast.ClassDef) -> Dict[str, Any]:
    values = {}
    for node in class_node.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return values


class ChallengeCatalog:
    def __init__(self, challenges_dir: Path):
        self.challenges_dir = challenges_dir.expanduser().resolve()
        if not self.challenges_dir.is_dir():
            raise FileNotFoundError(
                f"Challenge directory does not exist: {self.challenges_dir}. "
                "Clone the upstream LeetGPU repository or pass --challenges-dir."
            )
        self._challenges = self._discover()
        if not self._challenges:
            raise ValueError(f"No compatible challenges found in {self.challenges_dir}")

    def _discover(self) -> Dict[str, Dict[str, Any]]:
        challenges = {}
        for difficulty in ("easy", "medium", "hard"):
            difficulty_dir = self.challenges_dir / difficulty
            if not difficulty_dir.is_dir():
                continue
            for challenge_dir in difficulty_dir.iterdir():
                challenge_py = challenge_dir / "challenge.py"
                if not challenge_dir.is_dir() or not challenge_py.is_file():
                    continue
                try:
                    tree = ast.parse(challenge_py.read_text(encoding="utf-8"))
                    class_node = next(
                        node
                        for node in tree.body
                        if isinstance(node, ast.ClassDef) and node.name == "Challenge"
                    )
                    metadata = constant_assignments(class_node)
                    number_match = re.match(r"^(\d+)_", challenge_dir.name)
                    number = int(number_match.group(1)) if number_match else 1_000_000
                    key = f"{difficulty}/{challenge_dir.name}"
                    challenges[key] = {
                        "key": key,
                        "id": number,
                        "slug": challenge_dir.name,
                        "title": metadata.get("name", challenge_dir.name),
                        "difficulty": difficulty,
                        "access_tier": metadata.get("access_tier", "free"),
                        "path": challenge_dir,
                    }
                except (OSError, SyntaxError, StopIteration) as exc:
                    LOGGER.warning("Skipping invalid challenge %s: %s", challenge_dir, exc)
        return challenges

    def list_public(self) -> List[Dict[str, Any]]:
        fields = ("key", "id", "slug", "title", "difficulty", "access_tier")
        values = [{field: item[field] for field in fields} for item in self._challenges.values()]
        return sorted(values, key=lambda item: (item["id"], item["difficulty"]))

    def require(self, key: str) -> Dict[str, Any]:
        challenge = self._challenges.get(key)
        if challenge is None:
            raise KeyError(f"Unknown challenge: {key}")
        return challenge

    def detail(self, key: str) -> Dict[str, Any]:
        challenge = self.require(key)
        challenge_dir = challenge["path"]
        problem_path = challenge_dir / "challenge.html"
        starters = {}
        for language, filename in STARTER_FILES.items():
            starter = challenge_dir / "starter" / filename
            starters[language] = starter.read_text(encoding="utf-8") if starter.is_file() else None
        public = {name: value for name, value in challenge.items() if name != "path"}
        public.update(
            problem_html=problem_path.read_text(encoding="utf-8") if problem_path.is_file() else "",
            starters=starters,
        )
        return public


class ScoreStore:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS scores (
                        job_id TEXT PRIMARY KEY,
                        challenge_key TEXT NOT NULL,
                        challenge_name TEXT NOT NULL,
                        runner_name TEXT NOT NULL,
                        language TEXT NOT NULL,
                        gpu_name TEXT NOT NULL,
                        median_ms REAL NOT NULL,
                        min_ms REAL NOT NULL,
                        p95_ms REAL NOT NULL,
                        source_hash TEXT NOT NULL,
                        source_code TEXT NOT NULL DEFAULT '',
                        solution_filename TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        result_path TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS scores_challenge_time "
                    "ON scores(challenge_key, median_ms)"
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS submission_attempts (
                        job_id TEXT PRIMARY KEY,
                        challenge_key TEXT NOT NULL,
                        runner_name TEXT NOT NULL,
                        language TEXT NOT NULL,
                        status TEXT NOT NULL,
                        result_status TEXT,
                        created_at TEXT NOT NULL,
                        finished_at TEXT
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS attempts_challenge "
                    "ON submission_attempts(challenge_key, created_at)"
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO submission_attempts (
                        job_id, challenge_key, runner_name, language, status,
                        result_status, created_at, finished_at
                    )
                    SELECT job_id, challenge_key, runner_name, language, 'completed',
                           'passed', created_at, created_at
                    FROM scores
                    """
                )
                columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(scores)").fetchall()
                }
                if "source_code" not in columns:
                    connection.execute(
                        "ALTER TABLE scores ADD COLUMN source_code TEXT NOT NULL DEFAULT ''"
                    )
                if "solution_filename" not in columns:
                    connection.execute(
                        "ALTER TABLE scores ADD COLUMN solution_filename TEXT NOT NULL DEFAULT ''"
                    )
                self._backfill_legacy_sources(connection)

    @staticmethod
    def _backfill_legacy_sources(connection) -> None:
        rows = connection.execute(
            "SELECT job_id, result_path FROM scores WHERE source_code = ''"
        ).fetchall()
        for row in rows:
            result_path = Path(row["result_path"])
            solution = None
            for parent in (result_path.parent, *list(result_path.parents)[:5]):
                matches = sorted(parent.glob("solution.*"))
                if matches:
                    solution = matches[0]
                    break
            if solution is None:
                continue
            try:
                source = solution.read_text(encoding="utf-8")
            except OSError:
                continue
            connection.execute(
                "UPDATE scores SET source_code = ?, solution_filename = ? WHERE job_id = ?",
                (source, solution.name, row["job_id"]),
            )

    def record(self, job: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
        if result.get("status") != "passed" or not result.get("benchmark"):
            return False
        gpu_times = result["benchmark"]["gpu_ms"]
        environment = result.get("environment", {})
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO scores (
                        job_id, challenge_key, challenge_name, runner_name, language,
                        gpu_name, median_ms, min_ms, p95_ms, source_hash, source_code,
                        solution_filename, created_at, result_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job["id"],
                        job["challenge"],
                        result.get("challenge", job["challenge"]),
                        job["runner_name"],
                        job["language"],
                        environment.get("gpu", "Unknown GPU"),
                        float(gpu_times["median"]),
                        float(gpu_times["min"]),
                        float(gpu_times["p95"]),
                        job["source_hash"],
                        job["source_code"],
                        Path(job["solution_path"]).name,
                        job["created_at"],
                        str(job.get("result_path", "")),
                    ),
                )
        return True

    def leaderboard(
        self, challenge_key: str, language: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        where = "challenge_key = ?"
        parameters: List[Any] = [challenge_key]
        if language:
            where += " AND language = ?"
            parameters.append(language)
        parameters.append(limit)
        query = f"""
            WITH personal_best AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY runner_name, language ORDER BY median_ms ASC, created_at ASC
                ) AS personal_rank
                FROM scores
                WHERE {where}
            )
            SELECT job_id, runner_name, language, gpu_name, median_ms, min_ms,
                   p95_ms, source_hash, created_at
            FROM personal_best
            WHERE personal_rank = 1
            ORDER BY median_ms ASC, created_at ASC
            LIMIT ?
        """
        with closing(self._connect()) as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(rank=index, **dict(row)) for index, row in enumerate(rows, 1)]

    def submission(self, job_id: str) -> Dict[str, Any]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT job_id, challenge_key, challenge_name, runner_name, language,
                       gpu_name, median_ms, min_ms, p95_ms, source_hash, source_code,
                       solution_filename, created_at
                FROM scores WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return dict(row)

    def delete(self, job_id: str) -> bool:
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute("DELETE FROM scores WHERE job_id = ?", (job_id,))
                return cursor.rowcount > 0

    def begin_submission(self, job: Mapping[str, Any]) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO submission_attempts (
                        job_id, challenge_key, runner_name, language, status,
                        result_status, created_at, finished_at
                    ) VALUES (?, ?, ?, ?, 'queued', NULL, ?, NULL)
                    """,
                    (
                        job["id"],
                        job["challenge"],
                        job["runner_name"],
                        job["language"],
                        job["created_at"],
                    ),
                )

    def finish_submission(
        self, job: Mapping[str, Any], result: Optional[Mapping[str, Any]]
    ) -> None:
        if job.get("mode") != "submit":
            return
        result_status = result.get("status") if result else "error"
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE submission_attempts
                    SET status = ?, result_status = ?, finished_at = ?
                    WHERE job_id = ?
                    """,
                    (job.get("status", "error"), result_status, job.get("finished_at"), job["id"]),
                )

    def challenge_progress(self) -> Dict[str, Dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT challenge_key, COUNT(*) AS submission_count,
                       MAX(CASE WHEN result_status = 'passed' THEN 1 ELSE 0 END) AS passed
                FROM submission_attempts
                GROUP BY challenge_key
                """
            ).fetchall()
        return {
            row["challenge_key"]: {
                "submission_count": int(row["submission_count"]),
                "passed": bool(row["passed"]),
            }
            for row in rows
        }


class SettingsStore:
    def __init__(self, path: Path, default_ncu_dir: Path):
        self.path = path.resolve()
        self.default_ncu_dir = default_ncu_dir.resolve()
        self.lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.values = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.is_file():
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    return value
            except (OSError, json.JSONDecodeError):
                LOGGER.warning("Ignoring invalid settings file: %s", self.path)
        return {"ncu_export_dir": str(self.default_ncu_dir)}

    def public(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.values)

    def set_ncu_export_dir(self, value: str) -> Dict[str, Any]:
        if not value.strip():
            raise ValueError("NCU export directory cannot be empty")
        expanded = os.path.expandvars(value.strip())
        directory = Path(expanded).expanduser().resolve()
        if directory.exists() and not directory.is_dir():
            raise ValueError(f"NCU export path is not a directory: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.values["ncu_export_dir"] = str(directory)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(self.values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            temporary.replace(self.path)
            return dict(self.values)


def choose_directory(initial: str) -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("Tk directory picker is unavailable") from exc
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(initialdir=initial, mustexist=False, parent=root)
    finally:
        root.destroy()
    return selected or None


def find_ncu_ui() -> Optional[Path]:
    configured = os.getenv("NCU_UI_PATH")
    if configured and Path(configured).is_file():
        return Path(configured).resolve()
    found = shutil.which("ncu-ui") or shutil.which("ncu-ui.exe")
    if found:
        path = Path(found).resolve()
        if path.suffix.lower() == ".bat":
            executable = path.parent / "host" / "windows-desktop-win7-x64" / "ncu-ui.exe"
            if executable.is_file():
                return executable
        return path
    if os.name == "nt":
        root = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation"
        matches = sorted(root.glob("Nsight Compute */host/windows-desktop-win7-x64/ncu-ui.exe"))
        if not matches:
            matches = sorted(root.glob("Nsight Compute */ncu-ui.exe"))
        if matches:
            return matches[-1]
    return None


class JobManager:
    def __init__(
        self,
        root: Path,
        data_dir: Path,
        catalog: ChallengeCatalog,
        scores: ScoreStore,
        ncu_launch_count: int,
    ):
        self.root = root.resolve()
        self.data_dir = data_dir.resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.catalog = catalog
        self.scores = scores
        self.ncu_launch_count = ncu_launch_count
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.pending: queue.Queue = queue.Queue()
        self.active_process: Optional[subprocess.Popen] = None
        self.worker = threading.Thread(target=self._work, name="gpu-runner", daemon=True)
        self.worker.start()

    def submit(
        self,
        challenge: str,
        language: str,
        source: str,
        runner_name: str,
        profile: bool,
        mode: str,
    ) -> Dict[str, Any]:
        challenge_info = self.catalog.require(challenge)
        if language not in LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")
        if mode not in {"test", "submit"}:
            raise ValueError(f"Unsupported run mode: {mode}")
        if not source.strip():
            raise ValueError("Source code is empty")
        encoded = source.encode("utf-8")
        if len(encoded) > MAX_SOURCE_BYTES:
            raise ValueError("Source code exceeds 2 MiB")
        runner_name = re.sub(r"[^\w .-]", "", runner_name, flags=re.UNICODE).strip()[:32]
        if not runner_name:
            runner_name = "local"

        job_id = uuid.uuid4().hex[:12]
        challenge_dir_name = filesystem_name(
            f"{challenge_info['id']}_{challenge_info['title']}", challenge_info["slug"]
        )
        submission_dir_name = filesystem_name(runner_name, "local")
        job_dir = self.data_dir / "runs" / challenge_dir_name / submission_dir_name
        pending_dir = self.data_dir / ".pending" / job_id
        job = {
            "id": job_id,
            "challenge": challenge,
            "language": language,
            "runner_name": runner_name,
            "profile": profile,
            "mode": mode,
            "source_hash": hashlib.sha256(encoded).hexdigest(),
            "source_code": source,
            "status": "queued",
            "created_at": utc_now(),
            "started_at": None,
            "finished_at": None,
            "output": "",
            "result": None,
            "error": None,
            "job_dir": str(job_dir),
            "pending_dir": str(pending_dir),
        }
        pending_dir.mkdir(parents=True, exist_ok=False)
        solution = pending_dir / f"solution{SOLUTION_SUFFIXES[language]}"
        solution.write_text(source, encoding="utf-8")
        job["solution_path"] = str(solution)
        with self.lock:
            self.jobs[job_id] = job
        if mode == "submit":
            self.scores.begin_submission(job)
        self.pending.put(job_id)
        return self.public(job_id)

    def public(self, job_id: str) -> Dict[str, Any]:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            job = dict(self.jobs[job_id])
            job.pop("solution_path", None)
            job.pop("job_dir", None)
            job.pop("pending_dir", None)
            job.pop("source_code", None)
            job["output"] = job["output"][-50_000:]
            if job["status"] == "queued":
                queued = list(self.pending.queue)
                job["queue_position"] = queued.index(job_id) + 1 if job_id in queued else 1
            return job

    def artifact(self, job_id: str, kind: str) -> Optional[Path]:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            job_dir = Path(job["job_dir"])
        patterns = {
            "execution": "execution.log",
            "result": "result.json",
            "nsight": "nsight-compute.ncu-rep",
            "nsight-csv": "nsight-details.csv",
        }
        filename = patterns.get(kind)
        if not filename:
            return None
        matches = list(job_dir.rglob(filename))
        return matches[-1] if matches else None

    def _append_output(self, job_id: str, line: str) -> None:
        with self.lock:
            job = self.jobs[job_id]
            job["output"] = (job["output"] + line)[-100_000:]

    def _work(self) -> None:
        while True:
            job_id = self.pending.get()
            if job_id is None:
                self.pending.task_done()
                return
            try:
                self._run(job_id)
            except Exception as exc:
                LOGGER.exception("Job %s failed", job_id)
                with self.lock:
                    job = self.jobs[job_id]
                    job.update(status="error", error=str(exc), finished_at=utc_now())
                self.scores.finish_submission(job, None)
            finally:
                self._cleanup_pending(job_id)
                self.pending.task_done()

    def _cleanup_pending(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            pending_dir = Path(job["pending_dir"]) if job else None
        if pending_dir and pending_dir.is_dir():
            shutil.rmtree(pending_dir, ignore_errors=True)

    def _run(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs[job_id]
            job.update(status="running", started_at=utc_now())
            job_dir = Path(job["job_dir"])
            pending_solution = Path(job["solution_path"])
        challenge = self.catalog.require(job["challenge"])
        if job_dir.is_dir():
            shutil.rmtree(job_dir)
        job_dir.mkdir(parents=True, exist_ok=False)
        solution = job_dir / pending_solution.name
        shutil.copy2(pending_solution, solution)
        with self.lock:
            job["solution_path"] = str(solution)
        command = [
            sys.executable,
            str(self.root / "scripts" / "local_runner.py"),
            str(challenge["path"]),
            "--language",
            job["language"],
            "--solution",
            str(solution),
            "--run-dir",
            str(job_dir),
            "--tests",
            "correctness" if job["mode"] == "test" else "all",
            "--profile",
            "auto" if job["mode"] == "submit" and job["profile"] else "off",
            "--ncu-launch-count",
            str(self.ncu_launch_count),
        ]
        (job_dir / "command.txt").write_text(
            subprocess.list2cmdline(command) + "\n", encoding="utf-8"
        )
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            command,
            cwd=self.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creation_flags,
        )
        with self.lock:
            self.active_process = process
        web_log = job_dir / "web-execution.log"
        with web_log.open("w", encoding="utf-8", buffering=1) as output_file:
            assert process.stdout is not None
            for line in process.stdout:
                output_file.write(line)
                self._append_output(job_id, line)
        exit_code = process.wait()
        with self.lock:
            self.active_process = None

        results = sorted(job_dir.rglob("result.json"), key=lambda path: path.stat().st_mtime)
        result = None
        if results:
            result_path = results[-1]
            result = json.loads(result_path.read_text(encoding="utf-8"))
            job["result_path"] = str(result_path)
        status = "completed" if result is not None else "error"
        error = (
            None if result is not None else f"Runner exited with {exit_code} without result.json"
        )
        with self.lock:
            job.update(
                status=status,
                runner_exit_code=exit_code,
                result=result,
                error=error,
                finished_at=utc_now(),
            )
        self.scores.finish_submission(job, result)
        if result is not None and job["mode"] == "submit":
            self.scores.record(job, result)

    def stop(self) -> None:
        with self.lock:
            process = self.active_process
        if process and process.poll() is None:
            process.terminate()
        self.pending.put(None)


def system_info() -> Dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False, "error": result.stderr.strip() or "nvidia-smi failed"}
    name, driver, memory = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
    return {
        "available": True,
        "gpu": name,
        "driver": driver,
        "memory_mb": int(memory),
        "python": sys.executable,
    }


class WebApplication:
    def __init__(
        self,
        root: Path,
        data_dir: Path,
        ncu_launch_count: int,
        challenges_dir: Path,
    ):
        self.root = root.resolve()
        self.web_root = self.root / "web"
        self.catalog = ChallengeCatalog(challenges_dir)
        self.scores = ScoreStore(data_dir / "leaderboard.sqlite3")
        self.settings = SettingsStore(data_dir / "settings.json", self.root / "ncu_reports")
        self.jobs = JobManager(
            self.root, data_dir, self.catalog, self.scores, ncu_launch_count=ncu_launch_count
        )
        self.system = system_info()

    def challenge_list(self) -> List[Dict[str, Any]]:
        progress = self.scores.challenge_progress()
        challenges = self.catalog.list_public()
        for challenge in challenges:
            challenge.update(
                progress.get(
                    challenge["key"],
                    {"submission_count": 0, "passed": False},
                )
            )
        return challenges

    def export_nsight(self, job_id: str) -> Path:
        report = self.jobs.artifact(job_id, "nsight")
        if report is None:
            raise FileNotFoundError("Nsight Compute report is not available")
        job = self.jobs.public(job_id)
        challenge = self.catalog.require(job["challenge"])
        export_dir = Path(self.settings.public()["ncu_export_dir"])
        export_dir.mkdir(parents=True, exist_ok=True)
        filename = filesystem_name(f"{challenge['title']}_{job['runner_name']}") + ".ncu-rep"
        destination = export_dir / filename
        shutil.copy2(report, destination)
        return destination

    def open_nsight(self, job_id: str) -> Path:
        report = self.jobs.artifact(job_id, "nsight")
        if report is None:
            raise FileNotFoundError("Nsight Compute report is not available")
        executable = find_ncu_ui()
        if executable is None:
            raise FileNotFoundError("ncu-ui.exe was not found")
        subprocess.Popen([str(executable), str(report)], close_fds=True)
        return report


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "LeetGPULocal/1.0"

    @property
    def app(self) -> WebApplication:
        return self.server.app

    def log_message(self, format_string: str, *args) -> None:
        LOGGER.info("%s - %s", self.address_string(), format_string % args)

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")

    def send_json(self, value: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: int, message: str) -> None:
        self.send_json({"error": message}, status=status)

    def send_file(self, path: Path, attachment: bool = False) -> None:
        if not path.is_file():
            self.send_error_json(HTTPStatus.NOT_FOUND, "File not found")
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self._security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        if attachment:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length <= 0 or length > MAX_SOURCE_BYTES + 64 * 1024:
            raise ValueError("Request body is empty or too large")
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Request body must be UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("Request JSON must be an object")
        return value

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/challenges":
                self.send_json({"challenges": self.app.challenge_list()})
            elif parsed.path == "/api/challenge":
                self.send_json(self.app.catalog.detail(query.get("key", [""])[0]))
            elif parsed.path == "/api/system":
                self.send_json(self.app.system)
            elif parsed.path == "/api/leaderboard":
                challenge = query.get("challenge", [""])[0]
                self.app.catalog.require(challenge)
                language = query.get("language", [""])[0] or None
                if language and language not in LANGUAGES:
                    raise ValueError("Invalid leaderboard language")
                self.send_json(
                    {"scores": self.app.scores.leaderboard(challenge, language=language)}
                )
            elif parsed.path == "/api/leaderboard/source":
                self.send_json(self.app.scores.submission(query.get("job", [""])[0]))
            elif parsed.path == "/api/settings":
                self.send_json(self.app.settings.public())
            elif parsed.path.startswith("/api/jobs/"):
                parts = parsed.path.strip("/").split("/")
                job_id = parts[2] if len(parts) >= 3 else ""
                if len(parts) == 3:
                    self.send_json(self.app.jobs.public(job_id))
                elif len(parts) == 5 and parts[3] == "artifacts":
                    artifact = self.app.jobs.artifact(job_id, parts[4])
                    if artifact is None:
                        self.send_error_json(HTTPStatus.NOT_FOUND, "Artifact not available")
                    else:
                        self.send_file(artifact, attachment=True)
                else:
                    self.send_error_json(HTTPStatus.NOT_FOUND, "Unknown API path")
            elif parsed.path in {"/", "/index.html"}:
                self.send_file(self.app.web_root / "index.html")
            elif parsed.path.startswith("/static/"):
                relative = parsed.path.removeprefix("/static/")
                if "/" in relative or "\\" in relative or relative.startswith("."):
                    self.send_error_json(HTTPStatus.NOT_FOUND, "Invalid static path")
                else:
                    self.send_file(self.app.web_root / relative)
            else:
                self.send_error_json(HTTPStatus.NOT_FOUND, "Not found")
        except KeyError as exc:
            self.send_error_json(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            LOGGER.exception("GET %s failed", self.path)
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        parsed = urlsplit(self.path)
        try:
            if parsed.path == "/api/jobs":
                body = self.read_json()
                for field in ("challenge", "language", "source", "runner_name", "mode"):
                    if field in body and not isinstance(body[field], str):
                        raise ValueError(f"{field} must be a string")
                job = self.app.jobs.submit(
                    challenge=str(body.get("challenge", "")),
                    language=str(body.get("language", "")),
                    source=str(body.get("source", "")),
                    runner_name=str(body.get("runner_name", "local")),
                    profile=bool(body.get("profile", True)),
                    mode=str(body.get("mode", "submit")),
                )
                self.send_json(job, status=HTTPStatus.ACCEPTED)
            elif parsed.path == "/api/settings":
                body = self.read_json()
                directory = body.get("ncu_export_dir")
                if not isinstance(directory, str):
                    raise ValueError("ncu_export_dir must be a string")
                self.send_json(self.app.settings.set_ncu_export_dir(directory))
            elif parsed.path == "/api/settings/pick-directory":
                current = self.app.settings.public()["ncu_export_dir"]
                selected = choose_directory(current)
                if selected:
                    self.send_json(self.app.settings.set_ncu_export_dir(selected))
                else:
                    self.send_json(self.app.settings.public())
            elif parsed.path.startswith("/api/jobs/"):
                parts = parsed.path.strip("/").split("/")
                job_id = parts[2] if len(parts) >= 3 else ""
                action = parts[3] if len(parts) == 4 else ""
                if action == "export-nsight":
                    destination = self.app.export_nsight(job_id)
                    self.send_json({"destination": str(destination)})
                elif action == "open-nsight":
                    report = self.app.open_nsight(job_id)
                    self.send_json({"report": str(report)})
                else:
                    self.send_error_json(HTTPStatus.NOT_FOUND, "Unknown job action")
            else:
                self.send_error_json(HTTPStatus.NOT_FOUND, "Unknown API path")
        except KeyError as exc:
            self.send_error_json(HTTPStatus.NOT_FOUND, str(exc))
        except FileNotFoundError as exc:
            self.send_error_json(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            LOGGER.exception("POST %s failed", self.path)
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_DELETE(self) -> None:
        parsed = urlsplit(self.path)
        try:
            parts = parsed.path.strip("/").split("/")
            if len(parts) == 3 and parts[:2] == ["api", "leaderboard"]:
                deleted = self.app.scores.delete(parts[2])
                if not deleted:
                    self.send_error_json(HTTPStatus.NOT_FOUND, "Leaderboard entry not found")
                else:
                    self.send_json({"deleted": True})
            else:
                self.send_error_json(HTTPStatus.NOT_FOUND, "Unknown API path")
        except Exception as exc:
            LOGGER.exception("DELETE %s failed", self.path)
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))


class ApplicationServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, app):
        super().__init__(address, handler)
        self.app = app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--data-dir", type=Path, default=Path("local_data"))
    parser.add_argument(
        "--challenges-dir",
        type=Path,
        help=(
            "Path to the upstream repository's challenges directory. "
            "Defaults to LEETGPU_CHALLENGES_DIR or upstream/leetgpu-challenges/challenges."
        ),
    )
    parser.add_argument("--ncu-launch-count", type=int, default=20)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def resolve_challenges_dir(root: Path, configured: Optional[Path]) -> Path:
    value = configured
    if value is None:
        environment_value = os.getenv("LEETGPU_CHALLENGES_DIR")
        value = Path(environment_value) if environment_value else None
    candidate = value or (root / "upstream" / "leetgpu-challenges" / "challenges")
    candidate = candidate.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if (candidate / "challenges").is_dir():
        candidate = (candidate / "challenges").resolve()
    return candidate


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    root = Path(__file__).resolve().parent.parent
    challenges_dir = resolve_challenges_dir(root, args.challenges_dir)
    try:
        app = WebApplication(
            root,
            (root / args.data_dir).resolve(),
            args.ncu_launch_count,
            challenges_dir=challenges_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    server = ApplicationServer((args.host, args.port), RequestHandler, app)
    url = f"http://{args.host}:{args.port}"
    LOGGER.info("LeetGPU Local is running at %s", url)
    LOGGER.info("Challenges: %s", challenges_dir)
    LOGGER.info("Python: %s", sys.executable)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping local server")
    finally:
        app.jobs.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
