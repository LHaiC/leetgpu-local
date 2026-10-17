#!/usr/bin/env python3
"""Local web UI, submission queue, and per-challenge leaderboard."""

from __future__ import annotations

import argparse
import ast
import hashlib
import gzip
import hmac
import json
import logging
import mimetypes
import os
import queue
import re
import secrets
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
from urllib.parse import parse_qs, unquote, urlsplit

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
JOB_TIMEOUT_SAFETY = 30
LOGIN_ATTEMPT_LIMIT = 6
LOGIN_ATTEMPT_WINDOW = 60.0
LOGIN_LOCK_SECONDS = 300
MAX_QUEUED_PER_USER = 8


class ApiError(Exception):
    """An error with a stable code so the UI can render it in any language."""

    def __init__(self, message: str, status: int = 400, code: str = ""):
        super().__init__(message)
        self.status = status
        self.code = code


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
                        username TEXT NOT NULL DEFAULT '',
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
                        username TEXT NOT NULL DEFAULT '',
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
                        job_id, username, challenge_key, runner_name, language, status,
                        result_status, created_at, finished_at
                    )
                    SELECT job_id, '', challenge_key, runner_name, language, 'completed',
                           'passed', created_at, created_at
                    FROM scores
                    """
                )
                columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(scores)").fetchall()
                }
                if "username" not in columns:
                    connection.execute(
                        "ALTER TABLE scores ADD COLUMN username TEXT NOT NULL DEFAULT ''"
                    )
                if "source_code" not in columns:
                    connection.execute(
                        "ALTER TABLE scores ADD COLUMN source_code TEXT NOT NULL DEFAULT ''"
                    )
                if "solution_filename" not in columns:
                    connection.execute(
                        "ALTER TABLE scores ADD COLUMN solution_filename TEXT NOT NULL DEFAULT ''"
                    )
                attempt_columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(submission_attempts)"
                    ).fetchall()
                }
                if "username" not in attempt_columns:
                    connection.execute(
                        "ALTER TABLE submission_attempts "
                        "ADD COLUMN username TEXT NOT NULL DEFAULT ''"
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
                        job_id, username, challenge_key, challenge_name, runner_name,
                        language, gpu_name, median_ms, min_ms, p95_ms, source_hash,
                        source_code, solution_filename, created_at, result_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job["id"],
                        job.get("username") or job.get("runner_name") or "local",
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
                    PARTITION BY username, language
                    ORDER BY median_ms ASC, created_at ASC
                ) AS personal_rank
                FROM scores
                WHERE {where}
            )
            SELECT job_id, username, runner_name, language, gpu_name, median_ms, min_ms,
                   p95_ms, source_hash, created_at
            FROM personal_best
            WHERE personal_rank = 1
            ORDER BY median_ms ASC, created_at ASC
            LIMIT ?
        """
        with closing(self._connect()) as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(rank=index, **dict(row)) for index, row in enumerate(rows, 1)]

    def overall_leaderboard(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Cross-challenge ranking per user.

        `avg_percentile` is 1.0 when a user is first on every challenge they
        entered and 0.0 when last, which makes challenges of different sizes
        comparable. Single-entrant challenges are excluded from both averages
        (they carry no ranking information).
        """
        query = """
            WITH personal_best AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY challenge_key, username, language
                    ORDER BY median_ms ASC, created_at ASC
                ) AS personal_rank
                FROM scores
            ),
            ranked AS (
                SELECT
                    username,
                    challenge_key,
                    language,
                    RANK() OVER (
                        PARTITION BY challenge_key, language ORDER BY median_ms ASC
                    ) AS place,
                    COUNT(*) OVER (PARTITION BY challenge_key, language) AS field
                FROM personal_best
                WHERE personal_rank = 1
            )
            SELECT
                username,
                COUNT(DISTINCT challenge_key) AS challenges,
                COUNT(*) AS entries,
                ROUND(AVG(CASE WHEN field > 1 THEN place END), 2) AS avg_place,
                ROUND(AVG(CASE WHEN field > 1
                               THEN (field - place) * 1.0 / (field - 1) END), 4)
                    AS avg_percentile,
                SUM(CASE WHEN place = 1 THEN 1 ELSE 0 END) AS wins,
                MIN(place) AS best_place
            FROM ranked
            WHERE username != ''
            GROUP BY username
            ORDER BY avg_percentile DESC NULLS LAST, challenges DESC, wins DESC
            LIMIT ?
        """
        with closing(self._connect()) as connection:
            rows = connection.execute(query, (limit,)).fetchall()
        return [dict(rank=index, **dict(row)) for index, row in enumerate(rows, 1)]

    def user_profile(self, username: str) -> Dict[str, Any]:
        """Aggregate stats plus per-challenge bests for one user."""
        bests_query = """
            WITH personal_best AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY challenge_key, username, language
                    ORDER BY median_ms ASC, created_at ASC
                ) AS personal_rank
                FROM scores
                WHERE username = ?
            ),
            ranked AS (
                SELECT
                    challenge_key,
                    challenge_name,
                    language,
                    median_ms,
                    min_ms,
                    created_at,
                    RANK() OVER (
                        PARTITION BY challenge_key, language ORDER BY median_ms ASC
                    ) AS place,
                    COUNT(*) OVER (PARTITION BY challenge_key, language) AS field
                FROM personal_best
                WHERE personal_rank = 1
            )
            SELECT * FROM ranked
            ORDER BY challenge_key, language
        """
        attempts_query = """
            SELECT COUNT(*) AS submissions,
                   SUM(CASE WHEN result_status = 'passed' THEN 1 ELSE 0 END) AS passed
            FROM submission_attempts
            WHERE username = ?
        """
        with closing(self._connect()) as connection:
            bests = [dict(row) for row in connection.execute(bests_query, (username,))]
            row = connection.execute(attempts_query, (username,)).fetchone()
        summary = next(
            (entry for entry in self.overall_leaderboard(1000) if entry["username"] == username),
            None,
        )
        return {
            "username": username,
            "summary": summary,
            "submissions": int(row["submissions"] or 0) if row else 0,
            "passed": int(row["passed"] or 0) if row else 0,
            "bests": bests,
        }

    def submission(self, job_id: str) -> Dict[str, Any]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT job_id, username, challenge_key, challenge_name, runner_name,
                       language, gpu_name, median_ms, min_ms, p95_ms, source_hash,
                       source_code, solution_filename, created_at
                FROM scores WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return dict(row)

    def delete_user_data(self, username: str) -> None:
        """Drop a deleted account's leaderboard rows and submission history."""
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("DELETE FROM scores WHERE username = ?", (username,))
                connection.execute(
                    "DELETE FROM submission_attempts WHERE username = ?", (username,)
                )

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
                        job_id, username, challenge_key, runner_name, language, status,
                        result_status, created_at, finished_at
                    ) VALUES (?, ?, ?, ?, ?, 'queued', NULL, ?, NULL)
                    """,
                    (
                        job["id"],
                        job.get("username") or job.get("runner_name") or "local",
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

    def challenge_progress(
        self, username: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        where = "WHERE username = ?" if username else ""
        params = (username,) if username else ()
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""
                SELECT challenge_key, COUNT(*) AS submission_count,
                       MAX(CASE WHEN result_status = 'passed' THEN 1 ELSE 0 END) AS passed
                FROM submission_attempts
                {where}
                GROUP BY challenge_key
                """,
                params,
            ).fetchall()
        return {
            row["challenge_key"]: {
                "submission_count": int(row["submission_count"]),
                "passed": bool(row["passed"]),
            }
            for row in rows
        }




def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=16384, r=8, p=1)
    return "scrypt$" + salt.hex() + "$" + digest.hex()


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_hex, digest_hex = stored.split("$", 2)
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode("utf-8"), salt=bytes.fromhex(salt_hex), n=16384, r=8, p=1
        )
        return hmac.compare_digest(digest, bytes.fromhex(digest_hex))
    except (ValueError, TypeError, OSError):
        return False


SESSION_COOKIE = "lg_session"
SESSION_TTL_SECONDS = 14 * 24 * 3600
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{2,32}$")


class MultiuserStore:
    """Users, sessions, announcements, site settings, per-user solutions."""

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
                    CREATE TABLE IF NOT EXISTS users (
                        username TEXT PRIMARY KEY,
                        password_hash TEXT NOT NULL,
                        role TEXT NOT NULL DEFAULT 'user',
                        disabled INTEGER NOT NULL DEFAULT 0,
                        approved INTEGER NOT NULL DEFAULT 1,
                        deleted INTEGER NOT NULL DEFAULT 0,
                        deleted_at TEXT,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(users)").fetchall()
                }
                if "approved" not in columns:
                    connection.execute(
                        "ALTER TABLE users ADD COLUMN approved INTEGER NOT NULL DEFAULT 1"
                    )
                if "deleted" not in columns:
                    connection.execute(
                        "ALTER TABLE users ADD COLUMN deleted INTEGER NOT NULL DEFAULT 0"
                    )
                if "deleted_at" not in columns:
                    connection.execute("ALTER TABLE users ADD COLUMN deleted_at TEXT")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        token_hash TEXT PRIMARY KEY,
                        username TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS sessions_expiry ON sessions(expires_at)"
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS announcements (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        body TEXT NOT NULL,
                        author TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS site_settings (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_solutions (
                        username TEXT NOT NULL,
                        challenge_key TEXT NOT NULL,
                        language TEXT NOT NULL,
                        source_code TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (username, challenge_key, language)
                    )
                    """
                )

    # ---- users ----

    def user_count(self) -> int:
        with closing(self._connect()) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def create_user(
        self, username: str, password: str, role: str = "user", approved: bool = True
    ) -> Dict[str, Any]:
        if not USERNAME_PATTERN.match(username):
            raise ApiError("Username must be 2-32 characters: letters, digits, _ . -", code="username_format")
        if len(password) < 6:
            raise ApiError("Password must be at least 6 characters", code="password_short")
        existing = self.get_user(username)
        if existing is not None:
            if existing["deleted"]:
                raise ApiError(
                    "This name belonged to a deleted account and stays reserved",
                    code="username_retired",
                )
            raise ApiError("Username already taken", code="username_taken")
        with closing(self._connect()) as connection:
            with connection:
                try:
                    connection.execute(
                        "INSERT INTO users (username, password_hash, role, approved, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            username,
                            hash_password(password),
                            role,
                            1 if (role == "admin" or approved) else 0,
                            utc_now(),
                        ),
                    )
                except sqlite3.IntegrityError:
                    raise ApiError("Username already taken", code="username_taken")
        return {"username": username, "role": role}

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT username, role, disabled, approved, deleted, deleted_at, created_at "
                "FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        return dict(row) if row else None

    def verify_login(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT username, password_hash, role, disabled, approved, deleted "
                "FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        if row is None or row["disabled"] or row["deleted"]:
            return None
        if not verify_password(password, row["password_hash"]):
            return None
        if not row["approved"]:
            return {"username": row["username"], "role": row["role"], "pending": True}
        return {"username": row["username"], "role": row["role"]}

    def list_users(self, pending: Optional[bool] = None) -> List[Dict[str, Any]]:
        query = (
            "SELECT username, role, disabled, approved, deleted, deleted_at, created_at "
            "FROM users"
        )
        params: tuple = ()
        if pending is not None:
            query += " WHERE approved = ? AND deleted = 0"
            params = (0 if pending else 1,)
        query += " ORDER BY created_at"
        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def set_approved(self, username: str, approved: bool) -> bool:
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    "UPDATE users SET approved = ? WHERE username = ?",
                    (int(approved), username),
                )
                if cursor.rowcount and not approved:
                    connection.execute(
                        "DELETE FROM sessions WHERE username = ?", (username,)
                    )
                return cursor.rowcount > 0

    def set_role(self, username: str, role: str) -> bool:
        if role not in {"user", "admin"}:
            raise ValueError("role must be user or admin")
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    "UPDATE users SET role = ? WHERE username = ?", (role, username)
                )
                return cursor.rowcount > 0

    def admin_reset_password(self, username: str, new_password: str) -> bool:
        if len(new_password) < 6:
            raise ApiError("Password must be at least 6 characters", code="password_short")
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    "UPDATE users SET password_hash = ? WHERE username = ?",
                    (hash_password(new_password), username),
                )
                if cursor.rowcount:
                    connection.execute(
                        "DELETE FROM sessions WHERE username = ?", (username,)
                    )
                return cursor.rowcount > 0

    def set_disabled(self, username: str, disabled: bool) -> bool:
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    "UPDATE users SET disabled = ? WHERE username = ?",
                    (int(disabled), username),
                )
                if cursor.rowcount:
                    connection.execute(
                        "DELETE FROM sessions WHERE username = ?", (username,)
                    )
                return cursor.rowcount > 0

    def admin_count(self) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM users "
                "WHERE role = 'admin' AND disabled = 0 AND deleted = 0"
            ).fetchone()
        return int(row[0])

    def delete_user(self, username: str) -> bool:
        """Delete an account and its data, keeping the name permanently retired.

        Everything personal goes: credentials, sessions and saved drafts. The
        leaderboard rows are removed by ScoreStore.delete_user_data() (they live
        in a different database), so WebApplication.delete_user() orchestrates
        both. Only a tombstone row survives, so the name can never be registered
        again — that prevents a later account from impersonating a former member
        or inheriting their standing. The last active admin cannot be deleted:
        that would lock everyone out of administration.
        """
        user = self.get_user(username)
        if user is None or user["deleted"]:
            return False
        if user["role"] == "admin" and not user["disabled"] and self.admin_count() <= 1:
            raise ApiError(
                "The last admin cannot be deleted",
                status=400,
                code="last_admin",
            )
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE users
                       SET deleted = 1, deleted_at = ?, disabled = 1, approved = 0,
                           role = 'user', password_hash = ?
                     WHERE username = ?
                    """,
                    (utc_now(), hash_password(secrets.token_urlsafe(32)), username),
                )
                connection.execute(
                    "DELETE FROM sessions WHERE username = ?", (username,)
                )
                connection.execute(
                    "DELETE FROM user_solutions WHERE username = ?", (username,)
                )
        return True

    def purge_user(self, username: str) -> bool:
        """Hard-delete an account with no history (a rejected registration).

        Unlike delete_user this frees the name, which is safe because a pending
        account could never submit anything.
        """
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM users WHERE username = ?", (username,)
                )
                connection.execute(
                    "DELETE FROM sessions WHERE username = ?", (username,)
                )
                connection.execute(
                    "DELETE FROM user_solutions WHERE username = ?", (username,)
                )
                return cursor.rowcount > 0

    def change_password(self, username: str, old: str, new: str) -> bool:
        user = self.verify_login(username, old)
        if user is None:
            raise ApiError("Current password is incorrect", code="old_password_wrong")
        if len(new) < 6:
            raise ApiError("New password must be at least 6 characters", code="password_short")
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    "UPDATE users SET password_hash = ? WHERE username = ?",
                    (hash_password(new), username),
                )
        return True

    # ---- sessions ----

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create_session(self, username: str) -> str:
        token = secrets.token_urlsafe(32)
        expires = datetime.now(timezone.utc).timestamp() + SESSION_TTL_SECONDS
        expires_iso = datetime.fromtimestamp(expires, timezone.utc).isoformat()
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    "DELETE FROM sessions WHERE expires_at < ?", (utc_now(),)
                )
                connection.execute(
                    "INSERT INTO sessions (token_hash, username, created_at, expires_at) "
                    "VALUES (?, ?, ?, ?)",
                    (self._token_hash(token), username, utc_now(), expires_iso),
                )
        return token

    def session_user(self, token: str) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT u.username, u.role, u.disabled, u.deleted FROM sessions s
                JOIN users u ON u.username = s.username
                WHERE s.token_hash = ? AND s.expires_at > ?
                """,
                (self._token_hash(token), utc_now()),
            ).fetchone()
        if row is None or row["disabled"] or row["deleted"]:
            return None
        return {"username": row["username"], "role": row["role"]}

    def destroy_session(self, token: str) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    "DELETE FROM sessions WHERE token_hash = ?",
                    (self._token_hash(token),),
                )

    # ---- announcements ----

    def announcements(self, limit: int = 20) -> List[Dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT id, title, body, author, created_at FROM announcements "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_announcement(self, title: str, body: str, author: str) -> Dict[str, Any]:
        title, body = title.strip(), body.strip()
        if not title:
            raise ApiError("Announcement title is required", code="title_required")
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    "INSERT INTO announcements (title, body, author, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (title[:120], body[:4000], author, utc_now()),
                )
        return {"id": cursor.lastrowid, "title": title, "body": body, "author": author}

    def delete_announcement(self, announcement_id: int) -> bool:
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM announcements WHERE id = ?", (announcement_id,)
                )
                return cursor.rowcount > 0

    # ---- site settings ----

    def site_setting(self, key: str, default: str = "") -> str:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT value FROM site_settings WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else default

    def set_site_setting(self, key: str, value: str) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    "INSERT INTO site_settings (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, value),
                )

    # ---- user solutions ----

    def save_solution(
        self, username: str, challenge_key: str, language: str, source: str
    ) -> None:
        if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
            raise ValueError("Source code exceeds 2 MiB")
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO user_solutions (
                        username, challenge_key, language, source_code, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(username, challenge_key, language)
                    DO UPDATE SET source_code = excluded.source_code,
                                  updated_at = excluded.updated_at
                    """,
                    (username, challenge_key, language, source, utc_now()),
                )

    def delete_solution(self, username: str, challenge_key: str, language: str) -> bool:
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM user_solutions "
                    "WHERE username = ? AND challenge_key = ? AND language = ?",
                    (username, challenge_key, language),
                )
                return cursor.rowcount > 0

    def get_solution(
        self, username: str, challenge_key: str, language: str
    ) -> Optional[str]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT source_code FROM user_solutions "
                "WHERE username = ? AND challenge_key = ? AND language = ?",
                (username, challenge_key, language),
            ).fetchone()
        return row["source_code"] if row else None

    def list_solutions(self, username: str) -> List[Dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT challenge_key, language, updated_at FROM user_solutions "
                "WHERE username = ?",
                (username,),
            ).fetchall()
        return [dict(row) for row in rows]

    def solution_updated_at(
        self, username: str, challenge_key: str, language: str
    ) -> Optional[str]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT updated_at FROM user_solutions "
                "WHERE username = ? AND challenge_key = ? AND language = ?",
                (username, challenge_key, language),
            ).fetchone()
        return row["updated_at"] if row else None

    def solution_owners(self, challenge_key: str, language: str) -> List[Dict[str, Any]]:
        """Who has a saved draft for this challenge/language, newest first."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT username, updated_at, LENGTH(source_code) AS size
                FROM user_solutions
                WHERE challenge_key = ? AND language = ?
                ORDER BY updated_at DESC
                """,
                (challenge_key, language),
            ).fetchall()
        return [dict(row) for row in rows]


class JobManager:
    def __init__(
        self,
        root: Path,
        data_dir: Path,
        catalog: ChallengeCatalog,
        scores: ScoreStore,
        sandbox: bool = False,
        judge_image: str = "leetgpu-judge",
        sandbox_timeout: int = 600,
    ):
        self.root = root.resolve()
        self.data_dir = data_dir.resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.catalog = catalog
        self.scores = scores
        self.sandbox = sandbox
        self.judge_image = judge_image
        self.sandbox_timeout = sandbox_timeout
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
        mode: str,
        username: str = "",
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
            raise ApiError("Source code exceeds 2 MiB", code="source_too_large")
        if username:
            with self.lock:
                queued = sum(
                    1
                    for j in self.jobs.values()
                    if j.get("username") == username
                    and j.get("status") in {"queued", "running"}
                )
            if queued >= MAX_QUEUED_PER_USER:
                raise ApiError(
                    f"At most {MAX_QUEUED_PER_USER} queued jobs per user; wait for the queue to drain",
                    code="queue_full",
                )

        job_id = uuid.uuid4().hex[:12]
        challenge_dir_name = filesystem_name(
            f"{challenge_info['id']}_{challenge_info['title']}", challenge_info["slug"]
        )
        submission_dir_name = filesystem_name(runner_name, "local")
        job_dir = (
            self.data_dir / "runs" / challenge_dir_name / submission_dir_name / job_id
        )
        pending_dir = self.data_dir / ".pending" / job_id
        job = {
            "id": job_id,
            "username": username,
            "challenge": challenge,
            "language": language,
            "runner_name": runner_name,
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
            job.pop("result_path", None)
            job["output"] = self.scrub(job["output"][-50_000:])
            if job.get("error"):
                job["error"] = self.scrub(job["error"])
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
        }
        filename = patterns.get(kind)
        if not filename:
            return None
        matches = list(job_dir.rglob(filename))
        return matches[-1] if matches else None

    def _append_output(self, job_id: str, line: str) -> None:
        line = self.scrub(line)
        with self.lock:
            job = self.jobs[job_id]
            job["output"] = (job["output"] + line)[-100_000:]

    def scrub(self, text: str) -> str:
        """Strip host-specific paths so the UI never leaks local layout."""
        if not text:
            return text
        for host, placeholder in (
            (str(self.root), "/app"),
            (str(self.data_dir), "/data"),
        ):
            text = text.replace(host, placeholder)
        return text

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
        # Each job owns a fresh directory, so no pre-run cleanup is needed
        # and concurrent submissions can never read each other's output.
        job_dir.mkdir(parents=True, exist_ok=False)
        if self.sandbox:
            # The container runs as an unprivileged uid; /work must be writable.
            job_dir.chmod(0o777)
        solution = job_dir / pending_solution.name
        shutil.copy2(pending_solution, solution)
        os.chmod(solution, 0o644)
        with self.lock:
            job["solution_path"] = str(solution)
        if self.sandbox:
            command = self._sandbox_command(
                challenge, solution, job["language"], job["mode"],
                job_id=job_id,
            )
        else:
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
        timed_out = False
        timer = None
        if self.sandbox and self.sandbox_timeout > 0:
            def _kill():
                nonlocal timed_out
                timed_out = True
                # Killing the docker CLI client leaves the container alive;
                # the named container must be removed explicitly.
                subprocess.run(
                    ["docker", "rm", "-f", f"lgj-{job_id}"],
                    capture_output=True,
                    check=False,
                )
                process.kill()
            timer = threading.Timer(self.sandbox_timeout, _kill)
            timer.start()
        with web_log.open("w", encoding="utf-8", buffering=1) as output_file:
            assert process.stdout is not None
            for line in process.stdout:
                output_file.write(line)
                self._append_output(job_id, line)
        exit_code = process.wait()
        if timer is not None:
            timer.cancel()
        with self.lock:
            self.active_process = None

        results = sorted(job_dir.rglob("result.json"), key=lambda path: path.stat().st_mtime)
        result = None
        if results:
            result_path = results[-1]
            result = json.loads(result_path.read_text(encoding="utf-8"))
            job["result_path"] = str(result_path)
        status = "completed" if result is not None else "error"
        if timed_out:
            error = f"Sandbox killed the submission after {self.sandbox_timeout}s"
        else:
            error = (
                None
                if result is not None
                else f"Runner exited with {exit_code} without result.json"
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


    def _sandbox_command(
        self,
        challenge: Mapping[str, Any],
        solution: Path,
        language: str,
        mode: str,
        job_id: str,
    ) -> List[str]:
        challenges_dir = self.catalog.challenges_dir
        challenge_rel = Path(challenge["path"]).relative_to(challenges_dir)
        inner = [
            "sh",
            "-c",
            'python3 "$@" ; rc=$? ; chmod -R a+rwX /work 2>/dev/null ; exit $rc',
            "entry",
            "/judge/repo/scripts/local_runner.py",
            f"/judge/challenges/{challenge_rel.as_posix()}",
            "--language",
            language,
            "--solution",
            f"/work/{solution.name}",
            "--run-dir",
            "/work",
            "--tests",
            "correctness" if mode == "test" else "all",
        ]
        # uid 65534 = nobody: 逃逸容器后无 docker socket 权限、不落在宿主用户身上。
        return [
            "docker", "run", "--rm", "-i",
            "--name", f"lgj-{job_id}",
            "--network", "none",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--read-only",
            # Docker's tmpfs default is noexec/nodev, and just omitting noexec
            # is not enough — it must be overridden with an explicit `exec`.
            # Triton compiles a CUDA driver shim into its cache here and then
            # dlopen()s it; without `exec` those challenges die with
            # "failed to map segment from shared object".
            "--tmpfs", "/tmp:rw,nosuid,exec,size=2g",
            "--ipc", "none",
            "--pids-limit", "256",
            "--memory", "8g",
            "--cpus", "4",
            "--gpus", "all",
            "--user", "65534:65534",
            "--workdir", "/work",
            "-e", "HOME=/tmp",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-e", "TRITON_CACHE_DIR=/tmp/.triton-cache",
            "-e", "TORCHINDUCTOR_CACHE_DIR=/tmp/.inductor-cache",
            "-v", f"{self.root}:/judge/repo:ro",
            "-v", f"{challenges_dir}:/judge/challenges:ro",
            "-v", f"{solution.parent}:/work",
            self.judge_image,
            *inner,
        ]


def asset_version(web_root: Path) -> str:
    """Fingerprint of everything served from /static.

    The HTML references assets as `/static/x.js?v=<fingerprint>`; a request whose
    version matches may be cached immutably (by the browser and by Cloudflare),
    which keeps the ~20 assets off the origin on every load. A deploy changes the
    fingerprint, so the URLs change and nothing goes stale.
    """
    digest = hashlib.sha256()
    for path in sorted(web_root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        digest.update(path.relative_to(web_root).as_posix().encode())
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:12]


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
    }


_GPU_CACHE: Dict[str, Any] = {"at": 0.0, "value": None}
_GPU_CACHE_LOCK = threading.Lock()
_GPU_CACHE_TTL = 0.8


def gpu_stats() -> Dict[str, Any]:
    """Live GPU utilization + VRAM for the topbar widget.

    Cached for a fraction of a second so N open tabs share a single
    `nvidia-smi` invocation instead of spawning one process each poll.
    """
    now = time.monotonic()
    with _GPU_CACHE_LOCK:
        cached = _GPU_CACHE["value"]
        if cached is not None and now - _GPU_CACHE["at"] < _GPU_CACHE_TTL:
            return cached
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        value: Dict[str, Any] = {"available": False}
    else:
        util, used, total = [v.strip() for v in result.stdout.splitlines()[0].split(",")]
        value = {
            "available": True,
            "util_pct": int(float(util)),
            "mem_used_mb": int(float(used)),
            "mem_total_mb": int(float(total)),
            "mem_pct": round(100 * float(used) / float(total), 1),
        }
    with _GPU_CACHE_LOCK:
        _GPU_CACHE["at"] = now
        _GPU_CACHE["value"] = value
    return value


class WebApplication:
    def __init__(
        self,
        root: Path,
        data_dir: Path,
        challenges_dir: Path,
        sandbox: bool = False,
        judge_image: str = "leetgpu-judge",
        sandbox_timeout: int = 600,
        auth_required: bool = False,
        site_name: str = "LeetGPU Local",
    ):
        self.root = root.resolve()
        self.web_root = self.root / "web"
        self.catalog = ChallengeCatalog(challenges_dir)
        self.scores = ScoreStore(data_dir / "leaderboard.sqlite3")
        self.users = MultiuserStore(data_dir / "users.sqlite3")
        self.sandbox = sandbox
        self.auth_required = auth_required or sandbox
        self.site_name = site_name
        self.jobs = JobManager(
            self.root, data_dir, self.catalog, self.scores,
            sandbox=sandbox, judge_image=judge_image, sandbox_timeout=sandbox_timeout,
        )
        self.system = system_info()
        self.asset_version = asset_version(self.web_root)

    def registration_mode(self) -> str:
        if self.users.user_count() == 0:
            return "open"
        return self.users.site_setting("registration_mode", "open")

    def share_solutions(self) -> bool:
        """Whether members may read each other's saved drafts. Off by default:
        a draft is unfinished work, not a published solution."""
        return self.users.site_setting("share_solutions", "0") == "1"

    def can_read_solution(self, viewer: Mapping[str, Any], owner: str) -> bool:
        if viewer.get("username") == owner:
            return True
        if viewer.get("role") == "admin":
            return True
        return self.share_solutions()

    def delete_user(self, username: str) -> bool:
        """Delete an account and all of its data, across both databases."""
        if not self.users.delete_user(username):
            return False
        self.scores.delete_user_data(username)
        return True

    def registration_open(self) -> bool:
        return self.registration_mode() == "open"

    def challenge_list(self, username: str = "") -> List[Dict[str, Any]]:
        progress = self.scores.challenge_progress(username or None)
        challenges = self.catalog.list_public()
        for challenge in challenges:
            challenge.update(
                progress.get(
                    challenge["key"],
                    {"submission_count": 0, "passed": False},
                )
            )
        return challenges



class RequestHandler(BaseHTTPRequestHandler):
    server_version = "LeetGPULocal/1.0"
    # HTTP/1.0 (the http.server default) closes the socket after every
    # response, so each of the ~20 assets needs its own TCP handshake. Every
    # response here sets Content-Length, so HTTP/1.1 keep-alive is safe.
    protocol_version = "HTTP/1.1"

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

    def send_error_json(self, status: int, message: str, code: Optional[str] = None) -> None:
        payload: Dict[str, Any] = {"error": message}
        if code:
            payload["code"] = code
        self.send_json(payload, status=status)

    def _accepts_gzip(self) -> bool:
        return "gzip" in self.headers.get("Accept-Encoding", "").lower()

    def send_body(
        self,
        body: bytes,
        content_type: str,
        cache_control: Optional[str] = None,
        etag: Optional[str] = None,
        attachment_name: Optional[str] = None,
    ) -> None:
        """Single funnel for byte responses: ETag/304, gzip and length headers."""
        if etag and self.headers.get("If-None-Match") == etag:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self._security_headers()
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", cache_control or "no-cache")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        compressible = content_type.startswith(
            ("text/", "application/javascript", "application/json", "image/svg+xml")
        )
        compressed = False
        if compressible and len(body) > 512 and self._accepts_gzip():
            body = gzip.compress(body, 6)
            compressed = True

        self.send_response(HTTPStatus.OK)
        self._security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", cache_control or "no-store")
        if etag:
            self.send_header("ETag", etag)
        if compressed:
            self.send_header("Content-Encoding", "gzip")
        if compressible:
            self.send_header("Vary", "Accept-Encoding")
        if attachment_name:
            self.send_header(
                "Content-Disposition", f'attachment; filename="{attachment_name}"'
            )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(
        self,
        path: Path,
        attachment: bool = False,
        cache_control: Optional[str] = None,
    ) -> None:
        if not path.is_file():
            self.send_error_json(HTTPStatus.NOT_FOUND, "File not found")
            return
        stat = path.stat()
        # Always revalidate: a 304 costs one small round-trip, while a stale
        # max-age would keep serving old JS/CSS after an update.
        etag = f'W/"{stat.st_size:x}-{stat.st_mtime_ns:x}"'
        self.send_body(
            path.read_bytes(),
            mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            cache_control=cache_control or "no-cache",
            etag=etag,
            attachment_name=path.name if attachment else None,
        )

    def _content_length(self) -> int:
        try:
            return max(0, int(self.headers.get("Content-Length") or 0))
        except (TypeError, ValueError):
            return 0

    def _drain_body(self) -> None:
        """Consume any request body the handler did not read.

        With HTTP/1.1 keep-alive the next request is parsed from the same socket,
        so leftover bytes would be read as a request line — which shows up as
        `501 Unsupported method ('{}GET')` after a POST whose body was ignored
        (logout sends `{}` and never reads it).
        """
        remaining = self._content_length() - getattr(self, "_body_consumed", 0)
        if remaining > 0:
            try:
                self.rfile.read(remaining)
            except OSError:
                pass
        self._body_consumed = self._content_length()

    def read_json(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length <= 0 or length > MAX_SOURCE_BYTES + 64 * 1024:
            raise ValueError("Request body is empty or too large")
        raw = self.rfile.read(length)
        self._body_consumed = getattr(self, "_body_consumed", 0) + len(raw)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Request body must be UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("Request JSON must be an object")
        return value

    # ---- auth ----

    def _client_ip(self) -> str:
        # Behind a reverse proxy the real client is the first XFF hop.
        forwarded = self.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return self.client_address[0]

    def _login_throttled(self) -> bool:
        ip = self._client_ip()
        with self.server.login_lock:
            attempts = self.server.login_attempts.setdefault(ip, [])
            now = time.time()
            attempts[:] = [t for t in attempts if now - t < LOGIN_ATTEMPT_WINDOW]
            if len(attempts) >= LOGIN_ATTEMPT_LIMIT:
                return True
            attempts.append(now)
            return False

    def _session_token(self) -> str:
        cookie = self.headers.get("Cookie", "")
        for part in cookie.split(";"):
            name, _, value = part.strip().partition("=")
            if name == SESSION_COOKIE:
                return value
        return ""

    def current_user(self) -> Optional[Dict[str, Any]]:
        return self.app.users.session_user(self._session_token())

    def require_user(self) -> Dict[str, Any]:
        if not self.app.auth_required:
            return {"username": "", "role": "admin"}
        user = self.current_user()
        if user is None:
            raise ApiError("Login required", status=401, code="login_required")
        return user

    def require_admin(self) -> Dict[str, Any]:
        user = self.require_user()
        if user["role"] != "admin":
            raise ApiError("Admin permission required", status=403, code="admin_required")
        return user

    def set_session_cookie(self, token: str, max_age: int = SESSION_TTL_SECONDS) -> None:
        attrs = [f"{SESSION_COOKIE}={token}", "Path=/", "HttpOnly", "SameSite=Lax"]
        if max_age <= 0:
            attrs.append("Max-Age=0")
        else:
            attrs.append(f"Max-Age={max_age}")
        if self.server.secure_cookies:
            attrs.append("Secure")
        self.send_header("Set-Cookie", "; ".join(attrs))

    def send_json(self, value: Any, status: int = HTTPStatus.OK, cookie: Optional[str] = None) -> None:
        body = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        if cookie is not None:
            self.set_session_cookie(cookie)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        self._body_consumed = 0
        try:
            if parsed.path == "/api/me":
                user = self.current_user()
                self.send_json({
                    "user": user,
                    "auth_required": self.app.auth_required,
                    "registration_open": self.app.registration_open(),
                    "registration_mode": self.app.registration_mode(),
                    "maintenance": self.app.users.site_setting("maintenance") == "1",
                    "share_solutions": self.app.share_solutions(),
                    "site_name": self.app.site_name,
                })
            elif parsed.path == "/api/announcements":
                self.send_json({"announcements": self.app.users.announcements()})
            elif parsed.path == "/api/challenges":
                user = self.current_user() or {}
                self.send_json(
                    {"challenges": self.app.challenge_list(user.get("username", ""))}
                )
            elif parsed.path == "/api/challenge":
                self.send_json(self.app.catalog.detail(query.get("key", [""])[0]))
            elif parsed.path == "/api/system":
                self.send_json(self.app.system)
            elif parsed.path == "/api/system/gpu":
                self.send_json(gpu_stats())
            elif parsed.path == "/api/leaderboard":
                challenge = query.get("challenge", [""])[0]
                self.app.catalog.require(challenge)
                language = query.get("language", [""])[0] or None
                if language and language not in LANGUAGES:
                    raise ValueError("Invalid leaderboard language")
                self.send_json(
                    {"scores": self.app.scores.leaderboard(challenge, language=language)}
                )
            elif parsed.path == "/api/leaderboard/overall":
                self.send_json({"scores": self.app.scores.overall_leaderboard()})
            elif parsed.path == "/api/profile":
                user = self.require_user()
                target = query.get("username", [""])[0] or user["username"]
                if target != user["username"] and user["role"] != "admin":
                    raise ApiError(
                        "You can only view your own profile",
                        status=403,
                        code="own_profile_only",
                    )
                if self.app.users.get_user(target) is None:
                    raise ApiError("No such user", status=404, code="unknown_user")
                profile = self.app.scores.user_profile(target)
                profile["account"] = self.app.users.get_user(target)
                self.send_json(profile)
            elif parsed.path == "/api/leaderboard/source":
                user = self.require_user()
                submission = self.app.scores.submission(query.get("job", [""])[0])
                if (
                    user["role"] != "admin"
                    and submission.get("username", "") != user["username"]
                ):
                    raise ApiError("You can only view your own source", status=403, code="own_source_only")
                self.send_json(submission)
            elif parsed.path == "/api/solutions":
                user = self.require_user()
                self.send_json(
                    {"solutions": self.app.users.list_solutions(user["username"])}
                )
            elif parsed.path == "/api/solution":
                user = self.require_user()
                challenge = query.get("challenge", [""])[0]
                language = query.get("language", [""])[0]
                owner = query.get("username", [""])[0] or user["username"]
                if not self.app.can_read_solution(user, owner):
                    raise ApiError(
                        "This user's saved solution is not shared",
                        status=403,
                        code="solution_not_shared",
                    )
                if owner != user["username"] and self.app.users.get_user(owner) is None:
                    raise ApiError("No such user", status=404, code="unknown_user")
                self.send_json({
                    "source": self.app.users.get_solution(owner, challenge, language),
                    "username": owner,
                    "updated_at": self.app.users.solution_updated_at(owner, challenge, language),
                })
            elif parsed.path == "/api/solution/owners":
                user = self.require_user()
                challenge = query.get("challenge", [""])[0]
                language = query.get("language", [""])[0]
                self.app.catalog.require(challenge)
                if language not in LANGUAGES:
                    raise ValueError("Invalid language")
                owners = self.app.users.solution_owners(challenge, language)
                # Always surface the viewer's own row; hide the rest unless shared.
                if not self.app.share_solutions() and user["role"] != "admin":
                    owners = [row for row in owners if row["username"] == user["username"]]
                self.send_json({
                    "owners": owners,
                    "shared": self.app.share_solutions() or user["role"] == "admin",
                })
            elif parsed.path == "/api/admin/users":
                self.require_admin()
                self.send_json({"users": self.app.users.list_users()})
            elif parsed.path == "/api/admin/pending":
                self.require_admin()
                self.send_json(
                    {"users": self.app.users.list_users(pending=True)}
                )
            elif parsed.path.startswith("/api/jobs/"):
                user = self.require_user()
                parts = parsed.path.strip("/").split("/")
                job_id = parts[2] if len(parts) >= 3 else ""
                job = self.app.jobs.public(job_id)
                if user["role"] != "admin" and job.get("username", "") != user["username"]:
                    raise ApiError("You can only view your own job", status=403, code="own_job_only")
                if len(parts) == 3:
                    self.send_json(job)
                elif len(parts) == 5 and parts[3] == "artifacts":
                    artifact = self.app.jobs.artifact(job_id, parts[4])
                    if artifact is None:
                        self.send_error_json(HTTPStatus.NOT_FOUND, "Artifact not available")
                    else:
                        self.send_file(artifact, attachment=True)
                else:
                    self.send_error_json(HTTPStatus.NOT_FOUND, "Unknown API path")
            elif parsed.path in {"/", "/index.html"}:
                html = (self.app.web_root / "index.html").read_text(encoding="utf-8")
                html = html.replace("{{ASSET_VERSION}}", self.app.asset_version)
                self.send_body(
                    html.encode("utf-8"),
                    "text/html; charset=utf-8",
                    cache_control="no-cache",
                )
            elif parsed.path.startswith("/static/"):
                relative = unquote(parsed.path[len("/static/"):])
                root = self.app.web_root.resolve()
                target = (root / relative).resolve()
                # Allow subdirectories (vendor/, …) but never escape web_root.
                if target != root and root in target.parents and target.is_file():
                    requested = parse_qs(parsed.query).get("v", [""])[0]
                    cache = (
                        "public, max-age=31536000, immutable"
                        if requested == self.app.asset_version
                        else "no-cache"
                    )
                    self.send_file(target, cache_control=cache)
                else:
                    self.send_error_json(HTTPStatus.NOT_FOUND, "Not found")
            else:
                self.send_error_json(HTTPStatus.NOT_FOUND, "Not found")
        except ApiError as exc:
            self.send_error_json(exc.status, str(exc), exc.code)
        except PermissionError as exc:
            status = (
                HTTPStatus.UNAUTHORIZED
                if self.current_user() is None
                else HTTPStatus.FORBIDDEN
            )
            self.send_error_json(status, str(exc))
        except KeyError as exc:
            self.send_error_json(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            LOGGER.exception("GET %s failed", self.path)
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
        finally:
            self._drain_body()

    def do_POST(self) -> None:
        parsed = urlsplit(self.path)
        self._body_consumed = 0
        try:
            if parsed.path == "/api/auth/register":
                body = self.read_json()
                username = str(body.get("username", "")).strip()
                password = str(body.get("password", ""))
                first_user = self.app.users.user_count() == 0
                mode = self.app.registration_mode()
                if not first_user:
                    if mode == "closed":
                        raise PermissionError("Registration is closed")
                approved = first_user or mode == "open"
                user = self.app.users.create_user(
                    username,
                    password,
                    role="admin" if first_user else "user",
                    approved=approved,
                )
                if approved:
                    token = self.app.users.create_session(user["username"])
                    self.send_json(
                        {"user": self.app.users.get_user(user["username"])},
                        status=HTTPStatus.CREATED,
                        cookie=token,
                    )
                else:
                    self.send_json(
                        {"pending": True, "message": "Account created, pending admin approval"},
                        status=HTTPStatus.ACCEPTED,
                    )
            elif parsed.path == "/api/auth/login":
                if self._login_throttled():
                    self.send_error_json(
                        HTTPStatus.TOO_MANY_REQUESTS,
                        "Too many login attempts, try again later",
                        code="too_many_attempts",
                    )
                    return
                body = self.read_json()
                user = self.app.users.verify_login(
                    str(body.get("username", "")).strip(),
                    str(body.get("password", "")),
                )
                if user is None:
                    raise ApiError("Incorrect username or password", status=403, code="bad_credentials")
                if isinstance(user, dict) and user.get("pending"):
                    self.send_error_json(
                        HTTPStatus.FORBIDDEN,
                        "Account pending admin approval",
                        code="pending_approval",
                    )
                    return
                token = self.app.users.create_session(user["username"])
                self.send_json({"user": user}, cookie=token)
            elif parsed.path == "/api/auth/logout":
                token = self._session_token()
                if token:
                    self.app.users.destroy_session(token)
                self.send_response(HTTPStatus.NO_CONTENT)
                self._security_headers()
                self.set_session_cookie("", max_age=0)
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif parsed.path == "/api/auth/change-password":
                user = self.require_user()
                body = self.read_json()
                self.app.users.change_password(
                    user["username"],
                    str(body.get("old_password", "")),
                    str(body.get("new_password", "")),
                )
                self.send_json({"changed": True})
            elif parsed.path == "/api/jobs":
                user = self.require_user()
                if self.app.users.site_setting("maintenance") == "1":
                    self.send_error_json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "Submissions are paused for maintenance",
                        code="maintenance",
                    )
                    return
                body = self.read_json()
                for field in ("challenge", "language", "source", "runner_name", "mode"):
                    if field in body and not isinstance(body[field], str):
                        raise ValueError(f"{field} must be a string")
                job = self.app.jobs.submit(
                    challenge=str(body.get("challenge", "")),
                    language=str(body.get("language", "")),
                    source=str(body.get("source", "")),
                    runner_name=str(body.get("runner_name", "local")),
                    mode=str(body.get("mode", "submit")),
                    username=user["username"],
                )
                self.send_json(job, status=HTTPStatus.ACCEPTED)
            elif parsed.path == "/api/solutions":
                user = self.require_user()
                body = self.read_json()
                challenge = str(body.get("challenge", ""))
                language = str(body.get("language", ""))
                self.app.catalog.require(challenge)
                if language not in LANGUAGES:
                    raise ValueError(f"Unsupported language: {language}")
                source = str(body.get("source", ""))
                self.app.users.save_solution(
                    user["username"], challenge, language, source
                )
                self.send_json({"saved": True})
            elif parsed.path == "/api/admin/announcements":
                user = self.require_admin()
                body = self.read_json()
                self.send_json(
                    self.app.users.add_announcement(
                        str(body.get("title", "")), str(body.get("body", "")),
                        user["username"],
                    ),
                    status=HTTPStatus.CREATED,
                )
            elif parsed.path == "/api/admin/users":
                self.require_admin()
                body = self.read_json()
                username = str(body.get("username", "")).strip()
                password = str(body.get("password", ""))
                role = str(body.get("role", "user"))
                if role not in {"user", "admin"}:
                    raise ValueError("role must be user or admin")
                created = self.app.users.create_user(
                    username, password, role=role, approved=True
                )
                self.send_json(created, status=HTTPStatus.CREATED)
            elif parsed.path == "/api/admin/users/role":
                self.require_admin()
                body = self.read_json()
                self.send_json(
                    {"ok": self.app.users.set_role(
                        str(body.get("username", "")), str(body.get("role", "user"))
                    )}
                )
            elif parsed.path == "/api/admin/users/password":
                self.require_admin()
                body = self.read_json()
                self.send_json(
                    {"ok": self.app.users.admin_reset_password(
                        str(body.get("username", "")), str(body.get("password", ""))
                    )}
                )
            elif parsed.path == "/api/admin/settings":
                self.require_admin()
                body = self.read_json()
                if "registration_mode" in body:
                    mode = str(body["registration_mode"])
                    if mode not in {"open", "review", "closed"}:
                        raise ValueError("registration_mode must be open|review|closed")
                    self.app.users.set_site_setting("registration_mode", mode)
                if "maintenance" in body:
                    self.app.users.set_site_setting(
                        "maintenance", "1" if body["maintenance"] else "0"
                    )
                if "share_solutions" in body:
                    self.app.users.set_site_setting(
                        "share_solutions", "1" if body["share_solutions"] else "0"
                    )
                self.send_json({
                    "registration_mode": self.app.registration_mode(),
                    "maintenance": self.app.users.site_setting("maintenance") == "1",
                    "share_solutions": self.app.share_solutions(),
                })
            else:
                self.send_error_json(HTTPStatus.NOT_FOUND, "Unknown API path")
        except ApiError as exc:
            self.send_error_json(exc.status, str(exc), exc.code)
        except PermissionError as exc:
            status = (
                HTTPStatus.UNAUTHORIZED
                if self.current_user() is None
                else HTTPStatus.FORBIDDEN
            )
            self.send_error_json(status, str(exc))
        except KeyError as exc:
            self.send_error_json(HTTPStatus.NOT_FOUND, str(exc))
        except FileNotFoundError as exc:
            self.send_error_json(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            LOGGER.exception("POST %s failed", self.path)
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
        finally:
            self._drain_body()

    def do_DELETE(self) -> None:
        parsed = urlsplit(self.path)
        self._body_consumed = 0
        try:
            parts = parsed.path.strip("/").split("/")
            if len(parts) == 3 and parts[:2] == ["api", "leaderboard"]:
                user = self.require_user()
                entry = self.app.scores.submission(parts[2])
                if user["role"] != "admin" and entry.get("username", "") != user["username"]:
                    raise ApiError("You can only delete your own score", status=403, code="own_entry_only")
                deleted = self.app.scores.delete(parts[2])
                if not deleted:
                    self.send_error_json(HTTPStatus.NOT_FOUND, "Leaderboard entry not found")
                else:
                    self.send_json({"deleted": True})
            elif len(parts) == 4 and parts[:3] == ["api", "admin", "announcements"]:
                self.require_admin()
                self.send_json(
                    {"deleted": self.app.users.delete_announcement(int(parts[3]))}
                )
            elif len(parts) == 5 and parts[:3] == ["api", "admin", "users"]:
                self.require_admin()
                target, action = parts[3], parts[4]
                if action == "disable":
                    self.send_json(
                        {"disabled": self.app.users.set_disabled(target, True)}
                    )
                elif action == "enable":
                    self.send_json(
                        {"enabled": self.app.users.set_disabled(target, False)}
                    )
                elif action == "approve":
                    self.send_json(
                        {"approved": self.app.users.set_approved(target, True)}
                    )
                elif action == "reject":
                    # Rejecting a pending registration is a hard delete: the
                    # account never had any history to preserve.
                    self.send_json(
                        {"rejected": self.app.users.purge_user(target)}
                    )
                else:
                    self.send_error_json(HTTPStatus.NOT_FOUND, "Unknown user action")
            elif len(parts) == 2 and parts == ["api", "solutions"]:
                user = self.require_user()
                query = parse_qs(urlsplit(self.path).query)
                challenge = query.get("challenge", [""])[0]
                language = query.get("language", [""])[0]
                self.app.catalog.require(challenge)
                if language not in LANGUAGES:
                    raise ValueError("Invalid language")
                self.send_json({
                    "deleted": self.app.users.delete_solution(
                        user["username"], challenge, language
                    )
                })
            elif len(parts) == 4 and parts[:3] == ["api", "admin", "users"]:
                self.require_admin()
                if not self.app.delete_user(parts[3]):
                    raise ApiError("No such user", status=404, code="unknown_user")
                self.send_json({"deleted": True})
            else:
                self.send_error_json(HTTPStatus.NOT_FOUND, "Unknown API path")
        except ApiError as exc:
            self.send_error_json(exc.status, str(exc), exc.code)
        except PermissionError as exc:
            status = (
                HTTPStatus.UNAUTHORIZED
                if self.current_user() is None
                else HTTPStatus.FORBIDDEN
            )
            self.send_error_json(status, str(exc))
        except (KeyError, ValueError) as exc:
            self.send_error_json(HTTPStatus.NOT_FOUND, str(exc))
        except Exception as exc:
            LOGGER.exception("DELETE %s failed", self.path)
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
        finally:
            self._drain_body()


class ApplicationServer(ThreadingHTTPServer):
    daemon_threads = True
    # Default is 5, which refuses connections when a browser fetches the ~20
    # assets of this app in parallel (shows up as half-loaded pages).
    request_queue_size = 128
    allow_reuse_address = True

    def __init__(self, address, handler, app, secure_cookies: bool = False):
        super().__init__(address, handler)
        self.app = app
        self.secure_cookies = secure_cookies
        self.login_attempts: Dict[str, List[float]] = {}
        self.login_lock = threading.Lock()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--data-dir", type=Path, default=Path("local_data"))
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help=(
            "Run submissions inside a rootless Docker container "
            "(--network none, --cap-drop ALL, --read-only, GPU via nvidia runtime). "
            "Requires the leetgpu-judge image; see docker/."
        ),
    )
    parser.add_argument(
        "--judge-image",
        default="leetgpu-judge",
        help="Docker image used for sandboxed judging.",
    )
    parser.add_argument(
        "--sandbox-timeout",
        type=int,
        default=600,
        help="Seconds before a sandboxed submission is killed (0 disables).",
    )
    parser.add_argument(
        "--auth",
        action="store_true",
        help="Require login even without --sandbox (multi-user mode).",
    )
    parser.add_argument(
        "--secure-cookies",
        action="store_true",
        help="Mark session cookies Secure (use behind an HTTPS reverse proxy).",
    )
    parser.add_argument(
        "--challenges-dir",
        type=Path,
        help=(
            "Path to the upstream repository's challenges directory. "
            "Defaults to LEETGPU_CHALLENGES_DIR or upstream/leetgpu-challenges/challenges."
        ),
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--site-name",
        default="LeetGPU Local",
        help="Product name shown in the UI, browser tab and logs.",
    )
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
            challenges_dir=challenges_dir,
            sandbox=args.sandbox,
            judge_image=args.judge_image,
            sandbox_timeout=args.sandbox_timeout,
            auth_required=args.auth,
            site_name=args.site_name,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    server = ApplicationServer(
        (args.host, args.port), RequestHandler, app, secure_cookies=args.secure_cookies
    )
    url = f"http://{args.host}:{args.port}"
    LOGGER.info("%s is running at %s", args.site_name, url)
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
