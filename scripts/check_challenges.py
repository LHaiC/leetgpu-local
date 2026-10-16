#!/usr/bin/env python3
"""Check that every challenge can actually be evaluated in the judge image.

For each challenge it loads `challenge.py`, builds its example / functional /
performance test cases and runs the *reference* implementation on them. The
user's solution is never involved, so any failure here means the challenge is
broken for everyone — which is how the missing `Python.h` (python3-dev) problem
showed up: Triton compiles a small CUDA driver shim with gcc on first use, and
eight challenges failed with "fatal error: Python.h: No such file or directory".

Run it inside the judge image (paths default to the sandbox layout):

    docker run --rm --network none --gpus all --user 65534:65534 \\
        -v "$PWD/leetgpu-local:/judge/repo:ro" \\
        -v "$PWD/leetgpu-challenges/challenges:/judge/challenges:ro" \\
        -v "$PWD/scripts/check_challenges.py:/work/check.py:ro" \\
        -e HOME=/tmp -e TMPDIR=/tmp \\
        leetgpu-judge python3 /work/check.py

Exit status is 0 when every challenge passes.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

SCRIPTS = Path(os.environ.get("LEETGPU_SCRIPTS_DIR", "/judge/repo/scripts"))
CHALLENGES = Path(os.environ.get("LEETGPU_CHALLENGES_DIR", "/judge/challenges"))
DEVICE = os.environ.get("LEETGPU_DEVICE", "cuda:0")

sys.path.insert(0, str(SCRIPTS))

from local_runner import (  # noqa: E402
    invoke_reference,
    load_challenge,
    materialize_case,
    normalize_generated_cases,
    require_gpu,
    require_torch,
    signature_parts,
)


def check_challenge(challenge_dir: Path, torch) -> None:
    challenge = load_challenge(challenge_dir, DEVICE)
    _signature, names, _outputs = signature_parts(challenge)
    suites = {
        "example": challenge.generate_example_test(),
        "functional": challenge.generate_functional_test(),
        "performance": challenge.generate_performance_test(),
    }
    for suite, generated in suites.items():
        cases = normalize_generated_cases(generated, suite)
        for raw in cases:
            case = materialize_case(raw, torch, DEVICE)
            invoke_reference(challenge, case, names)
    torch.cuda.synchronize()


def main() -> int:
    torch = require_torch()
    require_gpu(torch, DEVICE)

    failures = []
    checked = 0
    for level_dir in sorted(path for path in CHALLENGES.iterdir() if path.is_dir()):
        for challenge_dir in sorted(path for path in level_dir.iterdir() if path.is_dir()):
            if not (challenge_dir / "challenge.py").is_file():
                continue
            checked += 1
            key = f"{level_dir.name}/{challenge_dir.name}"
            try:
                check_challenge(challenge_dir, torch)
            except Exception as exc:  # noqa: BLE001 - report and continue
                failures.append((key, f"{type(exc).__name__}: {exc}"))
                traceback.print_exc(limit=2)

    print("=" * 60)
    print(f"checked: {checked}   failed: {len(failures)}")
    for key, reason in failures:
        print(f"  {key}: {reason}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
