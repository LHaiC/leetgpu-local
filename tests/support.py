from pathlib import Path


def create_challenge_tree(root: Path) -> Path:
    challenges_dir = root / "challenges"
    challenge_dir = challenges_dir / "easy" / "1_synthetic_addition"
    starter_dir = challenge_dir / "starter"
    starter_dir.mkdir(parents=True)

    (challenge_dir / "challenge.py").write_text(
        """
class Challenge:
    name = "Synthetic Addition"
    access_tier = "free"
""".lstrip(),
        encoding="utf-8",
    )
    (challenge_dir / "challenge.html").write_text(
        "<h2>Synthetic test challenge</h2><p>Add two generated values.</p>\n",
        encoding="utf-8",
    )
    (starter_dir / "starter.cu").write_text(
        'extern "C" void solve(const float* a, const float* b, float* c, int n) {}\n',
        encoding="utf-8",
    )
    (starter_dir / "starter.triton.py").write_text(
        "def solve(a, b, c, n):\n    pass\n",
        encoding="utf-8",
    )
    (starter_dir / "starter.pytorch.py").write_text(
        "def solve(a, b, c, n):\n    pass\n",
        encoding="utf-8",
    )
    return challenges_dir
