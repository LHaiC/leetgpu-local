#!/usr/bin/env python3
"""Run LeetGPU challenges locally on an NVIDIA GPU.

Supported solution languages are CUDA, Triton, and PyTorch. Every invocation
creates a timestamped directory containing the execution log, JSON results,
build output, and (when available) an Nsight Compute report.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import datetime as dt
import hashlib
import importlib.util
import json
import locale
import logging
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

LOGGER = logging.getLogger("leetgpu.local")
SUPPORTED_LANGUAGES = ("cuda", "triton", "pytorch")
SOLUTION_NAMES = {
    "cuda": ("solution.cu",),
    "triton": ("solution.triton.py", "solution.py"),
    "pytorch": ("solution.pytorch.py", "solution.py"),
}


class LocalRunnerError(RuntimeError):
    pass


class CompilationError(LocalRunnerError):
    def __init__(self, message: str, compiler_output: str, log_path: Path):
        super().__init__(message)
        self.compiler_output = compiler_output
        self.log_path = log_path


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def configure_logging() -> None:
    LOGGER.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise LocalRunnerError(
            "PyTorch is not installed in this Python environment. Run scripts/setup_local.ps1 "
            "for the selected Conda environment."
        ) from exc
    return torch


def require_gpu(torch, device: str) -> None:
    if not device.startswith("cuda"):
        raise LocalRunnerError("The local runner currently requires an NVIDIA CUDA device.")
    if not torch.cuda.is_available():
        raise LocalRunnerError(
            f"PyTorch {torch.__version__} cannot access CUDA. Install a CUDA-enabled PyTorch build."
        )
    try:
        torch.cuda.set_device(torch.device(device))
    except Exception as exc:
        raise LocalRunnerError(f"Cannot select device {device}: {exc}") from exc


def safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return cleaned.strip("_") or "challenge"


def create_run_dir(log_root: Path, challenge_dir: Path, language: str) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = log_root / safe_name(challenge_dir.name) / f"{timestamp}_{language}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def resolve_challenge_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file() and path.name == "challenge.py":
        path = path.parent
    if not (path / "challenge.py").is_file():
        raise LocalRunnerError(f"No challenge.py found in {path}")
    return path


def resolve_solution(challenge_dir: Path, language: str, explicit: Optional[Path]) -> Path:
    if explicit:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise LocalRunnerError(f"Solution file does not exist: {path}")
        return path

    solution_dir = challenge_dir / "solution"
    for name in SOLUTION_NAMES[language]:
        candidate = solution_dir / name
        if candidate.is_file():
            return candidate.resolve()
    expected = " or ".join(str(solution_dir / name) for name in SOLUTION_NAMES[language])
    raise LocalRunnerError(f"No {language} solution found. Expected {expected}")


def load_python_module(path: Path, prefix: str):
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    module_name = f"_{prefix}_{digest}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise LocalRunnerError(f"Cannot import Python file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_challenge(challenge_dir: Path, device: str):
    challenges_root = challenge_dir.parent.parent
    sys.path.insert(0, str(challenges_root))
    try:
        module = load_python_module(challenge_dir / "challenge.py", "local_challenge")
        challenge_type = getattr(module, "Challenge", None)
        if challenge_type is None:
            raise LocalRunnerError(f"{challenge_dir / 'challenge.py'} has no Challenge class")
        return challenge_type(device=device)
    finally:
        try:
            sys.path.remove(str(challenges_root))
        except ValueError:
            pass


def torch_dtype(torch, name: str):
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise LocalRunnerError(f"Unsupported tensor dtype in challenge: {name}")
    return dtype


def materialize_value(value: Any, torch, device: str) -> Any:
    class_name = value.__class__.__name__
    if class_name == "RandTensor":
        result = torch.empty(value.shape, device=device, dtype=torch_dtype(torch, value.dtype))
        return result.uniform_(value.low, value.high)
    if class_name == "RandnTensor":
        result = torch.empty(value.shape, device=device, dtype=torch_dtype(torch, value.dtype))
        return result.normal_(value.mean, value.std)
    if class_name == "RandIntTensor":
        return torch.randint(
            value.low,
            value.high,
            value.shape,
            device=device,
            dtype=torch_dtype(torch, value.dtype),
        )
    if class_name == "FullTensor":
        return torch.full(
            value.shape, value.value, device=device, dtype=torch_dtype(torch, value.dtype)
        )
    if class_name == "OutTensor":
        return torch.empty(value.shape, device=device, dtype=torch_dtype(torch, value.dtype))
    if isinstance(value, dict):
        return {key: materialize_value(item, torch, device) for key, item in value.items()}
    if isinstance(value, list):
        return [materialize_value(item, torch, device) for item in value]
    if isinstance(value, tuple):
        return tuple(materialize_value(item, torch, device) for item in value)
    return value


def materialize_case(case: Mapping[str, Any], torch, device: str) -> Dict[str, Any]:
    if not isinstance(case, Mapping):
        raise LocalRunnerError(f"A test generator returned {type(case).__name__}, expected a dict")
    return {name: materialize_value(value, torch, device) for name, value in case.items()}


def normalize_generated_cases(value: Any, suite: str) -> List[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(item, Mapping) for item in value):
        return list(value)
    raise LocalRunnerError(f"{suite} test generator must return a dict or a list of dicts")


def clone_value(value: Any, torch) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone(memory_format=torch.preserve_format)
    if isinstance(value, dict):
        return {key: clone_value(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_value(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_value(item, torch) for item in value)
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def clone_case(case: Mapping[str, Any], torch) -> Dict[str, Any]:
    return {name: clone_value(value, torch) for name, value in case.items()}


def summarize_tensor(value, torch, full_limit: int = 64, sample_limit: int = 16) -> Dict[str, Any]:
    detached = value.detach()
    flat = detached.reshape(-1)
    summary: Dict[str, Any] = {
        "type": "tensor",
        "shape": list(detached.shape),
        "dtype": str(detached.dtype).removeprefix("torch."),
        "device": str(detached.device),
        "numel": detached.numel(),
    }
    if detached.numel() <= full_limit:
        summary["values"] = detached.cpu().tolist()
        return summary

    head_count = min(sample_limit, flat.numel())
    tail_count = min(4, max(0, flat.numel() - head_count))
    summary["values_truncated"] = True
    summary["head"] = flat[:head_count].cpu().tolist()
    if tail_count:
        summary["tail"] = flat[-tail_count:].cpu().tolist()
    if detached.dtype != torch.bool and not detached.is_complex():
        stats_value = detached.to(torch.float64)
        finite = stats_value[torch.isfinite(stats_value)]
        if finite.numel():
            summary["min"] = float(finite.min().item())
            summary["max"] = float(finite.max().item())
            summary["mean"] = float(finite.mean().item())
    return summary


def summarize_value(value: Any, torch, depth: int = 0) -> Any:
    if depth > 3:
        return {"type": type(value).__name__, "value": "<nested value omitted>"}
    if isinstance(value, torch.Tensor):
        return summarize_tensor(value, torch)
    if isinstance(value, dict):
        items = list(value.items())
        return {str(key): summarize_value(item, torch, depth + 1) for key, item in items[:16]}
    if isinstance(value, (list, tuple)):
        values = [summarize_value(item, torch, depth + 1) for item in value[:16]]
        return {
            "type": type(value).__name__,
            "length": len(value),
            "values": values,
            "values_truncated": len(value) > len(values),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    descriptor = {"type": type(value).__name__}
    for attribute in ("shape", "dtype", "low", "high", "mean", "std", "value"):
        if hasattr(value, attribute):
            descriptor[attribute] = getattr(value, attribute)
    return descriptor


def summarize_inputs(
    case: Mapping[str, Any], signature: Mapping[str, Tuple[Any, str]], torch
) -> Dict[str, Any]:
    return {
        name: summarize_value(case[name], torch)
        for name, (_, direction) in signature.items()
        if direction in {"in", "inout"} and name in case
    }


def find_vcvars64() -> Optional[Path]:
    vswhere = (
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    )
    if vswhere.is_file():
        result = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            candidate = Path(result.stdout.strip()) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if candidate.is_file():
                return candidate

    roots = (
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
    )
    for root in roots:
        pattern = "Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat"
        matches = sorted(root.glob(pattern), reverse=True)
        if matches:
            return matches[0]
    return None


def windows_build_environment() -> Dict[str, str]:
    environment = os.environ.copy()
    if shutil.which("cl", path=environment.get("PATH")):
        return environment
    vcvars = find_vcvars64()
    if vcvars is None:
        raise LocalRunnerError(
            "Microsoft Visual C++ Build Tools were not found. Install the Desktop development "
            "with C++ workload for Visual Studio 2022."
        )
    command = f'"{vcvars}" >nul && set'
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise LocalRunnerError(f"Failed to initialize Visual C++ environment from {vcvars}")
    encoding = locale.getpreferredencoding(False)
    for raw_line in result.stdout.decode(encoding, errors="replace").splitlines():
        if "=" not in raw_line or raw_line.startswith("="):
            continue
        key, value = raw_line.split("=", 1)
        environment[key] = value
    return environment


def ascii_temporary_directory(prefix: str, source: Path):
    candidates = [Path(tempfile.gettempdir()), *source.parents]
    for candidate in candidates:
        if not str(candidate).isascii() or not candidate.is_dir():
            continue
        try:
            return tempfile.TemporaryDirectory(prefix=prefix, dir=candidate)
        except OSError:
            continue
    raise LocalRunnerError(
        "CUDA compilation needs a writable ASCII-only temporary directory on Windows."
    )


def compile_cuda(source: Path, build_dir: Path) -> Tuple[Path, List[str], str]:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise LocalRunnerError(
            "nvcc was not found on PATH. Install the CUDA Toolkit or add it to PATH."
        )
    build_dir.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    temporary_directory = None
    if os.name == "nt":
        output = build_dir / f"solution-{source_hash}.dll"
        temporary_directory = ascii_temporary_directory(f"leetgpu-nvcc-{source_hash}-", source)
        staging_dir = Path(temporary_directory.name)
        compiler_source = staging_dir / "solution.cu"
        compiler_output = staging_dir / output.name
        shutil.copy2(source, compiler_source)
        command = [
            nvcc,
            "-O3",
            "-lineinfo",
            "--shared",
            str(compiler_source),
            "-o",
            str(compiler_output),
            "-Xcompiler=/MD",
            "-Xcompiler=/utf-8",
            "-Xlinker=/EXPORT:solve",
        ]
        environment = windows_build_environment()
        compiler_cwd = staging_dir
    else:
        output = build_dir / f"solution-{source_hash}.so"
        compiler_output = output
        command = [
            nvcc,
            "-O3",
            "-lineinfo",
            "--shared",
            "-Xcompiler",
            "-fPIC",
            str(source),
            "-o",
            str(output),
        ]
        environment = os.environ.copy()
        compiler_cwd = source.parent

    LOGGER.info("Compiling CUDA solution: %s", subprocess.list2cmdline(command))
    try:
        result = subprocess.run(
            command,
            cwd=str(compiler_cwd),
            env=environment,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        build_output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        if temporary_directory is not None:
            build_output = build_output.replace(str(compiler_source), str(source))
        build_log = build_dir / "build.log"
        build_log.write_text(build_output, encoding="utf-8")
        if result.returncode != 0 or not compiler_output.is_file():
            message = f"nvcc compilation failed with exit code {result.returncode}"
            LOGGER.error("Compiler output:\n%s", build_output or "<no compiler output>")
            raise CompilationError(message, build_output, build_log)
        if compiler_output != output:
            shutil.copy2(compiler_output, output)
        return output, command, build_output
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()


class PythonSolution:
    def __init__(self, source: Path, language: str):
        self.module = load_python_module(source, f"local_{language}_solution")
        self.solve = getattr(self.module, "solve", None)
        if not callable(self.solve):
            raise LocalRunnerError(f"{source} does not define a callable solve function")

    def invoke(self, arguments: Sequence[Any]):
        return self.solve(*arguments)


class CudaSolution:
    def __init__(self, binary: Path, signature: Mapping[str, Tuple[Any, str]], torch):
        self.torch = torch
        self.signature = signature
        try:
            self.library = ctypes.CDLL(str(binary))
            self.solve = self.library.solve
        except (OSError, AttributeError) as exc:
            raise LocalRunnerError(f"Cannot load solve from CUDA library {binary}: {exc}") from exc
        self.solve.argtypes = [ctype for ctype, _ in signature.values()]
        self.solve.restype = None

    def invoke(self, arguments: Sequence[Any]):
        converted = []
        for value, (ctype, _) in zip(arguments, self.signature.values()):
            if isinstance(value, self.torch.Tensor):
                converted.append(ctypes.cast(value.data_ptr(), ctype))
            else:
                if hasattr(value, "item"):
                    value = value.item()
                converted.append(ctype(value))
        self.solve(*converted)
        return None


def make_solution(
    language: str,
    source: Path,
    signature: Mapping[str, Tuple[Any, str]],
    torch,
    build_dir: Path,
    binary: Optional[Path] = None,
):
    build_info = None
    if language == "cuda":
        if binary is None:
            binary, command, output = compile_cuda(source, build_dir)
            build_info = {
                "binary": str(binary),
                "command": command,
                "output": output,
            }
        return CudaSolution(binary, signature, torch), build_info, binary
    return PythonSolution(source, language), build_info, None


def signature_parts(challenge) -> Tuple[Mapping[str, Tuple[Any, str]], List[str], List[str]]:
    signature = challenge.get_solve_signature()
    if not isinstance(signature, Mapping) or not signature:
        raise LocalRunnerError("get_solve_signature() must return a non-empty mapping")
    names = list(signature)
    outputs = [name for name, (_, direction) in signature.items() if direction in {"out", "inout"}]
    if not outputs:
        raise LocalRunnerError("The challenge signature has no out or inout parameters")
    return signature, names, outputs


def invoke_solution(solution, case: Mapping[str, Any], names: Sequence[str]):
    missing = [name for name in names if name not in case]
    if missing:
        raise LocalRunnerError(f"Test case is missing solve arguments: {', '.join(missing)}")
    return solution.invoke([case[name] for name in names])


def invoke_reference(challenge, case: Mapping[str, Any], names: Sequence[str]):
    missing = [name for name in names if name not in case]
    if missing:
        raise LocalRunnerError(f"Test case is missing reference arguments: {', '.join(missing)}")
    return challenge.reference_impl(*[case[name] for name in names])


def returned_outputs(returned: Any, outputs: Sequence[str]) -> Optional[Dict[str, Any]]:
    if returned is None:
        return None
    if len(outputs) == 1:
        return {outputs[0]: returned}
    if isinstance(returned, (tuple, list)) and len(returned) == len(outputs):
        return dict(zip(outputs, returned))
    raise LocalRunnerError(
        f"solve returned a value that cannot be mapped to outputs: {', '.join(outputs)}"
    )


def tensor_error(actual, expected, torch) -> Dict[str, float]:
    if actual.numel() == 0:
        return {"max_abs_error": 0.0, "mean_abs_error": 0.0}
    if actual.dtype == torch.bool or not (actual.is_floating_point() or actual.is_complex()):
        mismatches = int(torch.count_nonzero(actual != expected).item())
        return {"mismatched_elements": mismatches}
    difference = torch.abs(actual.to(torch.float64) - expected.to(torch.float64))
    finite = difference[torch.isfinite(difference)]
    if finite.numel() == 0:
        return {"max_abs_error": math.inf, "mean_abs_error": math.inf}
    return {
        "max_abs_error": float(finite.max().item()),
        "mean_abs_error": float(finite.mean().item()),
    }


def compare_outputs(
    actual_case: Mapping[str, Any],
    expected_case: Mapping[str, Any],
    actual_return: Any,
    expected_return: Any,
    outputs: Sequence[str],
    challenge,
    torch,
) -> Dict[str, Dict[str, float]]:
    returned_actual = returned_outputs(actual_return, outputs)
    returned_expected = returned_outputs(expected_return, outputs)
    diagnostics = {}
    for name in outputs:
        actual = returned_actual[name] if returned_actual is not None else actual_case[name]
        expected = returned_expected[name] if returned_expected is not None else expected_case[name]
        if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
            if actual.shape != expected.shape:
                shapes = f"{tuple(actual.shape)}, expected {tuple(expected.shape)}"
                raise AssertionError(f"output {name} has shape {shapes}")
            diagnostics[name] = tensor_error(actual, expected, torch)
            torch.testing.assert_close(
                actual,
                expected,
                atol=float(challenge.atol),
                rtol=float(challenge.rtol),
                equal_nan=True,
            )
        elif actual != expected:
            raise AssertionError(f"output {name} is {actual!r}, expected {expected!r}")
    return diagnostics


def run_correctness_case(
    label: str,
    raw_case: Mapping[str, Any],
    challenge,
    solution,
    names: Sequence[str],
    outputs: Sequence[str],
    signature: Mapping[str, Tuple[Any, str]],
    torch,
    device: str,
) -> Dict[str, Any]:
    started = time.perf_counter()
    record: Dict[str, Any] = {"name": label, "status": "error"}
    input_summary = None
    try:
        base_case = materialize_case(raw_case, torch, device)
        input_summary = summarize_inputs(base_case, signature, torch)
        expected_case = clone_case(base_case, torch)
        actual_case = clone_case(base_case, torch)
        with torch.no_grad():
            expected_return = invoke_reference(challenge, expected_case, names)
            torch.cuda.synchronize()
            actual_return = invoke_solution(solution, actual_case, names)
            torch.cuda.synchronize()
        diagnostics = compare_outputs(
            actual_case,
            expected_case,
            actual_return,
            expected_return,
            outputs,
            challenge,
            torch,
        )
        record.update(status="passed", outputs=diagnostics)
    except AssertionError as exc:
        record.update(status="failed", error=str(exc), inputs=input_summary)
    except Exception as exc:
        if input_summary is None:
            input_summary = summarize_inputs(raw_case, signature, torch)
        record.update(
            status="error",
            error=str(exc),
            traceback=traceback.format_exc(),
            inputs=input_summary,
        )
    record["duration_ms"] = (time.perf_counter() - started) * 1000.0
    return record


def reset_mutable_arguments(
    case: Mapping[str, Any],
    initial_inout: Mapping[str, Any],
    signature: Mapping[str, Tuple[Any, str]],
    torch,
) -> None:
    for name, (_, direction) in signature.items():
        value = case[name]
        if not isinstance(value, torch.Tensor):
            continue
        if direction == "inout":
            value.copy_(initial_inout[name])
        elif direction == "out":
            value.zero_()


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def run_benchmark(
    raw_case: Mapping[str, Any],
    solution,
    signature: Mapping[str, Tuple[Any, str]],
    names: Sequence[str],
    torch,
    device: str,
    warmup: int,
    repeat: int,
) -> Dict[str, Any]:
    case = materialize_case(raw_case, torch, device)
    initial_inout = {
        name: clone_value(case[name], torch)
        for name, (_, direction) in signature.items()
        if direction == "inout"
    }

    LOGGER.info("Benchmark warmup=%d repeat=%d", warmup, repeat)
    with torch.no_grad():
        for _ in range(warmup):
            reset_mutable_arguments(case, initial_inout, signature, torch)
            invoke_solution(solution, case, names)
        torch.cuda.synchronize()

        gpu_times = []
        wall_times = []
        for _ in range(repeat):
            reset_mutable_arguments(case, initial_inout, signature, torch)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start_event.record()
            invoke_solution(solution, case, names)
            end_event.record()
            end_event.synchronize()
            wall_times.append((time.perf_counter() - wall_start) * 1000.0)
            gpu_times.append(float(start_event.elapsed_time(end_event)))

    return {
        "warmup": warmup,
        "repeat": repeat,
        "gpu_ms": {
            "min": min(gpu_times),
            "median": statistics.median(gpu_times),
            "mean": statistics.fmean(gpu_times),
            "p95": percentile(gpu_times, 0.95),
            "samples": gpu_times,
        },
        "wall_ms": {
            "min": min(wall_times),
            "median": statistics.median(wall_times),
            "mean": statistics.fmean(wall_times),
            "p95": percentile(wall_times, 0.95),
            "samples": wall_times,
        },
    }


def find_ncu() -> Optional[Path]:
    configured = os.getenv("NCU_PATH")
    if configured and Path(configured).is_file():
        return Path(configured).resolve()
    found = shutil.which("ncu") or shutil.which("ncu.exe")
    if found:
        path = Path(found).resolve()
        if path.suffix.lower() == ".bat":
            executable = path.parent / "target" / "windows-desktop-win7-x64" / "ncu.exe"
            if executable.is_file():
                return executable
        return path
    if os.name == "nt":
        root = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation"
        matches = sorted(root.glob("Nsight Compute */target/windows-desktop-win7-x64/ncu.exe"))
        if matches:
            return matches[-1]
    return None


def run_nsight_compute(
    args,
    run_dir: Path,
    solution_path: Path,
    binary: Optional[Path],
) -> Dict[str, Any]:
    ncu = find_ncu()
    if ncu is None:
        return {"status": "unavailable", "error": "ncu was not found"}

    export_base = run_dir / "nsight-compute"
    command = [
        str(ncu),
        "--set",
        args.ncu_set,
        "--target-processes",
        "all",
        "--force-overwrite",
        "--export",
        str(export_base),
        "--launch-count",
        str(args.ncu_launch_count),
        sys.executable,
        str(Path(__file__).resolve()),
        str(args.challenge),
        "--language",
        args.language,
        "--solution",
        str(solution_path),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--_profile-target",
    ]
    if binary is not None:
        command.extend(("--_binary", str(binary)))

    (run_dir / "nsight-command.txt").write_text(
        subprocess.list2cmdline(command) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "Starting Nsight Compute (%s set, up to %d launches)", args.ncu_set, args.ncu_launch_count
    )
    try:
        result = subprocess.run(
            command,
            cwd=str(args.challenge),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=args.ncu_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + "\n" + (exc.stderr or "")
        (run_dir / "nsight-compute.log").write_text(output, encoding="utf-8")
        return {"status": "timeout", "error": f"Nsight Compute exceeded {args.ncu_timeout}s"}

    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    ncu_log = run_dir / "nsight-compute.log"
    ncu_log.write_text(output, encoding="utf-8")
    report_candidates = [export_base.with_suffix(".ncu-rep"), export_base]
    report = next((path for path in report_candidates if path.is_file()), None)
    profile = {
        "status": "passed" if result.returncode == 0 and report else "failed",
        "exit_code": result.returncode,
        "log": str(ncu_log),
        "report": str(report) if report else None,
    }
    if report:
        csv_command = [str(ncu), "--import", str(report), "--csv", "--page", "details"]
        csv_result = subprocess.run(
            csv_command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=120,
            check=False,
        )
        csv_path = run_dir / "nsight-details.csv"
        csv_path.write_text(csv_result.stdout, encoding="utf-8")
        profile["csv"] = str(csv_path)
    if profile["status"] == "failed":
        profile["error"] = f"ncu exited with {result.returncode}; see {ncu_log}"
    return profile


def profile_target(args) -> int:
    torch = require_torch()
    require_gpu(torch, args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    challenge_dir = resolve_challenge_dir(args.challenge)
    challenge = load_challenge(challenge_dir, args.device)
    signature, names, _ = signature_parts(challenge)
    solution_path = resolve_solution(challenge_dir, args.language, args.solution)
    binary = args._binary.resolve() if args._binary else None
    solution, _, _ = make_solution(
        args.language,
        solution_path,
        signature,
        torch,
        Path.cwd() / ".local_build",
        binary=binary,
    )
    generated = normalize_generated_cases(challenge.generate_performance_test(), "performance")
    case = materialize_case(generated[0], torch, args.device)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("leetgpu_solution")
    try:
        with torch.no_grad():
            invoke_solution(solution, case, names)
            torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()
    return 0


def environment_info(torch, device: str) -> Dict[str, Any]:
    triton_version = None
    try:
        import triton

        triton_version = triton.__version__
    except ImportError:
        pass
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton_version,
        "device": device,
        "gpu": torch.cuda.get_device_name(torch.device(device)),
    }


def run_local(args, run_dir: Path) -> Tuple[int, Dict[str, Any]]:
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    result: Dict[str, Any] = {
        "status": "error",
        "started_at": started_at,
        "challenge_path": str(args.challenge),
        "language": args.language,
        "run_dir": str(run_dir),
        "tests": [],
    }
    torch = require_torch()
    require_gpu(torch, args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    result["environment"] = environment_info(torch, args.device)
    LOGGER.info(
        "Environment: torch=%s CUDA=%s GPU=%s",
        torch.__version__,
        torch.version.cuda,
        result["environment"]["gpu"],
    )

    challenge = load_challenge(args.challenge, args.device)
    signature, names, outputs = signature_parts(challenge)
    solution_path = resolve_solution(args.challenge, args.language, args.solution)
    result.update(challenge=challenge.name, solution=str(solution_path), outputs=outputs)
    LOGGER.info("Challenge: %s", challenge.name)
    LOGGER.info("Solution: %s", solution_path)

    solution, build_info, binary = make_solution(
        args.language, solution_path, signature, torch, run_dir / "build"
    )
    if build_info:
        result["build"] = build_info

    run_example = args.tests in {"all", "correctness", "example"}
    run_functional = args.tests in {"all", "correctness", "functional"}
    run_performance = args.tests in {"all", "performance"}

    cases: List[Tuple[str, Mapping[str, Any]]] = []
    if run_example:
        generated = normalize_generated_cases(challenge.generate_example_test(), "example")
        cases.extend((f"example_{index:03d}", case) for index, case in enumerate(generated, 1))
    if run_functional:
        generated = normalize_generated_cases(challenge.generate_functional_test(), "functional")
        cases.extend((f"functional_{index:03d}", case) for index, case in enumerate(generated, 1))

    correctness_ok = True
    for label, case in cases:
        record = run_correctness_case(
            label, case, challenge, solution, names, outputs, signature, torch, args.device
        )
        result["tests"].append(record)
        if record["status"] == "passed":
            LOGGER.info("PASS %-18s %.3f ms", label, record["duration_ms"])
        else:
            correctness_ok = False
            LOGGER.error(
                "%s %-18s %s",
                record["status"].upper(),
                label,
                record.get("error", "unknown error"),
            )
            if record["status"] == "error":
                break

    if correctness_ok and run_performance:
        performance_cases = normalize_generated_cases(
            challenge.generate_performance_test(), "performance"
        )
        if len(performance_cases) != 1:
            LOGGER.warning(
                "Performance generator returned %d cases; benchmarking the first",
                len(performance_cases),
            )
        result["benchmark"] = run_benchmark(
            performance_cases[0],
            solution,
            signature,
            names,
            torch,
            args.device,
            args.warmup,
            args.repeat,
        )
        LOGGER.info(
            "Benchmark GPU median %.4f ms (min %.4f ms, p95 %.4f ms)",
            result["benchmark"]["gpu_ms"]["median"],
            result["benchmark"]["gpu_ms"]["min"],
            result["benchmark"]["gpu_ms"]["p95"],
        )

    should_profile = args.profile == "on" or (args.profile == "auto" and correctness_ok)
    if should_profile:
        profile = run_nsight_compute(args, run_dir, solution_path, binary)
        result["nsight_compute"] = profile
        if profile["status"] == "passed":
            LOGGER.info("Nsight Compute report: %s", profile["report"])
        elif args.profile == "on":
            correctness_ok = False
            LOGGER.error("Nsight Compute failed: %s", profile.get("error", profile["status"]))
        else:
            LOGGER.warning(
                "Nsight Compute skipped/failed: %s", profile.get("error", profile["status"])
            )
    elif args.profile != "off" and not correctness_ok:
        result["nsight_compute"] = {
            "status": "skipped",
            "reason": "correctness tests did not pass",
        }

    result["status"] = "passed" if correctness_ok else "failed"
    result["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return (0 if correctness_ok else 1), result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("challenge", type=Path, help="Challenge directory or challenge.py")
    parser.add_argument("--language", choices=SUPPORTED_LANGUAGES, default="cuda")
    parser.add_argument("--solution", type=Path, help="Explicit solution file")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--tests",
        choices=("all", "correctness", "example", "functional", "performance"),
        default="all",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-root", type=Path, default=Path("logs"))
    parser.add_argument("--run-dir", type=Path, help="Use this exact output directory")
    parser.add_argument(
        "--profile",
        choices=("auto", "on", "off"),
        default="auto",
        help="Nsight Compute behavior; auto profiles only after correctness passes",
    )
    parser.add_argument("--ncu-set", default="basic", help="Nsight Compute section set")
    parser.add_argument("--ncu-launch-count", type=int, default=20)
    parser.add_argument("--ncu-timeout", type=int, default=900)
    parser.add_argument("--_profile-target", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_binary", type=Path, help=argparse.SUPPRESS)
    return parser


def validate_args(args) -> None:
    if args.warmup < 0:
        raise LocalRunnerError("--warmup cannot be negative")
    if args.repeat < 1:
        raise LocalRunnerError("--repeat must be at least 1")
    if args.ncu_launch_count < 1:
        raise LocalRunnerError("--ncu-launch-count must be at least 1")
    args.challenge = resolve_challenge_dir(args.challenge)
    if args.solution:
        args.solution = args.solution.expanduser().resolve()
    args.log_root = args.log_root.expanduser().resolve()
    if args.run_dir:
        args.run_dir = args.run_dir.expanduser().resolve()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        if args._profile_target:
            configure_logging()
            return profile_target(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.run_dir:
        run_dir = args.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_dir(args.log_root, args.challenge, args.language)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    result: Dict[str, Any] = {
        "status": "error",
        "challenge_path": str(args.challenge),
        "language": args.language,
        "run_dir": str(run_dir),
    }
    exit_code = 2
    with (run_dir / "execution.log").open("w", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        configure_logging()
        LOGGER.info("Run directory: %s", run_dir)
        try:
            exit_code, result = run_local(args, run_dir)
        except Exception as exc:
            result.update(status="error", error=str(exc), traceback=traceback.format_exc())
            if isinstance(exc, CompilationError):
                result.update(
                    compiler_output=exc.compiler_output,
                    compiler_log=str(exc.log_path),
                )
            LOGGER.error("Local run failed: %s", exc)
            LOGGER.debug("%s", result["traceback"])
        finally:
            result["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            (run_dir / "result.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )
            LOGGER.info("Result: %s", result["status"])
            LOGGER.info("Saved execution log: %s", run_dir / "execution.log")
            sys.stdout, sys.stderr = original_stdout, original_stderr
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
