[CmdletBinding()]
param(
    [string]$CondaPrefix = $env:CONDA_PREFIX,
    [string]$TorchVersion = "2.5.1+cu124",
    [string]$TritonVersion = "3.1.0.post17",
    [string]$TorchWheelUrl = "",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu124",
    [string]$PyPiIndexUrl = "https://pypi.org/simple"
)

$ErrorActionPreference = "Stop"
if (-not $CondaPrefix) {
    throw "Activate a Conda environment or pass -CondaPrefix."
}
$python = Join-Path $CondaPrefix "python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python was not found at $python"
}

Write-Host "Using $python"
& $python -m pip install --upgrade pip --index-url $PyPiIndexUrl
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip" }

& $python -m pip install "numpy>=1.26,<2.3" filelock fsspec jinja2 networkx setuptools sympy typing-extensions --index-url $PyPiIndexUrl
if ($LASTEXITCODE -ne 0) { throw "Failed to install PyTorch runtime dependencies" }

if ($TorchWheelUrl) {
    & $python -m pip install --force-reinstall --no-deps $TorchWheelUrl
} else {
    & $python -m pip install --force-reinstall --no-deps "torch==$TorchVersion" --index-url $TorchIndexUrl
}
if ($LASTEXITCODE -ne 0) { throw "Failed to install CUDA PyTorch" }

& $python -m pip install --force-reinstall --no-deps "triton-windows==$TritonVersion" --index-url $PyPiIndexUrl
if ($LASTEXITCODE -ne 0) { throw "Failed to install Triton for Windows" }

& $python -c "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}'); assert torch.cuda.is_available(), 'CUDA is not available to PyTorch'; print(torch.cuda.get_device_name(0)); import triton; print(f'triton={triton.__version__}')"
if ($LASTEXITCODE -ne 0) { throw "Local GPU environment validation failed" }

Write-Host "Local LeetGPU environment is ready."
