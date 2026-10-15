# LeetGPU Local

> **Fork 说明（LHaiC）**：本仓库在 [xicheng-han/leetgpu-local](https://github.com/xicheng-han/leetgpu-local)
> 之上增加了多用户功能：
>
> - **注册登录**：`--auth` 开启；首个注册用户自动成为管理员；公告、解法云端保存、每题排行榜按用户名区分。
> - **公告**：管理员在「站点管理」里发布/删除，用户端顶部横幅显示。
> - **安全判题沙箱**：`--sandbox` 将每个提交丢进 `--network none --cap-drop ALL --read-only` 的
>   Docker 容器（镜像 `leetgpu-judge`，见 `docker/`）；管理员可关/开注册、禁用用户。
> - **Vim 编辑器**：顶栏键盘图标切换，支持 normal/insert/visual、hjkl/w/b/0/$/gg/G、d/y/c、dd/yy/cc、o/O、p、u/Ctrl-R、计数前缀。
>
> 部署多人服务见 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)。
> 不带 `--auth`/`--sandbox` 启动时行为与上游一致（单机、无登录、无沙箱）。


> **项目定位 / Project scope**
>
> LeetGPU Local 只是
> [AlphaGPU LeetGPU 题库](https://github.com/AlphaGPU/leetgpu-challenges)
> 的非官方本地扩展，不替代 LeetGPU 或其在线服务。本仓库只发布扩展功能，不打包上游题库。
>
> LeetGPU Local is an unofficial local extension for the
> [AlphaGPU LeetGPU challenge repository](https://github.com/AlphaGPU/leetgpu-challenges).
> It does not replace LeetGPU or its hosted service, and this repository does not bundle the
> upstream challenge set.

[中文](#中文) · [English](#english)

![LeetGPU Local workspace](pic/image1.png)

## 中文

LeetGPU Local 把官方题库变成一个可以完全在本机使用的 GPU 编程练习环境。连接官方题库后，
你可以直接浏览题目、编写 CUDA、Triton 或 PyTorch 解法，运行完整测试，比较性能，并保存
Nsight Compute 报告。

### 可以做什么

#### 浏览和练习官方题目

按题号、名称和难度查找题目，查看题面、示例和要求。已经通过的题目会保留完成状态，
每道题同时显示累计提交次数。

#### 在浏览器里编写三种解法

同一道题可以在 CUDA、Triton 和 PyTorch 之间切换。编辑器支持撤销、重做和批量切换注释，
无需为每次尝试手动创建文件。

#### 先验证，再计分

- **测试**：运行示例和完整功能测试，用于检查答案是否正确。测试不会写入排行榜。
- **提交并计分**：正确性通过后运行性能测试，并把成绩写入当前题目的排行榜。

编译错误会显示 NVCC 原始输出；测试失败时可以查看对应测试点和输入摘要。

#### 比较不同实现的性能

每道题都有独立排行榜。你可以为不同 kernel 命名，比较 CUDA、Triton 和 PyTorch 的成绩，
按语言筛选，并打开某条成绩对应的源码。

![Per-challenge local leaderboard](pic/image.png)

排行榜保存历史最佳成绩。删除当前最佳成绩后，同名解法的下一条成绩会自动递补。
排行榜只适合比较同一台机器上的结果，不同 GPU 的耗时不能直接比较。

#### 使用 Nsight Compute

提交时可以打开 Nsight 开关。完成后可以保存 `.ncu-rep` 报告，或直接使用本机
Nsight Compute 打开报告继续分析。

### 安装

目前提供 Windows 10/11 的自动安装流程。需要：

- NVIDIA GPU 和可用的 NVIDIA 驱动；
- Conda 或 Miniconda；
- CUDA Toolkit；
- Visual Studio 2022 Build Tools，并安装 **Desktop development with C++**；
- 可选的 Nsight Compute。

#### 1. 获取本项目

```powershell
git clone <this-repository-url> leetgpu-local
cd leetgpu-local
```

#### 2. 获取官方题库

官方题库不会包含在本项目中。运行以下脚本，将它单独下载到本地：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch_upstream.ps1
```

默认位置为 `upstream/leetgpu-challenges`，该目录不会被提交到本项目。

如果本机已经有官方题库，可以跳过下载，启动时指定它的 `challenges` 目录：

```powershell
python .\scripts\local_web.py --challenges-dir D:\path\to\leetgpu-challenges\challenges
```

也可以设置环境变量：

```powershell
$env:LEETGPU_CHALLENGES_DIR = "D:\path\to\leetgpu-challenges\challenges"
```

#### 3. 准备 Python 环境

```powershell
$CondaPrefix = "D:\conda-envs\leetgpu-local"
conda create --prefix $CondaPrefix python=3.10 -y

powershell -ExecutionPolicy Bypass -File .\scripts\setup_local.ps1 `
  -CondaPrefix $CondaPrefix
```

默认安装 `torch 2.5.1+cu124` 和匹配的 `triton-windows 3.1`。如需使用其他
CUDA 或 PyTorch 版本，可以通过 `setup_local.ps1` 的参数覆盖。

#### 4. 启动

```powershell
& "$CondaPrefix\python.exe" .\scripts\local_web.py
```

浏览器会打开 [http://127.0.0.1:8787](http://127.0.0.1:8787)。不希望自动打开浏览器时：

```powershell
& "$CondaPrefix\python.exe" .\scripts\local_web.py --no-browser
```

### 命令行使用

不打开网页也可以直接评测解法：

```powershell
$Challenge = "D:\path\to\leetgpu-challenges\challenges\easy\1_vector_add"

python .\scripts\local_runner.py $Challenge --language cuda
python .\scripts\local_runner.py $Challenge --language triton
python .\scripts\local_runner.py $Challenge --language pytorch
```

默认解法文件：

| 语言 | 文件 |
| --- | --- |
| CUDA | `solution/solution.cu` |
| Triton | `solution/solution.triton.py` 或 `solution/solution.py` |
| PyTorch | `solution/solution.pytorch.py` 或 `solution/solution.py` |

常用选项：

```powershell
# 只检查正确性
python .\scripts\local_runner.py $Challenge --language cuda --tests correctness --profile off

# 完整评测，不运行 Nsight Compute
python .\scripts\local_runner.py $Challenge --language triton --profile off

# 完整评测，并要求 Nsight Compute 成功
python .\scripts\local_runner.py $Challenge --language cuda --profile on

# 评测任意位置的解法文件
python .\scripts\local_runner.py $Challenge --language cuda --solution D:\work\solution.cu
```

### 数据与安全

提交记录、成绩、源码和运行结果都保存在本机，删除或重新克隆本项目不会自动备份这些数据。

> [!WARNING]
> 解法会以当前用户权限直接在本机运行，没有沙箱隔离。只运行可信代码。
> 服务默认仅监听 `127.0.0.1`，不要直接暴露到局域网或公网。

### 与上游项目的关系

本项目只读取用户自行获取的官方题库，不打包官方题库文件。
上游内容仍受 AlphaGPU 的
[CC BY-NC-ND 4.0](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/LICENSE)
许可证约束。本项目不隶属于 AlphaGPU，也未获得其官方背书。

---

## English

LeetGPU Local turns the official challenge repository into a GPU programming environment that runs
on your own machine. Once connected to an upstream checkout, it lets you browse challenges, write
CUDA, Triton, or PyTorch solutions, run the full test set, compare performance, and save Nsight
Compute reports.

### What You Can Do

#### Browse and practise official challenges

Find challenges by number, name, or difficulty and read their statements, examples, and
requirements. Completed challenges keep their passed state, and every challenge shows its
submission count.

#### Write three kinds of solution in the browser

Switch between CUDA, Triton, and PyTorch for the same challenge. The editor supports undo, redo,
and toggling comments without requiring you to create a new file for every attempt.

#### Test first, then score

- **Test** runs all examples and functional cases. It checks correctness without changing the
  leaderboard.
- **Submit & Score** runs the performance test after correctness passes and records the result on
  that challenge's leaderboard.

Compilation failures show the original NVCC output. Failed cases expose the relevant test and an
input summary.

#### Compare implementations

Every challenge has its own leaderboard. Name different kernels, compare CUDA, Triton, and PyTorch,
filter by language, and inspect the source code behind a result.

The leaderboard keeps the historical best result for each name and language. Removing the current
best promotes the next result. Scores are intended for comparisons on the same machine; timings
from different GPUs are not directly comparable.

#### Profile with Nsight Compute

Enable the Nsight option when submitting. After the run, save the `.ncu-rep` file or open it
directly in the local Nsight Compute application.

### Installation

The automated setup currently targets Windows 10/11. You need:

- an NVIDIA GPU with a working NVIDIA driver;
- Conda or Miniconda;
- CUDA Toolkit;
- Visual Studio 2022 Build Tools with **Desktop development with C++**;
- optional Nsight Compute.

#### 1. Get This Project

```powershell
git clone <this-repository-url> leetgpu-local
cd leetgpu-local
```

#### 2. Get the Official Challenges

The official challenge set is not included. Download it separately:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch_upstream.ps1
```

This uses `upstream/leetgpu-challenges` by default, and that directory is excluded from this
project.

To use an existing checkout instead:

```powershell
python .\scripts\local_web.py --challenges-dir D:\path\to\leetgpu-challenges\challenges
```

Or set it once for the current shell:

```powershell
$env:LEETGPU_CHALLENGES_DIR = "D:\path\to\leetgpu-challenges\challenges"
```

#### 3. Prepare the Python Environment

```powershell
$CondaPrefix = "D:\conda-envs\leetgpu-local"
conda create --prefix $CondaPrefix python=3.10 -y

powershell -ExecutionPolicy Bypass -File .\scripts\setup_local.ps1 `
  -CondaPrefix $CondaPrefix
```

The default installation uses `torch 2.5.1+cu124` and the matching
`triton-windows 3.1`. Override the `setup_local.ps1` parameters when another CUDA or PyTorch
version is required.

#### 4. Start

```powershell
& "$CondaPrefix\python.exe" .\scripts\local_web.py
```

The browser opens [http://127.0.0.1:8787](http://127.0.0.1:8787). To disable automatic opening:

```powershell
& "$CondaPrefix\python.exe" .\scripts\local_web.py --no-browser
```

### Command-Line Use

Solutions can also be evaluated without the web application:

```powershell
$Challenge = "D:\path\to\leetgpu-challenges\challenges\easy\1_vector_add"

python .\scripts\local_runner.py $Challenge --language cuda
python .\scripts\local_runner.py $Challenge --language triton
python .\scripts\local_runner.py $Challenge --language pytorch
```

Default solution files:

| Language | File |
| --- | --- |
| CUDA | `solution/solution.cu` |
| Triton | `solution/solution.triton.py` or `solution/solution.py` |
| PyTorch | `solution/solution.pytorch.py` or `solution/solution.py` |

Common options:

```powershell
# Correctness only
python .\scripts\local_runner.py $Challenge --language cuda --tests correctness --profile off

# Full evaluation without Nsight Compute
python .\scripts\local_runner.py $Challenge --language triton --profile off

# Full evaluation with required Nsight Compute profiling
python .\scripts\local_runner.py $Challenge --language cuda --profile on

# Evaluate a solution stored elsewhere
python .\scripts\local_runner.py $Challenge --language cuda --solution D:\work\solution.cu
```

### Data and Security

Submissions, scores, source code, and run results stay on the local machine. They are not
automatically backed up when the project is deleted or cloned again.

> [!WARNING]
> Solutions run directly with the current user's permissions and are not sandboxed. Run trusted
> code only. The server binds to `127.0.0.1` by default; do not expose it directly to a LAN or the
> public internet.

### Upstream Relationship

This project reads an official challenge checkout obtained by the user. It does not bundle the
official challenge files. Upstream content
remains governed by AlphaGPU's
[CC BY-NC-ND 4.0](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/LICENSE)
license. This project is not affiliated with or endorsed by AlphaGPU.
