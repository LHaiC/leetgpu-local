# Multi-User Deployment (公网 / 多人)

本 fork 在单机版之上加了：注册登录、公告、管理员面板、Docker 沙箱判题。

## 架构

```
浏览器 → nginx (TLS) → local_web.py (--sandbox --auth) → docker run leetgpu-judge (无网/无特权)
```

判题容器：`--network none`、drop 所有 capability、`--read-only`、非 root（容器内 nobody）、
限制 8G 内存 / 4 CPU / 256 pid，超时 600s 可配。

## 步骤

### 1. 依赖

```bash
# 宿主需要：NVIDIA 驱动、docker、nvidia-container-toolkit、Python 3.10+（仅跑 web 服务）
sudo apt install -y python3 docker.io nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

### 2. 题库

```bash
git clone https://github.com/AlphaGPU/leetgpu-challenges upstream/leetgpu-challenges
# 或启动时用 --challenges-dir 指定
```

### 3. 判题镜像

```bash
./scripts/build_judge_image.sh    # 构建 leetgpu-judge（cuda devel + torch + triton）
```

镜像里两个**必须**具备的东西（否则部分题目跑不起来，且报错很误导）：

| 依赖 | 不加会怎样 |
|---|---|
| `python3-dev`（提供 `Python.h`） | Triton 首次使用会用系统 gcc 编译一个 CUDA driver 小模块，报 `fatal error: Python.h: No such file or directory` |
| 容器 `/tmp` 必须可执行 | Triton 把该模块编译进缓存后再 `dlopen`。Docker 的 tmpfs 默认带 `noexec`，**必须显式写 `exec`** 才能覆盖，否则报 `failed to map segment from shared object` |

沙箱启动参数里已经带了 `--tmpfs /tmp:rw,nosuid,exec,size=2g`，Dockerfile 里也装了 `python3-dev`。

### 4. 自检：题库是否都能跑

换题库版本（或改镜像）后，先跑这个，别等用户报障：

```bash
cd /path/to/projects            # 含 leetgpu-local 与 leetgpu-challenges 的目录
docker run --rm --network none --gpus all --user 65534:65534 \
    -v "$PWD/leetgpu-local:/judge/repo:ro" \
    -v "$PWD/leetgpu-challenges/challenges:/judge/challenges:ro" \
    -e HOME=/tmp -e TMPDIR=/tmp \
    leetgpu-judge python3 /judge/repo/scripts/check_challenges.py
```

它逐题加载 `challenge.py`、生成测试用例并跑**参考实现**，不涉及用户解法。退出码非 0 说明有题目对所有人都跑不通。当前基线：`checked: 131   failed: 0`。

### 5. 启动

```bash
python3 scripts/local_web.py \
    --sandbox --auth --secure-cookies \
    --host 127.0.0.1 --port 8787 \
    --challenges-dir /path/to/leetgpu-challenges/challenges
```

`--secure-cookies` 只在 HTTPS 反代后面加；纯 http 内测别加（否则 cookie 发不出去）。

**首个注册的用户自动成为管理员。** 之后开放注册可在管理面板里关闭。

### 5. systemd（可选）

```ini
[Unit]
Description=LeetGPU Local
After=docker.service network.target

[Service]
WorkingDirectory=/opt/leetgpu-local
ExecStart=/usr/bin/python3 scripts/local_web.py --sandbox --auth --secure-cookies --challenges-dir /opt/leetgpu-challenges/challenges
Restart=on-failure
User=leetgpu

[Install]
WantedBy=multi-user.target
```

### 6. nginx 反代（HTTPS）

```nginx
server {
    listen 443 ssl;
    server_name gpu.example.com;
    ssl_certificate     /etc/letsencrypt/live/gpu.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/gpu.example.com/privkey.pem;
    client_max_body_size 3m;

    location / {
        proxy_pass http://127.0.0.1:8787;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 安全说明

- 判题在容器内跑，无外网、无宿主写权限。提交者只能动自己的解法/成绩/任务。
- 管理员可发公告、禁用/删除用户、关注册、删任意成绩、看任意提交源码。
- 排行榜成绩仅在同一 GPU 上可比。
- 沙箱挡的是"好奇的误用"而非坚定攻击者：容器逃逸仍有理论可能。对不信任的用户群，把服务放到一台没有敏感数据的机器上跑。

## 无登录单机模式（原行为）

不加 `--auth`/`--sandbox` 时行为和原版一致：无账号、无沙箱、只听 127.0.0.1。
