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

### 4. 启动

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
