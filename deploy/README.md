# 系统服务部署模板 (`systemd` / `launchd`)

本目录存放 `bt_studio` 在 **Linux（如 64C Threadripper 工作站）** 与 **macOS（Darwin）** 下的原生系统守护服务配置。所有服务启动前均自动调用 `source scripts/start_env.sh` 完成平台硬件并发预算与 Tailscale 内网 IP 绑定。

---

## 1. Linux 生产环境 (`systemd` user service)

文件位置：`deploy/systemd/`
- `bt-studio-monitor.service`：监控面板与 WebSocket 网关（`:8800`，崩溃 5s 自动拉起）
- `bt-studio-agent.service`：LLM 特征挖掘飞轮主进程（`RESUME=auto`，异常崩溃 30s 自动续跑）

### 安装与启动步骤

```bash
# 1) 在 .env 中配置 BT_STUDIO_API_TOKEN 与 LLM API Key
echo "BT_STUDIO_API_TOKEN=<your-secret-token>" >> ~/startup/bt_studio/.env

# 2) 软链或复制到用户级 systemd 目录
mkdir -p ~/.config/systemd/user
cp deploy/systemd/bt-studio-*.service ~/.config/systemd/user/

# 3) 开启用户脱离 SSH 会话常驻 (允许退出 Tailscale SSH 后服务继续在后台运行)
sudo loginctl enable-linger "$USER"

# 4) 重载并启动服务
systemctl --user daemon-reload
systemctl --user enable --now bt-studio-monitor.service
systemctl --user start bt-studio-agent.service

# 5) 查看状态与日志
systemctl --user status bt-studio-monitor.service bt-studio-agent.service
tail -f ~/startup/bt_studio/result/monitor.log ~/startup/bt_studio/result/agent.log
```

---

## 2. macOS 开发/研究环境 (`launchd` LaunchAgents)

文件位置：`deploy/launchd/`
- `ai.btstudio.monitor.plist`：监控网关守护进程
- `ai.btstudio.agent.plist`：LLM 特征挖掘守护进程

### 安装与启动步骤

```bash
# 1) 根据本机实际路径修改 plist 中的 WorkingDirectory 与 BT_STUDIO_API_TOKEN
#    验证 plist 语法合法性:
plutil -lint deploy/launchd/*.plist

# 2) 安装到 ~/Library/LaunchAgents
mkdir -p ~/Library/LaunchAgents
cp deploy/launchd/ai.btstudio.*.plist ~/Library/LaunchAgents/

# 3) 加载并启动监控服务
launchctl load -w ~/Library/LaunchAgents/ai.btstudio.monitor.plist

# 4) 按需触发挖掘服务
launchctl load ~/Library/LaunchAgents/ai.btstudio.agent.plist
launchctl start ai.btstudio.agent

# 5) 停止与卸载
launchctl unload ~/Library/LaunchAgents/ai.btstudio.agent.plist
launchctl unload ~/Library/LaunchAgents/ai.btstudio.monitor.plist
```

---

## 3. 产物层 (`result/artifacts/`) 定时备份策略

`result/` 分为 `artifacts/`（不可再生产物：runs / features / review / usage / tune）与 `cache/`（可再生缓存：bars / evals / monitor.sqlite）。生产环境只需定时增量备份 `result/artifacts/`：

```bash
# 添加每日凌晨 02:30 增量 rsync 备份至异地存储或 NAS (crontab -e):
30 2 * * * rsync -az --delete ~/startup/bt_studio/result/artifacts/ /mnt/backup/bt_studio_artifacts/
```

