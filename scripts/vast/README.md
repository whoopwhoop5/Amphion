# Vast.ai (RTX 4090) workflow

These scripts assume the repo is cloned on the GPU host and you are running from the repo root.

## One-time setup (on the GPU host)

```bash
cd ~/Amphion
bash scripts/vast/setup_vevo_venv.sh
```

## Pull latest code (on the GPU host)

```bash
cd ~/Amphion
bash scripts/vast/pull_latest.sh
```

## Offline smoke test (on the GPU host)

```bash
cd ~/Amphion
bash scripts/vast/run_offline_smoke.sh
```

## Live server (on the GPU host)

```bash
cd ~/Amphion
bash scripts/vast/run_live_server.sh
```

## Autotune (on the GPU host)

```bash
cd ~/Amphion
bash scripts/vast/run_search.sh
```

