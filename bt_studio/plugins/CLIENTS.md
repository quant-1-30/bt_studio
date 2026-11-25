# bt_studio Plugins

This directory contains server-side plugins that are part of the core `bt_studio` package.

## Layering (v3 refactor)

```
bt_studio/
├── pipeline/ + tune.py     # core compute (agent, features, diagnostics)
├── orchestrator/           # core HPO serial queue (AsyncTaskManager) — NOT a plugin
├── trade/                   # 实盘交易引擎 — imports core, NOT a plugin (见 bt_studio/trade/)
├── server/ + monitor/      # 网关 + 产物监控 — imports core, NOT a plugin (见 bt_studio/server/)
└── plugins/                # zero-core-dep transport/backend adapters (本目录)
    ├── zmq_client/         # generic ZMQ transport (ZmqClient + codec) — zero bt_studio.* imports
    └── xtp/                # XTP-specific quote cache (TopkQuoteCache) — zero bt_studio.* imports
```

Dependency rules (enforced by `tests/test_architecture.py`):

- `bt_studio.{pipeline,tune,orchestrator,visual,utils}` must NOT import `bt_studio.plugins.*`
- `plugins.zmq_client` and `plugins.xtp` must have zero `bt_studio` imports

## Available Plugins

- **`zmq_client/`**: Generic ZeroMQ transport (`ZmqClient` + `codec`) — a transport-agnostic DEALER client speaking the request/response protocol. Fully self-contained (asyncio + zmq + codec only). Backend-specific logic (XTP today) is NOT here.
- **`xtp/`**: XTP-specific market-data plumbing (`TopkQuoteCache`, `data_time_to_bar_idx`, `XTP_EXCHANGE_*`). Coupled only to the XTP quote schema; consumes any `ZmqClient`-like transport.

> 说明: 原 `plugins/api_server/` 已迁出至 `bt_studio/server/`(网关)+ `bt_studio/monitor/`(MonitorIndex),原 `plugins/live/` 已迁出至 `bt_studio/trade/`。下述 API 契约描述 `bt_studio/server/` 网关。

## Client Applications (Separated)

Client apps live in a separate repository: `~/starup/bt_clients/` (note: `bt_clients`)

- **iOS App** (`ios_app/`): Native Swift monitoring app. HTTP + WebSocket.
- **Qt App** (`qt/`): Desktop monitoring dashboard. HTTP + WebSocket.

## API Contract

### Active endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check (watcher status + registry sizes) |
| GET | `/api/results` | List watched artifact kinds + directories + counts |
| GET | `/api/results/{kind}` | List artifacts of `kind` (mtime desc) |
| GET | `/api/results/{kind}/{name}` | Detail: full JSON for `collapse`/`llm_run`; stat info for `model`/`score` (never unpickles) |
| WS | `/ws/{client_id}` | Real-time updates |

`kind` ∈ `model` | `score` | `collapse` | `llm_run` | `feature`.

### WebSocket events

| Event | Payload |
|-------|---------|
| `connected` | client_id |
| `initial_state` | `{results: {kind: [records]}}` snapshot |
| `artifact_created` / `artifact_updated` | `{kind, name, path, mtime, size, summary}` |
| `collapse_report` | same payload + verdict summary (clients may alert) |
| `pong` / `snapshot` | replies to `{action: "ping"}` / `{action: "refresh"}` |

### Deprecated (410 Gone)

`POST /api/tasks`, `POST /api/hpo`, `POST /api/inference`, `DELETE /api/tasks/{id}`, `GET /api/tasks/{id}/result`, `GET /api/stats` — execution moved to in-process `bt_studio.orchestrator`.

## Collapse reports

Every HPO run (`node_tune_monthly`) auto-validates its parameter space and writes `result/tune/collapse/collapse_{model_id}_{feature_col}.json`:

```json
{
  "verdict": "healthy_plateau | collapsed | isolated_spike | flat_landscape",
  "feature_col": "...", "model_id": 202412,
  "n_trials": 400, "search_bounds": {...},
  "diagnostics": {"downsample": {...}, "threshold_r": {...}}
}
```

The watcher pushes a `collapse_report` event as soon as a new report lands.