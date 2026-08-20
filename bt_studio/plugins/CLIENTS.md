# bt_studio Plugins

This directory contains server-side plugins that are part of the core `bt_studio` package.

## Layering (v3 refactor)

```
bt_studio/
├── pipeline/ + tune.py     # core compute (agent, features, diagnostics)
├── orchestrator/           # core HPO serial queue (AsyncTaskManager) — NOT a plugin
└── plugins/
    ├── api_server/         # thin resident gateway: polls result dirs → WS/REST push
    └── xtp_client/         # ZMQ trading client — zero bt_studio.* imports (true plugin)
```

Dependency rules (enforced by `tests/test_architecture.py`):

- `bt_studio.{pipeline,tune,orchestrator,visual,utils}` must NOT import `bt_studio.plugins.*`
- `plugins.api_server` may import core, must NOT import `xtp_client`
- `plugins.xtp_client` must have zero `bt_studio` imports

## Available Plugins

- **`api_server/`**: FastAPI HTTP + WebSocket gateway. **Resident results-watching service**: polls `result/tune/{models,scores,collapse}`, `result/llm/runs` and `result/features` every N seconds (`BT_STUDIO_WATCH_INTERVAL`, default 5s), analyzes new artifacts and pushes them to iOS/Qt clients. It no longer hosts HPO execution — task submission lives in-process in `bt_studio.orchestrator` (scripts call `get_task_manager()` directly).
- **`xtp_client/`**: ZMQ-based trading client for live order execution. Fully self-contained (asyncio + zmq + codec only) — deliberately kept as the reference shape of a "real plugin".

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