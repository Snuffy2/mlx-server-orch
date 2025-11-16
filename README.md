# MLX OpenAI Server Orchestrator

Wraps [MLX OpenAI Server](https://github.com/cubist38/mlx-openai-server) workers in a small CLI so you can launch and manage multiple OpenAI-compatible back-ends on the same machine. All model metadata lives in a single YAML file.

If a port is defined, it will use that port (if not already in use). Otherwise it will assign a port sequentially starting from `starting_port`.

## models.yaml

**base_path:** _(optional, defaults to `~/mlx-server-orch`)_ — root directory where `logs/` and `pids/` are written. Relative paths are resolved against the config file's location.

**starting_port:** _(optional, defaults to 5005)_

**models:**

| Field | Required | Default | Description |
| --- | :---: | --- | --- |
| name | ✓ |  | Name used by the Orchestrator (e.g., qwen3_30b) |
| model_path | ✓ |  | Huggingface model name (e.g., owner/model-name) |
| default |  | false | If true, the model is started when `start` is run without names |
| model_type |  | "lm" | |
| context_length |  | 32768 | Maximum token context length for the model |
| port |  | `auto-assigned` | TCP port the server will listen on; auto-assigned if omitted |
| host |  | "0.0.0.0" | Host/IP to bind the server to |
| max_concurrency |  | 1 | Max simultaneous requests the worker will accept |
| queue_timeout |  | 300 | How long (seconds) to wait for queued requests |
| queue_size |  | 100 | Maximum number of queued requests |
| disable_auto_resize |  | false | |
| quantize |  | 8 | |
| config_name |  |  | |
| lora_paths |  |  | |
| lora_scales |  |  | |
| log_file |  | <base_path>/logs/<name>.log | Path and name of per-model log file |
| no_log_file |  | false | When true, suppress writing a per-model log file |
| log_level |  | "INFO" | Logging level for the worker process |
| enable_auto_tool_choice |  | false | Allow the model to select tools automatically when available |
| tool_call_parser |  |  | |
| reasoning_parser |  |  | |
| trust_remote_code |  | false | Allow execution of remote model code when loading (use with caution) |

If you omit `log_file` (and `no_log_file` is `false`) the loader writes logs to `<base_path>/logs/<name>.log` automatically.

### Example snippet from `models.yaml`:

```yaml
base_path: ~/mlx-server-orch
starting_port: 5005
models:
  - name: qwen3_30b
    model_path: mlx-community/Qwen3-30B-A3B-4bit-DWQ
    default: true
    enable_auto_tool_choice: true
    trust_remote_code: true
    tool_call_parser: qwen3_moe
    reasoning_parser: qwen3_moe

  - name: medgemma_4b
    model_path: mlx-community/medgemma-4b-it-4bit
    enable_auto_tool_choice: true
    trust_remote_code: true
```

## Configuration File Location

The `models.yaml` file is searched for in the following order:

1. Path specified via `--config` CLI argument
2. Path specified in `MLXSERVER_MODELS_PATH` environment variable
3. `models.yaml` in the current working directory
4. `models.yaml` in the default base path (`~/mlx-server-orch`)

## CLI commands

All commands run through `mlx-server-orch <command>`. The `start` command launches child processes and runs detached. All commands support the global `--config` option to specify the path to `models.yaml`, or set the `MLXSERVER_MODELS_PATH` environment variable.

| Command | Description |
| --- | --- |
| `start [name ...]` | Start one or more models. Without names, every entry marked `default: true` is launched. |
| `stop [name ...]` | Stops the named models. Without names, every model is stopped. |
| `models` | Lists all configured models, marking which entries are defaults. |
| `status` | Shows which models are running along with PID and port information. |
| `help` | Prints CLI usage plus the `start` command-specific help. |

### Typical workflow

```bash
mlx-server-orch models
mlx-server-orch start                   # starts every default model
mlx-server-orch start medgemma_4b       # start one additional model
mlx-server-orch status                  # inspect running servers
mlx-server-orch stop medgemma_4b        # stop one model
```

You can specify a custom config file:

```bash
mlx-server-orch --config /path/to/models.yaml start
# or
MLXSERVER_MODELS_PATH=/path/to/models.yaml mlx-server-orch start
```

* Each started model writes logs to `<base_path>/logs/` (and per-process PID files in `<base_path>/pids/`).
* Edit `models.yaml` whenever you need to add, remove, or retune a model


## Local Install for Testing and Development

### Install & run

Install the CLI into a virtual environment (recommended) and run the
`mlx-server-orch` command:

```bash
git clone https://github.com/Snuffy2/mlx-server-orch.git
cd mlx-server-orch
python -m venv .venv
./.venv/bin/python -m pip install -e .
source .venv/bin/activate
# then run the installed CLI
mlx-server-orch models
```

Or install system-wide (or in your active environment):

```bash
pip install .
mlx-server-orch start
```

If you prefer not to install, you can run the CLI directly from the repo using
the virtualenv interpreter:

```bash
./.venv/bin/python -m mlx_server_orch.main models
```

### Build & package

A minimal example to build a wheel locally and install it for testing.

```bash
# build a wheel (requires the `build` package)
python -m pip install --upgrade build
python -m build --wheel

# install the produced wheel locally
pip install dist/*.whl
```
