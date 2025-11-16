"""MLX OpenAI Server Orchestrator CLI.

Command-line interface to start, stop, list, and inspect multiple MLX OpenAI
server worker instances on the same host. It manages per-process logging,
PID metadata, and port assignment so multiple servers can coexist.
"""

import argparse
import asyncio
import atexit
import contextlib
import json
import logging
import multiprocessing
import multiprocessing.util as mp_util
import os
from pathlib import Path
import re
import signal
import sys
import time

from loguru import logger

from . import paths
from .const import DEFAULT_STARTING_PORT
from .model_registry import ModelRegistryError, get_registry


def configure_cli_logging() -> None:
    """Configure logging for the CLI (stdout).

    This removes any existing Loguru handlers and sets up a simple
    stdout logger used when the CLI is executed interactively.
    """
    with contextlib.suppress(Exception):
        logger.remove()
    logger.add(sys.stdout, format="{message}", colorize=False)


def refresh_model_registry() -> None:
    """Reload the global `REGISTRY` of models, exiting on error.

    If the registry reload fails (invalid or missing config), log the
    error and exit the process with a non-zero code.
    """
    try:
        get_registry().reload()
    except ModelRegistryError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc


def ensure_models_file_exists() -> None:
    """Exit with a helpful error if the models.yaml file is missing."""

    config_path = paths.models_config_file()
    if not config_path.exists():
        logger.error(
            f"Missing models configuration: {config_path}. Copy models.yaml-example or create one.",
        )
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser for `mlx-server-orch`.

    The parser contains subcommands for starting, stopping and listing
    model server processes.
    """
    parser = argparse.ArgumentParser(
        prog="mlx-server-orch",
        description="MLX OpenAI Server Orchestrator — manage MLX model server processes.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser(
        "start",
        help="Start one or more models (runs in background).",
    )
    start_parser.add_argument(
        "names",
        nargs="*",
        metavar="name",
        help="Model nickname(s) to start.",
    )

    parser.start_parser = start_parser

    stop_parser = subparsers.add_parser(
        "stop",
        help="Stop running model processes.",
    )
    stop_parser.add_argument(
        "names",
        nargs="*",
        metavar="name",
        help="Model nickname(s) to stop (default: all running).",
    )

    subparsers.add_parser("models", help="List available model nicknames and paths.")
    subparsers.add_parser("status", help="Show running models and their PIDs.")
    subparsers.add_parser("help", help="Show this help message.")

    return parser


def _process_worker(name, model):
    """Entry point target for spawned worker processes.

    This configures per-process logging, registers PID cleanup and
    invokes the application entry point (`app.main.start`) for the
    given model configuration.
    """
    # Load registry first to set the correct base_path in this child process
    refresh_model_registry()

    try:
        configure_process_logging(name=name)
    except (OSError, PermissionError) as exc:  # pragma: no cover - logging best-effort
        logger.exception(f"Failed to configure process logging for {name}: {exc}")

    with contextlib.suppress(Exception):
        os.setsid()

    _register_pid_cleanup(name)

    from mlx_openai_server.main import start as _start  # noqa: PLC0415

    asyncio.run(_start(model))


def configure_process_logging(name: str | None = None):
    """Configure logging for a worker process.

    This redirects stdout/stderr to a per-process log file and
    configures Loguru to receive logging records. The function also
    registers an `atexit` handler to flush and close the file.
    """
    log_root = paths.log_root()
    log_root.mkdir(parents=True, exist_ok=True)

    pid = os.getpid()
    suffix = name or str(pid)

    app_log = log_root / f"{suffix}-app.log"

    app_file = app_log.open("a", buffering=1, encoding="utf-8")

    os.dup2(app_file.fileno(), 1)
    os.dup2(app_file.fileno(), 2)

    sys.stdout = app_file
    sys.stderr = app_file

    with contextlib.suppress(Exception):
        logger.remove()
    logger.add(str(app_log), backtrace=True, diagnose=True, enqueue=False, colorize=False)

    ansi_re = re.compile(r"\x1B?\[[0-9;]*[mK]")

    def _strip_ansi(s: str) -> str:
        return ansi_re.sub("", s)

    class InterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                level = logger.level(record.levelname).name
            except (LookupError, ValueError):
                level = record.levelno
            msg = record.getMessage()
            msg = _strip_ansi(msg)
            logger.opt(depth=6, exception=record.exc_info).log(level, msg)

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    def _close_files():
        with contextlib.suppress(Exception):
            app_file.flush()
        with contextlib.suppress(Exception):
            app_file.close()

    atexit.register(_close_files)


def _register_pid_cleanup(name: str) -> None:
    """Register an atexit handler to remove the PID metadata file.

    Ensures runtime directories exist and registers cleanup so that the
    PID file is removed when the process exits normally.
    """
    ensure_runtime_dirs()
    pid_path = pid_file(name)

    def _cleanup():
        with contextlib.suppress(FileNotFoundError):
            pid_path.unlink()

    atexit.register(_cleanup)


def pid_file(name: str) -> Path:
    """Return the path to the PID metadata file for a given model name."""
    return paths.pid_dir() / f"{name}.json"


def ensure_runtime_dirs() -> None:
    """Create runtime directories for logs and PID metadata.

    This is safe to call multiple times; it will create the directories
    if they do not already exist.
    """
    paths.log_root().mkdir(parents=True, exist_ok=True)
    paths.pid_dir().mkdir(parents=True, exist_ok=True)


def assign_ports(
    models: dict[str, object],
    reserved_ports: set[int] | None = None,
    starting_port: int | None = None,
) -> dict[str, object]:
    """Assign ports to a mapping of models, avoiding reserved ports.

    Existing model port attributes are preserved if valid and not
    conflicting. A new port (starting at the configured `starting_port`)
    is assigned when needed and the model object is mutated to include
    its port. Returns the updated mapping.
    """
    used_ports: dict[str, int] = {}
    claimed_ports: set[int] = set(reserved_ports or set())
    base_port = starting_port if starting_port is not None else DEFAULT_STARTING_PORT

    for name, model in models.items():
        port = getattr(model, "port", None)
        if port and port != 8000 and port not in claimed_ports:
            claimed_ports.add(port)
            used_ports[name] = port

    for name, model in models.items():
        port = getattr(model, "port", None)
        if port is None or port == 8000 or used_ports.get(name) != port:
            temp_port = base_port
            while temp_port in claimed_ports:
                temp_port += 1
            setattr(model, "port", temp_port)
            claimed_ports.add(temp_port)
            used_ports[name] = temp_port
            logger.info(
                f"Assigning port {getattr(model, 'port', 'unknown')} to {name} "
                f"({getattr(model, 'model_path', 'unknown')})"
            )

    return models


def write_pid_metadata(name: str, pid: int, model) -> None:
    """Write JSON metadata for a running model process to the PID dir.

    The metadata contains the PID, model name, model path, assigned
    port and start timestamp.
    """
    ensure_runtime_dirs()
    metadata = {
        "pid": pid,
        "name": name,
        "model_path": getattr(model, "model_path", ""),
        "port": getattr(model, "port", None),
        "started_at": time.time(),
    }
    pid_file(name).write_text(json.dumps(metadata))


def load_pid_metadata(name: str) -> dict | None:
    """Load and return PID metadata for `name` or None if missing/corrupt.

    If the PID file is corrupt it is removed and None is returned.
    """
    try:
        return json.loads(pid_file(name).read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        logger.warning(f"PID metadata for {name} is corrupt; removing.")
        with contextlib.suppress(FileNotFoundError):
            pid_file(name).unlink()
        return None


def discover_running_models() -> dict[str, dict]:
    """Discover running models by reading PID metadata files.

    Returns a mapping of model name -> metadata dictionary for each
    valid PID metadata file found in the PID directory.
    """
    ensure_runtime_dirs()
    running = {}
    for path in paths.pid_dir().glob("*.json"):
        name = path.stem
        meta = load_pid_metadata(name)
        if not meta:
            continue
        running[name] = meta
    return running


def process_alive(pid: int) -> bool:
    """Return True if a process with `pid` appears to be alive.

    Uses `os.kill(pid, 0)` which raises `ProcessLookupError` if no such
    process exists. Permission errors are treated as 'alive'.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def start_models(names: list[str] | None, detach: bool = True) -> None:
    """Start the requested models as background processes.

    If `names` is None the registry's default models are started. Ports
    are assigned to avoid collisions with existing running processes.
    When `detach` is True the launcher exits after spawning children.
    """
    refresh_model_registry()
    registry = get_registry()
    ensure_runtime_dirs()
    running = discover_running_models()
    reserved_ports: set[int] = set()
    for meta in running.values():
        pid = meta.get("pid")
        port = meta.get("port")
        if not port:
            continue
        if pid and not process_alive(pid):
            continue
        reserved_ports.add(port)
    try:
        requested_models = registry.build_model_map(names or None)
    except ModelRegistryError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc

    starting_port = registry.starting_port() or DEFAULT_STARTING_PORT
    models = assign_ports(
        requested_models,
        reserved_ports=reserved_ports,
        starting_port=starting_port,
    )
    processes: list[tuple[str, multiprocessing.Process]] = []
    started = 0

    for name, model in models.items():
        existing = running.get(name)
        if existing and process_alive(existing["pid"]):
            logger.warning(f"Model {name} already running with PID {existing['pid']}")
            continue

        logger.info(
            f"Starting MLX server {name} ({getattr(model, 'model_path', 'unknown')}) "
            f"on port {getattr(model, 'port', 'unknown')}"
        )
        proc = multiprocessing.Process(
            target=_process_worker,
            args=(name, model),
            name=f"server-{name}",
        )
        proc.start()
        write_pid_metadata(name, proc.pid, model)
        if detach:
            _detach_process(proc)
        else:
            processes.append((name, proc))
        started += 1

    if started == 0:
        logger.info("No models started.")
        return

    if detach:
        logger.info(f"Started {started} model(s)")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    supervise_processes(processes)


def _detach_process(proc: multiprocessing.Process) -> None:
    """Detach a child process from the multiprocessing parent tracking.

    This prevents the parent process from attempting to join the child
    when the parent exits.
    """
    with contextlib.suppress(Exception):
        mp_util._children.discard(proc)  # noqa: SLF001 - need to avoid auto-join on exit


def supervise_processes(processes: list[tuple[str, multiprocessing.Process]]) -> None:
    """Supervise and gracefully shutdown foreground child processes.

    Installs SIGINT/SIGTERM handlers to forward shutdown to children and
    waits for them to exit, terminating if necessary.
    """
    names = ", ".join(name for name, _ in processes)
    logger.info(f"Supervising model processes: {names}")

    def _shutdown(signum, frame):  # noqa: ARG001
        logger.info(f"Shutdown signal received ({signum}); stopping models.")
        for name, proc in processes:
            if proc.is_alive():
                try:
                    os.kill(proc.pid, signal.SIGINT)
                except ProcessLookupError:
                    logger.info(f"Process for {name} already exited.")
        for name, proc in processes:
            proc.join(timeout=10)
            if proc.is_alive():
                logger.warning(f"Process {name} did not exit; terminating.")
                proc.terminate()
                proc.join(timeout=5)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        for _, proc in processes:
            proc.join()
    finally:
        for name, _ in processes:
            with contextlib.suppress(FileNotFoundError):
                pid_file(name).unlink()


def stop_models(names: list[str]) -> None:
    """Stop running model processes specified by `names`.

    If `names` is empty, all discovered running models will be targeted.
    PID metadata is removed for stopped or stale entries.
    """
    refresh_model_registry()
    running = discover_running_models()
    target_names = names or list(running.keys())
    if not target_names:
        logger.info("No running models found.")
        return

    for name in target_names:
        meta = running.get(name)
        if not meta:
            logger.warning(f"Model {name} is not running.")
            continue
        pid = meta["pid"]
        if not process_alive(pid):
            logger.info(f"Model {name} had stale PID {pid}; cleaning up.")
            with contextlib.suppress(FileNotFoundError):
                pid_file(name).unlink()
            continue
        logger.info(f"Stopping model {name} (pid={pid})...")
        _terminate_process(pid)
        with contextlib.suppress(FileNotFoundError):
            pid_file(name).unlink()


def _terminate_process(pid: int) -> None:
    """Attempt to terminate `pid` gracefully, escalating to SIGKILL.

    Sends SIGINT then SIGTERM with short waits, finally SIGKILL where
    available if the process remains alive.
    """
    for sig, timeout in (
        (signal.SIGINT, 5),
        (signal.SIGTERM, 5),
    ):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return
        _wait_for_exit(pid, timeout)
        if not process_alive(pid):
            return

    if hasattr(signal, "SIGKILL"):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        _wait_for_exit(pid, 5)


def _wait_for_exit(pid: int, timeout: int) -> None:
    """Block until `pid` exits or `timeout` seconds elapse."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not process_alive(pid):
            return
        time.sleep(0.2)


def show_models() -> None:
    """Log all configured models and their filesystem paths.

    Default models are annotated in the output.
    """
    refresh_model_registry()
    registry = get_registry()
    default_names = set(registry.default_names())
    for entry in registry.all_entries():
        label = " (default)" if entry.name in default_names else ""
        logger.info(f"{entry.name}{label} -> {getattr(entry.config, 'model_path', 'unknown')}")


def status_models() -> None:
    """Report status for discovered running models (running/stopped)."""
    refresh_model_registry()
    running = discover_running_models()
    if not running:
        logger.info("No models are currently running.")
        return

    for name, meta in running.items():
        pid = meta.get("pid")
        port = meta.get("port")
        alive = process_alive(pid) if pid else False
        state = "running" if alive else "stopped"
        logger.info(f"{name} -> {state} (pid={pid}, port={port})")
        if not alive:
            with contextlib.suppress(FileNotFoundError):
                pid_file(name).unlink()


def show_help(parser: argparse.ArgumentParser) -> None:
    """Print help for the main CLI and additional start options.

    Prints the primary `argparse` help and, if available, prints
    help for the `start` subparser to show its extended options.
    """
    parser.print_help()
    start_parser = getattr(parser, "start_parser", None)
    if start_parser:
        logger.info("")
        logger.info("Start command options:")
        start_parser.print_help()


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for `mlx-server-orch`.

    Parses CLI arguments and dispatches to the appropriate subcommand
    handler (start/stop/models/status/help).
    """
    configure_cli_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {"start", "stop", "models", "status"}:
        ensure_models_file_exists()

    if args.command == "start":
        start_models(args.names, detach=True)
    elif args.command == "stop":
        stop_models(args.names)
    elif args.command == "models":
        show_models()
    elif args.command == "status":
        status_models()
    elif args.command == "help":
        show_help(parser)
    else:  # pragma: no cover - argparse enforces
        parser.print_help()


if __name__ == "__main__":
    main()
