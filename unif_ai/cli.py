"""Command-line launcher and headless batch runner for unif_ai."""

import argparse
import json
from pathlib import Path
from typing import Any, Final, Sequence
from wsgiref.simple_server import WSGIServer, make_server

from unif_ai.constants import (
    DEFAULT_TERRAIN_SECTOR,
    DEFAULT_WEATHER_STATE,
    ONE_INT,
    TRANSPORT_UAV_ROTARY,
    TWO_INT,
    VALID_TRANSPORT_MODES,
    ZERO_INT,
)
from unif_ai.security import load_safe_yaml_config, verify_condition
from unif_ai.server import (
    TacticalDesktopController,
    UnifAiWsgiApp,
    build_application_from_config,
)

DEFAULT_BIND_HOST: Final[str] = "127.0.0.1"
DEFAULT_BIND_PORT: Final[int] = 8080
MIN_UNPRIVILEGED_PORT: Final[int] = 1024
MAX_TCP_PORT: Final[int] = 65535
MODE_SERVE: Final[str] = "serve"
MODE_BATCH: Final[str] = "batch"
VALID_CLI_MODES: Final[tuple[str, ...]] = (MODE_SERVE, MODE_BATCH)
DEFAULT_BATCH_ID: Final[str] = "relief_flood_001"
DEFAULT_OWNER_ID: Final[str] = "responder_alpha"
DEFAULT_SUBJECT_ID: Final[str] = "person_tag_01"


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser for unif_ai.

    Returns:
        Configured ArgumentParser instance.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="unif_ai.cli",
        description="Launch UnifAiWsgiApp or run a headless relief batch.",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_CLI_MODES,
        default=MODE_SERVE,
        help="Execution mode: 'serve' starts WSGI; 'batch' runs one batch.",
    )
    parser.add_argument(
        "--workspace",
        default="./workspace",
        help="Path to the contained runtime workspace directory.",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Optional relative path to config.yaml inside workspace.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_BIND_HOST,
        help="Loopback address to bind the WSGI HTTP server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_BIND_PORT,
        help="Unprivileged TCP port for the WSGI HTTP server.",
    )
    parser.add_argument(
        "--batch-id",
        default=DEFAULT_BATCH_ID,
        help="Disaster relief batch identifier for headless execution.",
    )
    parser.add_argument(
        "--owner-id",
        default=DEFAULT_OWNER_ID,
        help="Authenticated responder ID for BOLA ownership.",
    )
    parser.add_argument(
        "--subject-id",
        default=DEFAULT_SUBJECT_ID,
        help="Affected person tag to pseudonymise via HKDF-SHA256.",
    )
    parser.add_argument(
        "--transport-mode",
        choices=VALID_TRANSPORT_MODES,
        default=TRANSPORT_UAV_ROTARY,
        help="Disaster relief transport mode for batch execution.",
    )
    parser.add_argument(
        "--terrain-sector",
        default=DEFAULT_TERRAIN_SECTOR,
        help="Disaster zone sector identifier.",
    )
    return parser


def run_headless_demo_batch(
    workspace_root: Path,
    config_rel_path: str,
    batch_id: str,
    owner_id: str,
    subject_id: str,
    transport_mode: str = TRANSPORT_UAV_ROTARY,
    terrain_sector: str = DEFAULT_TERRAIN_SECTOR,
) -> dict[str, Any]:
    """Execute a 4-stage relief batch and verify the SHA-256 audit ledger.

    Args:
        workspace_root: Trusted workspace root directory.
        config_rel_path: Optional relative config path inside workspace_root.
        batch_id: Mission batch identifier string.
        owner_id: Authenticated responder identifier string.
        subject_id: Raw affected person identifier to pseudonymise.
        transport_mode: Selected disaster relief transport vehicle mode.
        terrain_sector: Target disaster zone sector identifier.

    Returns:
        Dictionary containing the batch execution report and status.
    """
    resolved_root: Path = workspace_root.resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    cfg_dict: dict[str, Any] | None = None
    if len(config_rel_path.strip()) > ZERO_INT:
        cfg_dict = load_safe_yaml_config(resolved_root, config_rel_path)
    controller: TacticalDesktopController = TacticalDesktopController(
        workspace_root=resolved_root,
        config=cfg_dict,
    )
    batch_outcome: dict[str, Any] = controller.run_mission_batch(
        batch_id=batch_id,
        owner_id=owner_id,
        raw_subject_id=subject_id,
        transport_mode=transport_mode,
        terrain_sector=terrain_sector,
        weather_state=DEFAULT_WEATHER_STATE,
    )
    system_status: dict[str, Any] = controller.get_system_status()
    verify_condition(
        bool(system_status.get("ledger_chain_valid", False)),
        "Accountability ledger SHA-256 chain verification failed.",
    )
    return {
        "batch_result": batch_outcome,
        "system_status": system_status,
    }


def serve_wsgi_application(
    workspace_root: Path,
    config_rel_path: str,
    host: str,
    port: int,
) -> None:
    """Bind and run UnifAiWsgiApp on the specified loopback host and port.

    Args:
        workspace_root: Trusted workspace root directory.
        config_rel_path: Optional relative path to config.yaml.
        host: Loopback hostname or IP address.
        port: Unprivileged TCP port between 1024 and 65535.
    """
    clean_host: str = host.strip()
    verify_condition(
        clean_host in ("127.0.0.1", "localhost"),
        "WSGI server must bind strictly to a loopback interface.",
    )
    verify_condition(
        MIN_UNPRIVILEGED_PORT <= port <= MAX_TCP_PORT,
        "WSGI server port must reside within unprivileged TCP bounds.",
    )
    resolved_root: Path = workspace_root.resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    cfg_arg: str | None = (
        config_rel_path.strip()
        if len(config_rel_path.strip()) > ZERO_INT
        else None
    )
    wsgi_app: UnifAiWsgiApp = build_application_from_config(
        workspace_root=resolved_root,
        config_path=cfg_arg,
    )
    server_instance: WSGIServer = make_server(clean_host, port, wsgi_app)
    print(
        f"UNIF_AI Disaster Relief Console active at http://{clean_host}:{port}"
    )
    try:
        server_instance.serve_forever()
    except KeyboardInterrupt:
        server_instance.server_close()


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the unif_ai command-line interface.

    Args:
        argv: Optional sequence of CLI argument strings.

    Returns:
        Process exit code (0 on success, 1 on failure).
    """
    parser: argparse.ArgumentParser = build_arg_parser()
    args: argparse.Namespace = parser.parse_args(argv)
    workspace_path: Path = Path(str(args.workspace))
    if args.mode == MODE_BATCH:
        report: dict[str, Any] = run_headless_demo_batch(
            workspace_root=workspace_path,
            config_rel_path=str(args.config),
            batch_id=str(args.batch_id),
            owner_id=str(args.owner_id),
            subject_id=str(args.subject_id),
            transport_mode=str(args.transport_mode),
            terrain_sector=str(args.terrain_sector),
        )
        print(json.dumps(report, indent=TWO_INT, sort_keys=True))
        return ZERO_INT
    serve_wsgi_application(
        workspace_root=workspace_path,
        config_rel_path=str(args.config),
        host=str(args.host),
        port=int(args.port),
    )
    return ONE_INT


if __name__ == "__main__":
    raise SystemExit(main())
