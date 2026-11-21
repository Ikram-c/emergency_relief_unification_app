"""Hardened WSGI server and headless MVC controller for unif_ai."""

import io
import json
import sqlite3
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from unif_ai.constants import (
    ALL_CV_APPLICATIONS,
    APP_DOMAIN_TITLE,
    APP_NAME,
    APP_VERSION,
    DEFAULT_REFERENCE_LAYERS,
    DEFAULT_TERRAIN_SECTOR,
    DEFAULT_WEATHER_STATE,
    ENCODING_UTF8,
    FIVE_INT,
    FOUR_INT,
    HTTP_REASON_MAP,
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_NOT_FOUND,
    HTTP_STATUS_OK,
    LOG_DETAIL_MAX_LENGTH,
    LOG_LEVEL_INFO,
    LOG_RING_BUFFER_CAPACITY,
    MAX_PORTAL_BATCH_RESULTS,
    MAX_QUERY_PARAM_LENGTH,
    MAX_REQUEST_BODY_BYTES,
    ONE_INT,
    PASSWORD_Reset_TTL_SECONDS,
    RATE_LIMIT_REQUESTS_DEFAULT,
    RATE_LIMIT_WINDOW_SECONDS,
    SLOT_BLUE,
    THREE_INT,
    TRANSPORT_UAV_ROTARY,
    TWO_INT,
    VALID_TRANSPORT_MODES,
    ZERO_INT,
)
from unif_ai.gdpr import (
    AccountabilityLedger,
    execute_right_to_erasure,
    verify_uk_eu_residency,
)
from unif_ai.models import PipelineBatchState
from unif_ai.pipeline import (
    UnifAiPipelineOrchestrator,
    retrieve_tactical_rag_advisory,
)
from unif_ai.predeploy import (
    BlueGreenDeploymentManager,
    CostCappedRingLogger,
    PasswordResetTicket,
    PasswordResetTokenManager,
    PreDeployHttpError,
    SlidingWindowRateLimiter,
    TargetedDatabaseStore,
    evaluate_cors_origin,
    execute_with_error_boundary,
    sanitize_and_validate_input,
)
from unif_ai.sandbox import CloudSandboxClient
from unif_ai.security import (
    load_safe_yaml_config,
    parse_safe_json_payload,
    resolve_safe_path,
    verify_condition,
)

StartResponseCallable = Callable[
    [str, list[tuple[str, str]]],
    Callable[[bytes], Any],
]


class TacticalDesktopController:
    """Headless-compatible MVC controller for disaster relief pipelines."""

    def __init__(
        self,
        workspace_root: Path,
        config: Mapping[str, Any] | None = None,
        master_secret: str | None = None,
    ) -> None:
        """Initialize the MVC controller and all security/GDPR subsystems.

        Args:
            workspace_root: Trusted workspace root directory.
            config: Optional parsed configuration dictionary.
            master_secret: Optional override secret for deterministic tests.
        """
        self._workspace_root: Path = workspace_root.resolve()
        self._workspace_root.mkdir(parents=True, exist_ok=True)
        self._config: dict[str, Any] = (
            dict(config)
            if config is not None
            else self._default_configuration()
        )
        self._master_secret: str | None = master_secret
        pre_cfg: Mapping[str, Any] = self._config.get("predeploy", {})
        gdpr_cfg: Mapping[str, Any] = self._config.get("gdpr", {})
        app_cfg: Mapping[str, Any] = self._config.get("application", {})
        self._logger: CostCappedRingLogger = CostCappedRingLogger(
            capacity=int(
                pre_cfg.get(
                    "log_ring_buffer_capacity",
                    LOG_RING_BUFFER_CAPACITY,
                )
            ),
            max_detail_length=int(
                pre_cfg.get("log_detail_max_length", LOG_DETAIL_MAX_LENGTH)
            ),
        )
        self._rate_limiter: SlidingWindowRateLimiter = (
            SlidingWindowRateLimiter(
                max_requests=int(
                    pre_cfg.get(
                        "rate_limit_requests_per_minute",
                        RATE_LIMIT_REQUESTS_DEFAULT,
                    )
                ),
                window_seconds=int(
                    pre_cfg.get(
                        "rate_limit_window_seconds",
                        RATE_LIMIT_WINDOW_SECONDS,
                    )
                ),
            )
        )
        self._reset_manager: PasswordResetTokenManager = (
            PasswordResetTokenManager(
                ttl_seconds=int(
                    pre_cfg.get(
                        "password_reset_ttl_seconds",
                        PASSWORD_Reset_TTL_SECONDS,
                    )
                ),
                epoch=str(gdpr_cfg.get("hkdf_epoch", "2026-Q4")),
                master_secret=self._master_secret,
            )
        )
        self._deployment: BlueGreenDeploymentManager = (
            BlueGreenDeploymentManager(
                logger=self._logger,
                initial_active_slot=str(
                    app_cfg.get("active_slot", SLOT_BLUE)
                ),
                initial_version=str(app_cfg.get("version", APP_VERSION)),
            )
        )
        self._ledger: AccountabilityLedger = AccountabilityLedger(
            workspace_root=self._workspace_root,
            ledger_path="ledger/gdpr_accountability.jsonl",
        )
        self._db_store: TargetedDatabaseStore = TargetedDatabaseStore(
            workspace_root=self._workspace_root,
            database_path="data/unif_ai_missions.sqlite3",
        )
        region: str = verify_uk_eu_residency(
            str(gdpr_cfg.get("cloud_region", "uk-south"))
        )
        self._sandbox_client: CloudSandboxClient = CloudSandboxClient(
            workspace_root=self._workspace_root,
            cloud_region=region,
        )
        self._orchestrator: UnifAiPipelineOrchestrator = (
            UnifAiPipelineOrchestrator(
                workspace_root=self._workspace_root,
                config=self._config,
                ledger=self._ledger,
                db_store=self._db_store,
                sandbox_client=self._sandbox_client,
                master_secret=self._master_secret,
            )
        )

    @property
    def logger(self) -> CostCappedRingLogger:
        """Return the cost-capped production ring logger."""
        return self._logger

    @property
    def rate_limiter(self) -> SlidingWindowRateLimiter:
        """Return the sliding-window API rate limiter."""
        return self._rate_limiter

    @property
    def ledger(self) -> AccountabilityLedger:
        """Return the SHA-256 hash-chained accountability ledger."""
        return self._ledger

    @property
    def db_store(self) -> TargetedDatabaseStore:
        """Return the targeted B-tree indexed SQLite store."""
        return self._db_store

    @property
    def allowed_cors_origins(self) -> tuple[str, ...]:
        """Return the configured strict CORS origin allowlist."""
        sec_cfg: Mapping[str, Any] = self._config.get("security", {})
        origins: Sequence[str] = sec_cfg.get(
            "allowed_cors_origins",
            ("http://127.0.0.1:8080", "http://localhost:8080"),
        )
        return tuple(str(origin) for origin in origins)

    def run_mission_batch(
        self,
        batch_id: str,
        owner_id: str,
        raw_subject_id: str,
        transport_mode: str = TRANSPORT_UAV_ROTARY,
        terrain_sector: str = DEFAULT_TERRAIN_SECTOR,
        weather_state: str = DEFAULT_WEATHER_STATE,
    ) -> dict[str, Any]:
        """Validate inputs, execute the 4-stage pipeline, and return summary.

        Args:
            batch_id: Untrusted mission batch identifier.
            owner_id: Authenticated responder identifier.
            raw_subject_id: Raw affected person identifier to pseudonymise.
            transport_mode: Selected disaster relief transport mode.
            terrain_sector: Selected disaster zone sector.
            weather_state: Selected weather condition state.

        Returns:
            Dictionary containing the completed batch summary.
        """
        clean_batch: str = sanitize_and_validate_input(batch_id)
        clean_owner: str = sanitize_and_validate_input(owner_id)
        clean_subject: str = sanitize_and_validate_input(raw_subject_id)
        clean_transport: str = sanitize_and_validate_input(transport_mode)
        clean_sector: str = sanitize_and_validate_input(terrain_sector)
        clean_weather: str = sanitize_and_validate_input(weather_state)
        completed_state: PipelineBatchState = self._orchestrator.run_batch(
            batch_id=clean_batch,
            owner_id=clean_owner,
            raw_subject_id=clean_subject,
            transport_mode=clean_transport,
            terrain_sector=clean_sector,
            weather_state=clean_weather,
        )
        self._logger.log_event(
            level=LOG_LEVEL_INFO,
            component="mvc_controller",
            detail=f"Completed disaster relief batch {clean_batch}.",
        )
        return {
            "active_slot": self._deployment.active_slot,
            "batch": completed_state.to_summary_dict(),
            "ok": True,
        }

    def get_mission_batch(
        self,
        batch_id: str,
        authenticated_user_id: str,
    ) -> dict[str, Any]:
        """Retrieve a stored relief batch with strict BOLA verification.

        Args:
            batch_id: Requested batch identifier.
            authenticated_user_id: Logged-in responder identifier.

        Returns:
            Dictionary containing the authorized mission batch record.
        """
        record: dict[str, Any] = self._db_store.get_mission_batch_for_user(
            batch_id=batch_id,
            authenticated_user_id=authenticated_user_id,
        )
        return {"batch": record, "ok": True}

    def list_operator_rescues(
        self,
        authenticated_user_id: str,
    ) -> dict[str, Any]:
        """List Portal Dashboard relief batches owned by the responder.

        Args:
            authenticated_user_id: Logged-in responder identifier.

        Returns:
            Dictionary with BOLA-filtered relief mission summaries.
        """
        clean_user: str = sanitize_and_validate_input(authenticated_user_id)
        with sqlite3.connect(str(self._db_store.database_path)) as conn:
            cursor: sqlite3.Cursor = conn.cursor()
            cursor.execute(
                "SELECT batch_id, owner_id, subject_token, cloud_region, "
                "summary_json, created_at FROM mission_batches "
                "WHERE owner_id = ? ORDER BY created_at DESC LIMIT ?",
                (clean_user, MAX_PORTAL_BATCH_RESULTS),
            )
            rows: list[tuple[Any, ...]] = cursor.fetchall()
        rescues: list[dict[str, Any]] = [
            {
                "batch_id": str(row[ZERO_INT]),
                "cloud_region": str(row[THREE_INT]),
                "created_at": float(row[FIVE_INT]),
                "owner_id": str(row[ONE_INT]),
                "subject_token": str(row[TWO_INT]),
                "summary": json.loads(str(row[FOUR_INT])),
            }
            for row in rows
        ]
        return {
            "count": len(rescues),
            "ok": True,
            "owner_id": clean_user,
            "rescues": rescues,
        }

    def query_rag_advisor(
        self,
        query_text: str,
        transport_mode: str = TRANSPORT_UAV_ROTARY,
        weather_state: str = DEFAULT_WEATHER_STATE,
    ) -> dict[str, Any]:
        """Query the zero-PII RAG disaster relief protocol advisor.

        Args:
            query_text: Responder tactical or disaster sector query string.
            transport_mode: Selected relief transport vehicle mode.
            weather_state: Weather condition descriptor.

        Returns:
            Dictionary containing the sanitized RAG tactical advisory.
        """
        clean_query: str = sanitize_and_validate_input(query_text)
        clean_transport: str = sanitize_and_validate_input(transport_mode)
        clean_weather: str = sanitize_and_validate_input(weather_state)
        advisory: str = retrieve_tactical_rag_advisory(
            query_or_sector=clean_query,
            transport_mode=clean_transport,
            weather_state=clean_weather,
        )
        return {
            "advisory": advisory,
            "ok": True,
            "query": clean_query,
            "transport_mode": clean_transport,
        }

    def get_reference_layers_catalog(self) -> dict[str, Any]:
        """Return the Reference Layers catalog and CV application registry.

        Returns:
            Dictionary describing available reference layers and CV apps.
        """
        return {
            "cv_applications": list(ALL_CV_APPLICATIONS),
            "ok": True,
            "reference_layers": list(DEFAULT_REFERENCE_LAYERS),
            "transport_modes": list(VALID_TRANSPORT_MODES),
        }

    def issue_password_reset(
        self,
        user_id: str,
        now_epoch: int | None = None,
    ) -> dict[str, Any]:
        """Issue a single-use 30-minute expiring password reset link token.

        Args:
            user_id: Responder identifier requesting password reset.
            now_epoch: Optional explicit integer epoch for testing.

        Returns:
            Dictionary with token metadata and expiration epoch.
        """
        ticket: PasswordResetTicket = self._reset_manager.issue_reset_token(
            user_id=user_id,
            now_epoch=now_epoch,
        )
        self._logger.log_event(
            level=LOG_LEVEL_INFO,
            component="auth_reset",
            detail=f"Issued 30-min reset token for {ticket.user_id}.",
        )
        return {
            "expires_at": ticket.expires_at,
            "issued_at": ticket.issued_at,
            "ok": True,
            "reset_token": ticket.serialized_token,
            "user_id": ticket.user_id,
        }

    def redeem_password_reset(
        self,
        reset_token: str,
        now_epoch: int | None = None,
    ) -> dict[str, Any]:
        """Redeem a single-use 30-minute password reset link token.

        Args:
            reset_token: Serialized token string from the reset link.
            now_epoch: Optional explicit integer epoch for testing.

        Returns:
            Confirmation dictionary when token is valid and unconsumed.
        """
        verified_user: str = self._reset_manager.redeem_reset_token(
            serialized_token=reset_token,
            now_epoch=now_epoch,
        )
        self._logger.log_event(
            level=LOG_LEVEL_INFO,
            component="auth_reset",
            detail=f"Redeemed single-use reset token for {verified_user}.",
        )
        return {
            "ok": True,
            "status": "PASSWORD_RESET_CONFIRMED",
            "user_id": verified_user,
        }

    def trigger_gdpr_erasure(self, raw_subject_id: str) -> dict[str, Any]:
        """Execute GDPR Article 17 Right to Erasure across JSON, Zarr, and DB.

        Args:
            raw_subject_id: Raw affected person identifier requesting erasure.

        Returns:
            Dictionary reporting deleted database rows and scrubbed files.
        """
        clean_subject: str = sanitize_and_validate_input(raw_subject_id)
        outcome: dict[str, int] = execute_right_to_erasure(
            raw_subject_id=clean_subject,
            workspace_root=self._workspace_root,
            output_dir="output",
            database_path=self._db_store.database_path,
            ledger=self._ledger,
            epoch=self._orchestrator.hkdf_epoch,
            master_secret=self._master_secret,
        )
        self._logger.log_event(
            level=LOG_LEVEL_INFO,
            component="gdpr_erasure",
            detail=(
                f"Executed Art. 17 erasure: "
                f"rows={outcome['deleted_db_rows']}, "
                f"files={outcome['scrubbed_json_files']}."
            ),
        )
        return {"erasure_report": outcome, "ok": True}

    def switch_deployment_slot(
        self,
        candidate_version: str,
        health_verified: bool = True,
    ) -> dict[str, Any]:
        """Stage a release in standby slot and cut over or auto-rollback.

        Args:
            candidate_version: Version string for the standby deployment.
            health_verified: Health check status of the standby slot.

        Returns:
            Blue-green deployment status dictionary.
        """
        report: dict[str, Any] = self._deployment.deploy_and_switch(
            candidate_version=candidate_version,
            health_check=lambda: bool(health_verified),
        )
        return {"deployment": report, "ok": True}

    def get_system_status(self) -> dict[str, Any]:
        """Return operational status across security, GDPR, and slots.

        Returns:
            Dictionary of health, residency, index, and ledger telemetry.
        """
        return {
            "active_slot": self._deployment.active_slot,
            "alert_count": len(self._logger.alerts),
            "app_domain_title": APP_DOMAIN_TITLE,
            "app_name": APP_NAME,
            "app_version": APP_VERSION,
            "cloud_region": self._orchestrator.cloud_region,
            "custom_db_indexes": list(self._db_store.list_custom_indexes()),
            "cv_applications": list(ALL_CV_APPLICATIONS),
            "enabled_stages": list(self._orchestrator.enabled_stages),
            "ledger_chain_valid": self._ledger.verify_chain(),
            "log_ring_count": len(self._logger.entries),
            "ok": True,
            "reference_layers": list(DEFAULT_REFERENCE_LAYERS),
            "rollback_count": self._deployment.rollback_count,
            "standby_slot": self._deployment.standby_slot,
            "transport_modes": list(VALID_TRANSPORT_MODES),
        }

    def _default_configuration(self) -> dict[str, Any]:
        """Return safe default configuration when config.yaml is absent."""
        return {
            "application": {
                "active_slot": SLOT_BLUE,
                "domain_title": APP_DOMAIN_TITLE,
                "name": APP_NAME,
                "standby_slot": "green",
                "version": APP_VERSION,
                "weights_path": "models/sobel_otsu_kernel.npz",
            },
            "gdpr": {
                "cloud_region": "uk-south",
                "dp_epsilon": 0.85,
                "hkdf_epoch": "2026-Q4",
                "spatial_k_min": 3,
                "staging_ttl_seconds": 3600,
            },
            "pipeline": {},
            "predeploy": {},
            "security": {
                "allowed_cors_origins": [
                    "http://127.0.0.1:8080",
                    "http://localhost:8080",
                ]
            },
        }


class UnifAiWsgiApp:
    """PEP 3333 WSGI application enforcing the 10-point pre-deploy gate."""

    def __init__(
        self,
        controller: TacticalDesktopController,
        static_root: Path | None = None,
    ) -> None:
        """Initialize the WSGI application router.

        Args:
            controller: TacticalDesktopController instance.
            static_root: Optional path to static UI assets directory.
        """
        self._controller: TacticalDesktopController = controller
        default_static: Path = Path(__file__).resolve().parent / "static"
        self._static_root: Path = (
            static_root.resolve()
            if static_root is not None
            else default_static
        )

    def handle_http_request(
        self,
        method: str,
        path: str,
        query_string: str = "",
        headers: Mapping[str, str] | None = None,
        body_text: str = "",
        client_ip: str = "127.0.0.1",
    ) -> tuple[int, dict[str, str], bytes, str]:
        """Process an HTTP request through the 10-point pre-deployment gate.

        Args:
            method: HTTP verb (GET, POST, OPTIONS).
            path: URL path component.
            query_string: Raw URL query string.
            headers: Request header dictionary.
            body_text: Decoded UTF-8 request body string.
            client_ip: Client IP address for sliding-window rate limiting.

        Returns:
            Tuple of (status_code, response_headers, body_bytes, content_type).
        """
        req_headers: dict[str, str] = (
            {str(k).lower(): str(v) for k, v in headers.items()}
            if headers is not None
            else {}
        )
        origin_header: str | None = req_headers.get("origin")
        cors_headers: dict[str, str] = {"Vary": "Origin"}

        def routed_operation() -> dict[str, Any]:
            nonlocal cors_headers
            cors_headers = evaluate_cors_origin(
                request_origin=origin_header,
                allowed_origins=self._controller.allowed_cors_origins,
            )
            self._controller.rate_limiter.check_and_record(
                client_id=client_ip
            )
            norm_method: str = method.strip().upper()
            clean_path: str = path.strip()
            if norm_method == "OPTIONS":
                return {"ok": True, "preflight": "ALLOWED"}
            if clean_path == "/api/status" and norm_method == "GET":
                return self._controller.get_system_status()
            if clean_path == "/api/layers" and norm_method == "GET":
                return self._controller.get_reference_layers_catalog()
            if (
                clean_path in ("/api/batch/run", "/api/rescue/create")
                and norm_method == "POST"
            ):
                payload: dict[str, Any] = self._parse_body_dict(body_text)
                return self._controller.run_mission_batch(
                    batch_id=str(payload.get("batch_id", "")),
                    owner_id=str(payload.get("owner_id", "")),
                    raw_subject_id=str(payload.get("subject_id", "")),
                    transport_mode=str(
                        payload.get("transport_mode", TRANSPORT_UAV_ROTARY)
                    ),
                    terrain_sector=str(
                        payload.get("terrain_sector", DEFAULT_TERRAIN_SECTOR)
                    ),
                    weather_state=str(
                        payload.get("weather_state", DEFAULT_WEATHER_STATE)
                    ),
                )
            if clean_path == "/api/batch" and norm_method == "GET":
                params: dict[str, str] = self._parse_query_params(
                    query_string
                )
                operator_id: str = req_headers.get(
                    "x-operator-id",
                    params.get("owner_id", ""),
                )
                return self._controller.get_mission_batch(
                    batch_id=params.get("batch_id", ""),
                    authenticated_user_id=operator_id,
                )
            if clean_path == "/api/rescues" and norm_method == "GET":
                params = self._parse_query_params(query_string)
                operator_id = req_headers.get(
                    "x-operator-id",
                    params.get("owner_id", ""),
                )
                return self._controller.list_operator_rescues(
                    authenticated_user_id=operator_id
                )
            if clean_path == "/api/rag/query" and norm_method == "POST":
                payload = self._parse_body_dict(body_text)
                return self._controller.query_rag_advisor(
                    query_text=str(payload.get("query", "")),
                    transport_mode=str(
                        payload.get("transport_mode", TRANSPORT_UAV_ROTARY)
                    ),
                    weather_state=str(
                        payload.get("weather_state", DEFAULT_WEATHER_STATE)
                    ),
                )
            if clean_path == "/api/reset/issue" and norm_method == "POST":
                payload = self._parse_body_dict(body_text)
                return self._controller.issue_password_reset(
                    user_id=str(payload.get("user_id", ""))
                )
            if clean_path == "/api/reset/redeem" and norm_method == "POST":
                payload = self._parse_body_dict(body_text)
                return self._controller.redeem_password_reset(
                    reset_token=str(payload.get("reset_token", ""))
                )
            if clean_path == "/api/gdpr/erase" and norm_method == "POST":
                payload = self._parse_body_dict(body_text)
                return self._controller.trigger_gdpr_erasure(
                    raw_subject_id=str(payload.get("subject_id", ""))
                )
            if clean_path == "/api/deploy/switch" and norm_method == "POST":
                payload = self._parse_body_dict(body_text)
                return self._controller.switch_deployment_slot(
                    candidate_version=str(payload.get("version", "")),
                    health_verified=bool(
                        payload.get("health_verified", True)
                    ),
                )
            raise PreDeployHttpError(
                HTTP_STATUS_NOT_FOUND,
                "Requested endpoint was not found on this server.",
            )

        if path.strip() in ("/", "/index.html", "/styles.css", "/app.js"):
            static_result: tuple[int, dict[str, str], bytes, str] | None = (
                self._serve_static_asset(path.strip(), origin_header)
            )
            if static_result is not None:
                return static_result

        status_code, response_dict = execute_with_error_boundary(
            operation=routed_operation,
            logger=self._controller.logger,
            component="wsgi_http_router",
        )
        encoded_json: bytes = json.dumps(
            response_dict,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(ENCODING_UTF8)
        return (
            status_code,
            cors_headers,
            encoded_json,
            "application/json; charset=utf-8",
        )

    def __call__(
        self,
        environ: Mapping[str, Any],
        start_response: StartResponseCallable,
    ) -> list[bytes]:
        """Standard PEP 3333 WSGI entry point.

        Args:
            environ: WSGI environment dictionary.
            start_response: WSGI start_response callback.

        Returns:
            List containing the encoded response byte buffer.
        """
        method: str = str(environ.get("REQUEST_METHOD", "GET"))
        path: str = str(environ.get("PATH_INFO", "/"))
        query_string: str = str(environ.get("QUERY_STRING", ""))
        client_ip: str = str(environ.get("REMOTE_ADDR", "127.0.0.1"))
        content_len_str: str = str(environ.get("CONTENT_LENGTH", "")).strip()
        content_length: int = (
            int(content_len_str) if content_len_str.isdigit() else ZERO_INT
        )
        bounded_len: int = min(content_length, MAX_REQUEST_BODY_BYTES)
        wsgi_input: Any = environ.get("wsgi.input")
        body_bytes: bytes = (
            wsgi_input.read(bounded_len)
            if isinstance(wsgi_input, io.BufferedIOBase)
            or hasattr(wsgi_input, "read")
            else b""
        )
        body_text: str = body_bytes.decode(ENCODING_UTF8, errors="replace")
        headers: dict[str, str] = {
            key[FIVE_INT:].replace("_", "-").lower(): str(environ[key])
            for key in environ.keys()
            if key.startswith("HTTP_")
        }
        status_code, cors_headers, payload_bytes, content_type = (
            self.handle_http_request(
                method=method,
                path=path,
                query_string=query_string,
                headers=headers,
                body_text=body_text,
                client_ip=client_ip,
            )
        )
        status_line: str = HTTP_REASON_MAP.get(
            status_code,
            "500 Internal Server Error",
        )
        header_list: list[tuple[str, str]] = [
            ("Content-Type", content_type),
            ("Content-Length", str(len(payload_bytes))),
            ("X-Content-Type-Options", "nosniff"),
            ("X-Frame-Options", "DENY"),
        ]
        for h_key, h_val in cors_headers.items():
            header_list.append((h_key, h_val))
        start_response(status_line, header_list)
        return [payload_bytes]

    def _parse_body_dict(self, body_text: str) -> dict[str, Any]:
        """Parse and validate a JSON request body."""
        if not body_text.strip():
            raise PreDeployHttpError(
                HTTP_STATUS_BAD_REQUEST,
                "Request body must contain a non-empty JSON object.",
            )
        return parse_safe_json_payload(
            raw_text=body_text,
            max_bytes=MAX_REQUEST_BODY_BYTES,
        )

    def _parse_query_params(self, query_string: str) -> dict[str, str]:
        """Parse URL query parameters with length bounds."""
        verify_condition(
            len(query_string) <= MAX_QUERY_PARAM_LENGTH * FOUR_INT,
            "Query string length exceeds security ceiling.",
        )
        parsed: dict[str, list[str]] = urllib.parse.parse_qs(
            query_string,
            keep_blank_values=False,
        )
        return {
            str(key): str(values[ZERO_INT])[:MAX_QUERY_PARAM_LENGTH]
            for key, values in parsed.items()
            if len(values) >= ONE_INT
        }

    def _serve_static_asset(
        self,
        route_path: str,
        origin_header: str | None,
    ) -> tuple[int, dict[str, str], bytes, str] | None:
        """Serve static HTML, CSS, or JS assets from static_root safely."""
        asset_map: dict[str, tuple[str, str]] = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "application/javascript; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
        }
        if route_path not in asset_map or not self._static_root.exists():
            return None
        filename, mime_type = asset_map[route_path]
        safe_asset: Path = resolve_safe_path(self._static_root, filename)
        if not safe_asset.exists():
            return None
        cors_headers: dict[str, str] = evaluate_cors_origin(
            request_origin=origin_header,
            allowed_origins=self._controller.allowed_cors_origins,
        )
        return (
            HTTP_STATUS_OK,
            cors_headers,
            safe_asset.read_bytes(),
            mime_type,
        )


def build_application_from_config(
    workspace_root: Path,
    config_path: Path | str | None = None,
    master_secret: str | None = None,
) -> UnifAiWsgiApp:
    """Instantiate TacticalDesktopController and UnifAiWsgiApp from YAML.

    Args:
        workspace_root: Trusted workspace root directory.
        config_path: Optional path to config.yaml inside workspace_root.
        master_secret: Optional explicit master secret for testing.

    Returns:
        Configured UnifAiWsgiApp instance ready to serve requests.
    """
    cfg_dict: dict[str, Any] | None = None
    if config_path is not None:
        cfg_dict = load_safe_yaml_config(workspace_root, config_path)
    controller: TacticalDesktopController = TacticalDesktopController(
        workspace_root=workspace_root,
        config=cfg_dict,
        master_secret=master_secret,
    )
    return UnifAiWsgiApp(controller=controller)
