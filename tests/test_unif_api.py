"""Automated security, GDPR, relief CV, and pre-deployment test suite."""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any, Final

from unif_ai.cli import build_arg_parser, run_headless_demo_batch
from unif_ai.constants import (
    DOMAIN_OPERATOR,
    DOMAIN_SUBJECT,
    DP_EPSILON_DEFAULT,
    ENCODING_UTF8,
    FOUR_INT,
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_FORBIDDEN,
    HTTP_STATUS_GONE,
    HTTP_STATUS_INTERNAL_ERROR,
    HTTP_STATUS_NOT_FOUND,
    HTTP_STATUS_OK,
    HTTP_STATUS_TOO_MANY_REQUESTS,
    MAX_IMAGE_WIDTH_DEFAULT,
    ONE_INT,
    PASSWORD_Reset_TTL_SECONDS,
    REDACTED_PLACEHOLDER,
    SHA256_HEX_LENGTH,
    SLOT_BLUE,
    SLOT_GREEN,
    SPATIAL_K_MIN_DEFAULT,
    SYNTHETIC_BASE_LATITUDE,
    SYNTHETIC_BASE_LONGITUDE,
    THREE_INT,
    TRANSPORT_DASHCAM_4X4,
    TWO_INT,
    ZERO_FLOAT,
    ZERO_INT,
)
from unif_ai.gdpr import (
    AccountabilityLedger,
    SpatialAnonymisationResult,
    anonymise_spatial_coordinates,
    apply_crop_percentage_blur,
    get_master_secret_bytes,
    pseudonymise_identifier,
    purge_expired_staging_files,
    scrub_text_pii,
    verify_uk_eu_residency,
)
from unif_ai.models import CropConfig
from unif_ai.pipeline import (
    StageRegistry,
    deduplicate_by_graph_transitivity,
    ensure_verified_npz_weights_file,
)
from unif_ai.predeploy import (
    CostCappedRingLogger,
    SlidingWindowRateLimiter,
    execute_with_error_boundary,
)
from unif_ai.sandbox import read_dod_ome_zarr_chunk, write_dod_ome_zarr_store
from unif_ai.security import (
    SecurityViolationError,
    extract_safe_zip_archive,
    load_verified_npz_weights,
    parse_safe_xml_metadata,
    resolve_safe_path,
    validate_image_dimensions,
    validate_outbound_url,
    validate_safe_zarr_metadata,
)
from unif_ai.server import TacticalDesktopController, UnifAiWsgiApp

TEST_MASTER_SECRET: Final[str] = (
    "test_only_master_secret_key_for_unif_ai_gdpr_suite_2026"
)
TEST_EPOCH_BASE: Final[int] = 1762128000
TEST_EXPIRED_OFFSET: Final[int] = PASSWORD_Reset_TTL_SECONDS + ONE_INT
TEST_RATE_LIMIT_CAP: Final[int] = 3
TEST_LOG_RING_CAP: Final[int] = 4
TEST_LOG_DETAIL_CAP: Final[int] = 24
TEST_OVERSIZED_DIM: Final[int] = MAX_IMAGE_WIDTH_DEFAULT * TWO_INT
TEST_PIXEL_HIGH: Final[int] = 240
TEST_PIXEL_LOW: Final[int] = 10
TEST_GRID_DIM: Final[int] = 16
TEST_BOX_START: Final[int] = 4
TEST_BOX_END: Final[int] = 12
TEST_CROP_START_PCT: Final[float] = 25.0
TEST_CROP_END_PCT: Final[float] = 75.0
TEST_HIGH_RATE_LIMIT: Final[int] = 50
TEST_DISTANT_LAT_OFFSET: Final[float] = 0.05


class UnifAiSecurityGdprPreDeployTests(unittest.TestCase):
    """Comprehensive verification suite for unif_ai v2.0 Disaster Relief."""

    def setUp(self) -> None:
        """Create an isolated temporary workspace for each test case."""
        self._temp_dir: tempfile.TemporaryDirectory[str] = (
            tempfile.TemporaryDirectory()
        )
        self.workspace_root: Path = Path(self._temp_dir.name).resolve()

    def tearDown(self) -> None:
        """Clean up the temporary workspace directory."""
        self._temp_dir.cleanup()

    def test_owasp_asvs_and_ome_zarr_security_controls(self) -> None:
        """Verify CWE-22, CWE-502, CWE-611, CWE-400, and Zarr codec guards."""
        with self.assertRaises(SecurityViolationError):
            resolve_safe_path(self.workspace_root, "../escape.txt")
        with self.assertRaises(SecurityViolationError):
            resolve_safe_path(self.workspace_root, "bad\x00file.txt")

        npz_path, digest = ensure_verified_npz_weights_file(
            self.workspace_root
        )
        tensors = load_verified_npz_weights(
            self.workspace_root,
            npz_path,
            digest,
        )
        self.assertIn("sobel_x", tensors)
        bad_digest: str = "a" * SHA256_HEX_LENGTH
        with self.assertRaises(SecurityViolationError):
            load_verified_npz_weights(
                self.workspace_root,
                npz_path,
                bad_digest,
            )

        bad_zarray: Path = self.workspace_root / "bad.zarray"
        bad_zarray.write_text(
            json.dumps(
                {
                    "chunks": [ONE_INT, TEST_GRID_DIM, TEST_GRID_DIM],
                    "compressor": {"id": "pickle"},
                    "filters": [{"id": "pickle"}],
                    "shape": [ONE_INT, TEST_GRID_DIM, TEST_GRID_DIM],
                }
            ),
            encoding=ENCODING_UTF8,
        )
        with self.assertRaises(SecurityViolationError):
            validate_safe_zarr_metadata(self.workspace_root, bad_zarray)

        zip_path: Path = self.workspace_root / "malicious.zip"
        with zipfile.ZipFile(zip_path, mode="w") as zf:
            zf.writestr("../slip.txt", "payload")
        with self.assertRaises(SecurityViolationError):
            extract_safe_zip_archive(
                self.workspace_root,
                zip_path,
                self.workspace_root / "extracted",
            )

        xxe_xml: str = (
            "<!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]>"
            "<meta><tag>val</tag></meta>"
        )
        with self.assertRaises(SecurityViolationError):
            parse_safe_xml_metadata(xxe_xml)

        with self.assertRaises(SecurityViolationError):
            validate_image_dimensions(
                width=TEST_OVERSIZED_DIM,
                height=TEST_OVERSIZED_DIM,
            )

        with self.assertRaises(SecurityViolationError):
            validate_outbound_url("http://169.254.169.254/latest/meta-data")

    def test_uk_eu_gdpr_and_relief_vision_algorithms(self) -> None:
        """Verify CSPRNG secret, HKDF, CropConfig blur, Zarr, and H3 DP."""
        ephemeral_bytes: bytes = get_master_secret_bytes()
        self.assertGreaterEqual(len(ephemeral_bytes), SHA256_HEX_LENGTH)

        tok_op_1: str = pseudonymise_identifier(
            "relief_person_99",
            domain=DOMAIN_OPERATOR,
            master_secret=TEST_MASTER_SECRET,
        )
        tok_op_2: str = pseudonymise_identifier(
            "relief_person_99",
            domain=DOMAIN_OPERATOR,
            master_secret=TEST_MASTER_SECRET,
        )
        tok_sub: str = pseudonymise_identifier(
            "relief_person_99",
            domain=DOMAIN_SUBJECT,
            master_secret=TEST_MASTER_SECRET,
        )
        self.assertEqual(tok_op_1, tok_op_2)
        self.assertNotEqual(tok_op_1, tok_sub)

        dirty_text: str = (
            "Contact jane.doe@example.co.uk or +44 7428 810579 "
            "from 192.168.1.45"
        )
        scrubbed: str = scrub_text_pii(dirty_text)
        self.assertNotIn("jane.doe@example.co.uk", scrubbed)
        self.assertNotIn("192.168.1.45", scrubbed)
        self.assertIn(REDACTED_PLACEHOLDER, scrubbed)

        grid: list[list[int]] = [
            [
                TEST_PIXEL_HIGH
                if (r + c) % TWO_INT == ZERO_INT
                else TEST_PIXEL_LOW
                for c in range(TEST_GRID_DIM)
            ]
            for r in range(TEST_GRID_DIM)
        ]
        crop_cfg: CropConfig = CropConfig(
            top_pct=TEST_CROP_START_PCT,
            left_pct=TEST_CROP_START_PCT,
            bottom_pct=TEST_CROP_END_PCT,
            right_pct=TEST_CROP_END_PCT,
        )
        px_box: tuple[int, int, int, int] = crop_cfg.to_pixel_box(
            height=TEST_GRID_DIM,
            width=TEST_GRID_DIM,
        )
        self.assertEqual(
            px_box,
            (TEST_BOX_START, TEST_BOX_START, TEST_BOX_END, TEST_BOX_END),
        )
        blurred: list[list[int]] = apply_crop_percentage_blur(
            pixel_grid=grid,
            crop_percentages=(
                (
                    TEST_CROP_START_PCT,
                    TEST_CROP_START_PCT,
                    TEST_CROP_END_PCT,
                    TEST_CROP_END_PCT,
                ),
            ),
        )
        self.assertNotEqual(
            grid[TEST_BOX_START][TEST_BOX_START],
            blurred[TEST_BOX_START][TEST_BOX_START],
        )

        zarr_info: dict[str, Any] = write_dod_ome_zarr_store(
            workspace_root=self.workspace_root,
            store_rel_dir="zarr_stores/test_cube.ome.zarr",
            frame_grids=[blurred],
            h3_cell_id="h3_r9_0_0",
            subject_token=tok_sub,
        )
        self.assertEqual(len(zarr_info["root_sha256"]), SHA256_HEX_LENGTH)
        restored_chunk: list[list[int]] = read_dod_ome_zarr_chunk(
            workspace_root=self.workspace_root,
            store_rel_dir="zarr_stores/test_cube.ome.zarr",
            frame_index=ZERO_INT,
        )
        self.assertEqual(restored_chunk, blurred)

        chain_coords: list[tuple[float, float]] = [
            (SYNTHETIC_BASE_LATITUDE, SYNTHETIC_BASE_LONGITUDE),
            (SYNTHETIC_BASE_LATITUDE + 0.00005, SYNTHETIC_BASE_LONGITUDE),
            (SYNTHETIC_BASE_LATITUDE + 0.00010, SYNTHETIC_BASE_LONGITUDE),
            (
                SYNTHETIC_BASE_LATITUDE + TEST_DISTANT_LAT_OFFSET,
                SYNTHETIC_BASE_LONGITUDE,
            ),
        ]
        centroids, cluster_cnt = deduplicate_by_graph_transitivity(
            coordinates=chain_coords,
        )
        self.assertEqual(cluster_cnt, TWO_INT)
        self.assertEqual(len(centroids), TWO_INT)

        self.assertEqual(verify_uk_eu_residency("uk-south"), "uk-south")
        with self.assertRaises(SecurityViolationError):
            verify_uk_eu_residency("us-east-1")

        coords: list[tuple[float, float]] = [
            (SYNTHETIC_BASE_LATITUDE, SYNTHETIC_BASE_LONGITUDE),
            (SYNTHETIC_BASE_LATITUDE, SYNTHETIC_BASE_LONGITUDE),
            (SYNTHETIC_BASE_LATITUDE, SYNTHETIC_BASE_LONGITUDE),
        ]
        spatial_res: SpatialAnonymisationResult = (
            anonymise_spatial_coordinates(
                coordinates=coords,
                k_min=SPATIAL_K_MIN_DEFAULT,
                epsilon=DP_EPSILON_DEFAULT,
                master_secret=TEST_MASTER_SECRET,
            )
        )
        self.assertEqual(spatial_res.uniqueness_rate, ZERO_FLOAT)
        self.assertEqual(len(spatial_res.point_cells), THREE_INT)

        staging_dir: Path = self.workspace_root / "staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        expired_file: Path = staging_dir / "old_frame.raw"
        expired_file.write_text("stale", encoding=ENCODING_UTF8)
        purged: list[Path] = purge_expired_staging_files(
            workspace_root=self.workspace_root,
            staging_dir=staging_dir,
            ttl_seconds=ONE_INT,
            now_epoch=float(TEST_EPOCH_BASE + TEST_EXPIRED_OFFSET),
        )
        self.assertEqual(len(purged), ONE_INT)

        reg: StageRegistry = StageRegistry()
        self.assertEqual(len(reg), FOUR_INT)
        self.assertIn("ingress_validation_and_exif_strip", reg)

        ledger: AccountabilityLedger = AccountabilityLedger(
            self.workspace_root,
            "ledger/chain.jsonl",
        )
        ledger.append_event("TEST_EVT_1", tok_op_1, "First entry")
        ledger.append_event("TEST_EVT_2", tok_sub, "Second entry")
        self.assertTrue(ledger.verify_chain())

    def test_predeploy_ten_point_checklist_and_five_views_api(self) -> None:
        """Verify all 10 pre-deployment controls, 5 view routes, and CLI."""
        custom_cfg: dict[str, Any] = {
            "application": {
                "active_slot": SLOT_BLUE,
                "name": "unif_ai",
                "standby_slot": SLOT_GREEN,
                "version": "2.0.0",
                "weights_path": "models/sobel_otsu_kernel.npz",
            },
            "gdpr": {
                "cloud_region": "uk-south",
                "dp_epsilon": DP_EPSILON_DEFAULT,
                "hkdf_epoch": "2026-Q4",
                "spatial_k_min": SPATIAL_K_MIN_DEFAULT,
                "staging_ttl_seconds": 3600,
            },
            "pipeline": {},
            "predeploy": {
                "log_detail_max_length": TEST_LOG_DETAIL_CAP,
                "log_ring_buffer_capacity": TEST_LOG_RING_CAP,
                "password_reset_ttl_seconds": PASSWORD_Reset_TTL_SECONDS,
                "rate_limit_requests_per_minute": TEST_HIGH_RATE_LIMIT,
                "rate_limit_window_seconds": 60,
            },
            "security": {
                "allowed_cors_origins": ["http://127.0.0.1:8080"],
            },
        }
        controller: TacticalDesktopController = TacticalDesktopController(
            workspace_root=self.workspace_root,
            config=custom_cfg,
            master_secret=TEST_MASTER_SECRET,
        )
        app: UnifAiWsgiApp = UnifAiWsgiApp(controller=controller)

        status_run, hdr_run, body_run, _ = app.handle_http_request(
            method="POST",
            path="/api/rescue/create",
            headers={"Origin": "http://127.0.0.1:8080"},
            body_text=json.dumps(
                {
                    "batch_id": "relief_001",
                    "owner_id": "responder_alpha",
                    "subject_id": "person_subject_1",
                    "terrain_sector": "EARTHQUAKE_URBAN_GRID",
                    "transport_mode": TRANSPORT_DASHCAM_4X4,
                    "weather_state": "SEVERE_STORM_LOW_VISIBILITY",
                }
            ),
        )
        self.assertEqual(status_run, HTTP_STATUS_OK)
        self.assertEqual(
            hdr_run.get("Access-Control-Allow-Origin"),
            "http://127.0.0.1:8080",
        )
        run_data: dict[str, Any] = json.loads(body_run.decode(ENCODING_UTF8))
        self.assertTrue(run_data["batch"]["all_frames_redacted"])
        self.assertIsNotNone(run_data["batch"]["zarr_manifest"])
        self.assertGreaterEqual(
            len(run_data["batch"]["shape_descriptors"]),
            ONE_INT,
        )

        status_portal, _, body_portal, _ = app.handle_http_request(
            method="GET",
            path="/api/rescues",
            query_string="owner_id=responder_alpha",
            headers={"X-Operator-Id": "responder_alpha"},
        )
        self.assertEqual(status_portal, HTTP_STATUS_OK)
        portal_data: dict[str, Any] = json.loads(
            body_portal.decode(ENCODING_UTF8)
        )
        self.assertEqual(portal_data["count"], ONE_INT)

        status_rag, _, body_rag, _ = app.handle_http_request(
            method="POST",
            path="/api/rag/query",
            body_text=json.dumps(
                {
                    "query": "earthquake",
                    "transport_mode": TRANSPORT_DASHCAM_4X4,
                }
            ),
        )
        self.assertEqual(status_rag, HTTP_STATUS_OK)
        rag_data: dict[str, Any] = json.loads(body_rag.decode(ENCODING_UTF8))
        self.assertIn("Urban Structural Collapse", rag_data["advisory"])

        status_layers, _, body_layers, _ = app.handle_http_request(
            method="GET",
            path="/api/layers",
        )
        self.assertEqual(status_layers, HTTP_STATUS_OK)
        layers_data: dict[str, Any] = json.loads(
            body_layers.decode(ENCODING_UTF8)
        )
        self.assertEqual(len(layers_data["reference_layers"]), FOUR_INT)

        status_bola_deny, _, body_bola, _ = app.handle_http_request(
            method="GET",
            path="/api/batch",
            query_string="batch_id=relief_001",
            headers={"X-Operator-Id": "responder_intruder"},
        )
        self.assertEqual(status_bola_deny, HTTP_STATUS_FORBIDDEN)
        bola_err: dict[str, Any] = json.loads(body_bola.decode(ENCODING_UTF8))
        self.assertEqual(
            bola_err["error_screen"]["status_code"],
            HTTP_STATUS_FORBIDDEN,
        )

        status_sqli, _, _, _ = app.handle_http_request(
            method="POST",
            path="/api/batch/run",
            body_text=json.dumps(
                {
                    "batch_id": "relief_002 UNION SELECT *",
                    "owner_id": "responder_alpha",
                    "subject_id": "person_1",
                }
            ),
        )
        self.assertEqual(status_sqli, HTTP_STATUS_BAD_REQUEST)

        status_xss, _, _, _ = app.handle_http_request(
            method="POST",
            path="/api/batch/run",
            body_text=json.dumps(
                {
                    "batch_id": "<script>alert(1)</script>",
                    "owner_id": "responder_alpha",
                    "subject_id": "person_1",
                }
            ),
        )
        self.assertEqual(status_xss, HTTP_STATUS_BAD_REQUEST)

        status_cors, _, _, _ = app.handle_http_request(
            method="GET",
            path="/api/status",
            headers={"Origin": "https://evil.example.com"},
        )
        self.assertEqual(status_cors, HTTP_STATUS_FORBIDDEN)

        limiter: SlidingWindowRateLimiter = SlidingWindowRateLimiter(
            max_requests=TEST_RATE_LIMIT_CAP,
            window_seconds=PASSWORD_Reset_TTL_SECONDS,
        )
        ref_time: float = float(TEST_EPOCH_BASE)
        for _ in range(TEST_RATE_LIMIT_CAP):
            limiter.check_and_record("client_1", now_epoch=ref_time)
        with self.assertRaises(Exception) as rate_ctx:
            limiter.check_and_record("client_1", now_epoch=ref_time)
        self.assertEqual(
            getattr(rate_ctx.exception, "status_code", ZERO_INT),
            HTTP_STATUS_TOO_MANY_REQUESTS,
        )

        issue_res: dict[str, Any] = controller.issue_password_reset(
            user_id="responder_alpha",
            now_epoch=TEST_EPOCH_BASE,
        )
        token_str: str = issue_res["reset_token"]
        status_exp, _, _, _ = app.handle_http_request(
            method="POST",
            path="/api/reset/redeem",
            body_text=json.dumps({"reset_token": token_str}),
        )
        self.assertEqual(status_exp, HTTP_STATUS_OK)
        status_replay, _, _, _ = app.handle_http_request(
            method="POST",
            path="/api/reset/redeem",
            body_text=json.dumps({"reset_token": token_str}),
        )
        self.assertEqual(status_replay, HTTP_STATUS_GONE)

        status_404, _, _, _ = app.handle_http_request(
            method="GET",
            path="/api/nonexistent-route",
        )
        self.assertEqual(status_404, HTTP_STATUS_NOT_FOUND)

        ring_logger: CostCappedRingLogger = CostCappedRingLogger(
            capacity=TEST_LOG_RING_CAP,
            max_detail_length=TEST_LOG_DETAIL_CAP,
        )

        def failing_op() -> dict[str, Any]:
            raise RuntimeError("Database connection dropped unexpectedly.")

        status_500, err_500 = execute_with_error_boundary(
            operation=failing_op,
            logger=ring_logger,
            component="boundary_test",
        )
        self.assertEqual(status_500, HTTP_STATUS_INTERNAL_ERROR)
        self.assertNotIn("Traceback", json.dumps(err_500))
        self.assertEqual(len(ring_logger.alerts), ONE_INT)

        indexes: tuple[str, ...] = controller.db_store.list_custom_indexes()
        self.assertEqual(
            indexes,
            (
                "idx_mission_batches_batch_id",
                "idx_mission_batches_owner_id",
            ),
        )

        for idx in range(TEST_LOG_RING_CAP * TWO_INT):
            ring_logger.log_event(
                "INFO",
                "ring_test",
                f"Message_number_{idx}_with_extra_characters_for_truncation",
            )
        self.assertLessEqual(len(ring_logger.entries), TEST_LOG_RING_CAP)
        self.assertLessEqual(
            len(ring_logger.entries[-ONE_INT]["detail"]),
            TEST_LOG_DETAIL_CAP,
        )

        promote_ok: dict[str, Any] = controller.switch_deployment_slot(
            candidate_version="2.1.0",
            health_verified=True,
        )
        self.assertEqual(
            promote_ok["deployment"]["active_slot"],
            SLOT_GREEN,
        )
        promote_fail: dict[str, Any] = controller.switch_deployment_slot(
            candidate_version="2.2.0-candidate",
            health_verified=False,
        )
        self.assertTrue(promote_fail["deployment"]["rolled_back"])

        erasure_res: dict[str, Any] = controller.trigger_gdpr_erasure(
            raw_subject_id="person_subject_1"
        )
        self.assertEqual(
            erasure_res["erasure_report"]["deleted_db_rows"],
            ONE_INT,
        )
        self.assertEqual(
            erasure_res["erasure_report"]["scrubbed_json_files"],
            ONE_INT,
        )
        self.assertEqual(
            erasure_res["erasure_report"]["scrubbed_zarr_stores"],
            ONE_INT,
        )

        parser = build_arg_parser()
        parsed_args = parser.parse_args(["--mode", "batch"])
        self.assertEqual(parsed_args.mode, "batch")
        cli_report: dict[str, Any] = run_headless_demo_batch(
            workspace_root=self.workspace_root,
            config_rel_path="",
            batch_id="cli_relief_001",
            owner_id="responder_alpha",
            subject_id="person_cli_99",
        )
        self.assertTrue(cli_report["system_status"]["ledger_chain_valid"])


if __name__ == "__main__":
    unittest.main()
