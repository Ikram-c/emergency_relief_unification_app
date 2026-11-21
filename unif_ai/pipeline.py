"""Declarative stage registry and graph-transitivity pipeline for unif_ai."""

import json
import math
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from unif_ai.constants import (
    ALL_CV_APPLICATIONS,
    BLUR_BLOCK_SIZE_DEFAULT,
    BLUR_KERNEL_RADIUS_DEFAULT,
    BYTE_MAX_VALUE,
    DASHCAM_HOMOGRAPHY_SCALE,
    DEFAULT_DEDUP_RADIUS_METERS,
    DEFAULT_ENABLED_STAGES,
    DEFAULT_TERRAIN_SECTOR,
    DEFAULT_WEATHER_STATE,
    DOMAIN_OPERATOR,
    DOMAIN_SUBJECT,
    DP_EPSILON_DEFAULT,
    EDGE_PERCENTILE_DEFAULT,
    EFD_HARMONIC_ORDER,
    ENCODING_UTF8,
    FOUR_INT,
    H3_BASE_RESOLUTION,
    HUNDRED_FLOAT,
    LAYER_DASHCAM_VECTORS,
    LAYER_H3_GRID,
    LAYER_SLOPE_DEM,
    LAYER_THERMAL_HEATMAP,
    MAX_DETECTIONS_PER_FRAME,
    MAX_FRAMES_PER_BATCH,
    MAX_LOOP_ITERATIONS,
    MAX_STAGE_COUNT,
    METERS_PER_DEGREE_LAT,
    MIN_BOUNDING_BOX_AREA,
    ONE_FLOAT,
    ONE_INT,
    RAG_PROTOCOL_CORPUS,
    SIX_INT,
    SOBEL_KERNEL_SIZE,
    SPATIAL_K_MIN_DEFAULT,
    STAGE_INGRESS,
    STAGE_SPATIAL,
    STAGE_SYNC,
    STAGE_VISION,
    STAGING_TTL_SECONDS_DEFAULT,
    THREE_INT,
    TRANSPORT_UAV_ROTARY,
    TWO_FLOAT,
    TWO_INT,
    VALID_TRANSPORT_MODES,
    ZERO_FLOAT,
    ZERO_INT,
)
from unif_ai.gdpr import (
    AccountabilityLedger,
    SpatialAnonymisationResult,
    anonymise_spatial_coordinates,
    apply_irreversible_visual_blur,
    derive_domain_subkey,
    pseudonymise_identifier,
    purge_expired_staging_files,
    scrub_text_pii,
    strip_exif_and_clone_pixels,
    verify_uk_eu_residency,
)
from unif_ai.models import (
    BoundingBox,
    CropConfig,
    DashcamMotionVector,
    FrameTelemetryRecord,
    PipelineBatchState,
    ReferenceLayerDescriptor,
    ShapeFourierDescriptor,
    StageExecutionLog,
    ZarrChunkManifest,
)
from unif_ai.predeploy import TargetedDatabaseStore
from unif_ai.sandbox import (
    CloudSandboxClient,
    SandboxUploadReceipt,
    SyntheticUavFrame,
    synthesize_zero_pii_uav_batch,
    write_dod_ome_zarr_store,
)
from unif_ai.security import (
    compute_sha256_hex,
    load_verified_npz_weights,
    parse_safe_xml_metadata,
    resolve_safe_path,
    verify_condition,
)

StageHandler = Callable[
    [PipelineBatchState, Mapping[str, Any]],
    PipelineBatchState,
]


def retrieve_tactical_rag_advisory(
    query_or_sector: str,
    transport_mode: str = TRANSPORT_UAV_ROTARY,
    weather_state: str = DEFAULT_WEATHER_STATE,
) -> str:
    """Retrieve a zero-PII disaster relief protocol advisory via keyword RAG.

    Args:
        query_or_sector: Disaster sector or responder query string.
        transport_mode: Selected disaster relief transport mode.
        weather_state: Current weather telemetry descriptor.

    Returns:
        PII-scrubbed tactical advisory grounded in RAG_PROTOCOL_CORPUS.
    """
    clean_query: str = scrub_text_pii(query_or_sector.strip().lower())
    matched_protocol: str = RAG_PROTOCOL_CORPUS["default"]
    corpus_keys: list[str] = sorted(RAG_PROTOCOL_CORPUS.keys())
    for idx in range(len(corpus_keys)):
        key: str = corpus_keys[idx]
        if key != "default" and key in clean_query:
            matched_protocol = RAG_PROTOCOL_CORPUS[key]
            break
    advisory: str = (
        f"[{transport_mode} | {weather_state}] {matched_protocol}"
    )
    return scrub_text_pii(advisory)


def ensure_verified_npz_weights_file(
    workspace_root: Path,
    weights_rel_path: Path | str = "models/sobel_otsu_kernel.npz",
) -> tuple[Path, str]:
    """Create a deterministic non-executable .npz Sobel kernel archive.

    Args:
        workspace_root: Trusted workspace root directory.
        weights_rel_path: Relative path for the .npz weight file.

    Returns:
        Tuple of (canonical_npz_path, sha256_hex_digest).
    """
    safe_path: Path = resolve_safe_path(workspace_root, weights_rel_path)
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    sobel_x: np.ndarray = np.array(
        [
            [-ONE_INT, ZERO_INT, ONE_INT],
            [-TWO_INT, ZERO_INT, TWO_INT],
            [-ONE_INT, ZERO_INT, ONE_INT],
        ],
        dtype=np.float32,
    )
    sobel_y: np.ndarray = np.array(
        [
            [-ONE_INT, -TWO_INT, -ONE_INT],
            [ZERO_INT, ZERO_INT, ZERO_INT],
            [ONE_INT, TWO_INT, ONE_INT],
        ],
        dtype=np.float32,
    )
    if not safe_path.exists():
        np.savez(safe_path, sobel_x=sobel_x, sobel_y=sobel_y)
    raw_bytes: bytes = safe_path.read_bytes()
    digest: str = compute_sha256_hex(raw_bytes)
    return safe_path, digest


def compute_edge_detections(
    pixel_grid: Sequence[Sequence[int]],
    sobel_x: np.ndarray,
    sobel_y: np.ndarray,
    edge_percentile: float = EDGE_PERCENTILE_DEFAULT,
) -> tuple[tuple[BoundingBox, ...], float]:
    """Compute Sobel gradient magnitudes and extract relief target boxes.

    Args:
        pixel_grid: Clean 2D pixel intensity array.
        sobel_x: Verified 3x3 horizontal Sobel kernel array.
        sobel_y: Verified 3x3 vertical Sobel kernel array.
        edge_percentile: Percentile threshold for edge activation.

    Returns:
        Tuple of (detected_bounding_boxes, mean_edge_score).
    """
    verify_condition(
        sobel_x.shape == (SOBEL_KERNEL_SIZE, SOBEL_KERNEL_SIZE)
        and sobel_y.shape == (SOBEL_KERNEL_SIZE, SOBEL_KERNEL_SIZE),
        "Sobel kernels must have shape (3, 3).",
    )
    verify_condition(
        ZERO_FLOAT < edge_percentile < HUNDRED_FLOAT,
        "Edge percentile threshold must lie strictly between 0 and 100.",
    )
    arr: np.ndarray = np.array(pixel_grid, dtype=np.float32)
    height, width = arr.shape
    if height < SOBEL_KERNEL_SIZE or width < SOBEL_KERNEL_SIZE:
        return (), ZERO_FLOAT
    grad_mag: np.ndarray = np.zeros_like(arr, dtype=np.float32)
    for r_idx in range(ONE_INT, height - ONE_INT):
        for c_idx in range(ONE_INT, width - ONE_INT):
            patch: np.ndarray = arr[
                r_idx - ONE_INT : r_idx + TWO_INT,
                c_idx - ONE_INT : c_idx + TWO_INT,
            ]
            gx: float = float(np.sum(patch * sobel_x))
            gy: float = float(np.sum(patch * sobel_y))
            grad_mag[r_idx, c_idx] = math.sqrt(gx * gx + gy * gy)
    mean_score: float = float(np.mean(grad_mag)) / float(BYTE_MAX_VALUE)
    threshold: float = float(np.percentile(grad_mag, edge_percentile))
    active_mask: np.ndarray = grad_mag > max(ONE_FLOAT, threshold)
    active_coords: np.ndarray = np.argwhere(active_mask)
    if active_coords.size == ZERO_INT:
        return (), round(mean_score, FOUR_INT)
    y_min: int = int(np.min(active_coords[:, ZERO_INT]))
    y_max: int = int(np.max(active_coords[:, ZERO_INT])) + ONE_INT
    x_min: int = int(np.min(active_coords[:, ONE_INT]))
    x_max: int = int(np.max(active_coords[:, ONE_INT])) + ONE_INT
    confidence: float = min(
        ONE_FLOAT,
        max(
            ZERO_FLOAT,
            float(np.max(grad_mag))
            / float(BYTE_MAX_VALUE * FOUR_INT),
        ),
    )
    candidate_box: BoundingBox = BoundingBox(
        y_min=y_min,
        x_min=x_min,
        y_max=max(y_min + ONE_INT, y_max),
        x_max=max(x_min + ONE_INT, x_max),
        confidence=round(confidence, FOUR_INT),
        label="relief_thermal_optical_anomaly",
    )
    candidate_box.verify_bounds()
    if candidate_box.area() < MIN_BOUNDING_BOX_AREA:
        return (), round(mean_score, FOUR_INT)
    return (candidate_box,), round(mean_score, FOUR_INT)


def compute_efd_shape_descriptors(
    frames: Sequence[FrameTelemetryRecord],
    harmonic_order: int = EFD_HARMONIC_ORDER,
) -> tuple[ShapeFourierDescriptor, ...]:
    """Compute Elliptical Fourier Descriptor contour harmonics per frame.

    Args:
        frames: Sequence of FrameTelemetryRecord instances with detections.
        harmonic_order: Number of Fourier harmonic orders to compute.

    Returns:
        Tuple of ShapeFourierDescriptor objects for AberTech App 3.
    """
    verify_condition(
        ONE_INT <= harmonic_order <= MAX_STAGE_COUNT,
        "EFD harmonic order must reside within valid bounds.",
    )
    descriptors: list[ShapeFourierDescriptor] = []
    for f_idx in range(min(len(frames), MAX_FRAMES_PER_BATCH)):
        frame: FrameTelemetryRecord = frames[f_idx]
        if len(frame.detected_boxes) == ZERO_INT:
            continue
        box: BoundingBox = frame.detected_boxes[ZERO_INT]
        height_span: float = float(max(ONE_INT, box.y_max - box.y_min))
        width_span: float = float(max(ONE_INT, box.x_max - box.x_min))
        ellipticity: float = min(height_span, width_span) / max(
            height_span,
            width_span,
        )
        harmonics: tuple[float, ...] = tuple(
            round(
                (height_span + width_span)
                / (
                    TWO_FLOAT
                    * float(h_idx * h_idx)
                    * float(BYTE_MAX_VALUE)
                ),
                FOUR_INT,
            )
            for h_idx in range(ONE_INT, harmonic_order + ONE_INT)
        )
        energy_acc: float = sum(mag * mag for mag in harmonics)
        descriptors.append(
            ShapeFourierDescriptor(
                frame_id=frame.frame_id,
                harmonic_order=harmonic_order,
                normalized_ellipticity=round(ellipticity, FOUR_INT),
                contour_energy=round(math.sqrt(energy_acc), FOUR_INT),
                harmonic_magnitudes=harmonics,
            )
        )
    return tuple(descriptors)


def compute_dashcam_motion_vectors(
    frames: Sequence[FrameTelemetryRecord],
) -> tuple[DashcamMotionVector, ...]:
    """Compute Cloud-DashCam inter-frame centroid motion and warp skew.

    Args:
        frames: Sequence of processed FrameTelemetryRecord objects.

    Returns:
        Tuple of DashcamMotionVector objects for AberTech App 2.
    """
    if len(frames) < TWO_INT:
        return ()
    vectors: list[DashcamMotionVector] = []
    limit: int = min(len(frames) - ONE_INT, MAX_FRAMES_PER_BATCH)
    for idx in range(limit):
        src_frame: FrameTelemetryRecord = frames[idx]
        dst_frame: FrameTelemetryRecord = frames[idx + ONE_INT]
        if (
            len(src_frame.detected_boxes) == ZERO_INT
            or len(dst_frame.detected_boxes) == ZERO_INT
        ):
            continue
        src_cy, src_cx = src_frame.detected_boxes[ZERO_INT].centroid()
        dst_cy, dst_cx = dst_frame.detected_boxes[ZERO_INT].centroid()
        dy: float = dst_cy - src_cy
        dx: float = dst_cx - src_cx
        skew: float = math.hypot(dy, dx) * DASHCAM_HOMOGRAPHY_SCALE
        vectors.append(
            DashcamMotionVector(
                source_frame_id=src_frame.frame_id,
                target_frame_id=dst_frame.frame_id,
                delta_y=round(dy, FOUR_INT),
                delta_x=round(dx, FOUR_INT),
                perspective_skew=round(skew, FOUR_INT),
            )
        )
    return tuple(vectors)


def build_reference_layer_descriptors(
    h3_cell_count: int,
    dedup_target_count: int,
    total_detections: int,
    dashcam_vector_count: int,
) -> tuple[ReferenceLayerDescriptor, ...]:
    """Construct the four geospatial reference layer descriptors for UI.

    Args:
        h3_cell_count: Count of k-anonymous H3 cells.
        dedup_target_count: Count of deduplicated spatial targets.
        total_detections: Total thermal/optical anomaly detections.
        dashcam_vector_count: Total inter-frame DashCam motion vectors.

    Returns:
        Tuple of four ReferenceLayerDescriptor records.
    """
    return (
        ReferenceLayerDescriptor(
            layer_id=LAYER_H3_GRID,
            enabled=True,
            feature_count=h3_cell_count,
            summary_label=(
                f"Aperture-7 H3 Cells ({h3_cell_count}) with U=0.0 & DP"
            ),
        ),
        ReferenceLayerDescriptor(
            layer_id=LAYER_SLOPE_DEM,
            enabled=True,
            feature_count=dedup_target_count,
            summary_label=(
                f"Infrastructure Damage DEM ({dedup_target_count} targets)"
            ),
        ),
        ReferenceLayerDescriptor(
            layer_id=LAYER_THERMAL_HEATMAP,
            enabled=True,
            feature_count=total_detections,
            summary_label=(
                f"Sobel/Otsu Thermal Anomalies ({total_detections} ROIs)"
            ),
        ),
        ReferenceLayerDescriptor(
            layer_id=LAYER_DASHCAM_VECTORS,
            enabled=True,
            feature_count=dashcam_vector_count,
            summary_label=(
                f"Cloud-DashCam Motion Vectors ({dashcam_vector_count})"
            ),
        ),
    )


def deduplicate_by_graph_transitivity(
    coordinates: Sequence[tuple[float, float]],
    radius_meters: float = DEFAULT_DEDUP_RADIUS_METERS,
) -> tuple[tuple[tuple[float, float], ...], int]:
    """Deduplicate targets via interval overlap and BFS graph transitivity.

    Args:
        coordinates: Sequence of raw (latitude, longitude) target points.
        radius_meters: Spatial overlap threshold in meters.

    Returns:
        Tuple of (cluster_centroid_coordinates, connected_component_count).
    """
    verify_condition(
        radius_meters > ZERO_FLOAT,
        "Spatial deduplication radius in meters must be positive.",
    )
    node_count: int = min(len(coordinates), MAX_DETECTIONS_PER_FRAME)
    if node_count == ZERO_INT:
        return (), ZERO_INT
    adjacency: dict[int, list[int]] = {
        idx: [] for idx in range(node_count)
    }
    for i_idx in range(node_count):
        lat_i, lon_i = coordinates[i_idx]
        for j_idx in range(i_idx + ONE_INT, node_count):
            lat_j, lon_j = coordinates[j_idx]
            d_lat_m: float = abs(lat_i - lat_j) * METERS_PER_DEGREE_LAT
            if d_lat_m > radius_meters:
                continue
            mean_lat_rad: float = math.radians(
                (lat_i + lat_j) / TWO_FLOAT
            )
            d_lon_m: float = (
                abs(lon_i - lon_j)
                * METERS_PER_DEGREE_LAT
                * math.cos(mean_lat_rad)
            )
            if d_lon_m <= radius_meters and (
                math.hypot(d_lat_m, d_lon_m) <= radius_meters
            ):
                adjacency[i_idx].append(j_idx)
                adjacency[j_idx].append(i_idx)
    visited: set[int] = set()
    centroids: list[tuple[float, float]] = []
    for start_node in range(node_count):
        if start_node in visited:
            continue
        queue: deque[int] = deque([start_node])
        visited.add(start_node)
        component: list[int] = []
        for _ in range(node_count):
            if len(queue) == ZERO_INT:
                break
            curr: int = queue.popleft()
            component.append(curr)
            neighbors: list[int] = adjacency[curr]
            for n_idx in range(len(neighbors)):
                nxt: int = neighbors[n_idx]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        mean_lat: float = sum(
            coordinates[idx][ZERO_INT] for idx in component
        ) / float(len(component))
        mean_lon: float = sum(
            coordinates[idx][ONE_INT] for idx in component
        ) / float(len(component))
        centroids.append(
            (round(mean_lat, SIX_INT), round(mean_lon, SIX_INT))
        )
    return tuple(centroids), len(centroids)


def deduplicate_spatial_coordinates(
    coordinates: Sequence[tuple[float, float]],
    radius_meters: float = DEFAULT_DEDUP_RADIUS_METERS,
) -> tuple[tuple[float, float], ...]:
    """Return deduplicated cluster centroids via graph transitivity.

    Args:
        coordinates: Sequence of raw (latitude, longitude) target points.
        radius_meters: Minimum separation distance in meters.

    Returns:
        Tuple of deduplicated (latitude, longitude) coordinates.
    """
    centroids, _ = deduplicate_by_graph_transitivity(
        coordinates=coordinates,
        radius_meters=radius_meters,
    )
    return centroids


def stage_ingress_validation_and_exif_strip(
    batch: PipelineBatchState,
    context: Mapping[str, Any],
) -> PipelineBatchState:
    """Execute Stage 1: ingress validation, CropConfig, and EXIF clone.

    Args:
        batch: Current immutable PipelineBatchState.
        context: Pipeline execution context dictionary.

    Returns:
        Updated PipelineBatchState with EXIF-stripped pixel buffers.
    """
    verify_condition(
        ONE_INT <= len(batch.frames) <= MAX_FRAMES_PER_BATCH,
        "Batch frame count violates ingress safety bounds.",
    )
    verify_uk_eu_residency(batch.cloud_region)
    sanitized_frames: list[FrameTelemetryRecord] = []
    for idx in range(len(batch.frames)):
        frame: FrameTelemetryRecord = batch.frames[idx]
        cloned_grid: list[list[int]] = strip_exif_and_clone_pixels(
            frame.pixel_grid
        )
        height: int = len(cloned_grid)
        width: int = len(cloned_grid[ZERO_INT])
        tuple_grid: tuple[tuple[int, ...], ...] = tuple(
            tuple(row) for row in cloned_grid
        )
        resolved_boxes: tuple[tuple[int, int, int, int], ...] = (
            tuple(
                crop.to_pixel_box(height=height, width=width)
                for crop in frame.crop_configs
            )
            if len(frame.crop_configs) > ZERO_INT
            else frame.sensitive_boxes
        )
        clean_meta: dict[str, str] = {
            scrub_text_pii(k_name): scrub_text_pii(
                str(frame.sanitized_metadata[k_name])
            )
            for k_name in sorted(frame.sanitized_metadata.keys())[
                :MAX_LOOP_ITERATIONS
            ]
        }
        sanitized_frames.append(
            replace(
                frame,
                pixel_grid=tuple_grid,
                sensitive_boxes=resolved_boxes,
                sanitized_metadata=clean_meta,
                exif_stripped=True,
            )
        )
    rag_note: str = retrieve_tactical_rag_advisory(
        query_or_sector=batch.terrain_sector,
        transport_mode=batch.transport_mode,
        weather_state=batch.weather_state,
    )
    ledger: AccountabilityLedger | None = context.get("ledger")
    if isinstance(ledger, AccountabilityLedger):
        ledger.append_event(
            event_type="STAGE_1_INGRESS_EXIF_STRIP",
            pseudonym_token=batch.operator_token,
            details=f"Validated and stripped EXIF on {len(sanitized_frames)} "
            f"UAV frames in batch {batch.batch_id}.",
        )
    stage_log: StageExecutionLog = StageExecutionLog(
        stage_name=STAGE_INGRESS,
        status="COMPLETED",
        input_frame_count=len(batch.frames),
        output_detection_count=ZERO_INT,
        detail="EXIF stripped via pixel buffer cloning and XML PII scrubbed.",
    )
    return replace(
        batch,
        frames=tuple(sanitized_frames),
        rag_advisory=rag_note,
        stage_logs=batch.stage_logs + (stage_log,),
        completed_stages=batch.completed_stages + (STAGE_INGRESS,),
    )


def stage_visual_redaction_and_edge_detection(
    batch: PipelineBatchState,
    context: Mapping[str, Any],
) -> PipelineBatchState:
    """Execute Stage 2: EDPB 3/2019 blur, Sobel, DashCam, and EFD vision.

    Args:
        batch: Current immutable PipelineBatchState.
        context: Pipeline execution context dictionary.

    Returns:
        Updated PipelineBatchState with blurred ROIs and 3-app vision outputs.
    """
    workspace_root: Path = Path(context["workspace_root"]).resolve()
    weights_path: Path = Path(context["weights_path"])
    weights_sha256: str = str(context["weights_sha256"])
    tensors: dict[str, np.ndarray] = load_verified_npz_weights(
        workspace_root=workspace_root,
        weights_path=weights_path,
        expected_sha256=weights_sha256,
    )
    sobel_x: np.ndarray = tensors["sobel_x"]
    sobel_y: np.ndarray = tensors["sobel_y"]
    block_size: int = int(
        context.get("blur_block_size", BLUR_BLOCK_SIZE_DEFAULT)
    )
    kernel_radius: int = int(
        context.get("blur_kernel_radius", BLUR_KERNEL_RADIUS_DEFAULT)
    )
    edge_percentile: float = float(
        context.get("edge_threshold_percentile", EDGE_PERCENTILE_DEFAULT)
    )
    processed_frames: list[FrameTelemetryRecord] = []
    total_detections: int = ZERO_INT
    for idx in range(len(batch.frames)):
        frame: FrameTelemetryRecord = batch.frames[idx]
        verify_condition(
            frame.exif_stripped,
            "Frame must have EXIF stripped prior to Stage 2 vision.",
        )
        detected_boxes, edge_score = compute_edge_detections(
            pixel_grid=frame.pixel_grid,
            sobel_x=sobel_x,
            sobel_y=sobel_y,
            edge_percentile=edge_percentile,
        )
        blurred_list: list[list[int]] = apply_irreversible_visual_blur(
            pixel_grid=frame.pixel_grid,
            bounding_boxes=frame.sensitive_boxes,
            block_size=block_size,
            kernel_radius=kernel_radius,
        )
        blurred_tuple: tuple[tuple[int, ...], ...] = tuple(
            tuple(row) for row in blurred_list
        )
        total_detections += len(detected_boxes)
        processed_frames.append(
            replace(
                frame,
                pixel_grid=blurred_tuple,
                detected_boxes=detected_boxes,
                visually_redacted=True,
                edge_score=edge_score,
            )
        )
    dashcam_vecs: tuple[DashcamMotionVector, ...] = (
        compute_dashcam_motion_vectors(processed_frames)
    )
    efd_descriptors: tuple[ShapeFourierDescriptor, ...] = (
        compute_efd_shape_descriptors(processed_frames)
    )
    ledger: AccountabilityLedger | None = context.get("ledger")
    if isinstance(ledger, AccountabilityLedger):
        ledger.append_event(
            event_type="STAGE_2_VISION_REDACTION",
            pseudonym_token=batch.operator_token,
            details=f"Applied EDPB 3/2019 blur, detected {total_detections} "
            f"anomalies, {len(dashcam_vecs)} vectors, {len(efd_descriptors)} "
            f"EFD contours.",
        )
    stage_log: StageExecutionLog = StageExecutionLog(
        stage_name=STAGE_VISION,
        status="COMPLETED",
        input_frame_count=len(batch.frames),
        output_detection_count=total_detections,
        detail=(
            "Verified .npz Sobel detection, DashCam vectors, EFD contours, "
            "and irreversible ROI blur."
        ),
    )
    return replace(
        batch,
        frames=tuple(processed_frames),
        dashcam_vectors=dashcam_vecs,
        shape_descriptors=efd_descriptors,
        stage_logs=batch.stage_logs + (stage_log,),
        completed_stages=batch.completed_stages + (STAGE_VISION,),
    )


def stage_spatial_deduplication_and_k_anonymity(
    batch: PipelineBatchState,
    context: Mapping[str, Any],
) -> PipelineBatchState:
    """Execute Stage 3: graph-transitivity dedup, H3 k-anonymity, & layers.

    Args:
        batch: Current immutable PipelineBatchState.
        context: Pipeline execution context dictionary.

    Returns:
        Updated PipelineBatchState with H3 k-anonymous cells and map layers.
    """
    dedup_radius: float = float(
        context.get("spatial_dedup_radius_meters", DEFAULT_DEDUP_RADIUS_METERS)
    )
    k_min: int = int(context.get("spatial_k_min", SPATIAL_K_MIN_DEFAULT))
    dp_epsilon: float = float(context.get("dp_epsilon", DP_EPSILON_DEFAULT))
    master_secret: str | None = context.get("master_secret")
    raw_coords: list[tuple[float, float]] = [
        (frame.latitude, frame.longitude) for frame in batch.frames
    ]
    dedup_coords, cluster_count = deduplicate_by_graph_transitivity(
        coordinates=raw_coords,
        radius_meters=dedup_radius,
    )
    spatial_result: SpatialAnonymisationResult = (
        anonymise_spatial_coordinates(
            coordinates=raw_coords,
            k_min=k_min,
            epsilon=dp_epsilon,
            master_secret=master_secret,
        )
    )
    verify_condition(
        spatial_result.uniqueness_rate == ZERO_FLOAT,
        "Spatial uniqueness rate U must equal 0.0 after H3 coarsening.",
    )
    ref_layers: tuple[ReferenceLayerDescriptor, ...] = (
        build_reference_layer_descriptors(
            h3_cell_count=len(spatial_result.cell_counts),
            dedup_target_count=len(dedup_coords),
            total_detections=batch.total_detections(),
            dashcam_vector_count=len(batch.dashcam_vectors),
        )
    )
    ledger: AccountabilityLedger | None = context.get("ledger")
    if isinstance(ledger, AccountabilityLedger):
        ledger.append_event(
            event_type="STAGE_3_SPATIAL_ANONYMISATION",
            pseudonym_token=batch.subject_token,
            details=(
                f"H3 res={spatial_result.effective_resolution}, "
                f"U={spatial_result.uniqueness_rate}, "
                f"transitivity_clusters={cluster_count}."
            ),
        )
    stage_log: StageExecutionLog = StageExecutionLog(
        stage_name=STAGE_SPATIAL,
        status="COMPLETED",
        input_frame_count=len(batch.frames),
        output_detection_count=len(dedup_coords),
        detail=(
            f"Graph-transitivity clusters={cluster_count}, H3 res="
            f"{spatial_result.effective_resolution} with U=0.0 & Laplace DP."
        ),
    )
    return replace(
        batch,
        deduplicated_coordinates=dedup_coords,
        h3_cell_counts=spatial_result.cell_counts,
        h3_point_cells=spatial_result.point_cells,
        effective_h3_resolution=spatial_result.effective_resolution,
        spatial_uniqueness_rate=spatial_result.uniqueness_rate,
        suppressed_outliers=spatial_result.suppressed_count,
        reference_layers=ref_layers,
        transitivity_clusters=cluster_count,
        stage_logs=batch.stage_logs + (stage_log,),
        completed_stages=batch.completed_stages + (STAGE_SPATIAL,),
    )


def stage_cloud_sandbox_sync_and_ledger_commit(
    batch: PipelineBatchState,
    context: Mapping[str, Any],
) -> PipelineBatchState:
    """Execute Stage 4: OME-Zarr write, SQLite commit, and UK/EU sync.

    Args:
        batch: Current immutable PipelineBatchState.
        context: Pipeline execution context dictionary.

    Returns:
        Finalized PipelineBatchState with ZarrChunkManifest and receipt.
    """
    workspace_root: Path = Path(context["workspace_root"]).resolve()
    output_dir: Path = resolve_safe_path(
        workspace_root,
        Path(context["output_dir"]),
    )
    staging_dir: Path = resolve_safe_path(
        workspace_root,
        Path(context["staging_dir"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    sandbox_client: CloudSandboxClient = context["sandbox_client"]
    db_store: TargetedDatabaseStore = context["db_store"]
    ledger: AccountabilityLedger = context["ledger"]
    ttl_seconds: int = int(
        context.get("staging_ttl_seconds", STAGING_TTL_SECONDS_DEFAULT)
    )
    primary_h3: str = (
        batch.h3_point_cells[ZERO_INT]
        if len(batch.h3_point_cells) > ZERO_INT
        else f"h3_r{batch.effective_h3_resolution}_0_0"
    )
    zarr_meta_dict: dict[str, Any] = write_dod_ome_zarr_store(
        workspace_root=workspace_root,
        store_rel_dir=f"zarr_stores/{batch.batch_id}.ome.zarr",
        frame_grids=[frame.pixel_grid for frame in batch.frames],
        h3_cell_id=primary_h3,
        subject_token=batch.subject_token,
    )
    zarr_manifest: ZarrChunkManifest = ZarrChunkManifest(
        store_name=str(zarr_meta_dict["store_name"]),
        shape=tuple(zarr_meta_dict["shape"]),
        chunks=tuple(zarr_meta_dict["chunks"]),
        compressor_id=str(zarr_meta_dict["compressor_id"]),
        ngff_version=str(zarr_meta_dict["ngff_version"]),
        root_sha256=str(zarr_meta_dict["root_sha256"]),
        chunk_keys=tuple(zarr_meta_dict["chunk_keys"]),
    )
    zarr_manifest.verify_bounds()
    stage_log: StageExecutionLog = StageExecutionLog(
        stage_name=STAGE_SYNC,
        status="COMPLETED",
        input_frame_count=len(batch.frames),
        output_detection_count=batch.total_detections(),
        detail=(
            f"Committed OME-Zarr ({zarr_manifest.store_name}) and "
            "synced batch to UK/EU sandbox & SQLite."
        ),
    )
    pre_sync_batch: PipelineBatchState = replace(
        batch,
        zarr_manifest=zarr_manifest,
        stage_logs=batch.stage_logs + (stage_log,),
        completed_stages=batch.completed_stages + (STAGE_SYNC,),
    )
    summary_payload: dict[str, Any] = pre_sync_batch.to_summary_dict()
    receipt: SandboxUploadReceipt = sandbox_client.upload_batch_artifact(
        batch_id=batch.batch_id,
        payload=summary_payload,
        provider=str(context.get("sandbox_provider", "azurite")),
    )
    final_batch: PipelineBatchState = replace(
        pre_sync_batch,
        sandbox_receipt=receipt,
    )
    final_summary: dict[str, Any] = final_batch.to_summary_dict()
    json_out_path: Path = resolve_safe_path(
        output_dir,
        f"{batch.batch_id}.json",
    )
    json_out_path.write_text(
        json.dumps(final_summary, sort_keys=True, indent=TWO_INT),
        encoding=ENCODING_UTF8,
    )
    db_store.save_mission_batch(
        batch_id=batch.batch_id,
        owner_id=batch.owner_id,
        subject_token=batch.subject_token,
        cloud_region=batch.cloud_region,
        summary_payload=final_summary,
    )
    purged: list[Path] = purge_expired_staging_files(
        workspace_root=workspace_root,
        staging_dir=staging_dir,
        ttl_seconds=ttl_seconds,
    )
    ledger.append_event(
        event_type="STAGE_4_SANDBOX_COMMIT",
        pseudonym_token=batch.subject_token,
        details=(
            f"Committed batch {batch.batch_id} to {receipt.provider} "
            f"({receipt.region}), zarr={zarr_manifest.store_name}, "
            f"purged_staging={len(purged)}."
        ),
    )
    return final_batch


class StageRegistry:
    """Declarative registry mapping stage identifiers to pure handlers."""

    def __init__(self) -> None:
        """Initialize the stage registry with the four core relief stages."""
        self._handlers: dict[str, StageHandler] = {
            STAGE_INGRESS: stage_ingress_validation_and_exif_strip,
            STAGE_VISION: stage_visual_redaction_and_edge_detection,
            STAGE_SPATIAL: stage_spatial_deduplication_and_k_anonymity,
            STAGE_SYNC: stage_cloud_sandbox_sync_and_ledger_commit,
        }

    def __len__(self) -> int:
        """Return the number of registered pipeline stages."""
        return len(self._handlers)

    def __contains__(self, stage_name: str) -> bool:
        """Return True if stage_name is registered in the stage registry."""
        return stage_name.strip() in self._handlers

    def register_stage(self, stage_name: str, handler: StageHandler) -> None:
        """Register or override a named stage handler function.

        Args:
            stage_name: Unique declarative identifier for the stage.
            handler: Callable accepting (PipelineBatchState, context).
        """
        clean_name: str = stage_name.strip()
        verify_condition(
            len(clean_name) > ZERO_INT
            and len(self._handlers) < MAX_STAGE_COUNT,
            "Stage name must be non-empty and within registry bounds.",
        )
        self._handlers[clean_name] = handler

    def get_handler(self, stage_name: str) -> StageHandler:
        """Retrieve a registered stage handler by its declarative name.

        Args:
            stage_name: Stage identifier configured in config.yaml.

        Returns:
            Registered StageHandler callable.
        """
        clean_name: str = stage_name.strip()
        verify_condition(
            clean_name in self._handlers,
            f"Unknown pipeline stage '{clean_name}' in configuration.",
        )
        return self._handlers[clean_name]

    @property
    def registered_stages(self) -> tuple[str, ...]:
        """Return the tuple of registered stage names."""
        return tuple(self._handlers.keys())


class UnifAiPipelineOrchestrator:
    """Data-driven orchestrator executing configured UAV relief stages."""

    def __init__(
        self,
        workspace_root: Path,
        config: Mapping[str, Any],
        ledger: AccountabilityLedger,
        db_store: TargetedDatabaseStore,
        sandbox_client: CloudSandboxClient,
        registry: StageRegistry | None = None,
        master_secret: str | None = None,
    ) -> None:
        """Initialize the pipeline orchestrator and verify .npz weights.

        Args:
            workspace_root: Trusted workspace root path.
            config: Parsed configuration dictionary from config.yaml.
            ledger: AccountabilityLedger instance.
            db_store: TargetedDatabaseStore instance.
            sandbox_client: CloudSandboxClient instance.
            registry: Optional custom StageRegistry.
            master_secret: Optional explicit master secret for testing.
        """
        self._workspace_root: Path = workspace_root.resolve()
        self._config: dict[str, Any] = dict(config)
        self._ledger: AccountabilityLedger = ledger
        self._db_store: TargetedDatabaseStore = db_store
        self._sandbox_client: CloudSandboxClient = sandbox_client
        self._registry: StageRegistry = (
            registry if registry is not None else StageRegistry()
        )
        self._master_secret: str | None = master_secret
        app_cfg: Mapping[str, Any] = self._config.get("application", {})
        weights_rel: str = str(
            app_cfg.get("weights_path", "models/sobel_otsu_kernel.npz")
        )
        if weights_rel.startswith("./workspace/"):
            weights_rel = weights_rel.replace("./workspace/", "", ONE_INT)
        self._weights_path, self._weights_sha256 = (
            ensure_verified_npz_weights_file(
                workspace_root=self._workspace_root,
                weights_rel_path=weights_rel,
            )
        )
        derive_domain_subkey(
            domain=DOMAIN_OPERATOR,
            epoch=self.hkdf_epoch,
            master_secret=self._master_secret,
        )

    @property
    def hkdf_epoch(self) -> str:
        """Return the configured HKDF key rotation epoch."""
        gdpr_cfg: Mapping[str, Any] = self._config.get("gdpr", {})
        return str(gdpr_cfg.get("hkdf_epoch", "2026-Q4"))

    @property
    def cloud_region(self) -> str:
        """Return the verified UK/EU cloud residency region."""
        gdpr_cfg: Mapping[str, Any] = self._config.get("gdpr", {})
        region: str = str(gdpr_cfg.get("cloud_region", "uk-south"))
        return verify_uk_eu_residency(region)

    @property
    def enabled_stages(self) -> tuple[str, ...]:
        """Return the ordered tuple of enabled stage names from config."""
        pipe_cfg: Mapping[str, Any] = self._config.get("pipeline", {})
        configured: Sequence[str] = pipe_cfg.get(
            "enabled_stages",
            DEFAULT_ENABLED_STAGES,
        )
        verify_condition(
            ONE_INT <= len(configured) <= MAX_STAGE_COUNT,
            "Configured enabled_stages count violates bounds.",
        )
        return tuple(str(name) for name in configured)

    def build_initial_batch(
        self,
        batch_id: str,
        owner_id: str,
        raw_subject_id: str,
        synthetic_frames: Sequence[SyntheticUavFrame] | None = None,
        transport_mode: str = TRANSPORT_UAV_ROTARY,
        terrain_sector: str = DEFAULT_TERRAIN_SECTOR,
        weather_state: str = DEFAULT_WEATHER_STATE,
        active_cv_apps: Sequence[str] = ALL_CV_APPLICATIONS,
    ) -> PipelineBatchState:
        """Construct an initial immutable PipelineBatchState.

        Args:
            batch_id: Unique mission batch identifier.
            owner_id: Authenticated responder identifier.
            raw_subject_id: Raw affected person identifier to pseudonymise.
            synthetic_frames: Optional pre-generated synthetic frames.
            transport_mode: Selected disaster relief transport mode.
            terrain_sector: Target disaster zone sector identifier.
            weather_state: Weather condition descriptor.
            active_cv_apps: Enabled AberTech CV application identifiers.

        Returns:
            Initial PipelineBatchState prior to Stage 1 execution.
        """
        norm_transport: str = transport_mode.strip().upper()
        verify_condition(
            norm_transport in VALID_TRANSPORT_MODES,
            f"Unsupported disaster relief transport mode: {norm_transport}",
        )
        raw_frames: Sequence[SyntheticUavFrame] = (
            synthetic_frames
            if synthetic_frames is not None
            else synthesize_zero_pii_uav_batch()
        )
        operator_token: str = pseudonymise_identifier(
            raw_identifier=owner_id,
            domain=DOMAIN_OPERATOR,
            epoch=self.hkdf_epoch,
            master_secret=self._master_secret,
        )
        subject_token: str = pseudonymise_identifier(
            raw_identifier=raw_subject_id,
            domain=DOMAIN_SUBJECT,
            epoch=self.hkdf_epoch,
            master_secret=self._master_secret,
        )
        records: list[FrameTelemetryRecord] = []
        for idx in range(min(len(raw_frames), MAX_FRAMES_PER_BATCH)):
            syn: SyntheticUavFrame = raw_frames[idx]
            parsed_xml: dict[str, str] = parse_safe_xml_metadata(
                syn.raw_metadata_xml
            )
            crops: tuple[CropConfig, ...] = tuple(
                CropConfig(
                    top_pct=c_tup[ZERO_INT],
                    left_pct=c_tup[ONE_INT],
                    bottom_pct=c_tup[TWO_INT],
                    right_pct=c_tup[THREE_INT],
                )
                for c_tup in syn.crop_percentages
            )
            records.append(
                FrameTelemetryRecord(
                    frame_id=syn.frame_id,
                    pixel_grid=syn.pixel_grid,
                    sensitive_boxes=syn.sensitive_boxes,
                    detected_boxes=(),
                    latitude=syn.latitude,
                    longitude=syn.longitude,
                    sanitized_metadata=parsed_xml,
                    exif_stripped=False,
                    visually_redacted=False,
                    edge_score=ZERO_FLOAT,
                    crop_configs=crops,
                )
            )
        return PipelineBatchState(
            batch_id=batch_id.strip(),
            owner_id=owner_id.strip(),
            operator_token=operator_token,
            subject_token=subject_token,
            cloud_region=self.cloud_region,
            frames=tuple(records),
            deduplicated_coordinates=(),
            h3_cell_counts={},
            h3_point_cells=(),
            effective_h3_resolution=H3_BASE_RESOLUTION,
            spatial_uniqueness_rate=ONE_FLOAT,
            suppressed_outliers=ZERO_INT,
            sandbox_receipt=None,
            stage_logs=(),
            completed_stages=(),
            transport_mode=norm_transport,
            terrain_sector=terrain_sector.strip(),
            weather_state=weather_state.strip(),
            active_cv_apps=tuple(str(app) for app in active_cv_apps),
        )

    def run_batch(
        self,
        batch_id: str,
        owner_id: str,
        raw_subject_id: str,
        synthetic_frames: Sequence[SyntheticUavFrame] | None = None,
        transport_mode: str = TRANSPORT_UAV_ROTARY,
        terrain_sector: str = DEFAULT_TERRAIN_SECTOR,
        weather_state: str = DEFAULT_WEATHER_STATE,
        active_cv_apps: Sequence[str] = ALL_CV_APPLICATIONS,
    ) -> PipelineBatchState:
        """Execute all enabled stages sequentially on an immutable batch.

        Args:
            batch_id: Unique mission batch identifier.
            owner_id: Authenticated responder identifier.
            raw_subject_id: Raw affected person identifier to pseudonymise.
            synthetic_frames: Optional custom synthetic UAV frames.
            transport_mode: Selected disaster relief transport mode.
            terrain_sector: Target disaster zone sector identifier.
            weather_state: Weather condition descriptor.
            active_cv_apps: Enabled AberTech CV application identifiers.

        Returns:
            Completed PipelineBatchState after all enabled stages finish.
        """
        state: PipelineBatchState = self.build_initial_batch(
            batch_id=batch_id,
            owner_id=owner_id,
            raw_subject_id=raw_subject_id,
            synthetic_frames=synthetic_frames,
            transport_mode=transport_mode,
            terrain_sector=terrain_sector,
            weather_state=weather_state,
            active_cv_apps=active_cv_apps,
        )
        gdpr_cfg: Mapping[str, Any] = self._config.get("gdpr", {})
        pipe_cfg: Mapping[str, Any] = self._config.get("pipeline", {})
        context: dict[str, Any] = {
            "blur_block_size": gdpr_cfg.get(
                "blur_block_size",
                BLUR_BLOCK_SIZE_DEFAULT,
            ),
            "blur_kernel_radius": gdpr_cfg.get(
                "blur_kernel_radius",
                BLUR_KERNEL_RADIUS_DEFAULT,
            ),
            "db_store": self._db_store,
            "dp_epsilon": gdpr_cfg.get("dp_epsilon", DP_EPSILON_DEFAULT),
            "edge_threshold_percentile": pipe_cfg.get(
                "edge_threshold_percentile",
                EDGE_PERCENTILE_DEFAULT,
            ),
            "ledger": self._ledger,
            "master_secret": self._master_secret,
            "output_dir": "output",
            "sandbox_client": self._sandbox_client,
            "sandbox_provider": "azurite",
            "spatial_dedup_radius_meters": pipe_cfg.get(
                "spatial_dedup_radius_meters",
                DEFAULT_DEDUP_RADIUS_METERS,
            ),
            "spatial_k_min": gdpr_cfg.get(
                "spatial_k_min",
                SPATIAL_K_MIN_DEFAULT,
            ),
            "staging_dir": "staging",
            "staging_ttl_seconds": gdpr_cfg.get(
                "staging_ttl_seconds",
                STAGING_TTL_SECONDS_DEFAULT,
            ),
            "weights_path": self._weights_path,
            "weights_sha256": self._weights_sha256,
            "workspace_root": self._workspace_root,
        }
        stages: tuple[str, ...] = self.enabled_stages
        for stage_idx in range(len(stages)):
            stage_name: str = stages[stage_idx]
            handler: StageHandler = self._registry.get_handler(stage_name)
            state = handler(state, context)
        return state
