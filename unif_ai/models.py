"""Immutable Data-Oriented Design models for the unif_ai relief platform."""

from dataclasses import dataclass
from typing import Any, Mapping

from unif_ai.constants import (
    ALL_CV_APPLICATIONS,
    ALLOWED_ZARR_CODECS,
    CROP_PCT_MAX,
    CROP_PCT_MIN,
    DEFAULT_REFERENCE_LAYERS,
    DEFAULT_TERRAIN_SECTOR,
    DEFAULT_WEATHER_STATE,
    FOUR_INT,
    HALF_FLOAT,
    HUNDRED_FLOAT,
    MAX_ZARR_DIMENSIONS,
    OME_NGFF_VERSION,
    ONE_FLOAT,
    ONE_INT,
    SHA256_HEX_LENGTH,
    TRANSPORT_UAV_ROTARY,
    TWO_INT,
    ZERO_FLOAT,
    ZERO_INT,
)
from unif_ai.gdpr import convert_crop_percentages_to_box
from unif_ai.sandbox import SandboxUploadReceipt
from unif_ai.security import validate_image_dimensions, verify_condition


@dataclass(frozen=True)
class CropConfig:
    """Normalized percentage crop configuration for resolution-safe ROIs."""

    top_pct: float
    left_pct: float
    bottom_pct: float
    right_pct: float

    def verify_bounds(self) -> None:
        """Verify that crop percentages lie strictly within [0.0, 100.0]."""
        verify_condition(
            CROP_PCT_MIN <= self.top_pct < self.bottom_pct <= CROP_PCT_MAX,
            "CropConfig vertical bounds must satisfy 0.0 <= top < bottom.",
        )
        verify_condition(
            CROP_PCT_MIN <= self.left_pct < self.right_pct <= CROP_PCT_MAX,
            "CropConfig horizontal bounds must satisfy 0.0 <= left < right.",
        )

    def to_pixel_box(
        self,
        height: int,
        width: int,
    ) -> tuple[int, int, int, int]:
        """Convert normalized percentages into integer pixel coordinates.

        Args:
            height: Target frame height in pixels.
            width: Target frame width in pixels.

        Returns:
            Pixel bounding box tuple (y_min, x_min, y_max, x_max).
        """
        self.verify_bounds()
        return convert_crop_percentages_to_box(
            top_pct=self.top_pct,
            left_pct=self.left_pct,
            bottom_pct=self.bottom_pct,
            right_pct=self.right_pct,
            height=height,
            width=width,
        )

    def to_dict(self) -> dict[str, float]:
        """Serialize the normalized crop percentages to a dictionary.

        Returns:
            Dictionary with rounded top, left, bottom, and right percentages.
        """
        return {
            "bottom_pct": round(self.bottom_pct, FOUR_INT),
            "left_pct": round(self.left_pct, FOUR_INT),
            "right_pct": round(self.right_pct, FOUR_INT),
            "top_pct": round(self.top_pct, FOUR_INT),
        }


@dataclass(frozen=True)
class ZarrChunkManifest:
    """Data-Oriented Design OME-Zarr v2 chunked store manifest."""

    store_name: str
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    compressor_id: str
    ngff_version: str
    root_sha256: str
    chunk_keys: tuple[str, ...]

    def verify_bounds(self) -> None:
        """Verify OME-Zarr store dimensions, codec allowlist, and digest."""
        verify_condition(
            ONE_INT <= len(self.shape) <= MAX_ZARR_DIMENSIONS
            and len(self.shape) == len(self.chunks),
            "ZarrChunkManifest shape and chunks must match in dimensionality.",
        )
        verify_condition(
            self.compressor_id in ALLOWED_ZARR_CODECS,
            "ZarrChunkManifest compressor_id must be in ALLOWED_ZARR_CODECS.",
        )
        verify_condition(
            len(self.root_sha256) == SHA256_HEX_LENGTH,
            "ZarrChunkManifest root_sha256 must be 64 hex characters.",
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the OME-Zarr manifest to a JSON-compatible dictionary.

        Returns:
            Dictionary summarizing the OME-Zarr store layout and digest.
        """
        return {
            "chunk_count": len(self.chunk_keys),
            "chunk_keys": list(self.chunk_keys),
            "chunks": list(self.chunks),
            "compressor_id": self.compressor_id,
            "ngff_version": self.ngff_version or OME_NGFF_VERSION,
            "root_sha256": self.root_sha256,
            "shape": list(self.shape),
            "store_name": self.store_name,
        }


@dataclass(frozen=True)
class BoundingBox:
    """Immutable axis-aligned bounding box for aerial object detections."""

    y_min: int
    x_min: int
    y_max: int
    x_max: int
    confidence: float
    label: str

    def verify_bounds(self) -> None:
        """Verify geometric and confidence invariants on the bounding box."""
        verify_condition(
            ZERO_INT <= self.y_min < self.y_max,
            "BoundingBox y_min and y_max must satisfy 0 <= y_min < y_max.",
        )
        verify_condition(
            ZERO_INT <= self.x_min < self.x_max,
            "BoundingBox x_min and x_max must satisfy 0 <= x_min < x_max.",
        )
        verify_condition(
            ZERO_FLOAT <= self.confidence <= ONE_FLOAT,
            "BoundingBox confidence must reside within [0.0, 1.0].",
        )
        verify_condition(
            len(self.label.strip()) > ZERO_INT,
            "BoundingBox label must be a non-empty string.",
        )

    def to_tuple(self) -> tuple[int, int, int, int]:
        """Return the four-integer (y_min, x_min, y_max, x_max) coordinates.

        Returns:
            Tuple of (y_min, x_min, y_max, x_max).
        """
        return (self.y_min, self.x_min, self.y_max, self.x_max)

    def to_crop_config(self, height: int, width: int) -> CropConfig:
        """Convert pixel coordinates into a normalized CropConfig instance.

        Args:
            height: Frame height in pixels.
            width: Frame width in pixels.

        Returns:
            Normalized CropConfig in [0.0, 100.0] percentage space.
        """
        validate_image_dimensions(width=width, height=height)
        self.verify_bounds()
        top_p: float = (float(self.y_min) / float(height)) * HUNDRED_FLOAT
        bot_p: float = (float(self.y_max) / float(height)) * HUNDRED_FLOAT
        left_p: float = (float(self.x_min) / float(width)) * HUNDRED_FLOAT
        right_p: float = (float(self.x_max) / float(width)) * HUNDRED_FLOAT
        crop: CropConfig = CropConfig(
            top_pct=max(CROP_PCT_MIN, min(CROP_PCT_MAX, top_p)),
            left_pct=max(CROP_PCT_MIN, min(CROP_PCT_MAX, left_p)),
            bottom_pct=max(CROP_PCT_MIN, min(CROP_PCT_MAX, bot_p)),
            right_pct=max(CROP_PCT_MIN, min(CROP_PCT_MAX, right_p)),
        )
        crop.verify_bounds()
        return crop

    def area(self) -> int:
        """Compute the pixel area of the bounding box.

        Returns:
            Positive integer area in square pixels.
        """
        height: int = max(ZERO_INT, self.y_max - self.y_min)
        width: int = max(ZERO_INT, self.x_max - self.x_min)
        return height * width

    def centroid(self) -> tuple[float, float]:
        """Compute the (center_y, center_x) pixel centroid of the box.

        Returns:
            Tuple of (center_y, center_x) floats.
        """
        center_y: float = float(self.y_min + self.y_max) * HALF_FLOAT
        center_x: float = float(self.x_min + self.x_max) * HALF_FLOAT
        return (round(center_y, TWO_INT), round(center_x, TWO_INT))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the bounding box to a JSON-compatible dictionary.

        Returns:
            Dictionary representation of the bounding box.
        """
        center_y, center_x = self.centroid()
        return {
            "area": self.area(),
            "centroid_x": center_x,
            "centroid_y": center_y,
            "confidence": round(self.confidence, FOUR_INT),
            "label": self.label,
            "x_max": self.x_max,
            "x_min": self.x_min,
            "y_max": self.y_max,
            "y_min": self.y_min,
        }


@dataclass(frozen=True)
class DashcamMotionVector:
    """Inter-frame Cloud-DashCam motion and perspective warp descriptor."""

    source_frame_id: str
    target_frame_id: str
    delta_y: float
    delta_x: float
    perspective_skew: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize the DashCam motion vector to a dictionary.

        Returns:
            Dictionary of motion displacement and perspective skew metrics.
        """
        return {
            "delta_x": round(self.delta_x, FOUR_INT),
            "delta_y": round(self.delta_y, FOUR_INT),
            "perspective_skew": round(self.perspective_skew, FOUR_INT),
            "source_frame_id": self.source_frame_id,
            "target_frame_id": self.target_frame_id,
        }


@dataclass(frozen=True)
class ShapeFourierDescriptor:
    """Elliptical Fourier Descriptor (EFD) contour harmonic invariants."""

    frame_id: str
    harmonic_order: int
    normalized_ellipticity: float
    contour_energy: float
    harmonic_magnitudes: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the EFD shape descriptor to a dictionary.

        Returns:
            Dictionary containing harmonic magnitudes and contour metrics.
        """
        return {
            "contour_energy": round(self.contour_energy, FOUR_INT),
            "frame_id": self.frame_id,
            "harmonic_magnitudes": [
                round(val, FOUR_INT) for val in self.harmonic_magnitudes
            ],
            "harmonic_order": self.harmonic_order,
            "normalized_ellipticity": round(
                self.normalized_ellipticity,
                FOUR_INT,
            ),
        }


@dataclass(frozen=True)
class ReferenceLayerDescriptor:
    """Geospatial reference layer overlay state for the tactical map view."""

    layer_id: str
    enabled: bool
    feature_count: int
    summary_label: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the reference layer descriptor to a dictionary.

        Returns:
            Dictionary representing the reference layer status.
        """
        return {
            "enabled": self.enabled,
            "feature_count": self.feature_count,
            "layer_id": self.layer_id,
            "summary_label": self.summary_label,
        }


@dataclass(frozen=True)
class FrameTelemetryRecord:
    """Immutable state of a single UAV frame across pipeline stages."""

    frame_id: str
    pixel_grid: tuple[tuple[int, ...], ...]
    sensitive_boxes: tuple[tuple[int, int, int, int], ...]
    detected_boxes: tuple[BoundingBox, ...]
    latitude: float
    longitude: float
    sanitized_metadata: Mapping[str, str]
    exif_stripped: bool
    visually_redacted: bool
    edge_score: float
    crop_configs: tuple[CropConfig, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize frame telemetry without raw full-frame pixel bloat.

        Returns:
            Dictionary summarizing the frame state and detections.
        """
        height: int = len(self.pixel_grid)
        width: int = (
            len(self.pixel_grid[ZERO_INT])
            if height > ZERO_INT
            else ZERO_INT
        )
        return {
            "crop_configs": [
                crop.to_dict() for crop in self.crop_configs
            ],
            "detected_boxes": [
                box.to_dict() for box in self.detected_boxes
            ],
            "edge_score": round(self.edge_score, FOUR_INT),
            "exif_stripped": self.exif_stripped,
            "frame_id": self.frame_id,
            "height": height,
            "sanitized_metadata": dict(self.sanitized_metadata),
            "sensitive_box_count": len(self.sensitive_boxes),
            "visually_redacted": self.visually_redacted,
            "width": width,
        }


@dataclass(frozen=True)
class StageExecutionLog:
    """Immutable execution telemetry record for a single pipeline stage."""

    stage_name: str
    status: str
    input_frame_count: int
    output_detection_count: int
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the stage execution record to a dictionary.

        Returns:
            Dictionary containing stage metrics and status.
        """
        return {
            "detail": self.detail,
            "input_frame_count": self.input_frame_count,
            "output_detection_count": self.output_detection_count,
            "stage_name": self.stage_name,
            "status": self.status,
        }


@dataclass(frozen=True)
class PipelineBatchState:
    """Immutable batch state passed sequentially through the stage registry."""

    batch_id: str
    owner_id: str
    operator_token: str
    subject_token: str
    cloud_region: str
    frames: tuple[FrameTelemetryRecord, ...]
    deduplicated_coordinates: tuple[tuple[float, float], ...]
    h3_cell_counts: Mapping[str, float]
    h3_point_cells: tuple[str, ...]
    effective_h3_resolution: int
    spatial_uniqueness_rate: float
    suppressed_outliers: int
    sandbox_receipt: SandboxUploadReceipt | None
    stage_logs: tuple[StageExecutionLog, ...]
    completed_stages: tuple[str, ...]
    transport_mode: str = TRANSPORT_UAV_ROTARY
    terrain_sector: str = DEFAULT_TERRAIN_SECTOR
    weather_state: str = DEFAULT_WEATHER_STATE
    active_cv_apps: tuple[str, ...] = ALL_CV_APPLICATIONS
    dashcam_vectors: tuple[DashcamMotionVector, ...] = ()
    shape_descriptors: tuple[ShapeFourierDescriptor, ...] = ()
    reference_layers: tuple[ReferenceLayerDescriptor, ...] = ()
    rag_advisory: str = ""
    zarr_manifest: ZarrChunkManifest | None = None
    transitivity_clusters: int = ZERO_INT

    def total_detections(self) -> int:
        """Compute total bounding box detections across all batch frames.

        Returns:
            Sum of detected bounding boxes across frames.
        """
        return sum(len(frame.detected_boxes) for frame in self.frames)

    def all_frames_redacted(self) -> bool:
        """Verify whether every frame is EXIF-stripped and visually blurred.

        Returns:
            True if all frames have exif_stripped and visually_redacted set.
        """
        if len(self.frames) == ZERO_INT:
            return False
        return all(
            frame.exif_stripped and frame.visually_redacted
            for frame in self.frames
        )

    def to_summary_dict(self) -> dict[str, Any]:
        """Serialize the batch state into a zero-PII JSON summary dictionary.

        Returns:
            Dictionary suitable for SQLite, cloud sandbox, and UI rendering.
        """
        receipt_dict: dict[str, Any] | None = None
        if self.sandbox_receipt is not None:
            receipt_dict = {
                "blob_name": self.sandbox_receipt.blob_name,
                "container": self.sandbox_receipt.container,
                "offline_fallback_used": (
                    self.sandbox_receipt.offline_fallback_used
                ),
                "provider": self.sandbox_receipt.provider,
                "region": self.sandbox_receipt.region,
                "sha256_digest": self.sandbox_receipt.sha256_digest,
                "status_code": self.sandbox_receipt.status_code,
            }
        default_layers: list[dict[str, Any]] = (
            [layer.to_dict() for layer in self.reference_layers]
            if len(self.reference_layers) > ZERO_INT
            else [
                {
                    "enabled": True,
                    "feature_count": len(self.h3_cell_counts),
                    "layer_id": name,
                    "summary_label": f"Active overlay: {name}",
                }
                for name in DEFAULT_REFERENCE_LAYERS
            ]
        )
        zarr_dict: dict[str, Any] | None = (
            self.zarr_manifest.to_dict()
            if self.zarr_manifest is not None
            else None
        )
        return {
            "active_cv_apps": list(self.active_cv_apps),
            "all_frames_redacted": self.all_frames_redacted(),
            "batch_id": self.batch_id,
            "cloud_region": self.cloud_region,
            "completed_stages": list(self.completed_stages),
            "dashcam_vectors": [
                vec.to_dict() for vec in self.dashcam_vectors
            ],
            "deduplicated_target_count": len(
                self.deduplicated_coordinates
            ),
            "effective_h3_resolution": self.effective_h3_resolution,
            "frame_count": len(self.frames),
            "frames": [frame.to_dict() for frame in self.frames],
            "h3_cell_counts": dict(self.h3_cell_counts),
            "h3_point_cells": list(self.h3_point_cells),
            "operator_token": self.operator_token,
            "owner_id": self.owner_id,
            "rag_advisory": self.rag_advisory,
            "reference_layers": default_layers,
            "sandbox_receipt": receipt_dict,
            "shape_descriptors": [
                desc.to_dict() for desc in self.shape_descriptors
            ],
            "spatial_uniqueness_rate": round(
                self.spatial_uniqueness_rate,
                FOUR_INT,
            ),
            "stage_logs": [log.to_dict() for log in self.stage_logs],
            "subject_token": self.subject_token,
            "suppressed_outliers": self.suppressed_outliers,
            "terrain_sector": self.terrain_sector,
            "total_detections": self.total_detections(),
            "transitivity_clusters": self.transitivity_clusters,
            "transport_mode": self.transport_mode,
            "weather_state": self.weather_state,
            "zarr_manifest": zarr_dict,
        }
