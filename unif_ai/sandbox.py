"""DOD OME-Zarr persistence, UK/EU cloud sandbox, and relief synthesizer."""

import json
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from unif_ai.constants import (
    ALLOWED_HOSTS_DEFAULT,
    ALLOWED_UK_EU_REGIONS,
    ALLOWED_URL_SCHEMES,
    ALLOWED_ZARR_CODECS,
    CROP_PCT_MAX,
    CROP_PCT_STEP_OFFSET,
    DEFAULT_CROP_BOTTOM_PCT,
    DEFAULT_CROP_LEFT_PCT,
    DEFAULT_CROP_RIGHT_PCT,
    DEFAULT_CROP_TOP_PCT,
    ENCODING_UTF8,
    HTTP_STATUS_CREATED,
    HTTP_STATUS_OK,
    MAX_FRAMES_PER_BATCH,
    OME_NGFF_VERSION,
    ONE_FLOAT,
    ONE_INT,
    SYNTHETIC_BACKGROUND_VALUE,
    SYNTHETIC_BASE_LATITUDE,
    SYNTHETIC_BASE_LONGITUDE,
    SYNTHETIC_COORD_JITTER,
    SYNTHETIC_FRAME_COUNT,
    SYNTHETIC_FRAME_HEIGHT,
    SYNTHETIC_FRAME_WIDTH,
    SYNTHETIC_OBJECT_VALUE,
    TWO_INT,
    ZARR_FORMAT_VERSION,
    ZERO_FLOAT,
    ZERO_INT,
)
from unif_ai.gdpr import (
    convert_crop_percentages_to_box,
    scrub_text_pii,
    verify_uk_eu_residency,
)
from unif_ai.security import (
    compute_sha256_hex,
    resolve_safe_path,
    validate_image_dimensions,
    validate_outbound_url,
    validate_safe_zarr_metadata,
    verify_condition,
    verify_sha256_digest,
)


@dataclass(frozen=True)
class SandboxUploadReceipt:
    """Immutable receipt returned when a batch is synced to the sandbox."""

    provider: str
    region: str
    container: str
    blob_name: str
    sha256_digest: str
    offline_fallback_used: bool
    status_code: int


@dataclass(frozen=True)
class SyntheticUavFrame:
    """Zero-PII synthetic UAV frame with GPS telemetry and crop percentages."""

    frame_id: str
    pixel_grid: tuple[tuple[int, ...], ...]
    sensitive_boxes: tuple[tuple[int, int, int, int], ...]
    latitude: float
    longitude: float
    raw_metadata_xml: str
    crop_percentages: tuple[tuple[float, float, float, float], ...] = ()


def write_dod_ome_zarr_store(
    workspace_root: Path,
    store_rel_dir: Path | str,
    frame_grids: Sequence[Sequence[Sequence[int]]],
    h3_cell_id: str,
    subject_token: str,
    compressor_id: str = "zlib",
) -> dict[str, Any]:
    """Write a 3D [t, y, x] UAV frame cube as a verified OME-Zarr v2 store.

    Args:
        workspace_root: Trusted workspace root directory.
        store_rel_dir: Relative directory path for the .ome.zarr store.
        frame_grids: 3D sequence of 8-bit redacted pixel grids [t, y, x].
        h3_cell_id: Coarsened aperture-7 H3 cell identifier for .zattrs.
        subject_token: Pseudonymised subject token for Art. 17 traceability.
        compressor_id: Allowlisted codec name (default 'zlib').

    Returns:
        Dictionary manifest suitable for constructing ZarrChunkManifest.
    """
    norm_codec: str = compressor_id.strip().lower()
    verify_condition(
        norm_codec in ALLOWED_ZARR_CODECS,
        f"Codec '{norm_codec}' is not in ALLOWED_ZARR_CODECS.",
    )
    frame_count: int = len(frame_grids)
    verify_condition(
        ONE_INT <= frame_count <= MAX_FRAMES_PER_BATCH,
        "OME-Zarr frame count violates batch safety bounds.",
    )
    height: int = len(frame_grids[ZERO_INT])
    width: int = (
        len(frame_grids[ZERO_INT][ZERO_INT])
        if height > ZERO_INT
        else ZERO_INT
    )
    validate_image_dimensions(width=width, height=height)
    store_path: Path = resolve_safe_path(workspace_root, store_rel_dir)
    scale_zero_dir: Path = resolve_safe_path(store_path, "0")
    scale_zero_dir.mkdir(parents=True, exist_ok=True)
    zgroup_payload: dict[str, int] = {"zarr_format": ZARR_FORMAT_VERSION}
    zgroup_path: Path = resolve_safe_path(store_path, ".zgroup")
    zgroup_path.write_text(
        json.dumps(zgroup_payload, sort_keys=True),
        encoding=ENCODING_UTF8,
    )
    zattrs_payload: dict[str, Any] = {
        "h3_k_anonymous_cell": scrub_text_pii(h3_cell_id),
        "multiscales": [
            {
                "axes": [
                    {"name": "t", "type": "time"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [
                    {
                        "coordinateTransformations": [
                            {
                                "scale": [ONE_FLOAT, ONE_FLOAT, ONE_FLOAT],
                                "type": "scale",
                            },
                            {
                                "translation": [
                                    ZERO_FLOAT,
                                    ZERO_FLOAT,
                                    ZERO_FLOAT,
                                ],
                                "type": "translation",
                            },
                        ],
                        "path": "0",
                    }
                ],
                "name": store_path.name,
                "version": OME_NGFF_VERSION,
            }
        ],
        "subject_token": scrub_text_pii(subject_token),
    }
    zattrs_path: Path = resolve_safe_path(store_path, ".zattrs")
    zattrs_path.write_text(
        json.dumps(zattrs_payload, sort_keys=True, indent=TWO_INT),
        encoding=ENCODING_UTF8,
    )
    zarray_payload: dict[str, Any] = {
        "chunks": [ONE_INT, height, width],
        "compressor": {"id": norm_codec, "level": ONE_INT},
        "dimension_separator": ".",
        "dtype": "|u1",
        "fill_value": ZERO_INT,
        "filters": None,
        "order": "C",
        "shape": [frame_count, height, width],
        "zarr_format": ZARR_FORMAT_VERSION,
    }
    zarray_path: Path = resolve_safe_path(scale_zero_dir, ".zarray")
    zarray_path.write_text(
        json.dumps(zarray_payload, sort_keys=True, indent=TWO_INT),
        encoding=ENCODING_UTF8,
    )
    validate_safe_zarr_metadata(workspace_root, zarray_path)
    chunk_keys: list[str] = []
    chunk_digests: list[str] = []
    for t_idx in range(frame_count):
        grid: Sequence[Sequence[int]] = frame_grids[t_idx]
        flat_bytes: bytes = bytes(
            int(grid[r_idx][c_idx])
            for r_idx in range(height)
            for c_idx in range(width)
        )
        compressed_chunk: bytes = zlib.compress(flat_bytes, level=ONE_INT)
        chunk_key: str = f"0/{t_idx}.0.0"
        chunk_path: Path = resolve_safe_path(
            store_path,
            chunk_key,
        )
        chunk_path.write_bytes(compressed_chunk)
        chunk_keys.append(chunk_key)
        chunk_digests.append(compute_sha256_hex(compressed_chunk))
    combined_hashes: bytes = ":".join(chunk_digests).encode(ENCODING_UTF8)
    root_sha256: str = compute_sha256_hex(combined_hashes)
    return {
        "chunk_keys": tuple(chunk_keys),
        "chunks": (ONE_INT, height, width),
        "compressor_id": norm_codec,
        "ngff_version": OME_NGFF_VERSION,
        "root_sha256": root_sha256,
        "shape": (frame_count, height, width),
        "store_name": store_path.name,
    }


def read_dod_ome_zarr_chunk(
    workspace_root: Path,
    store_rel_dir: Path | str,
    frame_index: int = ZERO_INT,
) -> list[list[int]]:
    """Validate OME-Zarr metadata and decompress a single 2D frame chunk.

    Args:
        workspace_root: Trusted workspace root directory.
        store_rel_dir: Relative directory path of the .ome.zarr store.
        frame_index: Temporal chunk index t to read.

    Returns:
        Decompressed 2D list of 8-bit pixel intensities.
    """
    store_path: Path = resolve_safe_path(workspace_root, store_rel_dir)
    zarray_path: Path = resolve_safe_path(store_path, "0/.zarray")
    meta: dict[str, Any] = validate_safe_zarr_metadata(
        workspace_root,
        zarray_path,
    )
    shape: Sequence[int] = meta["shape"]
    frame_count, height, width = (
        int(shape[ZERO_INT]),
        int(shape[ONE_INT]),
        int(shape[TWO_INT]),
    )
    verify_condition(
        ZERO_INT <= frame_index < frame_count,
        "Requested OME-Zarr frame_index out of bounds.",
    )
    chunk_path: Path = resolve_safe_path(
        store_path,
        f"0/{frame_index}.0.0",
    )
    raw_bytes: bytes = zlib.decompress(chunk_path.read_bytes())
    verify_condition(
        len(raw_bytes) == height * width,
        "Decompressed OME-Zarr chunk byte length mismatch.",
    )
    return [
        [raw_bytes[r_idx * width + c_idx] for c_idx in range(width)]
        for r_idx in range(height)
    ]


class CloudSandboxClient:
    """Local Azurite and fake-gcs-server client with offline fallback."""

    def __init__(
        self,
        workspace_root: Path,
        cloud_region: str = "uk-south",
        azurite_endpoint: str = "http://127.0.0.1:10000/devstoreaccount1",
        fake_gcs_endpoint: str = "http://127.0.0.1:4443/storage/v1/b",
        container_name: str = "unif-ai-disaster-relief",
        offline_fallback_enabled: bool = True,
    ) -> None:
        """Initialize the UK/EU cloud storage sandbox adapter.

        Args:
            workspace_root: Trusted workspace root directory.
            cloud_region: UK/EU cloud residency region string.
            azurite_endpoint: Loopback Azurite Blob service URL.
            fake_gcs_endpoint: Loopback fake-gcs-server service URL.
            container_name: Target bucket or container identifier.
            offline_fallback_enabled: Whether to store locally when offline.
        """
        self._workspace_root: Path = workspace_root.resolve()
        self._region: str = verify_uk_eu_residency(
            region=cloud_region,
            allowed_regions=ALLOWED_UK_EU_REGIONS,
        )
        self._azurite_url: str = validate_outbound_url(
            url=azurite_endpoint,
            allowed_schemes=ALLOWED_URL_SCHEMES,
            allowed_hosts=ALLOWED_HOSTS_DEFAULT,
        )
        self._gcs_url: str = validate_outbound_url(
            url=fake_gcs_endpoint,
            allowed_schemes=ALLOWED_URL_SCHEMES,
            allowed_hosts=ALLOWED_HOSTS_DEFAULT,
        )
        self._container_name: str = container_name.strip()
        self._offline_fallback_enabled: bool = offline_fallback_enabled
        self._fallback_dir: Path = resolve_safe_path(
            self._workspace_root,
            "sandbox_blobs",
        )
        self._fallback_dir.mkdir(parents=True, exist_ok=True)
        self._memory_blobs: dict[str, tuple[bytes, str]] = {}

    @property
    def region(self) -> str:
        """Return the verified UK/EU residency region."""
        return self._region

    def upload_batch_artifact(
        self,
        batch_id: str,
        payload: Mapping[str, Any],
        provider: str = "azurite",
    ) -> SandboxUploadReceipt:
        """Upload an anonymized relief telemetry payload to the sandbox.

        Args:
            batch_id: Unique mission batch identifier.
            payload: Anonymized mission batch dictionary.
            provider: Target emulator ('azurite' or 'gcs').

        Returns:
            SandboxUploadReceipt detailing the storage commitment.
        """
        norm_provider: str = provider.strip().lower()
        verify_condition(
            norm_provider in ("azurite", "gcs"),
            "Sandbox provider must be either 'azurite' or 'gcs'.",
        )
        blob_name: str = f"{batch_id.strip()}.json"
        serialized_bytes: bytes = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
        ).encode(ENCODING_UTF8)
        digest: str = compute_sha256_hex(serialized_bytes)
        base_url: str = (
            self._azurite_url
            if norm_provider == "azurite"
            else self._gcs_url
        )
        target_url: str = (
            f"{base_url.rstrip('/')}/{self._container_name}/{blob_name}"
        )
        validate_outbound_url(target_url)
        offline_used: bool = False
        status_code: int = HTTP_STATUS_CREATED
        try:
            request_obj: urllib.request.Request = urllib.request.Request(
                url=target_url,
                data=serialized_bytes,
                headers={"Content-Type": "application/json"},
                method="PUT",
            )
            with urllib.request.urlopen(
                request_obj,
                timeout=float(ONE_INT),
            ) as response:
                status_code = int(response.status)
        except (urllib.error.URLError, TimeoutError, OSError):
            verify_condition(
                self._offline_fallback_enabled,
                "Cloud sandbox unreachable and offline fallback disabled.",
            )
            offline_used = True
            status_code = HTTP_STATUS_OK
        safe_blob_path: Path = resolve_safe_path(
            self._fallback_dir,
            blob_name,
        )
        safe_blob_path.write_bytes(serialized_bytes)
        self._memory_blobs[blob_name] = (serialized_bytes, digest)
        return SandboxUploadReceipt(
            provider=norm_provider,
            region=self._region,
            container=self._container_name,
            blob_name=blob_name,
            sha256_digest=digest,
            offline_fallback_used=offline_used,
            status_code=status_code,
        )

    def retrieve_batch_artifact(self, blob_name: str) -> dict[str, Any]:
        """Retrieve and integrity-check a stored batch artifact.

        Args:
            blob_name: Stored JSON blob filename.

        Returns:
            Deserialized batch payload dictionary.
        """
        clean_name: str = blob_name.strip()
        if clean_name in self._memory_blobs:
            raw_bytes, expected_digest = self._memory_blobs[clean_name]
            verify_condition(
                verify_sha256_digest(raw_bytes, expected_digest),
                "Sandbox blob SHA-256 integrity verification failed.",
            )
            return dict(json.loads(raw_bytes.decode(ENCODING_UTF8)))
        safe_blob_path: Path = resolve_safe_path(
            self._fallback_dir,
            clean_name,
        )
        verify_condition(
            safe_blob_path.exists(),
            "Requested sandbox blob does not exist in storage.",
        )
        disk_bytes: bytes = safe_blob_path.read_bytes()
        return dict(json.loads(disk_bytes.decode(ENCODING_UTF8)))


def synthesize_zero_pii_uav_batch(
    frame_count: int = SYNTHETIC_FRAME_COUNT,
    width: int = SYNTHETIC_FRAME_WIDTH,
    height: int = SYNTHETIC_FRAME_HEIGHT,
) -> list[SyntheticUavFrame]:
    """Generate deterministic, zero-PII synthetic UAV frames and crops.

    Args:
        frame_count: Number of synthetic frames to synthesize.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        List of SyntheticUavFrame objects with normalized crop percentages.
    """
    verify_condition(
        ONE_INT <= frame_count <= MAX_FRAMES_PER_BATCH,
        "Synthetic frame count violates batch bounds.",
    )
    validate_image_dimensions(width=width, height=height)
    frames: list[SyntheticUavFrame] = []
    for frame_idx in range(frame_count):
        pct_shift: float = float(frame_idx) * CROP_PCT_STEP_OFFSET
        top_pct: float = min(
            CROP_PCT_MAX - CROP_PCT_STEP_OFFSET * TWO_INT,
            DEFAULT_CROP_TOP_PCT + pct_shift,
        )
        left_pct: float = min(
            CROP_PCT_MAX - CROP_PCT_STEP_OFFSET * TWO_INT,
            DEFAULT_CROP_LEFT_PCT + pct_shift,
        )
        bot_pct: float = min(
            CROP_PCT_MAX,
            max(top_pct + CROP_PCT_STEP_OFFSET, DEFAULT_CROP_BOTTOM_PCT),
        )
        right_pct: float = min(
            CROP_PCT_MAX,
            max(left_pct + CROP_PCT_STEP_OFFSET, DEFAULT_CROP_RIGHT_PCT),
        )
        y_start, x_start, y_end, x_end = convert_crop_percentages_to_box(
            top_pct=top_pct,
            left_pct=left_pct,
            bottom_pct=bot_pct,
            right_pct=right_pct,
            height=height,
            width=width,
        )
        rows: tuple[tuple[int, ...], ...] = tuple(
            tuple(
                SYNTHETIC_OBJECT_VALUE
                if y_start <= r_idx < y_end and x_start <= c_idx < x_end
                else SYNTHETIC_BACKGROUND_VALUE
                for c_idx in range(width)
            )
            for r_idx in range(height)
        )
        lat: float = (
            SYNTHETIC_BASE_LATITUDE
            + float(frame_idx) * SYNTHETIC_COORD_JITTER
        )
        lon: float = (
            SYNTHETIC_BASE_LONGITUDE
            - float(frame_idx) * SYNTHETIC_COORD_JITTER
        )
        xml_meta: str = (
            "<uav_telemetry>"
            f"<frame_index>{frame_idx}</frame_index>"
            "<sensor_mode>RELIEF_SYNTHETIC_OPTICAL</sensor_mode>"
            "<responder_contact>relief-ops@abertech.example.org"
            "</responder_contact>"
            "</uav_telemetry>"
        )
        frames.append(
            SyntheticUavFrame(
                frame_id=f"frame_relief_{frame_idx:03d}",
                pixel_grid=rows,
                sensitive_boxes=((y_start, x_start, y_end, x_end),),
                latitude=lat,
                longitude=lon,
                raw_metadata_xml=xml_meta,
                crop_percentages=((top_pct, left_pct, bot_pct, right_pct),),
            )
        )
    return frames
