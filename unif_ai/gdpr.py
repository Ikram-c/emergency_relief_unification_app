"""UK/EU GDPR privacy-by-design controls and EDPB compliance routines."""

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from unif_ai.constants import (
    ALLOWED_UK_EU_REGIONS,
    BASE_GRID_STEP_DEGREES,
    BLUR_BLOCK_SIZE_DEFAULT,
    BLUR_KERNEL_RADIUS_DEFAULT,
    BYTE_MAX_VALUE,
    CROP_PCT_MAX,
    CROP_PCT_MIN,
    DOMAIN_SUBJECT,
    DP_EPSILON_DEFAULT,
    DP_EPSILON_MAX,
    DP_SENSITIVITY_DEFAULT,
    ENCODING_UTF8,
    ENV_MASTER_SECRET_KEY,
    EPHEMERAL_SECRET_BYTES,
    FOUR_INT,
    GENESIS_PREV_HASH,
    H3_APERTURE_FACTOR,
    H3_BASE_RESOLUTION,
    H3_MIN_RESOLUTION,
    HALF_FLOAT,
    HKDF_KEY_LENGTH_BYTES,
    HUNDRED_FLOAT,
    LAPLACE_UNIFORM_CLIP,
    LATITUDE_MAX_DEGREES,
    LATITUDE_MIN_DEGREES,
    LONGITUDE_MAX_DEGREES,
    LONGITUDE_MIN_DEGREES,
    MAX_DETECTIONS_PER_FRAME,
    MAX_IMAGE_HEIGHT_DEFAULT,
    MAX_IMAGE_PIXELS_DEFAULT,
    MAX_IMAGE_WIDTH_DEFAULT,
    MAX_LOOP_ITERATIONS,
    ONE_FLOAT,
    ONE_INT,
    PSEUDONYM_HEX_PREFIX_LEN,
    REDACTED_PLACEHOLDER,
    REGEX_EMAIL_PATTERN,
    REGEX_IPV4_PATTERN,
    REGEX_PHONE_PATTERN,
    SPATIAL_K_MIN_DEFAULT,
    SPATIAL_MAX_COARSEN_STEPS,
    STAGING_TTL_SECONDS_DEFAULT,
    THIRTY_TWO_INT,
    THREE_INT,
    TWO_FLOAT,
    TWO_INT,
    ZERO_FLOAT,
    ZERO_INT,
)
from unif_ai.security import (
    compute_sha256_hex,
    resolve_safe_path,
    validate_image_dimensions,
    verify_condition,
)

EMAIL_REGEX: re.Pattern[str] = re.compile(REGEX_EMAIL_PATTERN)
PHONE_REGEX: re.Pattern[str] = re.compile(REGEX_PHONE_PATTERN)
IPV4_REGEX: re.Pattern[str] = re.compile(REGEX_IPV4_PATTERN)
_PROCESS_EPHEMERAL_SECRET: str = secrets.token_hex(EPHEMERAL_SECRET_BYTES)


@dataclass(frozen=True)
class SpatialAnonymisationResult:
    """Immutable outcome of aperture-7 H3 spatial k-anonymisation."""

    cell_counts: Mapping[str, float]
    point_cells: tuple[str, ...]
    effective_resolution: int
    uniqueness_rate: float
    suppressed_count: int


def get_master_secret_bytes(explicit_secret: str | None = None) -> bytes:
    """Retrieve the master secret from argument, environment, or CSPRNG.

    Args:
        explicit_secret: Optional override secret for isolated testing.

    Returns:
        UTF-8 encoded master secret byte sequence.
    """
    env_secret: str = os.environ.get(ENV_MASTER_SECRET_KEY, "").strip()
    raw_secret: str = (
        explicit_secret.strip()
        if explicit_secret is not None
        else (env_secret if env_secret else _PROCESS_EPHEMERAL_SECRET)
    )
    verify_condition(
        len(raw_secret) >= PSEUDONYM_HEX_PREFIX_LEN,
        "Master secret must meet minimum entropy length requirement.",
    )
    return raw_secret.encode(ENCODING_UTF8)


def hkdf_extract_sha256(salt: bytes, input_key_material: bytes) -> bytes:
    """Execute RFC 5869 HKDF-Extract using HMAC-SHA256.

    Args:
        salt: Non-secret salt bytes (domain and epoch context).
        input_key_material: Secret input key material bytes.

    Returns:
        Pseudorandom key (PRK) of 32 bytes.
    """
    effective_salt: bytes = (
        salt if len(salt) > ZERO_INT else bytes(HKDF_KEY_LENGTH_BYTES)
    )
    return hmac.new(
        effective_salt,
        input_key_material,
        hashlib.sha256,
    ).digest()


def hkdf_expand_sha256(
    pseudorandom_key: bytes,
    info: bytes,
    length: int = HKDF_KEY_LENGTH_BYTES,
) -> bytes:
    """Execute RFC 5869 HKDF-Expand using HMAC-SHA256.

    Args:
        pseudorandom_key: 32-byte PRK from hkdf_extract_sha256.
        info: Context and application specific information bytes.
        length: Desired output key length in bytes.

    Returns:
        Derived subkey bytes of the requested length.
    """
    verify_condition(
        ONE_INT <= length <= HKDF_KEY_LENGTH_BYTES * TWO_INT,
        "Requested HKDF subkey length violates bounds.",
    )
    hash_len: int = HKDF_KEY_LENGTH_BYTES
    blocks_needed: int = (length + hash_len - ONE_INT) // hash_len
    output: bytearray = bytearray()
    previous_block: bytes = b""
    for block_index in range(ONE_INT, blocks_needed + ONE_INT):
        hmac_input: bytes = previous_block + info + bytes([block_index])
        previous_block = hmac.new(
            pseudorandom_key,
            hmac_input,
            hashlib.sha256,
        ).digest()
        output.extend(previous_block)
    return bytes(output[:length])


def derive_domain_subkey(
    domain: str,
    epoch: str,
    master_secret: str | None = None,
) -> bytes:
    """Derive a domain-separated subkey per EDPB 01/2025 guidelines.

    Args:
        domain: Logical data domain identifier.
        epoch: Key rotation epoch identifier (such as 2026-Q4).
        master_secret: Optional explicit master secret string.

    Returns:
        Isolated 32-byte domain subkey.
    """
    verify_condition(
        len(domain.strip()) > ZERO_INT and len(epoch.strip()) > ZERO_INT,
        "Domain and epoch strings must be non-empty for HKDF derivation.",
    )
    ikm: bytes = get_master_secret_bytes(master_secret)
    salt: bytes = f"unif_ai_salt:{epoch.strip()}".encode(ENCODING_UTF8)
    info: bytes = f"unif_ai_domain:{domain.strip()}".encode(ENCODING_UTF8)
    prk: bytes = hkdf_extract_sha256(salt=salt, input_key_material=ikm)
    return hkdf_expand_sha256(
        pseudorandom_key=prk,
        info=info,
        length=HKDF_KEY_LENGTH_BYTES,
    )


def pseudonymise_identifier(
    raw_identifier: str,
    domain: str,
    epoch: str = "2026-Q4",
    master_secret: str | None = None,
) -> str:
    """Compute a deterministic within-domain, unlinkable cross-domain token.

    Args:
        raw_identifier: Raw personal or device identifier string.
        domain: Purpose-specific domain string.
        epoch: Rotation epoch tag.
        master_secret: Optional override secret for testing.

    Returns:
        Prefixed pseudonym token string.
    """
    normalized_id: str = raw_identifier.strip()
    verify_condition(
        len(normalized_id) > ZERO_INT,
        "Raw identifier for pseudonymisation must not be empty.",
    )
    subkey: bytes = derive_domain_subkey(
        domain=domain,
        epoch=epoch,
        master_secret=master_secret,
    )
    digest_hex: str = hmac.new(
        subkey,
        normalized_id.encode(ENCODING_UTF8),
        hashlib.sha256,
    ).hexdigest()
    token_body: str = digest_hex[:PSEUDONYM_HEX_PREFIX_LEN]
    return f"pseudo_{domain}_{token_body}"


def scrub_text_pii(raw_text: str) -> str:
    """Scrub emails, UK phone numbers, and IPv4 addresses from text.

    Args:
        raw_text: Input log message or metadata string.

    Returns:
        Sanitized string with all matched PII patterns replaced.
    """
    verify_condition(
        isinstance(raw_text, str),
        "Input to scrub_text_pii must be a string.",
    )
    scrubbed: str = EMAIL_REGEX.sub(REDACTED_PLACEHOLDER, raw_text)
    scrubbed = PHONE_REGEX.sub(REDACTED_PLACEHOLDER, scrubbed)
    scrubbed = IPV4_REGEX.sub(REDACTED_PLACEHOLDER, scrubbed)
    return scrubbed


def strip_exif_and_clone_pixels(
    pixel_grid: Sequence[Sequence[int]],
    max_width: int = MAX_IMAGE_WIDTH_DEFAULT,
    max_height: int = MAX_IMAGE_HEIGHT_DEFAULT,
    max_pixels: int = MAX_IMAGE_PIXELS_DEFAULT,
) -> list[list[int]]:
    """Strip image metadata by cloning pixel intensities into a clean array.

    Args:
        pixel_grid: 2D sequence of 8-bit grayscale pixel intensities.
        max_width: Maximum permitted image width.
        max_height: Maximum permitted image height.
        max_pixels: Maximum permitted total pixels.

    Returns:
        Freshly allocated 2D list of clamped 8-bit pixel values with no EXIF.
    """
    height: int = len(pixel_grid)
    verify_condition(
        height > ZERO_INT,
        "Pixel grid must contain at least one row.",
    )
    width: int = len(pixel_grid[ZERO_INT])
    validate_image_dimensions(
        width=width,
        height=height,
        max_width=max_width,
        max_height=max_height,
        max_pixels=max_pixels,
    )
    clean_buffer: list[list[int]] = []
    for row_idx in range(height):
        row_data: Sequence[int] = pixel_grid[row_idx]
        verify_condition(
            len(row_data) == width,
            "All pixel grid rows must have uniform width.",
        )
        clean_row: list[int] = [
            max(ZERO_INT, min(BYTE_MAX_VALUE, int(row_data[col_idx])))
            for col_idx in range(width)
        ]
        clean_buffer.append(clean_row)
    return clean_buffer


def convert_crop_percentages_to_box(
    top_pct: float,
    left_pct: float,
    bottom_pct: float,
    right_pct: float,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    """Convert normalized [0.0, 100.0] crop percentages into pixel bounds.

    Args:
        top_pct: Top edge percentage in [0.0, 100.0).
        left_pct: Left edge percentage in [0.0, 100.0).
        bottom_pct: Bottom edge percentage in (top_pct, 100.0].
        right_pct: Right edge percentage in (left_pct, 100.0].
        height: Frame height in pixels.
        width: Frame width in pixels.

    Returns:
        Clamped integer pixel box tuple (y_min, x_min, y_max, x_max).
    """
    validate_image_dimensions(width=width, height=height)
    verify_condition(
        CROP_PCT_MIN <= top_pct < bottom_pct <= CROP_PCT_MAX,
        "Vertical crop percentages must satisfy 0.0 <= top < bottom <= 100.0.",
    )
    verify_condition(
        CROP_PCT_MIN <= left_pct < right_pct <= CROP_PCT_MAX,
        "Horizontal crop percentages must satisfy 0.0 <= left < right <= 100.",
    )
    y_min: int = max(
        ZERO_INT,
        min(height - ONE_INT, int(round((top_pct / HUNDRED_FLOAT) * height))),
    )
    y_max: int = max(
        y_min + ONE_INT,
        min(height, int(round((bottom_pct / HUNDRED_FLOAT) * height))),
    )
    x_min: int = max(
        ZERO_INT,
        min(width - ONE_INT, int(round((left_pct / HUNDRED_FLOAT) * width))),
    )
    x_max: int = max(
        x_min + ONE_INT,
        min(width, int(round((right_pct / HUNDRED_FLOAT) * width))),
    )
    return (y_min, x_min, y_max, x_max)


def apply_irreversible_visual_blur(
    pixel_grid: Sequence[Sequence[int]],
    bounding_boxes: Sequence[tuple[int, int, int, int]],
    block_size: int = BLUR_BLOCK_SIZE_DEFAULT,
    kernel_radius: int = BLUR_KERNEL_RADIUS_DEFAULT,
) -> list[list[int]]:
    """Apply block quantization plus separable 2D smoothing over ROIs.

    Args:
        pixel_grid: Clean 2D pixel intensity array.
        bounding_boxes: Sequence of (y_min, x_min, y_max, x_max) regions.
        block_size: Spatial block quantization size in pixels.
        kernel_radius: Separable 2D smoothing kernel radius.

    Returns:
        Redacted 2D pixel intensity array compliant with EDPB 3/2019.
    """
    redacted: list[list[int]] = strip_exif_and_clone_pixels(pixel_grid)
    height: int = len(redacted)
    width: int = len(redacted[ZERO_INT])
    verify_condition(
        block_size >= TWO_INT and kernel_radius >= ONE_INT,
        "Blur block size and kernel radius must meet minimum bounds.",
    )
    box_limit: int = min(len(bounding_boxes), MAX_DETECTIONS_PER_FRAME)
    for box_idx in range(box_limit):
        y_min, x_min, y_max, x_max = bounding_boxes[box_idx]
        r_start: int = max(ZERO_INT, min(height, int(y_min)))
        r_end: int = max(r_start, min(height, int(y_max)))
        c_start: int = max(ZERO_INT, min(width, int(x_min)))
        c_end: int = max(c_start, min(width, int(x_max)))
        for b_row in range(r_start, r_end, block_size):
            b_row_end: int = min(r_end, b_row + block_size)
            for b_col in range(c_start, c_end, block_size):
                b_col_end: int = min(c_end, b_col + block_size)
                pixel_sum: int = ZERO_INT
                pixel_count: int = ZERO_INT
                for rr in range(b_row, b_row_end):
                    for cc in range(b_col, b_col_end):
                        pixel_sum += redacted[rr][cc]
                        pixel_count += ONE_INT
                mean_val: int = (
                    pixel_sum // pixel_count
                    if pixel_count > ZERO_INT
                    else ZERO_INT
                )
                for rr in range(b_row, b_row_end):
                    for cc in range(b_col, b_col_end):
                        redacted[rr][cc] = mean_val
        horiz_pass: list[list[int]] = [list(row) for row in redacted]
        for rr in range(r_start, r_end):
            for cc in range(c_start, c_end):
                acc: int = ZERO_INT
                cnt: int = ZERO_INT
                k_left: int = max(c_start, cc - kernel_radius)
                k_right: int = min(c_end - ONE_INT, cc + kernel_radius)
                for kc in range(k_left, k_right + ONE_INT):
                    acc += redacted[rr][kc]
                    cnt += ONE_INT
                horiz_pass[rr][cc] = acc // max(ONE_INT, cnt)
        for rr in range(r_start, r_end):
            for cc in range(c_start, c_end):
                acc_v: int = ZERO_INT
                cnt_v: int = ZERO_INT
                k_top: int = max(r_start, rr - kernel_radius)
                k_bot: int = min(r_end - ONE_INT, rr + kernel_radius)
                for kr in range(k_top, k_bot + ONE_INT):
                    acc_v += horiz_pass[kr][cc]
                    cnt_v += ONE_INT
                redacted[rr][cc] = acc_v // max(ONE_INT, cnt_v)
    return redacted


def apply_crop_percentage_blur(
    pixel_grid: Sequence[Sequence[int]],
    crop_percentages: Sequence[tuple[float, float, float, float]],
    block_size: int = BLUR_BLOCK_SIZE_DEFAULT,
    kernel_radius: int = BLUR_KERNEL_RADIUS_DEFAULT,
) -> list[list[int]]:
    """Redact percentage-defined ROI crops using EDPB 3/2019 visual blur.

    Args:
        pixel_grid: 2D sequence of 8-bit grayscale pixel intensities.
        crop_percentages: Sequence of (top_pct, left_pct, bottom_pct, right).
        block_size: Spatial block quantization size in pixels.
        kernel_radius: Separable 2D smoothing kernel radius.

    Returns:
        Redacted 2D pixel intensity array.
    """
    height: int = len(pixel_grid)
    verify_condition(
        height > ZERO_INT,
        "Pixel grid must contain at least one row for percentage blur.",
    )
    width: int = len(pixel_grid[ZERO_INT])
    pixel_boxes: list[tuple[int, int, int, int]] = [
        convert_crop_percentages_to_box(
            top_pct=crop[ZERO_INT],
            left_pct=crop[ONE_INT],
            bottom_pct=crop[TWO_INT],
            right_pct=crop[THREE_INT],
            height=height,
            width=width,
        )
        for crop in crop_percentages[:MAX_DETECTIONS_PER_FRAME]
    ]
    return apply_irreversible_visual_blur(
        pixel_grid=pixel_grid,
        bounding_boxes=pixel_boxes,
        block_size=block_size,
        kernel_radius=kernel_radius,
    )


def verify_uk_eu_residency(
    region: str,
    allowed_regions: Sequence[str] = ALLOWED_UK_EU_REGIONS,
) -> str:
    """Verify that the target cloud storage region is in the UK/EU allowlist.

    Args:
        region: Configured cloud region identifier string.
        allowed_regions: Permitted UK and EEA region identifiers.

    Returns:
        Normalized region string.
    """
    normalized: str = region.strip().lower()
    verify_condition(
        normalized in allowed_regions,
        f"Cloud region '{normalized}' violates UK/EU residency policy.",
    )
    return normalized


def project_to_h3_aperture7_cell(
    latitude: float,
    longitude: float,
    resolution: int = H3_BASE_RESOLUTION,
) -> str:
    """Project WGS84 coordinates onto an aperture-7 hexagonal cell index.

    Args:
        latitude: Latitude in decimal degrees (-90.0 to 90.0).
        longitude: Longitude in decimal degrees (-180.0 to 180.0).
        resolution: H3 resolution level (between 3 and 9).

    Returns:
        Deterministic aperture-7 hexagonal cell identifier string.
    """
    verify_condition(
        LATITUDE_MIN_DEGREES <= latitude <= LATITUDE_MAX_DEGREES,
        "Latitude out of valid WGS84 bounds.",
    )
    verify_condition(
        LONGITUDE_MIN_DEGREES <= longitude <= LONGITUDE_MAX_DEGREES,
        "Longitude out of valid WGS84 bounds.",
    )
    verify_condition(
        H3_MIN_RESOLUTION <= resolution <= H3_BASE_RESOLUTION,
        "Resolution out of supported aperture-7 bounds.",
    )
    coarsen_exponent: int = H3_BASE_RESOLUTION - resolution
    step_deg: float = BASE_GRID_STEP_DEGREES * (
        H3_APERTURE_FACTOR**coarsen_exponent
    )
    q_coord: float = longitude / step_deg
    r_coord: float = latitude / step_deg - HALF_FLOAT * round(q_coord)
    q_idx: int = int(round(q_coord))
    r_idx: int = int(round(r_coord))
    return f"h3_r{resolution}_{q_idx}_{r_idx}"


def add_laplace_dp_noise(
    raw_count: int,
    cell_key: str,
    epsilon: float = DP_EPSILON_DEFAULT,
    sensitivity: float = DP_SENSITIVITY_DEFAULT,
    master_secret: str | None = None,
) -> float:
    """Add calibrated Laplace Differential Privacy noise to a cell count.

    Args:
        raw_count: Exact integer count in the spatial cell.
        cell_key: Hexagonal cell identifier for deterministic seeding.
        epsilon: Privacy budget parameter (must be in (0.0, 1.0]).
        sensitivity: L1 query sensitivity (default 1.0).
        master_secret: Optional secret for keyed uniform sampling.

    Returns:
        Non-negative noisy count rounded to two decimal places.
    """
    verify_condition(
        ZERO_FLOAT < epsilon <= DP_EPSILON_MAX,
        "Differential privacy epsilon must satisfy 0.0 < epsilon <= 1.0.",
    )
    verify_condition(
        sensitivity > ZERO_FLOAT,
        "Differential privacy sensitivity must be positive.",
    )
    scale_b: float = sensitivity / epsilon
    seed_key: bytes = get_master_secret_bytes(master_secret)
    digest: bytes = hmac.new(
        seed_key,
        f"dp_laplace:{cell_key}:{raw_count}".encode(ENCODING_UTF8),
        hashlib.sha256,
    ).digest()
    uint_val: int = int.from_bytes(digest[:FOUR_INT], byteorder="big")
    max_uint: float = float(ONE_INT << THIRTY_TWO_INT)
    uniform_u: float = (float(uint_val) / max_uint) - HALF_FLOAT
    clipped_u: float = max(
        -LAPLACE_UNIFORM_CLIP,
        min(LAPLACE_UNIFORM_CLIP, uniform_u),
    )
    sign_u: float = -ONE_FLOAT if clipped_u < ZERO_FLOAT else ONE_FLOAT
    laplace_noise: float = -scale_b * sign_u * math.log(
        ONE_FLOAT - TWO_FLOAT * abs(clipped_u)
    )
    noisy_total: float = max(ZERO_FLOAT, float(raw_count) + laplace_noise)
    return round(noisy_total, TWO_INT)


def anonymise_spatial_coordinates(
    coordinates: Sequence[tuple[float, float]],
    k_min: int = SPATIAL_K_MIN_DEFAULT,
    epsilon: float = DP_EPSILON_DEFAULT,
    master_secret: str | None = None,
) -> SpatialAnonymisationResult:
    """Coarsen H3 resolution until uniqueness U=0.0, then apply Laplace DP.

    Args:
        coordinates: Sequence of (latitude, longitude) pairs.
        k_min: Minimum crowd size k per spatial cell.
        epsilon: Laplace Differential Privacy epsilon (<= 1.0).
        master_secret: Optional secret for deterministic DP noise.

    Returns:
        SpatialAnonymisationResult with U=0.0 and DP-noised cell counts.
    """
    verify_condition(
        k_min >= TWO_INT,
        "Spatial k-anonymity threshold k_min must be at least 2.",
    )
    total_points: int = len(coordinates)
    verify_condition(
        total_points <= MAX_LOOP_ITERATIONS,
        "Coordinate sequence exceeds maximum iteration ceiling.",
    )
    if total_points == ZERO_INT:
        return SpatialAnonymisationResult(
            cell_counts={},
            point_cells=(),
            effective_resolution=H3_BASE_RESOLUTION,
            uniqueness_rate=ZERO_FLOAT,
            suppressed_count=ZERO_INT,
        )
    current_res: int = H3_BASE_RESOLUTION
    assigned_cells: list[str] = []
    counts_by_cell: dict[str, int] = {}
    for step_idx in range(SPATIAL_MAX_COARSEN_STEPS + ONE_INT):
        candidate_res: int = max(
            H3_MIN_RESOLUTION,
            H3_BASE_RESOLUTION - step_idx,
        )
        current_res = candidate_res
        assigned_cells = []
        counts_by_cell = {}
        for pt_idx in range(total_points):
            lat, lon = coordinates[pt_idx]
            cell_id: str = project_to_h3_aperture7_cell(
                latitude=lat,
                longitude=lon,
                resolution=candidate_res,
            )
            assigned_cells.append(cell_id)
            counts_by_cell[cell_id] = (
                counts_by_cell.get(cell_id, ZERO_INT) + ONE_INT
            )
        unique_violations: int = sum(
            count for count in counts_by_cell.values() if count < k_min
        )
        if unique_violations == ZERO_INT or candidate_res == H3_MIN_RESOLUTION:
            break
    suppressed_points: int = ZERO_INT
    filtered_cells: list[str] = []
    dp_cell_counts: dict[str, float] = {}
    cell_keys: list[str] = sorted(counts_by_cell.keys())
    for key_idx in range(min(len(cell_keys), MAX_LOOP_ITERATIONS)):
        cell_key: str = cell_keys[key_idx]
        raw_cnt: int = counts_by_cell[cell_key]
        if raw_cnt < k_min:
            suppressed_points += raw_cnt
        else:
            dp_cell_counts[cell_key] = add_laplace_dp_noise(
                raw_count=raw_cnt,
                cell_key=cell_key,
                epsilon=epsilon,
                master_secret=master_secret,
            )
    for pt_idx in range(total_points):
        cell_name: str = assigned_cells[pt_idx]
        if counts_by_cell.get(cell_name, ZERO_INT) >= k_min:
            filtered_cells.append(cell_name)
    return SpatialAnonymisationResult(
        cell_counts=dp_cell_counts,
        point_cells=tuple(filtered_cells),
        effective_resolution=current_res,
        uniqueness_rate=ZERO_FLOAT,
        suppressed_count=suppressed_points,
    )


def purge_expired_staging_files(
    workspace_root: Path,
    staging_dir: Path | str,
    ttl_seconds: int = STAGING_TTL_SECONDS_DEFAULT,
    now_epoch: float | None = None,
) -> list[Path]:
    """Purge staging files older than ttl_seconds per GDPR Art. 5(1)(e).

    Args:
        workspace_root: Trusted workspace root directory.
        staging_dir: Staging subdirectory path.
        ttl_seconds: Maximum retention age in seconds.
        now_epoch: Optional current UNIX timestamp for deterministic tests.

    Returns:
        List of canonical paths that were purged.
    """
    verify_condition(
        ttl_seconds > ZERO_INT,
        "Staging TTL seconds must be positive.",
    )
    safe_staging: Path = resolve_safe_path(workspace_root, staging_dir)
    if not safe_staging.exists():
        return []
    reference_time: float = (
        now_epoch if now_epoch is not None else time.time()
    )
    purged_files: list[Path] = []
    candidates: list[Path] = sorted(safe_staging.iterdir())
    loop_limit: int = min(len(candidates), MAX_LOOP_ITERATIONS)
    for idx in range(loop_limit):
        item_path: Path = candidates[idx]
        if item_path.is_file() and not item_path.is_symlink():
            age_seconds: float = reference_time - item_path.stat().st_mtime
            if age_seconds >= float(ttl_seconds):
                item_path.unlink()
                purged_files.append(item_path)
    return purged_files


class AccountabilityLedger:
    """PII-free, SHA-256 hash-chained JSONL accountability ledger."""

    def __init__(
        self,
        workspace_root: Path,
        ledger_path: Path | str,
    ) -> None:
        """Initialize the accountability ledger inside workspace_root.

        Args:
            workspace_root: Trusted workspace root path.
            ledger_path: Relative or absolute path to the JSONL ledger file.
        """
        self._workspace_root: Path = workspace_root.resolve()
        self._ledger_path: Path = resolve_safe_path(
            self._workspace_root,
            ledger_path,
        )
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def ledger_path(self) -> Path:
        """Return the canonical path to the JSONL ledger file."""
        return self._ledger_path

    def append_event(
        self,
        event_type: str,
        pseudonym_token: str,
        details: str,
        timestamp_epoch: float | None = None,
    ) -> dict[str, Any]:
        """Append a PII-scrubbed, SHA-256 chained event record to the ledger.

        Args:
            event_type: Categorical identifier of the GDPR/pipeline event.
            pseudonym_token: Domain-separated pseudonym or batch token.
            details: Operational detail string (automatically PII-scrubbed).
            timestamp_epoch: Optional explicit timestamp for testing.

        Returns:
            Dictionary representing the newly committed ledger entry.
        """
        prev_hash: str = self._get_last_entry_hash()
        clean_details: str = scrub_text_pii(details)
        clean_token: str = scrub_text_pii(pseudonym_token)
        event_time: float = (
            round(timestamp_epoch, TWO_INT)
            if timestamp_epoch is not None
            else round(time.time(), TWO_INT)
        )
        canonical_payload: dict[str, Any] = {
            "details": clean_details,
            "event_type": event_type.strip(),
            "prev_hash": prev_hash,
            "pseudonym_token": clean_token,
            "timestamp": event_time,
        }
        serialized_core: str = json.dumps(
            canonical_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        entry_hash: str = compute_sha256_hex(
            serialized_core.encode(ENCODING_UTF8)
        )
        full_entry: dict[str, Any] = dict(canonical_payload)
        full_entry["entry_hash"] = entry_hash
        line_text: str = (
            json.dumps(full_entry, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        with self._ledger_path.open("a", encoding=ENCODING_UTF8) as handle:
            handle.write(line_text)
        return full_entry

    def verify_chain(self) -> bool:
        """Verify cryptographic continuity across every JSONL ledger line.

        Returns:
            True if all hashes and prev_hash links are intact, False otherwise.
        """
        if not self._ledger_path.exists():
            return True
        raw_lines: list[str] = (
            self._ledger_path.read_text(encoding=ENCODING_UTF8).splitlines()
        )
        verify_condition(
            len(raw_lines) <= MAX_LOOP_ITERATIONS,
            "Ledger line count exceeds verification loop bound.",
        )
        expected_prev: str = GENESIS_PREV_HASH
        for idx in range(min(len(raw_lines), MAX_LOOP_ITERATIONS)):
            line_str: str = raw_lines[idx].strip()
            if not line_str:
                continue
            record: dict[str, Any] = json.loads(line_str)
            actual_prev: str = str(record.get("prev_hash", ""))
            stored_hash: str = str(record.get("entry_hash", ""))
            if not hmac.compare_digest(actual_prev, expected_prev):
                return False
            core_payload: dict[str, Any] = {
                "details": record.get("details"),
                "event_type": record.get("event_type"),
                "prev_hash": actual_prev,
                "pseudonym_token": record.get("pseudonym_token"),
                "timestamp": record.get("timestamp"),
            }
            recomputed_core: str = json.dumps(
                core_payload,
                sort_keys=True,
                separators=(",", ":"),
            )
            recomputed_hash: str = compute_sha256_hex(
                recomputed_core.encode(ENCODING_UTF8)
            )
            if not hmac.compare_digest(stored_hash, recomputed_hash):
                return False
            expected_prev = stored_hash
        return True

    def _get_last_entry_hash(self) -> str:
        """Return the entry_hash of the most recent ledger record."""
        if not self._ledger_path.exists():
            return GENESIS_PREV_HASH
        lines: list[str] = (
            self._ledger_path.read_text(encoding=ENCODING_UTF8).splitlines()
        )
        for idx in range(len(lines) - ONE_INT, -ONE_INT, -ONE_INT):
            candidate: str = lines[idx].strip()
            if candidate:
                parsed: dict[str, Any] = json.loads(candidate)
                return str(parsed.get("entry_hash", GENESIS_PREV_HASH))
        return GENESIS_PREV_HASH


def execute_right_to_erasure(
    raw_subject_id: str,
    workspace_root: Path,
    output_dir: Path | str,
    database_path: Path | str,
    ledger: AccountabilityLedger,
    epoch: str = "2026-Q4",
    master_secret: str | None = None,
) -> dict[str, int]:
    """Execute GDPR Article 17 Right to Erasure across JSON, Zarr, and SQLite.

    Args:
        raw_subject_id: Raw identifier of the data subject requesting erasure.
        workspace_root: Trusted workspace root path.
        output_dir: Directory containing JSON output batch artifacts.
        database_path: SQLite database file path.
        ledger: AccountabilityLedger instance to record the erasure audit.
        epoch: HKDF key rotation epoch string.
        master_secret: Optional explicit master secret for testing.

    Returns:
        Summary dictionary with deleted SQLite rows and scrubbed files.
    """
    subject_token: str = pseudonymise_identifier(
        raw_identifier=raw_subject_id,
        domain=DOMAIN_SUBJECT,
        epoch=epoch,
        master_secret=master_secret,
    )
    safe_output: Path = resolve_safe_path(workspace_root, output_dir)
    safe_db: Path = resolve_safe_path(workspace_root, database_path)
    safe_zarr_root: Path = resolve_safe_path(workspace_root, "zarr_stores")
    scrubbed_json_files: int = ZERO_INT
    if safe_output.exists():
        json_files: list[Path] = sorted(safe_output.glob("*.json"))
        for idx in range(min(len(json_files), MAX_LOOP_ITERATIONS)):
            file_path: Path = json_files[idx]
            if file_path.is_symlink():
                continue
            content_text: str = file_path.read_text(encoding=ENCODING_UTF8)
            if subject_token in content_text:
                updated_text: str = content_text.replace(
                    subject_token,
                    "[ERASED_ART17_SUBJECT]",
                )
                file_path.write_text(updated_text, encoding=ENCODING_UTF8)
                scrubbed_json_files += ONE_INT
    scrubbed_zarr_stores: int = ZERO_INT
    if safe_zarr_root.exists():
        zattrs_files: list[Path] = sorted(safe_zarr_root.rglob(".zattrs"))
        for idx in range(min(len(zattrs_files), MAX_LOOP_ITERATIONS)):
            zattr_path: Path = zattrs_files[idx]
            if zattr_path.is_symlink():
                continue
            z_text: str = zattr_path.read_text(encoding=ENCODING_UTF8)
            if subject_token in z_text:
                z_updated: str = z_text.replace(
                    subject_token,
                    "[ERASED_ART17_SUBJECT]",
                )
                zattr_path.write_text(z_updated, encoding=ENCODING_UTF8)
                scrubbed_zarr_stores += ONE_INT
    deleted_db_rows: int = ZERO_INT
    if safe_db.exists():
        with sqlite3.connect(str(safe_db)) as conn:
            cursor: sqlite3.Cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = ? AND name = ?",
                ("table", "mission_batches"),
            )
            if cursor.fetchone() is not None:
                cursor.execute(
                    "DELETE FROM mission_batches WHERE subject_token = ?",
                    (subject_token,),
                )
                deleted_db_rows = max(ZERO_INT, cursor.rowcount)
                conn.commit()
    ledger.append_event(
        event_type="GDPR_ARTICLE_17_ERASURE",
        pseudonym_token=subject_token,
        details=(
            f"Erased rows={deleted_db_rows}, "
            f"scrubbed_files={scrubbed_json_files}, "
            f"scrubbed_zarr={scrubbed_zarr_stores}"
        ),
    )
    return {
        "deleted_db_rows": deleted_db_rows,
        "scrubbed_json_files": scrubbed_json_files,
        "scrubbed_zarr_stores": scrubbed_zarr_stores,
    }
