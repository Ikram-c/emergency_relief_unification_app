"""OWASP ASVS v5.0 security and OME-Zarr hardening controls for unif_ai."""

import hashlib
import hmac
import io
import json
import math
import subprocess
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from unif_ai.constants import (
    ALLOWED_BINARIES_DEFAULT,
    ALLOWED_HOSTS_DEFAULT,
    ALLOWED_URL_SCHEMES,
    ALLOWED_ZARR_CODECS,
    ENCODING_UTF8,
    MAX_ARCHIVE_MEMBERS_DEFAULT,
    MAX_DTYPE_ITEMSIZE_BYTES,
    MAX_FILE_SIZE_BYTES_DEFAULT,
    MAX_IMAGE_HEIGHT_DEFAULT,
    MAX_IMAGE_PIXELS_DEFAULT,
    MAX_IMAGE_WIDTH_DEFAULT,
    MAX_LOOP_ITERATIONS,
    MAX_SUBPROCESS_ARGS,
    MAX_ZARR_CHUNK_BYTES,
    MAX_ZARR_DIMENSIONS,
    ONE_INT,
    SHA256_HEX_LENGTH,
    SUBPROCESS_TIMEOUT_SECONDS,
    XML_FORBIDDEN_TOKENS,
    ZERO_INT,
)


class SecurityViolationError(ValueError):
    """Raised when an OWASP ASVS security control or invariant fails."""


def verify_condition(condition: bool, message: str) -> None:
    """Verify a runtime precondition or postcondition invariant.

    Args:
        condition: Boolean expression that must evaluate to True.
        message: Diagnostic failure description without sensitive PII.

    Raises:
        SecurityViolationError: If the evaluated condition is False.
    """
    if not condition:
        raise SecurityViolationError(message)


def compute_sha256_hex(payload: bytes) -> str:
    """Compute the lowercase hexadecimal SHA-256 digest of a byte buffer.

    Args:
        payload: Raw byte sequence to hash.

    Returns:
        Hexadecimal SHA-256 digest string of length 64.
    """
    verify_condition(
        isinstance(payload, (bytes, bytearray)),
        "Payload must be a byte sequence for SHA-256 hashing.",
    )
    digest: str = hashlib.sha256(payload).hexdigest()
    verify_condition(
        len(digest) == SHA256_HEX_LENGTH,
        "Computed SHA-256 digest length invariant failed.",
    )
    return digest


def verify_sha256_digest(payload: bytes, expected_hex_digest: str) -> bool:
    """Verify a byte buffer against an expected SHA-256 hex digest.

    Args:
        payload: Raw bytes to verify.
        expected_hex_digest: Expected 64-character hex SHA-256 string.

    Returns:
        True if digests match in constant time, False otherwise.
    """
    verify_condition(
        len(expected_hex_digest) == SHA256_HEX_LENGTH,
        "Expected SHA-256 digest must be 64 hex characters.",
    )
    actual_digest: str = compute_sha256_hex(payload)
    return hmac.compare_digest(
        actual_digest.encode(ENCODING_UTF8),
        expected_hex_digest.lower().encode(ENCODING_UTF8),
    )


def resolve_safe_path(
    workspace_root: Path,
    candidate_path: Path | str,
    reject_symlinks: bool = True,
) -> Path:
    """Canonicalize a path and enforce containment inside workspace_root.

    Args:
        workspace_root: Trusted base directory path.
        candidate_path: Untrusted relative or absolute path input.
        reject_symlinks: Whether to reject symbolic links immediately.

    Returns:
        Canonicalized Path guaranteed to reside inside workspace_root.

    Raises:
        SecurityViolationError: On traversal, null bytes, or symlink escape.
    """
    raw_text: str = str(candidate_path)
    verify_condition(
        "\x00" not in raw_text,
        "Null byte detected in candidate file path.",
    )
    raw_parts: tuple[str, ...] = Path(raw_text).parts
    verify_condition(
        ".." not in raw_parts,
        "Parent directory traversal token rejected in path.",
    )
    canonical_root: Path = workspace_root.resolve()
    unverified_target: Path = (
        Path(raw_text)
        if Path(raw_text).is_absolute()
        else canonical_root / raw_text
    )
    if reject_symlinks and unverified_target.is_symlink():
        raise SecurityViolationError(
            "Symbolic link rejected during path resolution."
        )
    canonical_target: Path = unverified_target.resolve()
    verify_condition(
        canonical_target.is_relative_to(canonical_root),
        "Resolved path escapes the designated workspace root.",
    )
    return canonical_target


def validate_file_size(
    file_path: Path,
    max_bytes: int = MAX_FILE_SIZE_BYTES_DEFAULT,
) -> int:
    """Verify that an existing regular file fits within the byte ceiling.

    Args:
        file_path: Canonical path to the target file.
        max_bytes: Maximum permitted file size in bytes.

    Returns:
        Verified size of the file in bytes.
    """
    verify_condition(
        max_bytes > ZERO_INT,
        "Maximum byte ceiling must be positive.",
    )
    verify_condition(
        file_path.exists() and file_path.is_file(),
        "Target path must be an existing regular file.",
    )
    verify_condition(
        not file_path.is_symlink(),
        "Symbolic link file rejected during size check.",
    )
    byte_count: int = file_path.stat().st_size
    verify_condition(
        ZERO_INT <= byte_count <= max_bytes,
        "File byte size exceeds configured security ceiling.",
    )
    return byte_count


def load_safe_yaml_config(
    workspace_root: Path,
    config_path: Path | str,
    max_bytes: int = MAX_FILE_SIZE_BYTES_DEFAULT,
) -> dict[str, Any]:
    """Load and validate a YAML configuration file using yaml.safe_load.

    Args:
        workspace_root: Trusted workspace directory.
        config_path: Path to the YAML configuration file.
        max_bytes: Maximum allowed YAML byte size.

    Returns:
        Parsed top-level dictionary of configuration settings.
    """
    safe_path: Path = resolve_safe_path(workspace_root, config_path)
    validate_file_size(safe_path, max_bytes=max_bytes)
    raw_content: str = safe_path.read_text(encoding=ENCODING_UTF8)
    parsed_data: Any = yaml.safe_load(raw_content)
    verify_condition(
        isinstance(parsed_data, dict),
        "Top-level YAML configuration must deserialize to a dictionary.",
    )
    return dict(parsed_data)


def load_verified_npz_weights(
    workspace_root: Path,
    weights_path: Path | str,
    expected_sha256: str,
    max_bytes: int = MAX_FILE_SIZE_BYTES_DEFAULT,
) -> dict[str, np.ndarray]:
    """Load non-executable .npz model weights after SHA-256 verification.

    Args:
        workspace_root: Trusted workspace root directory.
        weights_path: Path to the .npz weights archive.
        expected_sha256: Expected SHA-256 hex digest of the .npz file.
        max_bytes: Maximum allowed file byte size.

    Returns:
        Dictionary mapping tensor names to NumPy arrays.
    """
    safe_path: Path = resolve_safe_path(workspace_root, weights_path)
    verify_condition(
        safe_path.suffix.lower() == ".npz",
        "Only non-executable .npz weight archives are permitted.",
    )
    validate_file_size(safe_path, max_bytes=max_bytes)
    raw_bytes: bytes = safe_path.read_bytes()
    verify_condition(
        verify_sha256_digest(raw_bytes, expected_sha256),
        "Model weight SHA-256 integrity verification failed.",
    )
    tensors: dict[str, np.ndarray] = {}
    with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as archive:
        key_list: list[str] = list(archive.files)
        verify_condition(
            len(key_list) <= MAX_ARCHIVE_MEMBERS_DEFAULT,
            "NPZ weight archive member count exceeds security ceiling.",
        )
        for index in range(min(len(key_list), MAX_ARCHIVE_MEMBERS_DEFAULT)):
            tensor_name: str = key_list[index]
            tensors[tensor_name] = np.array(archive[tensor_name], copy=True)
    verify_condition(
        len(tensors) > ZERO_INT,
        "Verified .npz archive must contain at least one tensor.",
    )
    return tensors


def parse_safe_json_payload(
    raw_text: str,
    max_bytes: int = MAX_FILE_SIZE_BYTES_DEFAULT,
) -> dict[str, Any]:
    """Parse a JSON string after enforcing byte-size bounds.

    Args:
        raw_text: Raw JSON text payload.
        max_bytes: Maximum permitted byte length.

    Returns:
        Parsed JSON object as a dictionary.
    """
    encoded: bytes = raw_text.encode(ENCODING_UTF8)
    verify_condition(
        ZERO_INT < len(encoded) <= max_bytes,
        "JSON payload byte size violates configured bounds.",
    )
    try:
        loaded: Any = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise SecurityViolationError("Malformed JSON payload.") from exc
    verify_condition(
        isinstance(loaded, dict),
        "Parsed JSON root must be a dictionary object.",
    )
    return dict(loaded)


def validate_safe_zarr_metadata(
    workspace_root: Path,
    zarray_json_path: Path | str,
    max_chunk_bytes: int = MAX_ZARR_CHUNK_BYTES,
) -> dict[str, Any]:
    """Verify a .zarray header against CWE-502 pickle codecs and chunk bombs.

    Args:
        workspace_root: Trusted workspace root directory.
        zarray_json_path: Path to the .zarray metadata file.
        max_chunk_bytes: Maximum uncompressed byte size per chunk.

    Returns:
        Validated .zarray metadata dictionary.
    """
    safe_meta_path: Path = resolve_safe_path(workspace_root, zarray_json_path)
    validate_file_size(safe_meta_path, max_bytes=MAX_FILE_SIZE_BYTES_DEFAULT)
    raw_text: str = safe_meta_path.read_text(encoding=ENCODING_UTF8)
    meta: dict[str, Any] = parse_safe_json_payload(raw_text)
    shape: Sequence[int] = meta.get("shape", ())
    chunks: Sequence[int] = meta.get("chunks", ())
    verify_condition(
        ONE_INT <= len(shape) <= MAX_ZARR_DIMENSIONS
        and len(shape) == len(chunks),
        "Zarr shape and chunks must have matching, bounded dimensions.",
    )
    total_elements: int = math.prod(int(dim) for dim in shape)
    chunk_elements: int = math.prod(int(c_dim) for c_dim in chunks)
    verify_condition(
        ONE_INT <= total_elements <= MAX_IMAGE_PIXELS_DEFAULT,
        "Zarr array total element count exceeds pixel ceiling.",
    )
    verify_condition(
        ONE_INT
        <= chunk_elements * MAX_DTYPE_ITEMSIZE_BYTES
        <= max_chunk_bytes,
        "Zarr chunk byte allocation exceeds decompression bomb ceiling.",
    )
    compressor: Mapping[str, Any] | None = meta.get("compressor")
    if compressor is not None:
        comp_id: str = str(compressor.get("id", "")).strip().lower()
        verify_condition(
            comp_id in ALLOWED_ZARR_CODECS,
            f"Forbidden or executable Zarr compressor rejected: {comp_id}",
        )
    filters: Sequence[Mapping[str, Any]] | None = meta.get("filters")
    if filters is not None:
        for idx in range(min(len(filters), MAX_ARCHIVE_MEMBERS_DEFAULT)):
            flt_id: str = str(filters[idx].get("id", "")).strip().lower()
            verify_condition(
                flt_id in ALLOWED_ZARR_CODECS,
                f"Forbidden or executable Zarr filter rejected: {flt_id}",
            )
    return meta


def extract_safe_zip_archive(
    workspace_root: Path,
    archive_path: Path | str,
    destination_dir: Path | str,
    max_members: int = MAX_ARCHIVE_MEMBERS_DEFAULT,
    max_total_bytes: int = MAX_FILE_SIZE_BYTES_DEFAULT,
) -> list[Path]:
    """Extract a ZIP archive with Zip-Slip and decompression bomb defenses.

    Args:
        workspace_root: Trusted workspace root path.
        archive_path: Path to the source ZIP archive.
        destination_dir: Target extraction directory inside workspace_root.
        max_members: Maximum number of archive entries allowed.
        max_total_bytes: Maximum cumulative uncompressed byte size.

    Returns:
        List of canonical extracted file paths.
    """
    safe_archive: Path = resolve_safe_path(workspace_root, archive_path)
    safe_dest: Path = resolve_safe_path(workspace_root, destination_dir)
    validate_file_size(safe_archive, max_bytes=max_total_bytes)
    safe_dest.mkdir(parents=True, exist_ok=True)
    extracted_paths: list[Path] = []
    cumulative_bytes: int = ZERO_INT
    with zipfile.ZipFile(safe_archive, mode="r") as zip_handle:
        infos: list[zipfile.ZipInfo] = zip_handle.infolist()
        verify_condition(
            len(infos) <= max_members,
            "Archive member count exceeds configured ceiling.",
        )
        loop_bound: int = min(len(infos), max_members)
        for idx in range(loop_bound):
            member: zipfile.ZipInfo = infos[idx]
            verify_condition(
                "\x00" not in member.filename,
                "Null byte detected in archive member filename.",
            )
            member_parts: tuple[str, ...] = Path(member.filename).parts
            verify_condition(
                ".." not in member_parts
                and not Path(member.filename).is_absolute(),
                "Zip-Slip path traversal detected in archive member.",
            )
            cumulative_bytes += member.file_size
            verify_condition(
                cumulative_bytes <= max_total_bytes,
                "Decompression bomb detected: cumulative size exceeded.",
            )
            target_path: Path = resolve_safe_path(safe_dest, member.filename)
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                payload: bytes = zip_handle.read(member.filename)
                verify_condition(
                    len(payload) <= max_total_bytes,
                    "Extracted payload exceeds byte ceiling.",
                )
                target_path.write_bytes(payload)
                extracted_paths.append(target_path)
    return extracted_paths


def parse_safe_xml_metadata(
    xml_text: str,
    max_bytes: int = MAX_FILE_SIZE_BYTES_DEFAULT,
) -> dict[str, str]:
    """Parse flat XML telemetry metadata while blocking XXE declarations.

    Args:
        xml_text: Raw XML string to inspect and parse.
        max_bytes: Maximum byte size permitted for the XML payload.

    Returns:
        Dictionary of child tag names to stripped text values.
    """
    encoded_xml: bytes = xml_text.encode(ENCODING_UTF8)
    verify_condition(
        ZERO_INT < len(encoded_xml) <= max_bytes,
        "XML payload size violates configured security bounds.",
    )
    upper_xml: str = xml_text.upper()
    for token_idx in range(len(XML_FORBIDDEN_TOKENS)):
        forbidden_token: str = XML_FORBIDDEN_TOKENS[token_idx]
        verify_condition(
            forbidden_token not in upper_xml,
            f"Forbidden XML declaration token detected: {forbidden_token}",
        )
    try:
        root_element: ET.Element = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise SecurityViolationError(
            "Malformed XML metadata rejected."
        ) from exc
    children: list[ET.Element] = list(root_element)
    verify_condition(
        len(children) <= MAX_LOOP_ITERATIONS,
        "XML element count exceeds iteration bound.",
    )
    metadata: dict[str, str] = {}
    for idx in range(min(len(children), MAX_LOOP_ITERATIONS)):
        elem: ET.Element = children[idx]
        metadata[str(elem.tag)] = (elem.text or "").strip()
    return metadata


def validate_image_dimensions(
    width: int,
    height: int,
    max_width: int = MAX_IMAGE_WIDTH_DEFAULT,
    max_height: int = MAX_IMAGE_HEIGHT_DEFAULT,
    max_pixels: int = MAX_IMAGE_PIXELS_DEFAULT,
) -> tuple[int, int]:
    """Verify image dimensions prior to pixel buffer allocation.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        max_width: Maximum permitted width in pixels.
        max_height: Maximum permitted height in pixels.
        max_pixels: Maximum permitted total pixel area.

    Returns:
        Verified (width, height) tuple.
    """
    verify_condition(
        ONE_INT <= width <= max_width,
        "Image width violates configured dimension bounds.",
    )
    verify_condition(
        ONE_INT <= height <= max_height,
        "Image height violates configured dimension bounds.",
    )
    total_pixels: int = width * height
    verify_condition(
        ONE_INT <= total_pixels <= max_pixels,
        "Image pixel count exceeds decompression ceiling.",
    )
    return width, height


def validate_outbound_url(
    url: str,
    allowed_schemes: Sequence[str] = ALLOWED_URL_SCHEMES,
    allowed_hosts: Sequence[str] = ALLOWED_HOSTS_DEFAULT,
) -> str:
    """Validate an outbound URL against SSRF scheme and host allowlists.

    Args:
        url: Candidate URL string to inspect.
        allowed_schemes: Permitted URL schemes (such as http, https).
        allowed_hosts: Permitted destination hostnames or loopback IPs.

    Returns:
        Normalized URL string verified safe for outbound requests.
    """
    verify_condition(
        isinstance(url, str) and len(url.strip()) > ZERO_INT,
        "Outbound URL must be a non-empty string.",
    )
    parsed: urllib.parse.SplitResult = urllib.parse.urlsplit(url.strip())
    scheme_lower: str = parsed.scheme.lower()
    host_lower: str = (parsed.hostname or "").lower()
    verify_condition(
        scheme_lower in allowed_schemes,
        "Outbound URL scheme is not in the security allowlist.",
    )
    verify_condition(
        host_lower in allowed_hosts,
        "Outbound URL hostname is not in the SSRF allowlist.",
    )
    verify_condition(
        parsed.username is None and parsed.password is None,
        "Embedded credentials in outbound URLs are prohibited.",
    )
    return urllib.parse.urlunsplit(parsed)


def run_safe_subprocess(
    command_args: Sequence[str],
    allowed_binaries: Sequence[str] = ALLOWED_BINARIES_DEFAULT,
    env_mapping: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute an external command with shell=False and binary allowlisting.

    Args:
        command_args: Ordered sequence of command binary and arguments.
        allowed_binaries: Permitted executable binary names.
        env_mapping: Optional sanitized environment variable mapping.

    Returns:
        CompletedProcess result containing stdout and stderr strings.
    """
    verify_condition(
        ONE_INT <= len(command_args) <= MAX_SUBPROCESS_ARGS,
        "Subprocess argument list violates length bounds.",
    )
    binary_name: str = Path(command_args[ZERO_INT]).name
    verify_condition(
        binary_name in allowed_binaries,
        "Requested executable is not in the subprocess binary allowlist.",
    )
    for idx in range(len(command_args)):
        arg_val: str = command_args[idx]
        verify_condition(
            isinstance(arg_val, str) and "\x00" not in arg_val,
            "Invalid or null-byte argument rejected in subprocess call.",
        )
    sanitized_env: dict[str, str] = {"PATH": "/usr/bin:/bin"}
    if env_mapping is not None:
        keys: list[str] = list(env_mapping.keys())
        for idx in range(min(len(keys), MAX_SUBPROCESS_ARGS)):
            key: str = keys[idx]
            sanitized_env[str(key)] = str(env_mapping[key])
    return subprocess.run(
        list(command_args),
        shell=False,
        check=True,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        env=sanitized_env,
    )
