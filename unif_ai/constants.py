"""Centralized typed constants for the unif_ai disaster relief platform."""

from typing import Final

APP_NAME: Final[str] = "unif_ai"
APP_DOMAIN_TITLE: Final[str] = (
    "AberTech Disaster Relief Vision & Telemetry Orchestrator"
)
APP_VERSION: Final[str] = "2.0.0"

ENCODING_UTF8: Final[str] = "utf-8"
ENCODING_ASCII: Final[str] = "ascii"

ZERO_INT: Final[int] = 0
ONE_INT: Final[int] = 1
TWO_INT: Final[int] = 2
THREE_INT: Final[int] = 3
FOUR_INT: Final[int] = 4
FIVE_INT: Final[int] = 5
SIX_INT: Final[int] = 6
EIGHT_INT: Final[int] = 8
SIXTEEN_INT: Final[int] = 16
THIRTY_TWO_INT: Final[int] = 32
SIXTY_FOUR_INT: Final[int] = 64
HUNDRED_INT: Final[int] = 100
BYTE_MAX_VALUE: Final[int] = 255

ZERO_FLOAT: Final[float] = 0.0
HALF_FLOAT: Final[float] = 0.5
ONE_FLOAT: Final[float] = 1.0
TWO_FLOAT: Final[float] = 2.0
HUNDRED_FLOAT: Final[float] = 100.0

CROP_PCT_MIN: Final[float] = 0.0
CROP_PCT_MAX: Final[float] = 100.0
DEFAULT_CROP_TOP_PCT: Final[float] = 18.75
DEFAULT_CROP_LEFT_PCT: Final[float] = 18.75
DEFAULT_CROP_BOTTOM_PCT: Final[float] = 43.75
DEFAULT_CROP_RIGHT_PCT: Final[float] = 43.75
CROP_PCT_STEP_OFFSET: Final[float] = 3.125

HTTP_STATUS_OK: Final[int] = 200
HTTP_STATUS_CREATED: Final[int] = 201
HTTP_STATUS_BAD_REQUEST: Final[int] = 400
HTTP_STATUS_FORBIDDEN: Final[int] = 403
HTTP_STATUS_NOT_FOUND: Final[int] = 404
HTTP_STATUS_GONE: Final[int] = 410
HTTP_STATUS_TOO_MANY_REQUESTS: Final[int] = 429
HTTP_STATUS_INTERNAL_ERROR: Final[int] = 500

HTTP_REASON_MAP: Final[dict[int, str]] = {
    HTTP_STATUS_OK: "200 OK",
    HTTP_STATUS_CREATED: "201 Created",
    HTTP_STATUS_BAD_REQUEST: "400 Bad Request",
    HTTP_STATUS_FORBIDDEN: "403 Forbidden",
    HTTP_STATUS_NOT_FOUND: "404 Not Found",
    HTTP_STATUS_GONE: "410 Gone",
    HTTP_STATUS_TOO_MANY_REQUESTS: "429 Too Many Requests",
    HTTP_STATUS_INTERNAL_ERROR: "500 Internal Server Error",
}

MAX_FILE_SIZE_BYTES_DEFAULT: Final[int] = 10485760
MAX_ARCHIVE_MEMBERS_DEFAULT: Final[int] = 32
MAX_IMAGE_WIDTH_DEFAULT: Final[int] = 4096
MAX_IMAGE_HEIGHT_DEFAULT: Final[int] = 4096
MAX_IMAGE_PIXELS_DEFAULT: Final[int] = 16777216
MAX_LOOP_ITERATIONS: Final[int] = 100000
MAX_STAGE_COUNT: Final[int] = 16
MAX_DETECTIONS_PER_FRAME: Final[int] = 512
MAX_FRAMES_PER_BATCH: Final[int] = 64
MAX_SUBPROCESS_ARGS: Final[int] = 32
SUBPROCESS_TIMEOUT_SECONDS: Final[int] = 15
MAX_REQUEST_BODY_BYTES: Final[int] = 2097152
MAX_QUERY_PARAM_LENGTH: Final[int] = 256
MAX_INPUT_STRING_LENGTH: Final[int] = 512
MAX_PORTAL_BATCH_RESULTS: Final[int] = 64

ZARR_FORMAT_VERSION: Final[int] = 2
OME_NGFF_VERSION: Final[str] = "0.4"
MAX_ZARR_DIMENSIONS: Final[int] = 5
MAX_ZARR_CHUNK_BYTES: Final[int] = 4194304
MAX_DTYPE_ITEMSIZE_BYTES: Final[int] = 8
ALLOWED_ZARR_CODECS: Final[tuple[str, ...]] = (
    "blosc",
    "gzip",
    "lz4",
    "zlib",
    "zstd",
)

SHA256_HEX_LENGTH: Final[int] = 64
SHA256_BLOCK_SIZE_BYTES: Final[int] = 64
HKDF_KEY_LENGTH_BYTES: Final[int] = 32
PSEUDONYM_HEX_PREFIX_LEN: Final[int] = 24
EPHEMERAL_SECRET_BYTES: Final[int] = 32
RESET_TOKEN_NONCE_BYTES: Final[int] = 16
GENESIS_PREV_HASH: Final[str] = "0" * SHA256_HEX_LENGTH

ENV_MASTER_SECRET_KEY: Final[str] = "UNIF_AI_MASTER_SECRET"

DOMAIN_TELEMETRY: Final[str] = "relief_uav_telemetry_v1"
DOMAIN_OPERATOR: Final[str] = "relief_responder_identity_v1"
DOMAIN_SUBJECT: Final[str] = "relief_affected_person_v1"
DOMAIN_PASSWORD_RESET: Final[str] = "relief_password_reset_v1"

ALLOWED_UK_EU_REGIONS: Final[tuple[str, ...]] = (
    "uk-south",
    "uk-west",
    "eu-west-1",
    "eu-west-2",
    "eu-central-1",
    "europe-west1",
    "europe-west2",
)

ALLOWED_URL_SCHEMES: Final[tuple[str, ...]] = ("http", "https")
ALLOWED_HOSTS_DEFAULT: Final[tuple[str, ...]] = (
    "127.0.0.1",
    "localhost",
    "azurite.local",
    "gcs.local",
)
ALLOWED_BINARIES_DEFAULT: Final[tuple[str, ...]] = ("python3", "git")

XML_FORBIDDEN_TOKENS: Final[tuple[str, ...]] = (
    "<!DOCTYPE",
    "<!ENTITY",
    "SYSTEM",
    "PUBLIC",
)

SQLI_FORBIDDEN_PATTERNS: Final[tuple[str, ...]] = (
    "UNION SELECT",
    "DROP TABLE",
    "OR 1=1",
    "--",
    "/*",
    "*/",
    ";--",
    "EXEC(",
)

XSS_FORBIDDEN_PATTERNS: Final[tuple[str, ...]] = (
    "<SCRIPT",
    "JAVASCRIPT:",
    "ONERROR=",
    "ONLOAD=",
    "<IFRAME",
    "<OBJECT",
)

REGEX_EMAIL_PATTERN: Final[str] = (
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
)
REGEX_PHONE_PATTERN: Final[str] = (
    r"\b(?:\+?44|0)(?:\d\s?){9,10}\b"
)
REGEX_IPV4_PATTERN: Final[str] = (
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)
REDACTED_PLACEHOLDER: Final[str] = "[REDACTED_PII]"

H3_APERTURE_FACTOR: Final[float] = 2.6457513110645907
H3_BASE_RESOLUTION: Final[int] = 9
H3_MIN_RESOLUTION: Final[int] = 3
SPATIAL_K_MIN_DEFAULT: Final[int] = 3
SPATIAL_MAX_COARSEN_STEPS: Final[int] = 6
DP_EPSILON_MAX: Final[float] = 1.0
DP_EPSILON_DEFAULT: Final[float] = 0.85
DP_SENSITIVITY_DEFAULT: Final[float] = 1.0
LAPLACE_UNIFORM_CLIP: Final[float] = 0.499999

LATITUDE_MIN_DEGREES: Final[float] = -90.0
LATITUDE_MAX_DEGREES: Final[float] = 90.0
LONGITUDE_MIN_DEGREES: Final[float] = -180.0
LONGITUDE_MAX_DEGREES: Final[float] = 180.0
METERS_PER_DEGREE_LAT: Final[float] = 111320.0
DEFAULT_DEDUP_RADIUS_METERS: Final[float] = 15.0
BASE_GRID_STEP_DEGREES: Final[float] = 0.001

BLUR_BLOCK_SIZE_DEFAULT: Final[int] = 8
BLUR_KERNEL_RADIUS_DEFAULT: Final[int] = 2
EDGE_PERCENTILE_DEFAULT: Final[float] = 75.0
MIN_BOUNDING_BOX_AREA: Final[int] = 9
SOBEL_KERNEL_SIZE: Final[int] = 3
EFD_HARMONIC_ORDER: Final[int] = 4
DASHCAM_HOMOGRAPHY_SCALE: Final[float] = 0.125

RATE_LIMIT_REQUESTS_DEFAULT: Final[int] = 60
RATE_LIMIT_WINDOW_SECONDS: Final[int] = 60
PASSWORD_Reset_TTL_SECONDS: Final[int] = 1800
STAGING_TTL_SECONDS_DEFAULT: Final[int] = 3600
LOG_RING_BUFFER_CAPACITY: Final[int] = 256
LOG_DETAIL_MAX_LENGTH: Final[int] = 200

SLOT_BLUE: Final[str] = "blue"
SLOT_GREEN: Final[str] = "green"
VALID_SLOTS: Final[tuple[str, ...]] = (SLOT_BLUE, SLOT_GREEN)

LOG_LEVEL_INFO: Final[str] = "INFO"
LOG_LEVEL_WARNING: Final[str] = "WARNING"
LOG_LEVEL_ERROR: Final[str] = "ERROR"
LOG_LEVEL_CRITICAL: Final[str] = "CRITICAL"
ALERT_TRIGGER_LEVELS: Final[tuple[str, ...]] = (
    LOG_LEVEL_ERROR,
    LOG_LEVEL_CRITICAL,
)

STAGE_INGRESS: Final[str] = "ingress_validation_and_exif_strip"
STAGE_VISION: Final[str] = "visual_redaction_and_edge_detection"
STAGE_SPATIAL: Final[str] = "spatial_deduplication_and_k_anonymity"
STAGE_SYNC: Final[str] = "cloud_sandbox_sync_and_ledger_commit"

DEFAULT_ENABLED_STAGES: Final[tuple[str, ...]] = (
    STAGE_INGRESS,
    STAGE_VISION,
    STAGE_SPATIAL,
    STAGE_SYNC,
)

CV_APP_UAV_ANOMALY: Final[str] = "uav_anomaly_detection"
CV_APP_CLOUD_DASHCAM: Final[str] = "cloud_dashcam_tracking"
CV_APP_SHAPE_EFD: Final[str] = "shape_fourier_analysis"
ALL_CV_APPLICATIONS: Final[tuple[str, ...]] = (
    CV_APP_UAV_ANOMALY,
    CV_APP_CLOUD_DASHCAM,
    CV_APP_SHAPE_EFD,
)

TRANSPORT_UAV_ROTARY: Final[str] = "UAV_ROTARY"
TRANSPORT_DASHCAM_4X4: Final[str] = "DASHCAM_4X4"
TRANSPORT_FOOT_TEAM: Final[str] = "FOOT_TEAM"
TRANSPORT_COASTAL_VESSEL: Final[str] = "COASTAL_VESSEL"
VALID_TRANSPORT_MODES: Final[tuple[str, ...]] = (
    TRANSPORT_UAV_ROTARY,
    TRANSPORT_DASHCAM_4X4,
    TRANSPORT_FOOT_TEAM,
    TRANSPORT_COASTAL_VESSEL,
)

LAYER_H3_GRID: Final[str] = "H3_K_ANONYMITY_GRID"
LAYER_SLOPE_DEM: Final[str] = "INFRASTRUCTURE_DAMAGE_DEM"
LAYER_THERMAL_HEATMAP: Final[str] = "THERMAL_ANOMALY_HEATMAP"
LAYER_DASHCAM_VECTORS: Final[str] = "DASHCAM_CONVOY_VECTORS"
DEFAULT_REFERENCE_LAYERS: Final[tuple[str, ...]] = (
    LAYER_H3_GRID,
    LAYER_SLOPE_DEM,
    LAYER_THERMAL_HEATMAP,
    LAYER_DASHCAM_VECTORS,
)

DEFAULT_TERRAIN_SECTOR: Final[str] = "FLOOD_ZONE_SECTOR_ALPHA"
DEFAULT_WEATHER_STATE: Final[str] = "SEVERE_STORM_LOW_VISIBILITY"

RAG_PROTOCOL_CORPUS: Final[dict[str, str]] = {
    "flood": (
        "Flood Relief Protocol: Deploy UAV_ROTARY below 350m AGL with "
        "thermal Sobel thresholding at P75 and H3 k>=3 spatial coarsening."
    ),
    "earthquake": (
        "Urban Structural Collapse Protocol: Enable DASHCAM_4X4 perspective "
        "warp stabilization along access corridors and monitor DEM contours."
    ),
    "coastal": (
        "Coastal Storm Surge Protocol: Activate Elliptical Fourier "
        "Descriptor (EFD order 4) invariants for debris discrimination."
    ),
    "default": (
        "AberTech Disaster Relief Protocol: Run 4-stage EXIF strip, EDPB "
        "3/2019 ROI blur, BFS graph-transitivity dedup, and UK/EU sync."
    ),
}

SYNTHETIC_FRAME_WIDTH: Final[int] = 64
SYNTHETIC_FRAME_HEIGHT: Final[int] = 64
SYNTHETIC_FRAME_COUNT: Final[int] = 3
SYNTHETIC_BACKGROUND_VALUE: Final[int] = 24
SYNTHETIC_OBJECT_VALUE: Final[int] = 220
SYNTHETIC_BASE_LATITUDE: Final[float] = 52.4761
SYNTHETIC_BASE_LONGITUDE: Final[float] = 1.7503
SYNTHETIC_COORD_JITTER: Final[float] = 0.00004
