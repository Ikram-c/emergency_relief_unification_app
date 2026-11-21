# UNIF_AI (`v2.0.0`) — AberTech Disaster Relief Vision & Telemetry Orchestrator

**OWASP ASVS v5.0 Hardened | UK/EU GDPR Privacy-by-Design | 10-Point Pre-Deployment Verified**

`unif_ai` is the central orchestration engine for the **AberTech** disaster relief computer vision suite. It transforms raw aerial UAV, 4x4 convoy dashcam, ground responder, and coastal storm-surge imagery into deduplicated, UK/EU GDPR-compliant situational maps in four deterministic steps.

---

## How `unif_ai` Works in Disaster Relief

1. **Multi-Modal Capture & EXIF Strip:** Ingests `UAV_ROTARY`, `DASHCAM_4X4`, `FOOT_TEAM`, and `COASTAL_VESSEL` feeds, strips EXIF metadata by cloning raw pixel buffers, and converts normalized `CropConfig` percentages into resolution-safe ROIs.
2. **Edge Vision & EDPB 3/2019 Visual Blur:** Applies irreversible block-quantization plus separable 2D smoothing over sensitive regions while executing Sobel/Otsu anomaly detection, Cloud-DashCam perspective motion tracking, and Elliptical Fourier Descriptor (EFD) contour analysis.
3. **Graph Transitivity & H3 $k$-Anonymity:** Collapses overlapping sightings via BFS graph transitivity, coarsens GPS points onto aperture-7 H3 hexagonal cells until spatial uniqueness $U = 0.0$ ($k \ge 3$), and adds calibrated Laplace Differential Privacy noise ($\epsilon \le 1.0$).
4. **DOD OME-Zarr & UK/EU Sovereign Sync:** Persists chunked OME-Zarr v2 frame cubes (`[t, y, x]`), commits a tamper-evident SHA-256 hash-chained JSONL ledger, and supports deterministic GDPR Article 17 Right to Erasure across SQLite, JSON, and OME-Zarr `.zattrs` stores.

---

## Quick Start (Tutorial)

### 1. Install Dependencies

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

### 2. Set Your Cryptographic Master Secret

```bash
export UNIF_AI_MASTER_SECRET="abertech_disaster_relief_secret_key_2026_uk_eu"
```

*(If omitted, `unif_ai` automatically generates a cryptographically secure per-process CSPRNG secret via `secrets.token_hex`.)*

### 3. Run a Headless Disaster Relief Batch

```bash
python3 -m unif_ai.cli \
  --mode batch \
  --workspace ./workspace \
  --batch-id relief_flood_001 \
  --owner-id responder_alpha \
  --subject-id person_tag_01
```

### 4. Launch the 5-Screen Tactical Web Console

```bash
python3 -m unif_ai.cli \
  --mode serve \
  --workspace ./workspace \
  --host 127.0.0.1 \
  --port 8080
```

Open `http://127.0.0.1:8080` to view the mobile-friendly **Startpage & Workflow Infographic**, **Base Home Menu**, **New Relief Flow**, **Portal Dashboard**, and **Reference Layers**.

---

## REST API & CLI Reference

| Endpoint / Flag | Method / Default | Purpose |
| :--- | :--- | :--- |
| `GET /api/status` | `GET` | UK/EU residency, active Blue-Green slot, B-tree indexes, and SHA-256 ledger state. |
| `POST /api/rescue/create` | `POST` | Execute the 4-stage disaster relief pipeline (`batch_id`, `owner_id`, `subject_id`, `transport_mode`, `terrain_sector`). |
| `GET /api/rescues` | `GET` | BOLA-filtered Portal Dashboard list for `X-Operator-Id`. |
| `POST /api/rag/query` | `POST` | Retrieve zero-PII disaster relief protocol guidance (`flood`, `earthquake`, `coastal`). |
| `GET /api/layers` | `GET` | Retrieve the four geospatial reference layers (`H3_K_ANONYMITY_GRID`, `INFRASTRUCTURE_DAMAGE_DEM`, `THERMAL_ANOMALY_HEATMAP`, `DASHCAM_CONVOY_VECTORS`). |
| `POST /api/gdpr/erase` | `POST` | Execute GDPR Article 17 Right to Erasure across SQLite, JSON outputs, and OME-Zarr `.zattrs`. |
| `POST /api/deploy/switch` | `POST` | Promote standby Blue-Green slot or trigger instant rollback if `health_verified` is `false`. |
| `--mode` | `"serve"` | `"serve"` starts the WSGI server; `"batch"` executes one headless relief batch. |

---

## Verification & CI

Run the linter, zero-comment check, and automated test suite locally before pushing:

```bash
ruff check unif_ai tests
python3 -m unittest discover -s tests -v
```
