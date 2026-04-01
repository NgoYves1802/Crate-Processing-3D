# crate_vision

3-D crate detection pipeline for the IFM O3D303 depth camera + S7-1200 PLC.

## Project structure

```
crate_vision/
│
├── config.py                 ← Single source of truth for all parameters
│
├── pipeline.py               ← Main orchestration: process_depth_layers()
│
├── pose.py                   ← Per-object pose: estimate_pose(), save_object(), build_meta()
│
├── ai_verifier.py            ← DenseNet-121 crate classifier (PyTorch / ONNX)
│
├── io/
│   ├── loader.py             ← load_depth_and_amplitude()
│   └── serializer.py         ← pack_crate(), build_crate_row(), write_crate_scans_json()
│
├── detection/
│   ├── depth.py              ← create_depth_masks(), remove_small_blobs(), apply_mask_to_image()
│   ├── ccl.py                ← ccl_on_mask(), size_filter()
│   ├── geometry.py           ← fit_min_area_rect(), detect_corners(), fit_plane_svd(),
│   │                            compute_mm_per_pixel_theoretical(), KDTree helpers, get_grid_anchor()
│   └── slots.py              ← analyze_crate_slots(), draw_slot_grid(), save_slot_figure()
│
└── hardware/
    ├── camera.py             ← GrabO3D300, configure_camera(), setup_pcic_stream()
    └── plc.py                ← PLCClient, readback_db()

main.py                       ← Entry point — wires hardware → pipeline
```

## Dependency graph (import direction →)

```
main.py
  → hardware/camera.py
  → hardware/plc.py
  → pipeline.py
      → config.py
      → io/loader.py
      → io/serializer.py
      → detection/depth.py
      → detection/ccl.py
      → detection/geometry.py
      → pose.py
          → config.py
          → ai_verifier.py   → config.py
          → detection/geometry.py
          → detection/slots.py
```

**hardware/** never imports from **pipeline** or **detection** — the boundary is clean.

## Quick start

```python
# Run the full pipeline on a saved snapshot
from crate_vision.pipeline import process_depth_layers
result = process_depth_layers("snapshots/snap0001_20260323_171356")
print(result["crates"])
```

## Configuration

```python
from crate_vision.config import get_config, override_config, load_config_from_json

# Read any parameter
cfg = get_config()
print(cfg.layer_distances_mm)         # [1020.0, 1300.0]

# Override at runtime
override_config(layer_distances_mm=[900.0, 1200.0], ai_conf_threshold=0.8)

# Load from JSON file
load_config_from_json("my_config.json")

# Save current config to JSON
get_config().save_json("my_config.json")
```

## Running the hardware loop

```bash
python main.py
python main.py --config my_config.json
```

## TIA Portal requirements

1. DB must have **"Optimized block access" UNCHECKED**
2. **PUT/GET must be enabled**:
   Device Config → PLC → PROFINET → Advanced options → Connections → ☑ Permit access with PUT/GET

## DB layout (92 bytes)

| Offset | Size | Type | Name         |
|--------|------|------|--------------|
| 0      | 2    | INT  | crate_count  |
| 2      | 2    | INT  | snap_index   |
| 4+     | 22×4 | —    | crate array  |

Each crate record (22 bytes):

| +Offset | Type | Name        |
|---------|------|-------------|
| 0       | INT  | crate_number|
| 2       | REAL | Rx          |
| 6       | REAL | Ry          |
| 10      | REAL | Rz          |
| 14      | REAL | theta       |
| 18      | WORD | S_flags (S1–S12) |
| 20      | BYTE | ai_class    |
| 21      | BYTE | padding     |
