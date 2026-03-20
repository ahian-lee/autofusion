# Constructor OCC Baseline

This baseline is isolated from the original MOFFusion training code.

It compares two input variants for the MOF constructor classifiers:

- `sdf`: the original 4-channel SDF input
- `sdf_occ`: the original 4-channel SDF input plus one extra occupancy channel computed as `(sdf[channel] < 0)`

The goal is to answer whether explicit occupancy helps the downstream topology / node / edge prediction.

## Files

- `tools/constructor_occ_baseline.py`
- `launchers/train_constructor_occ_baseline.sh`

## Quick Start

Run all four experiments:

```bash
cd /opt/data/private/moffusion/autofusion
bash ./launchers/train_constructor_occ_baseline.sh
```

Run a single experiment:

```bash
cd /opt/data/private/moffusion/autofusion
python ./tools/constructor_occ_baseline.py \
  --task topo \
  --input_variant sdf_occ \
  --dataroot ./data \
  --dataset_mode pormake-pld2 \
  --batch_size 32 \
  --epochs 20 \
  --lr 1e-4 \
  --save_dir ./outputs/constructor_occ_baseline/topo_sdf_occ
```

## Outputs

Each run writes:

- a best checkpoint: `*_best.pt`
- a report file: `*_report.json`

For `topo`, the main metric is:

- `topo_accuracy`

For `bb`, the main metrics are:

- `total_accuracy`
- `node1_accuracy`
- `node2_accuracy`
- `edge_accuracy`

## Suggested Comparison

Compare:

1. `topo / sdf` vs `topo / sdf_occ`
2. `bb / sdf` vs `bb / sdf_occ`

If `sdf_occ` is consistently better, then explicit occupancy is helping even though it is derived from SDF.
