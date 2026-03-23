# Constructor OCC Baseline Summary

Date: 2026-03-23

## Goal

Evaluate whether explicitly adding `occ_from_sdf` helps downstream MOF constructor prediction, and whether occupancy carries information beyond the SDF sign structure.

## Files

- `topo_sdf_report.json`
- `topo_sdf_occ_report.json`
- `bb_sdf_report.json`
- `bb_sdf_occ_report.json`
- `occ_sanity_check_report_gpu.json`

## Main Results

### Topology prediction

- `topo_sdf`: best `topo_accuracy = 0.9871`
- `topo_sdf_occ`: best `topo_accuracy = 0.9846`

Result: `sdf` is slightly better than `sdf_occ`.

### Building-block prediction

- `bb_sdf`: best `total_accuracy = 0.0655`
- `bb_sdf_occ`: best `total_accuracy = 0.0235`

Result: `sdf` is clearly better than `sdf_occ`.

## Sanity Check: Occupancy vs SDF

From `occ_sanity_check_report_gpu.json`:

- Hard threshold on upsampled SDF vs GT occupancy:
  - `accuracy = 1.0000`
  - `iou = 1.0000`
  - `dice = 1.0000`

- Fitted logistic map `sigmoid(a * sdf + b)` vs GT occupancy:
  - `accuracy = 0.9837`
  - `iou = 0.9578`
  - `dice = 0.9784`

- Existing `occ_predictor_full.pt` vs GT occupancy:
  - `accuracy = 0.5777`
  - `iou = 0.0000`
  - `dice = 0.0000`

## Interpretation

1. The current occupancy ground truth is effectively a direct thresholded transform of SDF.
2. A simple fitted logistic function already approximates occupancy very well.
3. Explicitly concatenating `occ_from_sdf` does not improve downstream constructor prediction under the current setup.
4. Under the current data construction pipeline, occupancy does not appear to provide substantial new information beyond the SDF sign structure. It is better interpreted as a derived or auxiliary representation rather than an independent modality.

## Practical Conclusion

- For the current constructor baseline, keep `sdf` as the main input.
- Do not assume `occ_from_sdf` will improve topology or BB prediction.
- If occupancy is used, treat it primarily as an auxiliary supervision or analysis target.
