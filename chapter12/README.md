# Chapter 12: real-data reproducibility (work in progress)

This branch contains section-organized code for published or independently measured datasets. **No illustrative, synthetic or fabricated measurements can be published as case-study results.** Each case requires the source dataset, original license, checksum, date acquired, preprocessing log, training/test split, benchmark comparison, generated figures and machine-readable metrics.

## Section layout
- `chapter12/section_12_distributed_training/`: published Megatron-LM SC21 Table 1 reconstruction.
- `chapter12/section_12_manufacturing/`: real UCI SECOM measurements (data must be acquired separately).
- `chapter12/section_12_telecommunications/`: measured network telemetry and adjudicated incident labels required.
- `chapter12/section_12_pharmaceutical/`: experimental ChEMBL activity records required.
- `chapter12/section_12_clinical_imaging/`: study-level adjudicated external-site inference predictions required.
- `chapter12/section_12_geospatial_fleet/`: official NYC TLC trip records (real trip ETA proxy, **not** proof of dispatch efficiency).

**Status:** The distributed-training figure is reproduced from verified published numbers. Raw data for the other five applications is not in the working files, so their real-data benchmarks are **blocked**. Never substitute a synthetic dataset or an unrelated dataset, and never publish an unexecuted example as a result.

## Reproducibility gates
1. Obtain raw data from an authoritative source; record source URL, license, checksum and publication/collection date.
2. Validate required columns and provenance before execution.
3. Prevent leakage using chronological, scaffold, site, patient, or grouped splits as appropriate.
4. Compare against a simple baseline with identical data partition.
5. Export source-bearing figures and numerical metrics; document sample size, units, confidence limitations and methods.
6. Keep evaluation claims limited to the acquired dataset and experiment.

**Source:** Narayanan et al. (2021), *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*, SC21; Table 1. Official NYC TLC source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page. UCI SECOM dataset: https://archive.ics.uci.edu/dataset/179/secom.

This branch does not contain the unpublished book manuscript or any protected clinical records.