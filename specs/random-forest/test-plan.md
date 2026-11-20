# Random Forest Test Plan

| Field | Value |
|---|---|
| Owner | Nate, Mohsan, Ivan |
| Status | Draft |
| Last updated | 2025-11-20 |
| Related | [summary.md](summary.md), [implementation-plan.md](implementation-plan.md), [shared test plan](../shared/test-plan.md) |

> Each test should reference the requirement IDs it covers.

Test case format: `ID | description | requirement IDs | expected result | status`

## Unit tests

- [ ] `UT-RF-01` | Worker trains specified tree count $n_k$ on local synthetic data | R1, R6 | Returns list of valid decision tree objects matching $n_k$ | Not started
- [ ] `UT-RF-02` | Tree serialization converts scikit-learn tree structure to dictionary and back without precision loss | R5 | Deserialized tree predictions match original tree exactly | Not started
- [ ] `UT-RF-03` | Coordinator merges lists of trees from multiple sites into one unified `FederatedRandomForestClassifier` | R2 | Unified model contains sum of all worker trees and allows `.predict()` | Not started
- [ ] `UT-RF-04` | Hyperparameter validation catches invalid inputs (`n_estimators < 1`, `max_depth <= 0`) | R2 | Raises `ValueError` before job dispatch | Not started

## Integration tests

- [ ] `IT-RF-01` | Mock 3-site federated training run locally without live DCP network | R1, R2, R6 | Successfully produces a combined forest with $N$ trees | Not started
- [ ] `IT-RF-02` | Schema alignment validation across simulated sites | R2 | Fails gracefully with informative error if Site B has different columns than Site A | Not started

## DCP end-to-end tests

- [ ] `E2E-RF-01` | Train 100-tree forest across 2 real DCP worker nodes with remote data mapping | R1, R2, R6 | Complete DCP job finishes, returns trees, coordinator produces valid forest | Not started

## Invalid-input tests

- [ ] `INV-RF-01` | Site provides dataset with fewer rows than `min_samples_split` | R2 | Worker handles gracefully, fits single-leaf tree or reports error | Not started
- [ ] `INV-RF-02` | Site dataset contains unexpected `NaN` values without imputer | R2 | Worker reports validation failure through DCP slice error | Not started

## Partial-participation tests

- [ ] `PART-RF-01` | 3 sites planned, 1 site drops out or times out | R8 | Coordinator logs warning, builds forest from remaining 2 sites (e.g. 66 trees instead of 100) | Not started

## Failed-job recovery tests

- [ ] `REC-RF-01` | DCP job fails halfway through, rerun initiated | R9 | Completed slices are retrieved from DCP cache; only incomplete slices re-execute | Not started

## Model-quality checks

- [ ] `QUAL-RF-01` | Centralized scikit-learn RF vs Federated RF on synthetic classification data (IID split across 3 sites) | R1, R2 | Federated RF test accuracy within 3% of centralized baseline | Not started
- [ ] `QUAL-RF-02` | Quality test on skewed non-IID split (Site A has 80% Class 0, Site B has 80% Class 1) | R1, R2 | Model still achieves > 80% of centralized baseline accuracy | Not started

## Data-locality checks

- [ ] `PRIV-RF-01` | Inspect slice return payload bytes | R6 | Ensure zero row-level training instances, user IDs, or raw features exist in payload | Not started

## Serialization tests

- [ ] `SER-RF-01` | Export trained model to artifact and load in a bare Python environment with only `scikit-learn` installed | R5 | Model loads without DCP package and generates identical `.predict()` outputs | Not started

## Pass/fail criteria

- All unit tests and integration tests pass in CI.
- At least one E2E test passes on real DCP compute infrastructure.
- Privacy check confirms no raw data rows are transmitted in slice payloads.
- Inference works standalone with scikit-learn.
