# Random Forest Implementation Plan

| Field | Value |
|---|---|
| Owner | Nate, Mohsan, Ivan |
| Status | Draft |
| Last updated | 2025-02-14 |
| Related | [summary.md](summary.md), [test-plan.md](test-plan.md), [shared implementation plan](../shared/implementation-plan.md) |

## Architecture

The Random Forest track implements a **Federated Bagging** (Ensemble Aggregation) architecture:
1. **Coordinator Initialization:** The coordinator instantiates `FederatedRandomForestClassifier(n_estimators=N, ...)`.
2. **Slice Partitioning:** The coordinator divides `N` trees across `K` registered sites (`n_k = N / K` trees per site).
3. **DCP Slice Execution:** Each DCP slice executes on a worker with access to site-local data. It trains `n_k` standard decision trees using local feature bagging and bootstrapping.
4. **Result Gathering:** Each worker serializes its `n_k` fitted decision tree estimators and returns them via DCP.
5. **Coordinator Assembly:** The coordinator merges the lists of decision trees into a single unified forest structure.
6. **Inference:** Inference is performed locally on the coordinator or downstream client via majority voting over all `N` trees.

```
+-------------------------------------------------------------+
|                        Coordinator                          |
|  1. Computes per-site tree allocation: n_k = n_estimators / K|
|  2. Dispatches DCP Job with K slices                        |
+-------------------------------------------------------------+
        |                                    |
  [Slice 1] (n_1 trees)                [Slice 2] (n_2 trees)
        v                                    v
+-----------------------+            +-----------------------+
|   DCP Worker Site 1   |            |   DCP Worker Site 2   |
|  (Local Data Site 1)  |            |  (Local Data Site 2)  |
|  Trains n_1 Decision  |            |  Trains n_2 Decision  |
|  Trees locally        |            |  Trees locally        |
+-----------------------+            +-----------------------+
        |                                    |
        +---- Returns Serialized Trees ------+
                            |
                            v
+-------------------------------------------------------------+
|                        Coordinator                          |
|  3. Combines all n_k trees into single global forest        |
|  4. Serializes unified model for inference                  |
+-------------------------------------------------------------+
```

## Modules and responsibilities

| Module | Responsibility | Runs where | Owner |
|---|---|---|---|
| `rf_coordinator.py` | Validates hyperparameters, partitions tree counts, creates DCP job, gathers slice results, and constructs final forest. | Coordinator (Client) | Mohsan |
| `rf_worker.py` | Reads local site data, trains assigned subset of decision trees, extracts tree state structures, returns serialized trees. | DCP Worker | Nate |
| `rf_model.py` | Contains `FederatedRandomForestClassifier` container, voting logic, and scikit-learn export/import compatibility. | Coordinator & Inference clients | Ivan |
| `rf_serialization.py` | Encodes and decodes decision tree node arrays, thresholds, and leaf values to JSON-safe dictionaries. | Worker & Coordinator | Mohsan, Nate |

## Data flow

1. **Coordinator -> Worker (Job Input):**
   - Slice configuration: number of trees to train (`n_k`), hyperparameter dictionary (`max_depth`, `criterion`, `min_samples_split`, `random_state`), and local data file identifier.
2. **Worker Computation:**
   - Worker loads local tabular data.
   - Worker trains `n_k` `DecisionTreeClassifier` models using local bootstrap samples.
3. **Worker -> Coordinator (Slice Output):**
   - Serialized array of `n_k` trees. Each tree contains node arrays: `children_left`, `children_right`, `feature`, `threshold`, `value`, and class label mappings.
4. **Privacy Validation (R6):**
   - Raw data records, sample IDs, and feature values never leave the worker. Only decision boundaries (split features, thresholds, and aggregate class counts at leaves) are transmitted.

## DCP job design

- **Job Type:** 1 discrete DCP compute job.
- **Slices:** Exactly 1 slice per data-holding site (or 1 slice per batch of trees if a site trains many trees).
- **Compute Groups / Remote Data:** Uses DCP Compute Groups and Remote Data specifications so that each slice executes on the specific worker node co-located with that site's private data storage.

## Training rounds or aggregation

### How trees are trained and combined
- **Decision:** Trees are trained completely locally at each site and combined via ensemble concatenation (Federated Bagging).
- **Communication:** Exactly **1 round** of communication. Slices are dispatched simultaneously and collected once.

### What each participant sends
- Each site returns an array of `n_k` decision tree states.
- Size per tree is proportional to the number of nodes: `O(2^max_depth)`. For `max_depth=10`, a binary tree has at most 1,023 nodes (~20–40 KB per tree). For 25 trees, payload size is ~0.5–1.0 MB, well within DCP payload thresholds.

### How heterogeneous sites are handled
- **Unequal Sample Sizes:** Larger sites can optionally be allocated more trees proportionally, or tree votes can be weighted by site sample count `N_k / N_total`.
- **Schema Requirements:** All sites must share identical feature names, types, and column order. Categorical features must be pre-encoded or one-hot encoded identically across sites.

### Privacy risks of shared information

| Shared item | What it could reveal | Mitigation | Accepted? |
|---|---|---|---|
| Split thresholds | Exact feature values present in local training set | Bounding tree depth (`max_depth <= 10`) prevents single-sample isolations. | Yes (Documented) |
| Leaf sample counts | Exact class distributions in narrow partitions | Minimum samples per split (`min_samples_split >= 5`) ensures leaf aggregation. | Yes (Documented) |
| Tree structures | Distribution of feature importances at that site | Natural property of tree ensembles; no row reconstruction possible. | Yes |

### Hyperparameters

| Hyperparameter | Description | Default | Valid values | Validated where |
|---|---|---|---|---|
| `n_estimators` | Total number of trees across all sites | `100` | Integer >= 2 | Coordinator |
| `max_depth` | Maximum depth of each decision tree | `10` | Integer >= 1 or `None` | Coordinator |
| `min_samples_split` | Minimum samples required to split an internal node | `2` | Integer >= 2 | Coordinator |
| `criterion` | Split quality metric | `"gini"` | `"gini"`, `"entropy"` | Coordinator |
| `random_state` | Seed for deterministic training | `None` | Integer or `None` | Coordinator |

## Serialization for inference

- The model assembles into a standard format that can be loaded into a scikit-learn `RandomForestClassifier` object by assigning `estimators_`, `classes_`, and `n_features_in_`.
- Enables running inference (R5) using solely `numpy` or `scikit-learn` without installing the DCP SDK.

## Failure handling

- **Site Dropout (R8):** If a site fails or times out, the coordinator checks if a quorum of sites finished (e.g., >= 50% of planned trees). If so, it warns the user and builds the forest using the available trees.
- **Slice Reuse (R9):** If a job fails partially, DCP slice caching allows completed site trees to be reused without recomputation.

## Dependencies

- Worker: `scikit-learn`, `numpy`
- Coordinator: `dcp-client` (or Distributive DCP JS/Python bridge), `numpy`, `scikit-learn`
- Inference: `scikit-learn` and `numpy` only (R5).

## Work breakdown

| Task | Description | Requirement IDs | Owner | Status |
|---|---|---|---|---|
| Worker local tree trainer | Script executed on worker to fit local trees and serialize node arrays | R1, R6 | Nate | In progress |
| Coordinator slice manager | Job creation, slice distribution, and tree gathering via DCP | R1, R2, R8 | Mohsan | In progress |
| Model assembly & inference | Stitching tree estimators into unified scikit-learn compatible object | R2, R5 | Ivan | In progress |
| Serialization / Deserialization | JSON/binary tree parser between worker payload and coordinator | R5, R6 | Mohsan | Not started |

## Owners

| Area | Owner(s) |
|---|---|
| Tree / split-statistic training | Nate |
| Aggregation / forest assembly | Ivan |
| Coordinator & DCP Job flow | Mohsan |
| Heterogeneous sites & Privacy analysis | Mohsan, Ivan |
| Serialization for inference | Nate, Mohsan |
