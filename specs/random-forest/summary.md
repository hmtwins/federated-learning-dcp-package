# Random Forest Summary

| Field | Value |
|---|---|
| Owner | Nate, Mohsan, Ivan |
| Status | Draft |
| Last updated | 2025-02-14 |
| Related | [implementation-plan.md](implementation-plan.md), [test-plan.md](test-plan.md), [shared summary](../shared/summary.md) |

## Purpose

The Random Forest track provides an ensemble-based non-linear model for the federated learning package on DCP. While Logistic Regression handles linearly separable problems, Random Forest enables privacy-preserving learning on complex tabular datasets with non-linear decision boundaries, feature interactions, and mixed continuous/discrete features without requiring feature scaling.

## Users

This track serves the same personas identified in the shared summary:
- **Data scientists / ML engineers** who want to train tree-based models across decentralized data silos (e.g., cross-hospital patient data, multi-bank fraud detection) using a familiar API.
- **Data custodians** who host private datasets and require guarantees that raw data rows never leave their compute environment.

## Scope

- **In Scope:**
  - Tabular classification (Binary classification MVP; multiclass support built on the same architecture).
  - Single-round Federated Bagging architecture: sites train independent sets of decision trees locally on private data; the coordinator ensembles them into a single global forest.
  - Integration with the DCP Python/JS client for slice dispatch and result gathering.
  - Serialization to a lightweight, standard format (scikit-learn compatible) for inference.
- **Out of Scope (for PoC):**
  - Multi-round federated split-finding (e.g., FedTree/quantile-histogram streaming).
  - Gradient Boosted Decision Trees (GBDT / XGBoost) which require sequential dependent rounds.
  - Regression (deferred to post-PoC).
  - Advanced differential privacy noise addition (leaf clipping/Laplace noise on counts deferred).

## Non-goals

- Iterative multi-round tree construction across sites.
- Supporting unstructured data (images, text) natively.
- Dynamic automated hyperparameter search over DCP during training.

## Functional requirements

This track maps directly to shared requirements (R1–R10):
- **R1 (DCP Compute):** Slice computation maps local tree fitting to DCP workers.
- **R2 (Coordinator API):** Coordinator exposes a clean `FederatedRandomForestClassifier` fit/predict interface.
- **R5 (Model Export & Minimal Inference):** Global forest serializes to standard estimator structures loadable with scikit-learn without requiring DCP dependencies.
- **R6 (Data Locality / Privacy):** Workers return serialized tree branch/leaf parameters only; zero raw feature rows leave the worker.
- **R7 (Reproducibility):** Hyperparameters include distributed random seeds for deterministic bagging.
- **R8 (Partial Participation):** If a site fails or times out, the coordinator aggregates trees from remaining sites and produces a valid forest with reduced estimator count.
- **R9 (Failure Recovery):** Completed slices from successful sites are cached and reused if the job is rerun.
- **R10 (Logging & Observability):** Progress reporting on slice completion and forest assembly.

## Constraints

- **Payload Size:** Serialized decision trees grow with depth ($O(2^D)$). Maximum depth must be bounded (default `max_depth=10`) to prevent slice results from exceeding DCP return-value payload limits (typically a few megabytes).
- **Data Non-IID Skew:** If sites have vastly unequal sample counts or class imbalances, unweighted aggregation can bias the forest.

## Inputs

- **Hyperparameters:**
  - `n_estimators` (total trees in forest, distributed among sites)
  - `max_depth` (maximum tree depth)
  - `criterion` (`gini` or `entropy`)
  - `min_samples_split` (minimum samples required to split a node)
  - `random_state` (seed for reproducibility)
- **Local Data Format:** Tabular numeric matrices (CSV, NumPy arrays, or Pandas DataFrames) located on worker-local storage mapped to the DCP compute environment.

## Outputs

- A unified `FederatedRandomForestClassifier` object containing the aggregated collection of trained decision trees.
- Serialized portable JSON / joblib model artifact suitable for prediction in production.

## Open questions

- (See `specs/shared/open-questions.md`) How should tree count be divided when `n_estimators` is not cleanly divisible by the number of participating sites?
- Should site trees be weighted proportionally to the site's local dataset size during voting?
