# Reproducibility

## Environment

See:

configs/environment.txt

for the original experimental environment.

## Dataset

The dataset is not redistributed in this repository.

See:

data/README.md

for the official dataset source, DOI, and license information.

## Experimental Design

The final study uses:

- 1,170 physical side-scan sonar images
- 668 annotated bounding boxes
- nested 5-fold evaluation
- separate calibration and test data
- reliability calibration
- selective prediction
- conformal risk control

## Canonical Results

The primary reported benchmark is:

esults/FINAL_CANONICAL_BENCHMARK.csv

The conformal certification ledger is:

esults/FINAL_CONFORMAL_GUARANTEE_LEDGER.csv

The provenance certificate is:

esults/FINAL_PROVENANCE_CERTIFICATE.json

## Reproducibility Principle

The repository intentionally excludes raw datasets, trained model
weights, temporary training outputs, caches, and intermediate
development artifacts.

This keeps the public repository focused on the final scientific
protocol and results.
