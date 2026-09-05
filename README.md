# AquaTrust: Calibrated Reliability and Conformal Risk Control for Side-Scan Sonar Object Detection

A research release for a reliability-aware AI decision layer for side-scan sonar object detection.

## Overview
AquaTrust evaluates whether detector confidence is trustworthy by combining detection confidence with local image evidence, perturbation consistency, and predicted-class information. A conformal risk-control layer then determines whether a detection can be accepted under a predefined risk budget or deferred for human review.

## Experimental Setup
- Detector: YOLOv8s at 832-pixel input resolution
- Dataset: 1,170 real side-scan sonar images with 668 annotated objects
- Split: 819 training, 175 calibration, and 176 final evaluation images
- Candidate detections: 2,115 across the complete dataset
- Evaluation: calibration, risk, discrimination, selective prediction, and conformal risk control

## Main Results
AquaTrust M6 achieved ECE-10 = 0.0689, Risk@80% = 0.4965, Brier = 0.2219, AUC = 0.7047, PR-AUC = 0.6778, and AURC = 0.3630.

The main contribution is a detector-agnostic reliability and decision layer. The experiments deliberately keep YOLOv8s fixed so that improvements can be attributed to the reliability framework rather than a simultaneous detector change.

## Repository Structure
- code/ - Research and evaluation scripts
- configs/ - Experimental configuration files
- data/ - Dataset documentation and metadata; raw dataset files are not redistributed
- docs/ - Research documentation and supporting material
- figures/ - Final figures and visual results
- poster/ - Smart Amrita Hackathon poster materials
- results/ - Final experimental results

## Dataset
The underlying side-scan sonar dataset is from Santos et al. (2024) and is available through Figshare. DOI: 10.6084/m9.figshare.24574879. Please obtain the dataset from its official source and follow its stated license and citation requirements.

## Reproducibility
The repository contains the curated code, configurations, figures, results, and documentation required to understand the reported experiments. Raw dataset archives and trained checkpoints are intentionally not included.

## Citation
See CITATION.cff for the repository citation metadata.

## License
See LICENSE for licensing information.
