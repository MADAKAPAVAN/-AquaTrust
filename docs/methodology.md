# AquaTrust Methodology

## Research Objective

AquaTrust adds a calibrated reliability and conformal decision layer
to an underwater object detector.

The detector produces a candidate detection and confidence score.
AquaTrust evaluates whether that confidence is supported by additional
evidence before allowing the detection to be treated as an
autonomous decision.

## Evidence

The reliability layer uses detector confidence together with
additional evidence including:

- local patch contrast,
- speckle/noise variability,
- perturbation consistency,
- IoU survival,
- predicted class interaction.

## Calibration

The study evaluates raw detector confidence and post-hoc calibration
baselines before evaluating the proposed AquaTrust reliability model.

Calibration is assessed using:

- Expected Calibration Error (ECE),
- Brier score,
- negative log-likelihood where applicable.

## Selective Decision Making

AquaTrust supports selective prediction. Predictions can be accepted
when estimated risk is sufficiently low and deferred when uncertainty
or estimated risk is too high.

The resulting risk-coverage relationship is evaluated empirically.

## Conformal Risk Control

The conformal component uses calibration data to determine operating
policies under the stated exchangeability and monotonicity assumptions.

The appropriate claim is:

"Under the stated exchangeability, monotonicity, and calibration
protocol, CRC provides a finite-sample expected-risk guarantee."

PAC-CRC results are described separately as high-probability control
over expected risk under the stated assumptions.

These guarantees should not be interpreted as unconditional
per-mission or per-image safety guarantees.

## Detector Backbone

YOLOv8s is deliberately fixed as the detector backbone.

The purpose of the study is to isolate the contribution of the
reliability and conformal decision layer rather than compare detector
architectures.

Changing the detector architecture simultaneously would introduce
an additional experimental variable and make attribution of the
observed reliability improvements less direct.

Generalization across detector families and newer detector
generations remains future work.

## Experimental Protocol

The final protocol uses nested evaluation with held-out test data.

The public repository contains the canonical scripts and result
artifacts needed to understand and reproduce the reported analysis.
