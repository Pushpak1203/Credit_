# Fairness Protocol

1. Compute selection-rate disparity and adverse-impact ratio (minimum group selection rate / maximum group selection rate).
2. Compare approval-good-rate and bad-decline-rate by group; retain TPR/FPR definitions in the report.
3. Compare approved bad rates / calibration by group.
4. Check feature/protected-attribute correlations as a proxy diagnostic; high association triggers investigation, not automatic exclusion.
5. Escalate if adverse-impact ratio, error-rate gaps, calibration gaps, or proxy correlations materially deteriorate versus a documented baseline. Thresholds must be set by the lender's compliance/model-risk governance rather than inferred from synthetic data.
6. Protected attributes are never passed to the model.
