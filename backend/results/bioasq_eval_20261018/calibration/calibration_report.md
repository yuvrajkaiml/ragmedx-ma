# Phase D — Confidence Calibration (reused phase5 artifacts)

- Selected source: `D:\Ragmed_2\RAGMedXMA_Chapter5_Validated_Results\predictions.jsonl`
- Source consistency: **ok**
- Config rows: 40, BioASQ: 13, pubmedqa excluded: 27, abstained excluded: 11
- **N (non-abstained BioASQ) = 2**, **P (correct) = 0**
- Data regime: **DEGENERATE**

> DEGENERATE — single-class / insufficient data, provisional, not a calibration claim

The correctness labeller found P=0 correct answers among N=2 non-abstained BioASQ questions in config 'phase5-full-pipeline'. With no positive class (or no negative class), ECE/Brier are degenerate and a selective-accuracy threshold cannot be meaningfully optimized, so the 0.65 proposal can be neither confirmed nor refuted from this data. No calibration map is fit; all reported numbers are the RAW heuristic over the labelled rows, provisional.

- Raw ECE: 0.8493, Raw Brier: 0.721315
- At incumbent 0.65: coverage=1.0, selective_accuracy=0.0, n_answered=2

**The 0.65 threshold can be neither confirmed nor refuted from this data.**
