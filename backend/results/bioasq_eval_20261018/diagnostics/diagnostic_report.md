# Phase A — BioASQ PMID Diagnostic

- Queries: **250** (with gold: 250)
- Raw gold PMID entries: 2398
- Distinct normalized gold PMIDs: **2394**
- Unparsable gold entries: 0
- eval_split chunks: 58173 (distinct PMIDs: 37395)
- Gold PMIDs matched in eval_split: **0** (0.0%)
- eval_split exclusion confirmed: **True**

## 789 / 2,394 / 172 / 250 reconciliation

- `all_250_distinct_gold_pmids` — distinct normalized gold PMIDs over all 250 eval questions: **2394**
- `phase5_40_subset_distinct_gold_pmids` — distinct gold PMIDs over the 40-question phase5 subset: **172**
- `phase5_40_subset_sum_n_gold` — sum of per-query n_gold over the 40 phase5 queries: **172**
- `phase5_40_subset_bioasq_only_distinct` — distinct gold PMIDs over the BioASQ-only phase5 subset: **145**

- 789 reproducible: **False**

The originally quoted 789 figure could not be reproduced from any artifact present in the repository; it most plausibly originated from a run configuration or question subset not retained on disk. The canonical measured figures are 2,394/250 (full) and 172/40 (phase5 subset).

## Coverage gap vs retrieval failure

- Missing gold document (coverage gap): The corpus does not contain the gold PMID at all; the retriever cannot return it. This is a corpus-coverage gap, not a ranking failure.
- Retrieval failure: The gold PMID IS present in the corpus but does not appear in the retriever's top-k. This is a ranking failure attributable to the retriever.
- eval_split case: All gold PMIDs are absent from eval_split by deliberate leakage exclusion, so every eval_split 'miss' is a coverage gap, never a retrieval failure.

## Retrieval-eval corpus availability

- Corpus: `D:\Ragmed_2\backend\data\indexes\bioasq_retrieval_eval\v1` (distinct PMIDs: 2010)
- Queries with >=1 available gold: 250
- Queries with all gold available: 188
- Gold PMIDs covered in corpus: 2010 (83.9599%)

## PMID->chunk round-trip

- Sampled: 20, round-trip OK: 20, all OK: True
