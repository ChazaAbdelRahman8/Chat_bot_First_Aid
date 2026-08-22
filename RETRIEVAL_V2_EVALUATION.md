# Retrieval Evaluation

## Overview

This project evaluates the retrieval part of the RAG system using a fixed golden dataset.

The goal is to measure whether the retriever finds the correct manual pages or chunks before they are sent to the LLM.

The main retrieval methods tested were:

* BGE-M3 dense retrieval
* BM25 keyword retrieval
* Dense + BM25 RRF fusion
* Query translation
* Multi-query expansion

---

## Dataset

Main evaluation dataset:

`data/evaluation/golden_dataset_v2.json`

* 220 total records
* 203 retrieval questions
* 17 behavior cases excluded
* 7 manuals
* 3,492 chunks

The dense embedding model used is:

`BAAI/bge-m3`

Embedding size:

`1024`

The main evaluation depth is:

`Top-8`

---

## Metrics

### Precision@K

Measures how many retrieved results are relevant.

Example:

If 4 of the Top-8 retrieved chunks are relevant:

`Precision@8 = 4 / 8 = 0.50`

### Recall@K

Measures how many of the expected relevant results were retrieved.

Higher recall means the retriever is finding more of the useful information.

### Hit@K

Checks whether at least one correct result appears in the Top-K.

### MRR@K

Measures how high the first relevant result appears.

A relevant result at rank 1 gives the best score.



# Baseline Comparison

Two initial retrieval strategies were compared.

| Method              | Recall@8 |  MRR@8 |  Hit@8 |
| ------------------- | -------: | -----: | -----: |
| BGE-M3 Dense        |   0.5948 | 0.5268 | 0.7931 |
| BM25                |   0.4142 | 0.3304 | 0.5468 |


BGE-M3 dense retrieval was the strongest single baseline.



---

# RRF Experiment

A proper Reciprocal Rank Fusion strategy was tested instead of directly appending BM25 results.

Several dense/BM25 weights were tested.

The best policy was:

### English

Use BGE-M3 dense retrieval only.

`Dense Top-8`

### Arabic and Code-Switched Queries

Retrieve:

* Dense Top-30
* BM25 Top-30

Then combine them using weighted RRF:

`70% Dense + 30% BM25`

Return the final Top-8.

---

## Selected Production Policy

| Policy          |   Recall@8 |      MRR@8 |      Hit@8 |
| --------------- | ---------: | ---------: | ---------: |
| Dense Baseline  |     0.5948 |     0.5268 |     0.7931 |
| Selected Policy | **0.7047** | **0.5378** | **0.8030** |

The selected policy improves all three main metrics.

Production therefore uses:

```text
English
    ↓
BGE-M3 Dense
    ↓
Top-8
```

```text
Arabic / Code-Switched
    ↓
Dense Top-30 + BM25 Top-30
    ↓
70/30 Weighted RRF
    ↓
Top-8
```

---

# Translation Experiment

Arabic and code-switched queries were also tested with English translation.

The translation policy did not improve retrieval.

For the Arabic/code-switched evaluation set:

| Method             |   Recall@8 |
| ------------------ | ---------: |
| Translation Policy |     0.5040 |
| Dense Only         |     0.5317 |

Translation was therefore rejected and is not used during production retrieval.

---

# Multi-Query Experiment

The project also tested generating alternative versions of the user's question.

Examples included:

* Medical keyword rewrite
* Synonym rewrite
* Multiple query variants

These approaches reduced retrieval performance.

| Method              |   Recall@8 |      MRR@8 |      Hit@8 |
| ------------------- | ---------: | ---------: | ---------: |
| Production Baseline | **0.7047** | **0.7378** | **0.8030** |
| Medical Rewrite     |     0.5866 |     0.5190 |     0.7635 |
| Two Variants        |     0.5817 |     0.5118 |     0.7586 |

Multi-query expansion was therefore rejected.

---

# Exact-Chunk Evaluation

The original evaluation mainly used correct PDF pages as ground truth.

A second benchmark was created to evaluate exact chunk relevance:

`data/evaluation/golden_retrieval_multichunk.json`

It contains:

* 69 questions
* 56 English
* 13 Arabic
* 5–8 relevant chunks per question

Two local models independently graded candidate chunks:

* `granite4:3b`
* `qwen3:4b`

A chunk was considered relevant only when both models judged it sufficiently relevant.

---

## Exact-Chunk Production Results

| Metric      | Result |
| ----------- | -----: |
| Precision@5 | 0.6391 |
| Recall@5    | 0.5621 |
| MRR@5       | 0.8406 |
| Precision@8 | 0.5638 |
| Recall@8    | 0.8446 |
| MRR@8       | 0.8406 |
| nDCG@8      | 0.7402 |

These results show that the production retriever usually places a relevant chunk near the top and retrieves a meaningful portion of the available relevant context.

---

# Final Retrieval Strategy

The current production retrieval strategy is:

```text
User Question
      ↓
Language Detection
      ↓
 ┌───────────────┬─────────────────────┐
 │               │                     │
English      Arabic / Code-Switched
 │               │
 ↓               ↓
Dense          Dense Top-30
Top-8             +
               BM25 Top-30
                  ↓
             70/30 RRF
                  ↓
                Top-8
```

Translation and multi-query expansion remain experimental features and are not used in production.

---

# Running the Evaluations

Run the retrieval policy experiment with:

```powershell
python eval/retrieval_policy_experiment.py
```

Run the exact-chunk evaluation with:

```powershell
python eval/evaluate_retrieval_multichunk.py
```

Evaluation results are stored under:

`data/evaluation/results/`

---

# Conclusion

The main findings are:

* BGE-M3 is stronger than BM25 as the main retriever.
* English queries work best with dense retrieval.
* Arabic and code-switched queries improve with 70/30 dense/BM25 RRF.
* Translation did not improve retrieval.
* Multi-query expansion did not improve retrieval.

# limitation 
The production system retrieves the Top-8 chunks for each query. Using eight chunks can introduce additional irrelevant context and therefore reduce precision compared with a smaller retrieval depth.

However, this project prioritizes recall over precision because the domain is high-stakes and missing relevant information may be more harmful than retrieving some additional irrelevant context.
Increasing retrieval depth from 5 to 8 decreases precision but substantially increases recall. Therefore, Top-8 was retained to provide the generation model with broader relevant evidence.

A limitation of this approach is that the additional chunks may introduce noise, increase context size, and potentially distract the generation model. Future evaluation should compare different retrieval depths using end-to-end answer quality in addition to retrieval metrics.