# RAGAS Evaluation Report

## 1. Purpose

This document reports the standalone RAGAS evaluation of the first-aid retrieval-augmented generation system. It is separate from retrieval-only evaluation because RAGAS evaluates the generated answer and retrieved context together.

The evaluation used the frozen 220-record golden dataset. The dataset was not regenerated or modified for this run.

## 2. Evaluated configuration

| Component | Configuration |
|---|---|
| Generator | `qwen3:4b` through local Ollama |
| Independent RAGAS judge | `granite4:3b` through local Ollama |
| Judge temperature | `0` |
| RAGAS version | `0.4.3` |
| Dense embedding model | `BAAI/bge-m3`, 1024 dimensions, L2-normalized |
| Vector database | Qdrant, collection `first_aid_chunks` |
| English retrieval | Dense top-5 plus up to 3 BM25 additions |
| Arabic retrieval | Weighted hybrid reciprocal-rank fusion |
| Corpus | 7 first-aid manuals, 3,492 structure-aware chunks |
| Evaluation dataset | Frozen `golden_dataset.json`, 220 records |

The generator and judge were deliberately different models. This avoids having Qwen directly evaluate its own answers, although Granite is still a relatively small 3.4B-parameter judge and its scores should not be treated as human clinical review.

## 3. Dataset composition

| Expected behavior | Records | Evaluation method |
|---|---:|---|
| Answer | 203 | RAGAS answer and context metrics plus deterministic behavior check |
| Abstain | 11 | Deterministic abstention-behavior check |
| Safety refusal | 6 | Deterministic `policy_gate` behavior check |
| **Total** | **220** | |

RAGAS answer-quality metrics were applied only when an answerable record produced a validated answer. Expected abstentions and safety refusals were not assigned artificial RAGAS scores.

## 4. RAGAS metrics

| Metric | Mean | Median | Completed rows | Coverage of 203 answerable rows |
|---|---:|---:|---:|---:|
| Faithfulness | **0.9268** | 1.0000 | 186 | 91.63% |
| Factual correctness | **0.7790** | 0.8000 | 188 | 92.61% |
| Context precision | **0.9312** | 1.0000 | 188 | 92.61% |
| Context recall | **0.9740** | 1.0000 | 186 | 91.63% |

Metric interpretation:

- **Faithfulness** measures whether claims in the generated answer are supported by retrieved context.
- **Factual correctness** compares claims in the generated answer with the curated reference answer.
- **Context precision** measures whether highly ranked retrieved passages are useful for the question and reference answer.
- **Context recall** measures whether the retrieved context supports the information required by the reference answer.

The averages above use only completed metric values. They do not assign zero to generation failures or judge failures, so the coverage denominators must always be reported alongside the means.

## 5. Deterministic behavior and safety results

| Expected behavior | Passed | Total | Accuracy |
|---|---:|---:|---:|
| Answer | 188 | 203 | **92.61%** |
| Abstain | 11 | 11 | **100.00%** |
| Safety refusal | 6 | 6 | **100.00%** |
| **Overall** | **205** | **220** | **93.18%** |

The deterministic Phase 8 input guard blocked exactly the 17 expected non-answer records in the frozen dataset and blocked none of the 203 answerable records during the collision audit.

The guard distinguishes:

- `policy_gate`: exact medication dose/injection requests, prompt injection, and clearly out-of-scope requests;
- `retrieval_induced`: retrieved evidence is empty or insufficient;
- `model_generated`: the model abstains or repeatedly fails deterministic answer validation.

## 6. Results by question language

The following RAGAS means include only completed metric values in each language group.

| Language | Answerable | Validated answers | Answer rate | Faithfulness | Factual correctness | Context precision | Context recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| English | 161 | 153 | 95.03% | 0.9395 | 0.7822 | 0.9327 | 0.9698 |
| Arabic | 36 | 29 | 80.56% | 0.8867 | 0.7790 | 0.9090 | 0.9911 |
| Arabic-English | 6 | 6 | 100.00% | 0.8000 | 0.6983 | 1.0000 | 1.0000 |

Arabic generation reliability is the clearest language-specific weakness: 7 of 36 answerable Arabic questions failed deterministic generation validation.

## 7. Incomplete evaluations and failure analysis

The final run contains 19 error entries:

| Failure class | Count | Effect |
|---|---:|---|
| Answerable generation failed validation | 15 | No RAGAS metrics were assigned to these records |
| Granite faithfulness schema failure | 2 | Faithfulness missing for two otherwise answered records |
| Granite context-recall schema failure | 2 | Context recall missing for two otherwise answered records |
| **Total** | **19** | |

### 7.1 Generation failures

The 15 answerable records without validated answers are:

`FAE075`, `FAE089`, `FAE104`, `FAE105`, `FAE107`, `FAE113`, `FAE120`, `FAE140`, `FAE141`, `FAE143`, `FAE145`, `FAE181`, `FAE182`, `FAE184`, and `FAE188`.

These failures were safe abstentions rather than unvalidated medical answers being returned. The main causes were per-sentence citation formatting, Arabic response-language validation, or the model producing a non-abstaining answer without an inline citation.

### 7.2 Judge failures

Granite produced invalid RAGAS structured output after retries for:

- `FAE060`: faithfulness;
- `FAE106`: context recall;
- `FAE157`: context recall;
- `FAE185`: faithfulness.

These are evaluation-tool failures, not evidence that the corresponding generated answers were incorrect. They remain missing rather than being imputed as zero or one.

## 8. Interpretation

The results support four conclusions:

1. **Retrieved context was usually relevant and sufficiently complete.** Context precision was 0.9312 and context recall was 0.9740 on completed judgments.
2. **Generated answers were usually grounded.** Faithfulness was 0.9268, with a median of 1.0.
3. **Answer completeness and agreement with the curated reference remain weaker.** Factual correctness was 0.7790, lower than the grounding and retrieval metrics.
4. **Safety behavior was reliable after deterministic gating.** All expected abstentions and safety refusals passed, with no answerable collision in the frozen dataset audit.

The run is marked `valid: false` in the machine-generated summary because 15 answerable records failed generation behavior and four RAGAS judgments are missing. This flag describes evaluation completeness; it does not mean that all completed scores are invalid.

## 9. Limitations

- Granite 4 3B is a small local judge. Human expert review or a stronger independent judge could produce different semantic scores.
- Arabic and English are not equally represented in the dataset.
- RAGAS means exclude failed generations and missing judge outputs. Reporting means without coverage would overstate system-wide performance.
- The deterministic guard was verified against this frozen set. Future, differently worded attacks and out-of-scope requests require additional adversarial testing.
- RAGAS metrics assess semantic grounding and agreement; they do not replace clinical validation of first-aid recommendations.
- Repeated tuning against the frozen set can cause evaluation overfitting. The remaining failures should be preserved as documented limitations or checked on a separate development set before further tuning.

## 10. Reproduction

Prerequisites:

- Qdrant is running and contains 3,492 points in `first_aid_chunks`.
- Ollama is running with `qwen3:4b` and `granite4:3b` installed.
- The `FirstAidFinal` environment contains the pinned RAGAS and LangChain dependencies.

Run or resume the complete evaluation from the project root:

```powershell
python eval\ragas_eval.py --all --generation-attempts 6 --metric-attempts 3
```

The evaluator is checkpointed. Existing successful rows and metric values are reused unless `--restart` is explicitly supplied.

Machine-readable artifacts:

- `data/evaluation/results/ragas_full/summary.json`
- `data/evaluation/results/ragas_full/rows.jsonl`
- `data/evaluation/results/ragas_full/run_stdout.log`
- `data/evaluation/results/ragas_full/run_stderr.log`

## 11. Final reported result

For the final project, report the result as:

> On the frozen 220-record evaluation set, the system achieved 93.18% expected-behavior accuracy, including 100% on 11 expected abstentions and 100% on 6 safety-refusal tests. Among successfully validated answerable responses, RAGAS measured 0.9268 faithfulness, 0.7790 factual correctness, 0.9312 context precision, and 0.9740 context recall. Metric coverage ranged from 186 to 188 of 203 answerable records; 15 answerable generations safely abstained and four judge outputs remained incomplete.

## 12. Four-axis answer-quality pilot on the production retrieval set

The evaluator was extended to report the four answer-quality dimensions required for the final project. This pilot uses 10 answerable questions selected deterministically from `golden_retrieval_multichunk.json` and the final production Top-8 retrieval configuration. It is a validation run, not a replacement for the frozen 220-record headline evaluation.

| Required metric | Operational definition | Mean | Completed rows |
|---|---|---:|---:|
| Faithfulness | Proportion of answer claims supported by retrieved evidence | **0.8611** | 9/10 |
| Correctness | F1 agreement between generated-answer claims and the reference answer | **0.6450** | 10/10 |
| Relevance | Semantic relevance of the generated answer to the user question | **0.8197** | 10/10 |
| Completeness | Recall of reference-answer claims in the generated answer | **0.6170** | 10/10 |

Retrieval diagnostics for the same pilot were context precision `0.9737` (10/10) and context recall `0.9778` (9/10). All 10 records produced validated answers and passed the expected-behavior check.

One faithfulness judgment and one context-recall judgment are missing because `granite4:3b` returned invalid structured judge output after all retries. Missing values were excluded from their means and were not converted to zeros. The machine summary therefore reports `valid: false`; this denotes incomplete judge coverage, not an invalid run.

These pilot results show a specific gap: retrieved evidence was strong and answers were generally grounded and relevant, while correctness and especially completeness were lower. The next improvement target is therefore answer synthesis from all relevant evidence, not simply increasing retrieval Top-K.

Pilot artifacts:

- `data/evaluation/results/ragas_final_generation_smoke/summary.json`
- `data/evaluation/results/ragas_final_generation_smoke/rows.jsonl`

Run or resume the pilot:

```powershell
python eval\ragas_eval.py --golden data\evaluation\golden_retrieval_multichunk.json --sample-size 10 --output-dir data\evaluation\results\ragas_final_generation_smoke --generation-attempts 6 --metric-attempts 3
```

For a final score over every eligible record in this production-retrieval dataset, use the same command with `--all` and a separate output directory.
