# Supervisor Routing and Latency Evaluation

Evaluation date: 2026-08-18  
Dataset: `data/evaluation/supervisor_routing_dataset.json` version 1.0  
Records: 40  
Endpoint: `POST /v1/route`  
Production supervisor: Groq `openai/gpt-oss-120b`, with the configured fallback chain

## Method

The dataset contains explicit expected guard decisions and route sets. It covers
manual-grounded RAG, current web information, manual visuals, attached images,
synthetic mental-health appointments, appointment follow-ups, supervisor scope
refusals, greetings, conversation turns, prompt injection, and dose/injection
safety gates. English, Arabic, and French examples are included.

The route-only endpoint executes the same input guard, structured supervisor,
route normalization, provider fallback, and deterministic routing rules as the
production graph. It intentionally stops before specialist execution. This
separates supervisor decision latency from retrieval, vision, web, Agent B, and
answer-generation latency.

Route comparison is set-based because parallel route order is not semantically
important. A decision passes only when both the guard decision and the complete
route set match their labels.

## Results

| Metric | Result |
|---|---:|
| Completed records | 40 / 40 |
| Overall decision accuracy | 1.0000 |
| Exact supervisor route accuracy | 1.0000 |
| Input-guard accuracy | 1.0000 |
| Route micro precision | 1.0000 |
| Route micro recall | 1.0000 |
| Route micro F1 | 1.0000 |
| First-pass completion rate | 0.9750 |
| Records successfully retried | 1 |

### Per-route results

| Route | Positive support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `rag` | 18 | 1.0000 | 1.0000 | 1.0000 |
| `visual` | 10 | 1.0000 | 1.0000 | 1.0000 |
| `web_search` | 6 | 1.0000 | 1.0000 | 1.0000 |
| `appointment` | 6 | 1.0000 | 1.0000 | 1.0000 |
| `scope_guard` | 4 | 1.0000 | 1.0000 | 1.0000 |

Supports sum to more than the number of supervisor records because visual
first-aid requests correctly use the multi-label route `rag + visual`.

### Latency

| Measurement | Mean | Median | P95 | Maximum |
|---|---:|---:|---:|---:|
| Guard + supervisor decision | 963.634 ms | 1001.415 ms | 1495.263 ms | 3435.582 ms |
| Supervisor only (34 routed records) | 1133.551 ms | 1051.188 ms | 1615.916 ms | 3435.413 ms |
| Client/API round trip | 987.293 ms | 1017.919 ms | 1504.772 ms | 3445.898 ms |

The six deterministic guard cases did not call the supervisor and completed in
approximately 0.0-0.1 ms internally. All 34 routed records used Groq; the six
guarded records report `not_used` as their provider.

One request timed out during the first pass and succeeded when retried. The
accuracy score therefore describes all completed decisions, while the 97.5%
first-pass completion rate separately exposes the observed transient provider
or network reliability issue.

## Interpretation and limitations

These results demonstrate that the implemented routing policy behaves correctly
on the defined balanced regression set. They do not prove perfect routing on
arbitrary user traffic. The examples have unambiguous intent, the dataset is
small, and deterministic normalization enforces several safety-critical routes
such as attached first-aid images and appointment requests. Future evaluation
should add noisy spelling, longer conversations, implicit intent, ambiguous
multi-intent queries, and a held-out set written independently of the routing
prompt.

## Reproduce

With Agent A running at port 8000:

```powershell
python eval\supervisor_routing_eval.py --timeout 90 --overwrite
```


