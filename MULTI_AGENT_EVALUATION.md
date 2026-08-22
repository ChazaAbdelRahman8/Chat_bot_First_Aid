# Hard Multi-Agent Evaluation

## Purpose

This evaluation measures the complete production workflow of Agent System A,
including its HTTP communication with Agent System B. It complements the
40-question routing regression set with deliberately difficult cases. The hard
set is intended to reveal integration failures rather than produce a perfect
score.

## Test set and method

- Dataset: `data/evaluation/multi_agent_hard_dataset.json`
- Size: 24 tasks and 26 user turns
- Execution: real Docker services, production `/v1/chat` and
  `/v1/chat/visual` endpoints, Qdrant, MongoDB, Groq/Ollama, live web search,
  and Agent System B over HTTP
- Languages: English, Arabic, French, and code-switched queries
- Difficult cases: misspellings, follow-ups, multi-specialist requests,
  attached and manual visuals, current information, appointment booking,
  prompt injection, exact-dose requests, crisis escalation, and mixed
  in-scope/out-of-scope requests

Expected routes, tools, parallel-aware trajectories, minimum useful steps, and
deterministic completion criteria are defined before execution in the dataset.

## Results after remediation

| Dimension | Metric | Result |
|---|---|---:|
| Routing accuracy | Exact task route set | 100.00% |
| Routing accuracy | Turn-level exact match | 100.00% |
| Routing accuracy | Micro precision / recall / F1 | 100.00% / 100.00% / 100.00% |
| Tool accuracy | Exact tool names and valid arguments | 100.00% |
| Tool accuracy | Argument validity | 100.00% |
| Tool accuracy | Micro precision / recall / F1 | 100.00% / 100.00% / 100.00% |
| Trajectory match | Parallel-aware exact match | 100.00% |
| Task completion | All deterministic criteria passed | 100.00% (24/24) |
| Step efficiency | Graph-node efficiency | 100.00% |
| Step efficiency | Tool-call efficiency | 90.97% |
| Step efficiency | Combined action efficiency | 95.59% |

The graph executed exactly the required 128 nodes, with no missing or extra
graph nodes. It executed 47 logical tool calls against a minimum of 36. The
additional calls come primarily from bounded search retry and answer-validation
cycles, so tool-loop efficiency remains the main optimization opportunity.

### Before-and-after comparison

| Metric | Initial hard run | After fixes |
|---|---:|---:|
| Exact routing accuracy | 87.50% | 100.00% |
| Exact tool and argument accuracy | 79.17% | 100.00% |
| Trajectory match | 87.50% | 100.00% |
| Task completion | 66.67% | 100.00% |
| Combined action efficiency | 88.46% | 95.59% |

## Latency and API-token cost

| Measure | Mean | Median | P95 | Maximum |
|---|---:|---:|---:|---:|
| End-to-end latency | 26.561 s | 9.987 s | 129.900 s | 239.167 s |
| Model tokens/task | 3,785 | 1,467 | 15,191 | 17,868 |
| Model calls/task | 3.125 | 2 | 8 | 9 |
| Estimated API cost/task | $0.0021 | $0.0008 | $0.0085 | $0.0106 |

Estimated API-token cost for all 24 tasks was **$0.051412**. The calculation
uses the configured model/provider token counts and published Groq prices:
GPT OSS 120B at $0.15/M input and $0.60/M output, and Qwen 3.6 27B at $0.60/M
input and $3.00/M output. Local Ollama inference, hardware, electricity, Docker,
and free search-service costs are not monetized, so this is an API-token-cost
estimate rather than total infrastructure cost.

References: [Groq pricing](https://groq.com/pricing) and
[Qwen 3.6 27B model details](https://console.groq.com/docs/model/qwen/qwen3.6-27b).

## Failure remediation log

The dataset labels and expected trajectories were not relaxed. Each original
failure was fixed in the production workflow and rerun against its original
question.

| ID | Original failure | Implemented fix | Verified result |
|---|---|---|---|
| `MAH002` | Arabic code-switched RAG answer intermittently abstained or failed language validation. | Preserve the original utterance for generation, retain the supervisor rewrite separately, and use it only for a bounded retrieval retry. | RAG route, retrieval tool, Arabic answer, and citation all pass. One evaluation attempt abstained and the recorded retry passed, so provider variability remains visible. |
| `MAH008` | Lebanon search was geographically ambiguous; final output omitted a URL and could take more than 180 seconds. | Preserve verbatim search results and URLs, bound DDGS calls to 8 seconds, reduce the web ReAct cycle, cap local generation at 20 seconds, and use validated evidence-only answer fallback. | RAG + web, citations, URL, and completion pass in 25.161 seconds. |
| `MAH009` | Manual page rendered, but outer ReAct prose dropped the canonical RAG citation. | Make the validated RAG generation payload—not wrapper prose—the specialist evidence contract. | RAG + visual, both tools, citation, and rendered page pass. |
| `MAH010` | Arabic manual-visual intent missed the visual route; RAG could abstain even when a source page rendered. | Add Arabic manual-visual signals and a non-interpretive, cited rendered-page fallback using only page metadata. | Arabic RAG + visual route, two tools, citation, and page image pass. |
| `MAH014` | Agent B received an English router rewrite and returned English or missing preferences. | Send the original Arabic utterance over HTTP; parse Arabic language, online mode, and budget words; localize Agent B messages. | `matches_found`, Arabic response, and matching profiles pass. |
| `MAH020` | Specialist models could finish without mandatory tools; heat-alert path timed out or omitted manual citations. | Enforce direct mandatory-tool fallback for RAG and web, preserve canonical RAG evidence and exact web URLs, and bound model/search fallbacks. | Both routes, all expected tools, citation, URL, and completion pass in 6.944 seconds. |
| `MAH021` | The input guard rejected the whole Bitcoin + bleeding request. | Detect valid first-aid intent inside mixed-domain input and decompose current mixed requests into RAG + web instead of blocking the urgent portion. | Guard, RAG + web routing, tools, citation, and completion pass. |
| `MAH022` | French manual-visual request was treated as RAG-only. | Add French manual/illustration signals and the same cited rendered-page evidence contract. | French RAG + visual route, both tools, citation, and rendered page pass. |

Additional safeguards introduced during remediation:

- RAG and visual specialists always receive the original user utterance, while
  optional retrieval rewrites remain separate metadata;
- mandatory RAG, web, and manual-visual tools execute even if a ReAct provider
  fails before calling them;
- deterministic synthesis concatenates only validated specialist evidence and
  is checked by the same citation/URL guard;
- the web ReAct loop is limited to one complete reason/action/observation cycle;
- remote failure no longer launches an unbounded second local synthesis pass.

## Interpretation

The separate 40-question routing set remains useful as a deterministic
regression test, where it achieved 100%. It is not the headline production
quality result. This hard end-to-end set is the fairer baseline because it tests
tool calls, trajectories, final outcomes, efficiency, latency, and cost under
multilingual and multi-agent conditions.

All eight failed cases remain unchanged in the dataset and now act as regression
tests. The 100% result describes this 24-task hard set, not universal production
accuracy. Live provider behavior is still variable—most visibly in `MAH002`—and
the high P95 includes older checkpointed visual/web tasks. A larger independently
authored holdout should therefore be the next generalization test.

## Reproduce

With all Docker services running:

```powershell
python eval/multi_agent_eval.py `
  --dataset data/evaluation/multi_agent_hard_dataset.json `
  --output-dir data/evaluation/results/multi_agent_hard
```

Use `--refresh-ids MAH002,MAH008,MAH009,MAH010,MAH014,MAH020,MAH021,MAH022`
to rerun all originally failed cases while retaining the other checkpointed rows.
