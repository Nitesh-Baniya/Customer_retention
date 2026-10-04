# Track B — Agentic AI MLOps (W15 assistant + W16 verification agent)

MLOps for the assistant built in W15/W16: **uv** (reproducible env), **MLflow** (prompt/config
experiment tracking with full agent traces), **Evidently AI** (LLM-as-a-judge regression test
suite), optional **Airflow** (nightly regression eval). There is no trained model here, so
what gets versioned, tracked and monitored is the *system prompt + agent configuration* and
the *behaviour* of the agent loop.

> **Read this first — status of the numbers in this repo**
> This was developed in a sandbox with no Hugging Face token and no outbound network, so the
> **live** experiment has **not been run yet**. Everything was verified end-to-end with an
> **offline mode** (`--offline`): a *scripted rule-based policy* standing in for the LLM, canned
> weather data, and a heuristic judge standing in for the Evidently LLM judge. Offline runs go to a
> separate MLflow experiment (`agentic-assistant-prompts-OFFLINE-STUB`), are tagged
> `mode=offline_stub`, and every file they produce has `_OFFLINE` in its name. **They prove the
> plumbing works; they are not evidence about any real model.** Run the live command below
> and paste your real table into section 2 before submitting.

```
agentic_mlops/        MLOps layer (config, traced agent, scoring, regression, tracking, compare, diagnose)
app/                  W15/W16 assistant code (agent, LLM client, tools, skills) - copied, with 2 bug fixes (below)
prompts/              prompt_v1.txt, prompt_v2.txt, prompt_v3.txt   <- the versioned thing we experiment on
data/golden.jsonl     approved reference answers for the regression test set
reports/              exported run comparison, Evidently HTML reports, traces, failure diagnosis, promotion decision
airflow/dags/         agent_regression_dag.py (optional bonus)
tests/                pytest (incl. the live code path against a fake OpenAI endpoint)
pyproject.toml, uv.lock
```

## 0. Quick start

```bash
git clone <repo> && cd <repo>/track_b_agentic
uv sync                                   # one command: exact env from uv.lock
uv run pytest -q                          # 5 tests, no network

# A) offline smoke test (no keys, ~10 s)
uv run python -m agentic_mlops.run_experiment --offline
uv run python -m agentic_mlops.diagnose --offline

# B) real experiment
cp .env.example .env                      # fill HF_TOKEN, JUDGE_API_KEY (+ optional MONID_API_KEY)
uv run python -m agentic_mlops.run_experiment          # runs prompt v1, v2, v3
uv run python -m agentic_mlops.diagnose                # trace-driven failure report
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000   # compare runs / open Traces tab
```

## a. Environment & Reproducibility (uv)

Problems uv solves **for this project specifically**:

* **Three fast-moving, mutually sensitive libraries.** `mlflow` (3.x tracing API, `mlflow.start_span`),
  `evidently` (the 0.7 LLM-judge API: `LLMEval`, `BinaryClassificationPromptTemplate`, new `Report`/`tests`
  — it differs from 0.4.x/legacy) and `openai` (the W15 client targets the Hugging Face router through
  it). Bounds in `pyproject.toml` (`mlflow>=3.1,<4`, `evidently>=0.7.14,<0.8`) plus the committed `uv.lock`
  freeze the full resolved dependency tree (~230 packages) so a grader gets the same APIs we tested.
* **The W15 backend pinned `python >=3.11,<3.13`** and carried a heavy unrelated stack (asyncpg, qdrant,
  sentence-transformers, redis). The experiment harness only needs the agent/LLM/tool code, so this
  project installs just that slice → much smaller, faster environment (no Postgres/Redis/Qdrant needed to
  evaluate a prompt).
* **Airflow is an optional extra** (`uv sync --extra airflow`) because its constraint set is heavy and
  conflict-prone.

Clean-clone path (verified in a fresh copy without `.venv`): `uv sync` → `uv run pytest -q` (5 passed) →
`uv run python -m agentic_mlops.run_experiment --offline` (3 MLflow runs, 3 Evidently reports).

## b. Experiment Tracking Strategy (MLflow)

**What is varied.** Each MLflow run = one *configuration* (`agentic_mlops/config.py → EXPERIMENTS`):

| Param | v1 | v2 | v3 |
|---|---|---|---|
| `prompt_version` (file in `prompts/`) | v1 | v2 | v3 |
| `pass_draft_to_verifier` | False | False | **True** |
| `temperature`, `max_iterations`, `max_verification_rounds`, `model` | 0.2 / 5 / 3 / hf default | same | same |

* **v1** – the unchanged W15 system prompt with W16 agent behaviour (baseline).
* **v2** – v1 + explicit *tool-routing rules* ("ALWAYS call `web_search` for statistics/latest…").
  Motivated by the failure mode **`answered_from_memory`**: the agent answered population / "latest
  developments" questions from parametric knowledge and never called a tool.
* **v3** – v2 + *completeness / stop rules* (one tool call per named entity, never treat a placeholder
  result as evidence) **and** the verifier is now shown the draft answer. Motivated by two failure modes
  that v2's traces still showed: **`verifier_skipped`** (the verifier's decision step reported "no draft
  answer was provided", so the whole verification loop never ran — see bug #2 below) and
  **`incomplete_entity_coverage`** (a three-city question stopped after one `get_current_weather` call).

Each step is a *response to a failure recorded in the previous version's traces*; `agentic_mlops.diagnose`
generates that evidence (`reports/failure_diagnosis*.md`).

**What is measured** (all logged per run; `reports/run_comparison*.md`):

* W16 harness metrics, same definitions: `task_completion_rate`, `tool_call_correctness`,
  `verification_recall`, avg iterations, plus new `strict_completion_rate` (one tool call per entity),
  `entity_coverage_rate`, `error_rate`, `max_iterations_rate`.
* **Cost**: real `prompt/completion/total_tokens` and LLM-call count read from API `usage` (the W16 harness
  *estimated* tokens as `len(text)/4`), tool calls per query, latency.
* **Regression signal** from Evidently: `pct_tests_passed`, `correctness_pass_rate`, `completeness_pass_rate`,
  `suite_gate_passed`.
* **Promotion**: a version is promotable only if the Evidently gate passes; then highest strict completion →
  highest judge pass rate → fewest tokens (`reports/promotion_decision*.json`).

**Traces** (the primary diagnostic). Every query's full trace is logged two ways:

1. as structured JSON — `traces/all_traces.jsonl` (every query) and `traces/representative/{success,failure}_<case>.json`
   per run. One record per step: `{step, iteration, phase, tool, args, result, success, reasoning, latency_s}`
   with phases `tool_call → draft_answer → verification_decision → verification_search → final_answer`,
   and a per-query `termination_reason` (`success_no_verification_needed`, `success_after_verification`,
   `max_iterations_reached`, `error`, …) and `llm_iterations`. `reasoning` is the model's reasoning field or the
   assistant text sent with the tool call (gpt-oss/DeepSeek expose it), and the verifier's own `reason`.
2. as native MLflow traces (spans for each LLM turn and tool call; Traces tab, linked to the run).

Representative traces copied into the repo: `reports/traces*/<version>/`.

### Results

**Offline stub run (plumbing check only — scripted policy, NOT a model):** `reports/run_comparison_OFFLINE.md`

| version | gate | completion | strict | tool_correct | verif_recall | coverage | pct_tests_passed | avg_tokens* |
|---|---|---|---|---|---|---|---|---|
| v1 | FAIL | 0.375 | 0.375 | 0.75 | 0.00 | 0.625 | 81.25 | 1669 |
| v2 | FAIL | 0.375 | 0.375 | 1.00 | 0.00 | 0.75 | 81.25 | 2467 |
| v3 | PASS | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 93.75 | 4025 |

\*offline token counts are `len(json)/4` estimates. The *shape* is what the stub is built to show: v2 fixes
tool choice but not verification; v3 fixes both at ~2.4× the tokens of v1 — i.e. the quality/cost
trade-off you should expect to report.

**Live run — fill in after `run_experiment` (required before submitting):**

| version | gate | completion | strict | verif_recall | pct_tests_passed | avg_tokens | avg_latency_s |
|---|---|---|---|---|---|---|---|
| v1 | | | | | | | |
| v2 | | | | | | | |
| v3 | | | | | | | |

Then write 3–4 sentences: which version won, what it cost (tokens/latency vs. completion), and which
*real* failure modes `diagnose` found (edit the v2/v3 motivations above if your live traces differ — the
rule is that each revision answers a failure you actually saw).

## c. Monitoring & Regression Testing (Evidently AI)

* **Reference vs current.** *Reference* = approved golden answers (`data/golden.jsonl`, one per query; seed
  file hand-written, replace with your best live version via `uv run python -m agentic_mlops.freeze_golden --version v3`
  after reading them). *Current* = the responses the version under test produces for the **same 8 fixed
  queries** (the W16 harness set, with the `current_time`→`current_utc_time` tool-name fix).
* **Test suite** (`agentic_mlops/regression.py`) – Evidently 0.7 `Dataset` + `LLMEval` descriptors +
  `Report` with suite-level tests, two judge checks:
  1. **Correctness** (reference-based, `BinaryClassificationPromptTemplate`): *incorrect* if the response contradicts the
     reference or drops a key fact/number/entity/unit; live values (weather/time) may differ numerically.
  2. **Completeness** (question-based): *incomplete* if any named entity / sub-question is not addressed.
  Each per-row verdict becomes a pass/fail test; suite tests then require **each check's pass rate ≥ 0.8**
  (`CHECK_PASS_RATE_THRESHOLD`). **A failing suite blocks promotion**; it is not just a number that moved.
* **Logged to MLflow:** `pct_tests_passed` (share of case×check tests passed), `pct_cases_passed`,
  per-check pass rates, `suite_gate_passed`, tag `promotion_gate=PASS|FAIL`; artifacts: Evidently HTML report,
  `judge_audit.csv` (response, reference, verdict, judge reasoning), `suite_tests.json`. HTML copies:
  `reports/evidently_regression_<version>.html`.
* **Judge model.** Any OpenAI-compatible endpoint (`JUDGE_BASE_URL`/`JUDGE_MODEL`/`JUDGE_API_KEY`; default HF
  router + the W15 *fallback* model). Use a different model than the agent to limit self-preference bias.
* **Sanity-checking the judge.** `uv run python -m agentic_mlops.judge_audit --version v3` writes
  `reports/judge_audit_v3.csv`; read the responses yourself, fill `human_correct`/`human_complete`, re-run
  → agreement % (logged as `judge_human_agreement`) and every disagreement.
  In the offline run the heuristic judge (not an LLM) mis-scored `78.5398` vs reference `≈78.54` until its
  numeric tolerance was fixed — the same class of error an LLM judge can make, which is why the audit step exists.
* **Interpretation of the offline run** (what to do with live results the same way): v3 passes the gate (93.75 %);
  its one failing case, `population_compare`, fails because **`web_search` is a stub** (W15/W16 returns
  `"Information related to: <query>"`, never real data) — the response correctly has no population figures, so
  the judge flags *lost information vs. reference*. That is a true regression signal about the **tool**, not the
  prompt; wire a real search API (or relax those goldens) before trusting those two cases.
* **Action on a failing gate:** do not promote; read `judge_audit.csv` + the failing trace, decide
  prompt-bug vs. tool-bug vs. judge-bug, and iterate.

## Bugs found in the W15/W16 code while instrumenting it (fixed in this copy)

1. **`web_search` always crashed** – `web_search.py` called `.model_dump()` on a plain `@dataclass`
   (`SearchResult`) → `AttributeError`. The registry only catches validation/arith errors, so the tool call
   raised, and in `_perform_verification` the exception was swallowed ("Verification search failed"), so
   **verification could never actually run**. Fixed with `dataclasses.asdict`.
2. **Verification decision crashed with `response_model=None`** – `VerificationAssistantAgent` calls
   `llm.complete(..., response_model=None)`, but `LLMClient._response_format(None)` dereferences
   `None.__name__` (`AttributeError`, not caught) — every query that reached verification would fail.
   `agentic_mlops/usage_client.py` handles `None` (plain JSON mode). Also,
   the verifier was handed `completion.parsed`, which is always `None` on tool-enabled turns, so it never saw the
   draft answer — now optionally fed the draft (`pass_draft_to_verifier`, enabled in v3).
3. W16 harness expected tool `current_time`; the registered tool is `current_utc_time` (case could never pass).

Both crashes were reproduced (bug 2 against the original client) / covered by `tests/`.

## d. Orchestration (Airflow bonus)

`airflow/dags/agent_regression_dag.py` — DAG `agent_regression_eval`, `@daily`, `catchup=False`.

```
run_eval → check_degradation ─┬─(degraded)→ alert_degradation (task fails → Airflow alert, writes degradation_alert.json)
                              └─(healthy) → mark_healthy
```

* `run_eval` re-runs the promoted prompt version (from `reports/promotion_decision.json`) over the fixed query
  set → new MLflow run (harness metrics + Evidently suite).
* `check_degradation` branches on thresholds: `pct_tests_passed < 80`, `task_completion_rate < 0.75`,
  `error_rate > 0.10`, or token cost/query > +30 % vs. the previous run.
* **Not executed here:** Airflow is not installed in this sandbox; the file is syntax-checked and its logic
  reuses code that is tested. `AGENT_EVAL_OFFLINE=1` runs it without network for a dry run. Setup:
  `uv sync --extra airflow && export AIRFLOW_HOME=$PWD/airflow && airflow standalone`.

## What I could and could not verify

| Verified here | Not verified here |
|---|---|
| `uv sync` from a clean copy; `uv.lock` committed | Live HF-router runs (no token/network) |
| 5 pytest tests, incl. the live client path against a fake OpenAI endpoint (token usage, `response_model=None`, trace shape) | Real LLM-judge verdicts (Evidently `openai` provider path with custom `api_url` is configured, not exercised) |
| Full offline pipeline: 24 MLflow traces, 3 runs, Evidently HTML + suite tests, gate/promotion, diagnose, exports | Airflow DAG execution |
