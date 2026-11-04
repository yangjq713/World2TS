---
name: world2ts-offline-discover
version: 1.0.0
description: "Use for World2TS Greenfield offline discovery from local Arrow shards plus strict OfflineMetadata JSON, including readiness, two user review gates, Agent query feedback, canonical events, semantic projection, quality audit, and minimal publication."
metadata:
  requires:
    bins: ["python3"]
---

# World2TS Offline Discover

Read [`../world2ts-shared/SKILL.md`](../world2ts-shared/SKILL.md) first.
The Agent owns semantic decisions; the CLI validates, executes searches, projects
approved rules, audits deterministic constraints, and persists artifacts.

## Non-negotiable contract

- Start from an empty run directory. Do not inherit events, links, selections,
  or judgments from another run.
- Current internal contracts are run schema `3.3`, data-readiness schema `1.1`,
  and curation strategy schema `2.0`. Start a new run for older internal
  artifacts; do not add a compatibility
  mode. Already published three-file datasets remain valid.
- Inputs are Arrow shards plus one strict OfflineMetadata JSON file.
- Offline discovery operates directly on time coverage.
- Tavily and AnySearch are the only Web search sources. Every search window
  also runs an approved local `python-holidays` calendar and retains only
  holidays with deterministic nearby-series movement support. Do not silently
  fall back to GDELT or mechanically invent queries or selections.
- Search requires a user-approved strategy hash. Publication requires a
  user-approved quality-report hash.
- The Agent writes dataset/series semantics, strategy, query intents, event
  selections, canonical events, projection rules, and the content quality report.
- Forecast, concurrent, decision, or retrospective context is allowed when
  accurately labeled. Never rewrite it as a prior cause.
- Public event summaries must be self-contained Agent prose, not copied
  headlines or snippets.
- Public output is exactly `dataset_info.json`,
  `unified_event_series.jsonl`, and `unified_events.jsonl`.

## One-task, two-review workflow

Stay in the same user task throughout. Work autonomously until the strategy
review, pause for the user's explicit decision, continue through search and
curation until the quality review, then pause again. Publish only after explicit
quality approval. A revision decision routes back within the same task.

All Agent-authored files below must use the exact run-local filename shown.

### 1. Prepare strict inputs

```bash
world2ts-core --json offline-discover \
  --phase prepare \
  --run-dir <new-empty-run-dir> \
  --dataset-id <dataset-id> \
  --arrow-file <shard.arrow> \
  --offline-metadata-file <dataset.offline_metadata.json>
```

Repeat `--arrow-file` for multiple shards. Read:

- `data_readiness_report.json`
- `series_semantic_dimensions.jsonl`
- `agent_semantic_briefing.json`
- `run_state.json`

If readiness fails, fix the input facts or start a corrected Greenfield run. Do
not mask ID, frequency, time-range, channel, or empty-series errors.

### 2. Write and validate Agent semantics

Write:

- `agent_dataset_semantics.json`
- `agent_series_semantics.jsonl`

Use only verified facts. Cover every series exactly once, state the verified
frequency explicitly, and preserve the exact start/end range.

```bash
world2ts-core --json offline-discover \
  --phase validate-semantics \
  --run-dir <run-dir>
```

Continue only when the result requests `write_curation_strategy`.

### 3. Strategy and first user review

Read readiness, verified semantics, and available dimensions. Write
`curation_strategy.draft.json` with:

- relevant event-family outlines and why each can add context;
- time period and query budgets;
- Tavily/AnySearch source policy;
- holiday country/subdivision calendars, IANA timezones, and exact target
  series IDs inferred from verified geography;
- permitted context roles and canonical-date preferences;
- approved projection dimensions and dataset-wide policy;
- quality policy, including whether zero-event series are allowed.

Do not write concrete queries in the strategy.

```bash
world2ts-core --json offline-discover \
  --phase validate-strategy \
  --run-dir <run-dir>
```

Read `strategy_review.md` and present its concise scope, budget, projection
policy, risks, and SHA-256 to the user. Ask for `approve`, `revise`, or `stop`.
Never infer approval.

Write `strategy_review_decision.json`:

```json
{
  "decision": "approve",
  "reviewed_strategy_sha256": "<exact displayed hash>",
  "user_instructions": "",
  "reviewed_at": "<ISO-8601 timestamp>"
}
```

Run `validate-strategy` again. Approval freezes
`curation_strategy.approved.json` and creates `time_windows.json`. On revision,
edit the draft, rerun validation, and ask the user again with the new hash.

### 4. Author window priors and queries, then run the feedback loop

Generate the briefing:

```bash
world2ts-core --json offline-discover \
  --phase intents \
  --run-dir <run-dir>
```

Read `agent_query_briefing.json`. Write one record per window to
`agent_window_search_plans.jsonl`; each record must contain concrete,
Agent-known event hypotheses that might affect the series. These are query
priors only, never event evidence.

Write all initial records to `agent_query_intents.jsonl`. Each initial intent
must:

- identify one `window_id` and one approved `outline_id`;
- use `stage="initial"`;
- use the exact window bounds;
- state a concrete query, context hypothesis, and relevance pathway;
- use `generated_by="agent_active_review"`;
- omit `source_names` to use the approved defaults unless an explicit
  Tavily/AnySearch override is needed.

Advance one complete-window feedback step:

```bash
world2ts-core --json offline-discover \
  --phase agent-loop \
  --run-dir <run-dir>
```

Follow `blocked_reason` and `next_agent_action`:

- `write_agent_search_inputs`: write all window plans and initial intents.
- `write_agent_window_search_plans`: the plans file is still missing; write one
  strict record for every window.
- `write_initial_query_intents`: plans exist but the complete approved
  outline-query set is still missing; write the initial intents.
- `write_agent_event_selection`: after every outline query and the structured
  holiday analysis for the current window have completed, review the combined
  candidates, account for every candidate not reviewed in an earlier round,
  and append one strict window decision to
  `agent_event_selections.jsonl`. A `continue_search` decision may retain
  selected candidates while requesting another query.
- `write_window_refined_query`: add one topic-unrestricted
  `stage="window_refinement"` intent. Reference all material prior queries with
  `feedback_query_intent_ids`, optionally reference
  `prior_event_hypothesis_ids`, and include `refinement_reason` plus
  `failed_query_feedback`.
- `retry_agent_loop`: rerun with `--retry-failed` after a transient source
  failure; the CLI retries the failed Agent-authored intents rather than
  inventing a replacement query.
- `write_canonical_events`: all search windows reached an accepted terminal state.

Never pre-generate fallback refinements. Never repeat an equivalent query in the
same window. Read `search_trace.jsonl`, `holiday_impacts.jsonl`,
`discovered_events.jsonl`, and the current window plan before each selection or
refinement. A refinement is deliberately not confined to an outline: combine
all themes, retrieved evidence, rejection feedback, and Agent priors from that
window. Preserve useful weak context with honest confidence; reject noise,
wrong scope, duplicates, or unsupported claims.

### 5. Canonicalize events

Write `canonical_events.jsonl`. The Agent decides merge versus split, canonical
dates/ranges, context role, affected scope, and final summary. Source IDs must
come from this run's selected candidates.

```bash
world2ts-core --json offline-discover \
  --phase validate-canonical-events \
  --run-dir <run-dir>
```

Inspect duplicate hints and any selected candidates intentionally omitted from
the final inventory. Resolve material issues before projection.

### 6. Write semantic projection rules

Write `projection_rules.jsonl` using only strategy-approved dimensions. Hold
ambiguous scope instead of broadcasting it. Dataset-wide rules require explicit
strategy authorization.

```bash
world2ts-core --json offline-discover \
  --phase project \
  --run-dir <run-dir>
```

Inspect `projected_event_series_links.jsonl`, including held events, zero-link
series, and broad rules.

### 7. Audit and second user review

Generate deterministic audit inputs:

```bash
world2ts-core --json offline-discover \
  --phase audit \
  --run-dir <run-dir>
```

Read `quality_audit_briefing.json`. Write `quality_report.json` with the exact
`audit_input_sha256`, unchanged hard gates/statistics, and Agent judgments for
all five content gates. State whether more search is worthwhile and name the
return phase when recommending revision.

The structural hard gate requires valid artifacts, matching dataset/series
references, terminal search windows, at least one trace for each attempted
query, valid canonical events and projections, and compliance with the
zero-event-series policy. Per-source trace completeness and holiday-impact
completeness remain visible in statistics and the five Agent content gates;
they do not independently block the structural gate.

Run `audit` again to validate the report and create `quality_review.md`. Present
that concise packet and hash to the user. Ask for `approve`, `revise_search`,
`revise_events`, `revise_projection`, or `stop`.

Write `quality_review_decision.json`:

```json
{
  "decision": "approve",
  "reviewed_quality_report_sha256": "<exact displayed hash>",
  "user_instructions": "",
  "target_window_ids": [],
  "reviewed_at": "<ISO-8601 timestamp>"
}
```

Run `audit` a third time to apply the decision. A revision must change the
routed inputs and produce a new audit/report/hash before approval.

### 8. Publish

```bash
world2ts-core --json offline-discover \
  --phase publish \
  --run-dir <run-dir>
```

Publication revalidates the quality approval, Agent semantics, canonical event
references, projection rules, exact schemas, and current hashes. It installs
the three public files plus published run state transactionally.

## Operational guidance

- Inspect `run_state.json` rather than guessing phase from files.
- Use `run_history.jsonl` to audit transitions and input hashes.
- Pace real searches one complete window at a time. A slow or failed source is
  not evidence that no event exists.
- Skip unavailable Tavily or AnySearch sources for the current execution, but
  fail loudly when neither approved Web source is available. Retry the same
  Agent intent when all sources that actually ran failed.
- Keep internal evidence rich, but keep public schemas minimal.
- On any schema or reference error, fix the closest Agent artifact. Do not add a
  compatibility alias or fallback.
