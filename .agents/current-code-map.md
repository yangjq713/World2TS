# World2TS Current Code Map

CLI modules are thin boundaries. Domain logic belongs in package modules.

## Unified dataset pipeline

Remote inputs cross one strict acquisition boundary:

```text
Agent collection plan -> collect -> data.arrow + offline_metadata.json
```

The collection command is `world2ts_core/commands/collect.py`; strict plan,
materialization, and provider-contract logic lives in
`world2ts_core/collection.py`. Remote and existing local datasets then enter
the same Greenfield discovery workflow. Collection writes only the strict input
pair.

## Greenfield discovery

Entry point:

- `world2ts_core/commands/offline_discover.py`

Explicit phase orchestration and state transitions:

- `world2ts_core/offline_discover/workflow.py`

Domain modules:

| Module | Responsibility |
|---|---|
| `schema.py` | Strict OfflineMetadata and normalized dimension schema |
| `io.py` | Arrow decoding only |
| `runtime.py` | Strict metadata-to-series projection |
| `readiness.py` | Arrow/metadata facts and Agent semantic validation |
| `strategy.py` | Strategy validation, review packet, hash-bound approval |
| `time_windows.py` | Time coverage windows |
| `query_intents.py` | Agent window priors, strict intents, cross-theme refinement lineage |
| `source_runtime.py` | Registry-backed Tavily/AnySearch execution, cache, concurrency, failures |
| `holiday_calendar.py` | Local geography-approved holiday dates |
| `holiday_impact.py` | Geography-approved holidays and deterministic nearby-series support |
| `search.py` | Window-level feedback state, parallel sources, candidates, provenance trace |
| `canonicalization.py` | Current-run Agent canonical event validation |
| `projection.py` | Approved-dimension rules and stable event-series links |
| `quality_audit.py` | Deterministic hard gates, Agent content report, user gate |
| `publication.py` | Transactional strict three-file publication |
| `models.py` | Current v3 artifact contracts |

There is no mechanical full phase, fallback selection, judgment/finding, or
legacy run-loader path.

## Discovery state and artifacts

`run_state.json` is authoritative. Every transition appends
`run_history.jsonl`.

Main internal artifacts, in workflow order:

```text
data_readiness_report.json
series_semantic_dimensions.jsonl
agent_semantic_briefing.json
agent_dataset_semantics.json
agent_series_semantics.jsonl
semantic_readiness_report.json
curation_strategy.draft.json
strategy_review.md
strategy_review_decision.json
curation_strategy.approved.json
time_windows.json
agent_query_briefing.json
agent_window_search_plans.jsonl
agent_query_intents.jsonl
search_state.json
search_trace.jsonl
holiday_impacts.jsonl
discovered_events.jsonl
agent_event_selections.jsonl
canonical_events.jsonl
projection_rules.jsonl
projected_event_series_links.jsonl
quality_audit_briefing.json
quality_report.json
quality_review.md
quality_review_decision.json
```

Published files:

```text
dataset_info.json
unified_event_series.jsonl
unified_events.jsonl
```

Internal evidence may be rich. The public contract remains minimal.

## Tests

Discovery tests are organized by current contract:

- readiness/runtime/schema
- strategy and first review gate
- time windows and query intents
- search feedback loop and source fan-out
- canonicalization and projection
- quality audit and second review gate
- publication and workflow state
- four frozen golden cases and quality metrics

Collection tests cover strict plans, provider boundaries, transactional output,
LOTSA-compatible metadata, explicit UTC timestamps, and direct offline
readiness.

Run focused tests first, then:

```bash
python -m pytest tests/ -q
python -m ruff check world2ts_core tests
git diff --check
```
