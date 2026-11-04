---
name: world2ts-shared
version: 1.1.0
description: "World2TS shared rules for strict remote acquisition and the unified Greenfield discovery workflow."
metadata:
  requires:
    bins: ["python3"]
---

# World2TS Shared Rules

World2TS constructs event-driven time-series benchmarks.

## Responsibility boundary

- Agent = semantic brain: inspect evidence, form queries, select candidates,
  reason about scope and timing, write canonical summaries, and judge quality.
- CLI = mechanical hands: load data, execute adapters, validate strict
  contracts, calculate deterministic statistics, and persist artifacts.
- Fail loudly. Do not hide missing inputs, invalid references, or unavailable
  sources behind templates, aliases, or silent fallbacks.
- JSON goes to stdout; diagnostics go to stderr. Put `--json` before the command.

```bash
world2ts-core --json <command> [options]
world2ts-core <command> --help
```

## Unified input flow

Remote and existing local datasets share the same discovery workflow. Remote
series first cross one strict acquisition boundary:

```text
Agent collection plan -> collect -> data.arrow + offline_metadata.json
```

Then use the same time-only workflow for either input origin:

```text
prepare -> Agent semantics -> strategy review -> time-window Agent search
-> canonical events -> semantic projection -> quality review -> publish
```

Collection writes exactly `data.arrow` and `offline_metadata.json`. The Agent
selects the exact source and semantics, and the CLI never switches source or
invents metadata.

Discovery is time-only and Greenfield. Events may be actual, forecast, decision,
concurrent, or retrospective context when honestly labeled. Do not present
non-causal context as a cause.

Skills: `world2ts-collect`, then `world2ts-offline-discover`.

Store root is normally `store/offline_discovery/{dataset}/runs/{run_id}/`.
The authoritative phase is `run_state.json`; transition history is
`run_history.jsonl`. Published output is exactly:

- `dataset_info.json`
- `unified_event_series.jsonl`
- `unified_events.jsonl`

## Source routing

Greenfield discovery uses Tavily plus AnySearch for complementary Web recall and,
for every window, a local `python-holidays` calendar with Agent-approved
country/subdivision, IANA timezone, and exact target series scope. Calendar
events must pass deterministic nearby-series movement support on the real
timestamp axis. It has no automatic GDELT fallback. Unavailable Tavily or
AnySearch sources are skipped for the current execution, but at least one of
the two approved Web sources must be available.

## Data-first work

- Inspect verified series facts and metadata before forming hypotheses.
- Make queries and event summaries specific.
- Cross-check important claims across available evidence.
- Keep provenance and uncertainty explicit.
- Keep rich evidence in internal artifacts and minimal fields in public output.

## Proxy configuration

HTTP source access may require:

```toml
# world2ts.toml
http_proxy = "http://127.0.0.1:7890"
```

This setting does not authorize routing offline discovery to GDELT.

## Skill index

| Skill | Scope |
|---|---|
| `world2ts-collect` | Exact remote collection into Arrow + OfflineMetadata |
| `world2ts-offline-discover` | Greenfield time-only offline workflow with two user gates |
| `world2ts-metadata` | LOTSA preparation, Agent-authored strict metadata, and validation |
