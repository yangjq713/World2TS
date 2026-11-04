# World2TS Collaboration Rules

## What this is

World2TS constructs event-driven time-series benchmarks through one strict
Arrow + OfflineMetadata boundary and one Greenfield discovery workflow.
World2TS-core is the repository's CLI and Python distribution.

**Agent = semantic brain. CLI = mechanical hands.**

Read `.agents/skills/world2ts-shared/SKILL.md` for the workflow entry point
and `.agents/current-code-map.md` for code ownership. The checked-in Agent
skills are the authoritative task procedures.

## Canonical architecture

- **8 series sources:** yahoo, fred, noaa, noaa_isd, owid, coingecko, eia,
  polymarket.
- **2 Web-search sources:** tavily and anysearch.
- **Local holiday analysis:** geography-approved calendars filtered by real
  series movement; it is not a registry source or a standalone CLI.
- **12 leaf CLI commands:** collect, offline-discover, config show/set, sources
  list, and seven metadata commands.

Remote collection writes only `data.arrow` and `offline_metadata.json`.
Existing local Arrow data enters through the same strict metadata contract.
Discovery then runs explicit phases with two user review gates and publishes
the minimal three-file dataset.

## Working rules

1. Keep Agent judgment explicit: the CLI validates and persists decisions but
   must not invent semantics, queries, selections, canonical events, or links.
2. Preserve honest time roles. Actual, forecast, decision, concurrent, and
   retrospective context are valid when labeled accurately; do not claim
   causality that the evidence does not establish.
3. Use specific, self-contained event descriptions and cross-check evidence
   before canonicalization.
4. Fail loudly. Do not introduce silent source, schema, or compatibility
   fallbacks.
5. Emit JSON to stdout and diagnostics to stderr; every command must support
   `--help`.

## Verification

Start with focused tests, then run:

```bash
python -m pytest tests/ -q
python -m ruff check world2ts_core tests
git diff --check
```

The test count is intentionally not documented; pytest output is authoritative.
