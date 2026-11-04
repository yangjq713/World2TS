---
name: world2ts-collect
version: 1.2.0
description: "Fetch Agent-selected online time series and materialize the canonical Arrow plus strict OfflineMetadata inputs for offline discovery."
metadata:
  requires:
    bins: ["python3"]
  cliHelp: "world2ts-core collect --help"
---

# World2TS Remote Collection

Read [`../world2ts-shared/SKILL.md`](../world2ts-shared/SKILL.md) first.

This skill covers only the remote-input boundary:

```text
Agent collection plan -> exact source fetch -> data.arrow + offline_metadata.json
-> offline-discover prepare
```

The Agent owns source selection, symbols, scope, time interpretation, target
semantics, metadata dimensions, and query hints. The CLI executes the exact
plan, validates source output, and writes the two standard local inputs. It
does not choose another source, infer semantic defaults, or fill observations.

## 1. Inspect capabilities

```bash
world2ts-core --json sources list
world2ts-core --json sources list --domain finance
```

Treat capability output as the executable collection contract. Each series
source reports `primary_target_fields`, `target_option`, allowed
`adapter_options`, `metadata_dimensions`, `timestamp_semantics`, and the
half-open `range_semantics`. The plan's `domain` is the dataset's semantic
category; it does not mechanically route or replace the explicit `source`.

Collection currently accepts fixed nominal frequencies only:

- `15min`
- `1h`
- `1d`
- `1w`

Do not request `1mo` or `1q` until calendar-frequency readiness is supported.

## 2. Write the strict plan

Write a JSON file outside the new output directory. Every shown field is
required, including deliberate empty `{}` / `[]` values. Unknown fields,
duplicate keys, duplicate symbols, non-canonical frequencies, timezone-aware
`start` or `end` values, and invalid IANA timezones are rejected.

```json
{
  "schema_version": "1.0",
  "dataset_id": "us_semiconductor_equities_daily",
  "source": "yahoo",
  "domain": "semiconductors",
  "symbols": ["NVDA", "AMD", "INTC"],
  "start": "2024-01-01",
  "end": "2025-01-01",
  "frequency": "1d",
  "timezone": "America/New_York",
  "adapter_options": {},
  "metadata": {
    "subcategory": "public semiconductor company equity prices",
    "geo": "United States",
    "description_short": "Daily closing prices for selected US-listed semiconductor companies.",
    "description_long": "Daily primary closing-price series collected from Yahoo Finance with original trading-day timestamps and company metadata. Trading-session gaps are preserved, and only the primary close target is materialized.",
    "entity_type": "public_company",
    "spatial_dimension": "company_location",
    "target": {
      "source_field": "close",
      "name": "close",
      "description": "Unadjusted closing market price for the trading session.",
      "unit": "USD"
    },
    "dimensions": ["sector", "industry", "exchange", "currency"],
    "query_hints": {
      "dataset_terms": ["semiconductor equities", "chip industry"],
      "event_bias": ["earnings", "export controls", "product launches"]
    }
  }
}
```

### Plan decisions

- `dataset_id`: stable offline dataset identity; letters, digits, `.`, `_`, and
  `-` only.
- `source`: exact registered adapter. There is no source fallback.
- `domain`: Agent-selected semantic category passed through to metadata.
- `symbols`: expected Arrow `item_id` values in physical order. Every adapter
  result must match these IDs exactly once.
- `start` / `end`: naive date or datetime interpreted in `timezone`; the range
  is half-open `[start, end)`, so adjacent collection windows never duplicate
  their boundary observation.
- `frequency`: nominal series frequency. Real timestamps may contain market
  closures or missing observations and are never regularized.
- `adapter_options`: source-specific constructor choices. Frequency-owned
  options such as Yahoo `interval` cannot contradict `frequency`. Keep API
  credentials in config or environment variables.
- `metadata.target.source_field`: exact primary value field advertised by
  `sources list`; it binds the Agent's semantic name, description, and unit to
  the values actually returned by the adapter. Auxiliary adapter `columns` are
  intentionally not materialized in this first contract.
- `metadata.dimensions`: exact adapter metadata keys to promote per item. If
  any selected key is absent or empty for any symbol, collection fails. Do not
  request a dimension unless the source evidence supports it consistently.
- `metadata.query_hints`: Agent-authored retrieval guidance; the CLI does not
  invent it. Put honest source limitations and materialization scope directly
  in `metadata.description_long`, matching the original LOTSA metadata schema.

Use the exact `adapter_options`, primary target fields, metadata dimensions,
timestamp semantics, and range semantics reported by `sources list`. Semantic
options are mandatory when advertised: common examples are NOAA
`{"datatype":"TMAX"}`, NOAA ISD `{"field":"TMP"}`, and OWID
`{"metric":"new_cases"}`. Keep FRED/EIA credentials in normal config.
CoinGecko, OWID, and Polymarket plans must use `"timezone":"UTC"` because
those sources emit UTC observations or UTC civil dates.
Run a small representative collection before scaling a source whose absolute
range or target semantics have not been independently verified.

## 3. Materialize

The destination must not exist.

```bash
world2ts-core --json collect \
  --plan collection_plan.json \
  --output-dir collected/us_semiconductor_equities_daily
```

Success writes exactly:

```text
collected/us_semiconductor_equities_daily/
  data.arrow
  offline_metadata.json
```

`data.arrow` contains one row per symbol with:

- `item_id`
- `start`
- `freq`
- `target`
- `timestamps`

All stored timestamps are UTC. For source timestamps without timezone
information, the plan timezone is authoritative. Original observation gaps are
preserved.

`offline_metadata.json` uses the original LOTSA wire field names. In
particular, per-item semantic dimensions are serialized under `attributes`;
the offline runtime normalizes them to its internal `dimensions` model.

Collection is transactional. Missing or extra symbols, empty data, duplicate
or unordered timestamps, length mismatches, out-of-range observations,
frequency mismatches, infinite/all-missing targets, invalid promoted
dimensions, invalid metadata, or failed offline readiness leave no output
directory.

## 4. Hand off to offline discovery

Use the standard pair directly; do not convert it again.

```bash
world2ts-core --json offline-discover \
  --phase prepare \
  --run-dir store/offline_discovery/us_semiconductor_equities_daily/runs/run-001 \
  --dataset-id us_semiconductor_equities_daily \
  --arrow-file collected/us_semiconductor_equities_daily/data.arrow \
  --offline-metadata-file collected/us_semiconductor_equities_daily/offline_metadata.json
```

Continue with `world2ts-offline-discover`.
