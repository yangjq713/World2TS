---
name: world2ts-metadata
description: Prepare strict LOTSA OfflineMetadata directly from registry, HuggingFace, Arrow, and Agent research; validate it against Arrow; optionally render human-readable cards.
---

# World2TS LOTSA Metadata

Use this skill to prepare the strict `OfflineMetadata` JSON required by
`offline-discover`.

The Agent owns semantic research and description. The CLI gathers deterministic
facts and validates contracts. Markdown is optional documentation generated
from validated JSON; it is not a production input.

## Boundary

Metadata describes the dataset:

- exact dataset and series identity;
- frequency and time range;
- target and covariate meanings and units;
- entity, geography, provider, and per-series semantic dimensions;
- optional factual item groups.

Do not put event families, event hypotheses, holiday decisions, source policy,
or search queries in metadata. Those belong to the later curation strategy,
window search plan, and query intents.

`OfflineMetadata` schema `2.0` still contains `query_hints` for compatibility.
For this workflow write:

```json
{"dataset_terms": [], "event_bias": []}
```

## Workflow

### 1. Optionally enrich structural inventory

Inspect one dataset without writing:

```bash
world2ts-core --json metadata hf-fetch --dataset <dataset>
```

Materialize a phase registry enriched with HuggingFace structure:

```bash
world2ts-core --json metadata enrich --phase <1-6>
```

This records structural facts such as example count, first-row frequency,
target dimensionality, covariate dimensionality, and byte size.

Enrichment improves the briefing but is not a prerequisite. Existing local
Arrow data may be prepared even when coarse or enriched phase registries do not
cover it. In that case the briefing marks `registry_context.metadata_status` as
`arrow_only`, and the Agent researches missing semantics from verified sources.
Arrow facts remain authoritative in both cases.

### 2. Prepare an Agent briefing

```bash
world2ts-core --json metadata prepare \
  --dataset <dataset> \
  --arrow-file <shard.arrow>
```

Repeat `--arrow-file` for multiple shards. Success writes:

```text
store/LOTSA/meta_data_info/<DATASET>.metadata_briefing.json
```

Read the full briefing. Its `arrow_facts` are authoritative for base IDs,
runtime series IDs, target-channel expansion, frequencies, exact time ranges,
and per-series profiles.

If `legacy_card` is non-null, it may be read as migration evidence. It is not
authoritative and must not override Arrow facts or independently verified
sources.

### 3. Write strict metadata directly

Research the dataset and write the exact path in `agent_output.path`:

```text
store/LOTSA/meta_data_info/<DATASET>.offline_metadata.json
```

Use `schema_version: "2.0"` and the model in
`world2ts_core/offline_discover/schema.py`.

Required rules:

- `dataset_semantics.dataset_name` equals the briefing `dataset_id`;
- `dataset_semantics.num_series` equals `arrow_facts.base_series_count`;
- `item_schema.item_semantics` covers every `base_series_id` exactly once;
- `item_schema.target` has one entry per target channel;
- frequency and dataset time range exactly match Arrow facts;
- semantic claims come from verified research, never filename-only guesses;
- dimensions use stable factual attributes suitable for later projection;
- unknown optional values remain empty rather than invented;
- `query_hints` remains empty.

For a multichannel Arrow target, item semantics are keyed by base series ID,
not by generated `__target_<n>` runtime IDs.

### 4. Validate against Arrow

```bash
world2ts-core --json metadata validate \
  --offline-metadata-file \
    store/LOTSA/meta_data_info/<DATASET>.offline_metadata.json \
  --arrow-file <shard.arrow>
```

This reuses the same strict readiness implementation as
`offline-discover --phase prepare`. It checks schema, IDs, base-series count,
target-channel expansion, frequency, exact time range, non-empty targets, and
recoverable per-series semantics.

Do not continue until the result is `status: "ready"`.

### 5. Optionally render a card

```bash
world2ts-core --json metadata render-card \
  --offline-metadata-file \
    store/LOTSA/meta_data_info/<DATASET>.offline_metadata.json
```

The resulting `<DATASET>.md` is generated documentation. The JSON remains
authoritative. Use `--force` only when replacing an existing legacy card is
intentional.

### 6. Check coverage

```bash
world2ts-core --json metadata status
world2ts-core --json metadata status --phase <1-6>
```

Status reports structural enrichment, briefing presence, strict JSON presence,
schema validity, and optional card presence. Runtime Arrow alignment is always
confirmed by `metadata validate` and again by offline `prepare`.

## Continue to Offline Discovery

```bash
world2ts-core --json offline-discover \
  --phase prepare \
  --run-dir <new-empty-run-dir> \
  --dataset-id <dataset> \
  --arrow-file <shard.arrow> \
  --offline-metadata-file \
    store/LOTSA/meta_data_info/<DATASET>.offline_metadata.json
```

The later Agent-authored strategy decides event families, holiday scope, and
search direction. Metadata does not pre-commit those decisions.
