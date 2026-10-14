# Contributing to Jahan

Thank you for helping build an honest simulator of human society. The most valuable contributions are
**sourced data** and **calibration**, not more mechanisms.

## Ground rules

1. **Every number has a source.** Distributions live in `data/countries/{ISO3}.yaml`. Each block cites an
   id in `data/registry.yaml` with source, URL, year, license and confidence. Never commit an unsourced
   number without marking it `provisional` or `placeholder`.
2. **The core is deterministic.** Code under `src/jahan/{sim,gen,model,store}` must not use wall-clock time,
   unseeded randomness, network access, or iteration order of sets/dicts keyed by strings. All randomness
   goes through `jahan.rng` keyed by stable ids. The determinism tests must pass.
3. **The LLM never decides outcomes.** LLM output may become scenario parameters or text; every state change
   is a typed event produced by the engine.
4. **Heuristics are labelled.** Modelling choices without a source carry a `HEURISTIC` comment and appear in
   report caveats.

## Adding or improving a country

1. Copy `data/countries/USA.yaml` to `data/countries/{ISO3}.yaml` and replace every block.
2. Add registry entries for each `source:` id. Prefer primary sources: UN World Population Prospects
   (age/sex, fertility, mortality), ILOSTAT (employment, occupation by education), Barro-Lee / UNESCO UIS
   (education), World Bank WDI (internet, urbanisation), national censuses (ethnicity, languages, households),
   WHO world reports (vision, hearing), World Values Survey (values), Pew (religion).
3. Add `data/names/{ISO3}.yaml` with frequency-weighted names and their source.
4. Run `uv run jahan validate-data --countries {ISO3}` and the property tests.

## Development

```
uv sync
uv run pytest                 # fast suite
uv run pytest -m "slow or perf"
uv run ruff check src tests scripts && uv run ruff format src tests scripts && uv run mypy
```

Pull requests need green CI on Linux and Windows. Changes to `rng.py` encodings, enum values or schema
columns change the snapshot format and need a `SCHEMA_VERSION` bump and a note in the PR.
