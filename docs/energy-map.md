# QGRE Energy Map

Generated: 2026-04-07

## Energy Map Table

| Module | LOC | Energy | Churn | Imports From | Imported By | Role |
|--------|-----|--------|-------|--------------|-------------|------|
| **trainer** | 3252 | HIGH | 65 | advantages, attention_bonds, checkpoint, logging, segments, types, nemo_* | — | Central orchestrator, training loop |
| **types** | 1745 | HIGH | 34 | — | 8 modules | State machines, dataclasses, CheckpointState |
| **advantages** | 1483 | HIGH | 37 | attention_bonds, segments, spans | trainer | Advantage estimation, ERIC, SPO math |
| **generation** | 643 | MED | 24 | weight_bus, weight_export, weight_load | __main__ | Completion generation, weight sync |
| **config** | 631 | MED | 23 | — | 4 modules | QGREConfig, validation |
| **weight_load** | 601 | MED | 19 | types | generation, weight_bus | WeightLoader lifecycle state machine |
| **data** | 584 | MED | 11 | — | trainer, __main__ | DataLoader, priority sampling |
| **segments** | 565 | MED | — | — | 4 modules | Segmenter interface, HIF parsing |
| **critic** | 564 | MED | 18 | types | trainer | VPRM critic network |
| **checkpoint** | 532 | MED | 23 | types | trainer | Save/load, schema migration |
| **hints** | 493 | LOW | 8 | — | trainer | HintRegistry for EGRS |
| **attention_bonds** | 473 | LOW | 10 | — | advantages, trainer | Bond strength, causal decay |
| **schema** | 391 | LOW | 8 | — | 4 modules | FieldSpec validation, schemas |
| **spans** | 254 | LOW | 14 | — | advantages, trainer | Token-to-span mapping |
| **weight_bus** | — | LOW | 11 | — | generation, trainer | SyncStrategy interface |

## Dependency Diagram

```
                           ┌─────────────────────────────────────────┐
                           │            HIGH ENERGY                  │
                           └─────────────────────────────────────────┘

                                    ┌──────────────┐
                                    │   trainer    │ ◄── 3252 LOC, 65 churn
                                    │  (orchestr.) │     imports 10+ modules
                                    └──────┬───────┘
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              │                            │                            │
              ▼                            ▼                            ▼
     ┌────────────────┐           ┌────────────────┐           ┌────────────────┐
     │   advantages   │           │     types      │           │   checkpoint   │
     │   (SPO/ERIC)   │           │ (state specs)  │           │  (save/load)   │
     │   1483 LOC     │           │   1745 LOC     │           │   532 LOC      │
     └───────┬────────┘           └───────┬────────┘           └───────┬────────┘
             │                            │                            │
             │                            │ ◄── imported by 8 modules  │
             ▼                            │                            │
     ┌───────────────────────┐            │                            │
     │  attention_bonds      │            │                            │
     │  segments, spans      │            ▼                            ▼
     │  LOW energy           │    ┌───────────────┐            ┌───────────────┐
     └───────────────────────┘    │    schema     │            │    config     │
                                  │   391 LOC     │            │   631 LOC     │
                                  └───────────────┘            └───────────────┘


                           ┌─────────────────────────────────────────┐
                           │          WEIGHT SYNC SUBSYSTEM          │
                           └─────────────────────────────────────────┘

                                    ┌──────────────┐
                                    │  generation  │ ◄── entry point
                                    │   643 LOC    │
                                    └──────┬───────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
           ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
           │  weight_load   │     │   weight_bus   │     │ weight_export  │
           │   601 LOC      │     │      LOW       │     │      LOW       │
           │  state machine │     │  SyncStrategy  │     │  LoRA export   │
           └───────┬────────┘     └────────────────┘     └────────────────┘
                   │
                   ▼
           ┌────────────────┐
           │     types      │ ◄── WeightLoaderLifecycle enum
           └────────────────┘


                           ┌─────────────────────────────────────────┐
                           │            DEPENDENCY FLOW              │
                           └─────────────────────────────────────────┘

    ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
    │ trainer │ ──► │advantag.│ ──► │ spans   │     │ schema  │ ◄── leaf
    └────┬────┘     └────┬────┘     └─────────┘     └─────────┘
         │               │
         │               ▼
         │          ┌─────────┐
         │          │segments │ ◄── no internal deps
         │          └─────────┘
         │
         ▼
    ┌─────────┐     ┌─────────┐
    │checkpnt │ ──► │  types  │ ◄── HUB: 8 consumers
    └─────────┘     └─────────┘
                         ▲
                         │
    ┌─────────┐          │
    │wgt_load │ ─────────┘
    └─────────┘
```

## Risk Patterns

| Risk Pattern | Modules | Mitigation |
|--------------|---------|------------|
| **Hub vulnerability** | types (8 consumers) | Changes require atomic rollout |
| **Orchestration cascade** | trainer → 10+ modules | Single failure affects entire pipeline |
| **Numerical hazards** | advantages (37 churn) | NaN/inf guards need exhaustive coverage |
| **State machine race** | weight_load | Threading safety under concurrent load |

## Energy Distribution

- **HIGH (3):** trainer, types, advantages
- **MEDIUM (6):** generation, config, weight_load, data, segments, critic, checkpoint
- **LOW (11+):** hints, attention_bonds, schema, spans, weight_bus, weight_export, logging, lora_*, nemo_extracted/*

## Leverage Points (from hardening)

1. **Schema validation as single path** — Implemented in types.py StateSpec.from_dict methods
2. **WeightLoader lifecycle state machine** — 4-state FSM prevents invalid transitions
3. **Checkpoint boundary validation** — All deserialization goes through schema.py

## Notes

- **Churn** = number of file changes in last 2 weeks (from git log)
- **Energy** = f(LOC, churn, import count, bug fix history)
- Modules with no internal qgre imports are leaf nodes (low coupling risk)
