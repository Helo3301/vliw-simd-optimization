# Phase 2: Mapping - Data Dependencies

## Computation Graph

```
ROUND LOOP (16 iterations)
└── BATCH LOOP (256 elements, or 32 vector groups of 8)
    │
    ├── PHASE 1: LOAD
    │   ├── addr_idx = inp_indices_p + i
    │   ├── addr_val = inp_values_p + i
    │   ├── idx = mem[addr_idx]          ← Can parallel with above
    │   ├── val = mem[addr_val]          ← Can parallel with above
    │   ├── addr_node = forest_values_p + idx  ← DEPENDS ON idx
    │   └── node_val = mem[addr_node]    ← BOTTLENECK: 8 different addresses!
    │
    ├── PHASE 2: HASH (6 stages)
    │   └── For each stage:
    │       ├── tmp1 = val op1 const  ─┬─ PARALLEL
    │       ├── tmp2 = val op3 shift  ─┘
    │       └── val = tmp1 op2 tmp2      SEQUENTIAL
    │
    ├── PHASE 3: BRANCH
    │   ├── is_even = (val % 2) == 0
    │   ├── offset = is_even ? 1 : 2
    │   ├── idx_next = 2 * idx + offset
    │   ├── in_bounds = idx_next < n_nodes
    │   └── idx_final = in_bounds ? idx_next : 0
    │
    └── PHASE 4: STORE
        ├── mem[inp_indices_p + i] = idx_final
        └── mem[inp_values_p + i] = val
```

## Critical Path Analysis

```
           ┌─────┐   ┌─────┐   ┌─────────┐   ┌─────┐   ┌────────────┐   ┌──────┐   ┌─────┐
Timeline:  │LOAD │ → │LOAD │ → │LOAD     │ → │XOR  │ → │HASH        │ → │BRANCH│ → │STORE│
           │idx  │   │val  │   │node_val │   │     │   │(6 stages)  │   │      │   │     │
           └─────┘   └─────┘   └─────────┘   └─────┘   └────────────┘   └──────┘   └─────┘
              │         │           │
              └────┬────┘           │
             PARALLEL          DEPENDS ON idx
```

## Dependency Classes

### True Dependencies (RAW - Read After Write)
- `addr_node` depends on `idx`
- `node_val` depends on `addr_node`
- Each hash stage depends on previous stage output
- `idx_next` depends on `idx` and `val` (after hash)
- Store depends on final `idx` and `val`

### Independent Operations (can parallelize)
- Load `idx` and load `val`
- Within hash stage: `tmp1` and `tmp2` computation
- Across different batch elements (until they need to store back)

## The Gather Bottleneck

```
8 elements need 8 different tree nodes:
  indices = [42, 17, 99, 3, 88, 51, 72, 6]  (random)

No vgather, so:
  Cycle 1: load node[0], load node[1]
  Cycle 2: load node[2], load node[3]
  Cycle 3: load node[4], load node[5]
  Cycle 4: load node[6], load node[7]

  = 4 cycles minimum per vector group
```

## Pipelining Insight

Best solution: 1,363 cycles for 512 vector-iterations = **2.6 cycles each**

But gather alone = 4 cycles. How?

**Answer: Multi-stage pipeline across iterations**

```
Cycle:    1    2    3    4    5    6    7    8    9   10   11   12
          ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
Group A:  [  GATHER  ][    HASH    ][STORE]
Group B:       [  GATHER  ][    HASH    ][STORE]
Group C:            [  GATHER  ][    HASH    ][STORE]
Group D:                 [  GATHER  ][    HASH    ][STORE]
```

With enough pipeline stages and scratch space, we can have:
- Group A: storing
- Group B: hashing
- Group C: gathering
- Group D: loading indices/values

This overlaps the 4-cycle gather with other work!

## Resource Utilization per Cycle

| Resource | Available | Baseline Uses | Optimal Target |
|----------|-----------|---------------|----------------|
| ALU | 12 | 1 | 8-12 (scalar scatter prep) |
| VALU | 6 | 0 | 3-4 (hash, branch) |
| Load | 2 | 1 | 2 (gather) |
| Store | 2 | 1 | 1-2 (results) |
| Flow | 1 | 0.1 | 1 (loop control) |

## Key Insights

1. **Gather dominates** - but can be hidden with pipelining
2. **Hash is embarrassingly parallel** - 6 VALU slots can run multiple stages
3. **Memory bandwidth** - 2 loads + 2 stores per cycle is the ceiling
4. **Scratch space** - 1536 words, enough for many pipeline stages

## Minimum Cycle Calculation (aggressive pipelining)

If we can perfectly overlap everything:
- 512 vector-iterations
- Each needs 4 cycles of gather (2 loads/cycle)
- 512 × 4 / overlap_factor

With 4-deep pipeline: 512 × 4 / 4 = 512 cycles + pipeline fill/drain
With 8-deep pipeline: could get even lower

Target zone: **1,000 - 1,500 cycles** seems achievable with heavy pipelining.
