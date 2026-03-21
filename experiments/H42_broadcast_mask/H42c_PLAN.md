# H42c: Surgical Broadcast Integration Plan

## Goal
Integrate broadcast-and-mask INTO H38's pipeline without breaking round fusion or interleaving.

## H38 Structure Analysis

```
Total iterations: 32
Per iteration: 8 desks × 8 elements × 2 rounds = 128 element-rounds

Iteration mapping:
  Iter 1-4:  All 256 elements, rounds 0-1  ← ROUND 0 HERE
  Iter 5-8:  All 256 elements, rounds 2-3
  Iter 9-12: All 256 elements, rounds 4-5
  ...
  Iter 29-32: All 256 elements, rounds 14-15
```

## Where Broadcast Helps

| Iterations | Rounds | Round 0 Status | Opportunity |
|------------|--------|----------------|-------------|
| 1-4 | 0-1 | Round 0 active | **BROADCAST for R0, 2-load for R1** |
| 5-8 | 2-3 | Past round 0 | Normal gather |
| 9-32 | 4-15 | Past round 0 | Normal gather |

## The Surgical Change

### Current H38 Flow (per iteration with round fusion):
```
1. Compute offsets (1 cycle)
2. Compute addresses (2 cycles)
3. Load idx/val for 8 desks (8 cycles)
4. Prepare gather addresses (4 cycles)
5. ROUND 1: Interleaved gather + hash for 8 desks (~36 cycles)
6. Prepare gather addresses for round 2 (4 cycles)
7. ROUND 2: Interleaved gather + hash for 8 desks (~36 cycles)
8. Store all desks (8 cycles)
9. Loop control (3 cycles)
```

### Proposed H42c Flow:

**Iterations 1-4 (rounds 0-1): MODIFIED**
```
1. Compute offsets (1 cycle)
2. Compute addresses (2 cycles)
3. Load idx/val for 8 desks (8 cycles)
4. BROADCAST: Copy tree[0] to all 8 desks' node_val (2 cycles) ← CHANGE
5. ROUND 0: Hash only, no gather (interleaved structure, ~20 cycles)
6. Prepare gather addresses for round 1 (4 cycles)
7. ROUND 1: Normal interleaved gather + hash (~36 cycles)
8. Store all desks (8 cycles)
9. Loop control (3 cycles)
```

**Iterations 5-32 (rounds 2-15): UNCHANGED**
```
[Exact H38 code]
```

## Cycle Savings Estimate

### Per round-0-iteration:
- Current: 4 cycles gather × 8 desks = 32 cycles
- Proposed: 2 cycles broadcast to all desks = 2 cycles
- Savings: **30 cycles per iteration**

### Total for iterations 1-4:
- 4 iterations × 30 cycles = **120 cycles saved**

### Overhead:
- Need to detect iteration 1-4 vs 5-32: ~2-3 cycles
- Structural duplication for two paths: code size increase

### Net expected:
- H38: 4,062 cycles
- H42c target: ~3,950 cycles (if clean integration)
- Improvement: ~2.8%

## Implementation Options

### Option A: Two Separate Loops (Cleanest)
```
Loop 1: Iterations 1-4 (rounds 0-1, broadcast for R0)
Loop 2: Iterations 5-32 (rounds 2-15, normal H38)
```
- Pro: No conditionals inside hot loop
- Con: Code duplication, loop overhead between phases

### Option B: Conditional Inside Loop
```
Main loop:
  if iter < 4:
    broadcast path for round 0
    normal path for round 1
  else:
    normal H38 path
```
- Pro: Single loop structure
- Con: Conditional overhead every iteration

### Option C: Fully Unrolled First 4 Iterations (Most Aggressive)
```
Unrolled: Iter 1-4 with broadcast R0
Loop: Iter 5-32 with normal H38
```
- Pro: No loop overhead for first phase, no conditionals
- Con: Largest code size, most complex

## Recommendation: Option A

**Rationale:**
1. Two loops keeps each loop's hot path simple
2. The transition between loops is just 2-3 instructions
3. Each loop can be independently optimized
4. Matches the structure of H42b but WITH round fusion in loop 2

## Key Integration Points

### 1. Tree[0] Preload (before any loop)
```python
# Load tree[0] once
self.add("load", ("load", root_node_val, self.scratch["forest_values_p"]))
self.add("valu", ("vbroadcast", v_root_node, root_node_val))
```

### 2. Round 0 Path (inside first loop, replace gather with broadcast copy)
```python
# Instead of 8 × 4-cycle gathers:
self.instrs.append({
    "valu": [
        ("+", desks[0]['node_val'], v_root_node, v_zero),  # copy
        ("+", desks[1]['node_val'], v_root_node, v_zero),
        ("+", desks[2]['node_val'], v_root_node, v_zero),
        ("+", desks[3]['node_val'], v_root_node, v_zero),
        ("+", desks[4]['node_val'], v_root_node, v_zero),
        ("+", desks[5]['node_val'], v_root_node, v_zero),
    ],
})
self.instrs.append({
    "valu": [
        ("+", desks[6]['node_val'], v_root_node, v_zero),
        ("+", desks[7]['node_val'], v_root_node, v_zero),
    ],
})
# Total: 2 cycles instead of 32 cycles
```

### 3. Round 1 Path (immediately after R0 hash in same loop)
```python
# Normal H38 gather + hash interleaving
# Elements are now at indices 1 or 2
# Can't use broadcast (2 unique values per 8 lanes)
```

### 4. Transition to Main Loop
```python
# After iterations 1-4 complete:
self.add("load", ("const", iter_counter, 4))  # Start at iteration 5
# Fall through to normal H38 loop for iterations 5-32
```

## Data Flow Diagram

```
         ┌─────────────────────────────────────┐
         │  PROLOGUE: Load tree[0], broadcast  │
         └──────────────┬──────────────────────┘
                        │
         ┌──────────────▼──────────────────────┐
         │  LOOP 1: Iterations 1-4             │
         │  ┌─────────────────────────────────┐│
         │  │ Round 0: BROADCAST (2 cycles)   ││
         │  │ Hash R0: Interleaved (~20 cyc)  ││
         │  │ Round 1: Normal gather (32 cyc) ││
         │  │ Hash R1: Interleaved (~20 cyc)  ││
         │  │ Store (8 cycles)                ││
         │  └─────────────────────────────────┘│
         │  (4 iterations)                     │
         └──────────────┬──────────────────────┘
                        │
         ┌──────────────▼──────────────────────┐
         │  LOOP 2: Iterations 5-32            │
         │  ┌─────────────────────────────────┐│
         │  │ Round N: Normal gather (32 cyc) ││
         │  │ Hash RN: Interleaved (~20 cyc)  ││
         │  │ Round N+1: Normal gather        ││
         │  │ Hash RN+1: Interleaved          ││
         │  │ Store (8 cycles)                ││
         │  └─────────────────────────────────┘│
         │  (28 iterations - exact H38 code)   │
         └─────────────────────────────────────┘
```

## Success Criteria

| Metric | H38 | H42c Target | Pass? |
|--------|-----|-------------|-------|
| Cycles | 4,062 | < 4,000 | Improvement |
| Correctness | ✓ | ✓ | Must pass |
| vs Baseline | 36.4x | > 36.9x | Speedup |

## Next Steps

1. [ ] Copy H38 as base
2. [ ] Extract iteration 1-4 logic into Loop 1
3. [ ] Replace Round 0 gather with broadcast in Loop 1
4. [ ] Keep Loop 2 as exact H38 code for iterations 5-32
5. [ ] Test correctness
6. [ ] Measure cycles
7. [ ] Compare to H38
