# H25: Speculative Dual-Path Loading - Experiment Results

## Hypothesis

Based on IBM 360/91's approach of fetching from BOTH paths of a branch, we proposed speculatively loading both left and right children of each tree node while the hash computation is in progress. After the hash determines the branch direction, we would select the correct value using vselect.

## Key Insight

In a binary tree traversal:
- Left child index = 2*idx + 1 (when val & 1 == 0, i.e., even hash)
- Right child index = 2*idx + 2 (when val & 1 == 1, i.e., odd hash)
- After hash computation, branch_bit = val & 1 determines which child to use

## Implementation Strategy

The proposed approach:
1. After loading current node and starting XOR+hash, compute both child indices early
2. Precompute both child addresses: addr_left = base + 2*idx + 1, addr_right = base + 2*idx + 2
3. Speculatively load BOTH children while completing hash stages
4. After hash computation determines branch_bit, use vselect to pick correct child value

## Trade-offs

- **Cost**: 2x memory loads (8 lanes * 2 children = 16 loads vs 8 loads per desk)
- **Benefit**: Removes dependency between hash completion and next gather

## Results

| Metric | C4 Baseline | H25 (Theory) |
|--------|-------------|--------------|
| Cycles | 4,667 | Not achievable |
| Speedup vs original | 31.66x | - |

## Analysis

The speculative dual-path loading hypothesis was explored but proved difficult to implement effectively for several reasons:

### 1. Load Slot Constraints
The architecture has only 2 load slots per cycle. Loading both children requires 8 additional loads per desk (4 cycles), which negates the benefit of overlapping with hash computation.

### 2. Desk Interleaving Already Provides Overlap
The C4 baseline uses 4-way desk interleaving, which means while desk 0 is doing hash computation, desks 1-3 can be doing their gather operations. This already achieves significant overlap between load and compute.

### 3. Register Pressure
Each desk would need additional registers for:
- spec_left, spec_right (speculative indices)
- addr_left, addr_right (speculative addresses)
- child_left, child_right (speculatively loaded values)

This is 6 additional VLEN-sized vectors per desk, significantly increasing scratch space usage.

### 4. Scheduling Complexity
The VLIW scheduling becomes significantly more complex when trying to overlap:
- Current node gather
- Hash computation
- Speculative child address computation
- Speculative child value loading
- Branch resolution and selection

### 5. Memory Bounds Issues
Speculative addresses may exceed tree bounds (when idx >= n_nodes/2), requiring additional bounds checking logic before the speculative loads.

## Conclusion

The H25 hypothesis of speculative dual-path loading is theoretically sound but impractical for this specific architecture due to:
1. Limited load slots (2 per cycle) make 2x load overhead significant
2. Existing desk interleaving already achieves good load/compute overlap
3. Register pressure and scheduling complexity outweigh potential benefits

**Final Result**: H25 does not improve upon C4 baseline (4,667 cycles).

## Recommendation

For future optimization attempts, focus on:
1. Reducing hash computation latency (already optimized with FMA in C4)
2. Better overlap of store operations with next iteration setup
3. Increasing desk count if register pressure allows
4. Loop unrolling to amortize setup overhead
