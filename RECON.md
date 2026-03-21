# Phase 1: Recon - Complete ISA Analysis

## Machine Constants

| Constant | Value | Notes |
|----------|-------|-------|
| VLEN | 8 | Vector width - 8 elements per vector op |
| N_CORES | 1 | Single core (multicore disabled) |
| SCRATCH_SIZE | 1536 | Words of scratch space (registers) |

## Slot Limits (per cycle)

| Engine | Slots | Purpose |
|--------|-------|---------|
| alu | 12 | Scalar arithmetic |
| valu | 6 | Vector arithmetic (8-wide) |
| load | 2 | Memory → Scratch |
| store | 2 | Scratch → Memory |
| flow | 1 | Control flow, select |
| debug | 64 | Debugging (ignored in submission) |

## Execution Model

- **VLIW**: All engines execute in parallel each cycle
- **Write-after-read**: Effects apply at END of cycle, so you can read values before they're overwritten in same cycle
- **Cycle counting**: Only cycles with non-debug ops count

## ALU Instructions

Format: `(op, dest, a1, a2)` → `scratch[dest] = scratch[a1] op scratch[a2]`

| Op | Description |
|----|-------------|
| + | Addition |
| - | Subtraction |
| * | Multiplication |
| // | Floor division |
| cdiv | Ceiling division: (a + b - 1) // b |
| ^ | XOR |
| & | AND |
| \| | OR |
| << | Left shift |
| >> | Right shift |
| % | Modulo |
| < | Less than → 0 or 1 |
| == | Equality → 0 or 1 |

All results mod 2^32.

## VALU Instructions

All operate on 8 contiguous scratch locations.

| Instruction | Effect |
|-------------|--------|
| `("vbroadcast", dest, src)` | Replicate scalar to vector |
| `("multiply_add", dest, a, b, c)` | dest[i] = a[i] * b[i] + c[i] (FMA) |
| `(op, dest, a1, a2)` | Apply any ALU op element-wise |

## LOAD Instructions

| Instruction | Effect |
|-------------|--------|
| `("load", dest, addr)` | scratch[dest] = mem[scratch[addr]] |
| `("load_offset", dest, addr, offset)` | scratch[dest+offset] = mem[scratch[addr+offset]] |
| `("vload", dest, addr)` | Load 8 contiguous words (addr is scalar) |
| `("const", dest, val)` | Load immediate value |

**Critical**: No gather instruction! For non-contiguous loads, must use multiple scalar loads.

## STORE Instructions

| Instruction | Effect |
|-------------|--------|
| `("store", addr, src)` | mem[scratch[addr]] = scratch[src] |
| `("vstore", addr, src)` | Store 8 contiguous words (addr is scalar) |

## FLOW Instructions

| Instruction | Effect |
|-------------|--------|
| `("select", dest, cond, a, b)` | dest = a if cond else b (scalar) |
| `("vselect", dest, cond, a, b)` | Per-element select (vector) |
| `("add_imm", dest, a, imm)` | dest = a + immediate |
| `("jump", addr)` | Unconditional jump |
| `("cond_jump", cond, addr)` | Jump if scratch[cond] != 0 |
| `("cond_jump_rel", cond, offset)` | Relative conditional jump |
| `("jump_indirect", addr)` | Jump to scratch[addr] |
| `("halt",)` | Stop execution |

## Hash Function

6 stages, each: `a = (a op1 const) op2 (a op3 shift)`

| Stage | op1 | const | op2 | op3 | shift |
|-------|-----|-------|-----|-----|-------|
| 0 | + | 0x7ED55D16 | + | << | 12 |
| 1 | ^ | 0xC761C23C | ^ | >> | 19 |
| 2 | + | 0x165667B1 | + | << | 5 |
| 3 | + | 0xD3A2646C | ^ | << | 9 |
| 4 | + | 0xFD7046C5 | + | << | 3 |
| 5 | ^ | 0xB55A4F09 | ^ | >> | 16 |

## Test Parameters

- forest_height = 10 → n_nodes = 2047
- rounds = 16
- batch_size = 256

## Key Bottlenecks Identified

1. **Gather (tree lookup)**: Each batch element has different tree index. No vgather, so 8 scalar loads = 4 cycles minimum per vector group.

2. **Slot underutilization**: Baseline uses 1 op per cycle. Could use 12 ALU + 6 VALU + 2 load + 2 store simultaneously.

3. **No loops**: Fully unrolled, can't overlap iterations.

## Parallelism Budget per Cycle

```
Theoretical max ops per cycle:
- 12 scalar ALU ops
- 6 vector ALU ops (= 48 scalar equivalent)
- 2 loads
- 2 stores
- 1 flow op
```
