import sys, os, itertools, subprocess

base_file = 'experiments/BF300/theory_hashorder_WIN.py'
with open(base_file) as f:
    base_code = f.read()

# Try: each group gets a different hash ordering from the top-4 best
# Best orders: [2,0,1,3], [2,0,3,1], [3,0,1,2], [3,0,2,1]
orderings = [
    [[2,0,1,3], [3,0,2,1], [2,0,3,1], [3,0,1,2]],  # rotating top-4
    [[2,0,1,3], [2,0,1,3], [3,0,2,1], [3,0,2,1]],  # alternating pairs
    [[3,0,2,1], [2,0,1,3], [3,0,1,2], [2,0,3,1]],  # reverse rotate
    [[2,0,1,3], [3,0,1,2], [2,0,1,3], [3,0,1,2]],  # alternating 2
]

# To implement per-group ordering, we need to pass the group index
# Replace the hash function to accept a group parameter
new_hash_func = """
        group_hash_orders = GROUP_HASH_ORDERS
        def emit_hash_interleaved(group_desks, group_idx=0):
            order = group_hash_orders[group_idx % len(group_hash_orders)]
            gd = [group_desks[i] for i in order]
"""

for idx, order_set in enumerate(orderings):
    modified = base_code.replace(
        '''        def emit_hash_interleaved(group_desks):
            # BF300: Desk ordering [2,0,1,3] = non-zero desk first, then desk 0
            # Saves 8 cycles vs previous [0,2,1,3] ordering
            gd = [group_desks[2], group_desks[0], group_desks[1], group_desks[3]]''',
        new_hash_func.replace('GROUP_HASH_ORDERS', str(order_set))
    )
    # Also need to pass group_idx to emit_hash_interleaved calls
    # This is complex - let's use a different approach: set a class variable
    # Actually simpler: use a counter
    modified = modified.replace(
        '''        def emit_hash_interleaved(group_desks):
            # BF300: Desk ordering [2,0,1,3] = non-zero desk first, then desk 0
            # Saves 8 cycles vs previous [0,2,1,3] ordering
            gd = [group_desks[2], group_desks[0], group_desks[1], group_desks[3]]''',
        new_hash_func.replace('GROUP_HASH_ORDERS', str(order_set))
    )
    # Actually this won't work well because emit_hash_interleaved is called many times
    # without group context. Skip this approach.
    break

# Simpler: just test single orderings we haven't tried yet with R10 dead code removal
# We already found [2,0,1,3] is best at 1,469. Let's combine with R10 removal variations.

# Try adding R10 dead code removal AND removing unused inits
# We know: hash order alone = 1,469, R10+hash = 1,469 (same), R10+unused = 1,479 (worse)
# So the bottleneck is purely scheduling, not VALU count.

print("Per-group hash ordering requires deeper code changes. Skipping.")
print("Focus on other scheduling improvements instead.")
