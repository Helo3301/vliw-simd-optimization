import sys, os, itertools, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

base_file = 'experiments/BF200/theory_1b_gs4_WIN.py'
with open(base_file) as f:
    base_code = f.read()

results = []
for perm in itertools.permutations(range(4)):
    perm_str = str(list(perm))
    modified = base_code.replace(
        'gd = [group_desks[i] for i in range(0, len(group_desks), 2)] + [group_desks[i] for i in range(1, len(group_desks), 2)]',
        f'perm = {perm_str}; gd = [group_desks[i] for i in perm]'
    )
    
    tmpf = f'experiments/BF300/_tmp_perm.py'
    with open(tmpf, 'w') as f:
        f.write(modified)
    
    result = subprocess.run(
        ['/usr/bin/python3.11', tmpf, '--check'],
        capture_output=True, text=True, timeout=120
    )
    output = result.stdout + result.stderr
    
    cycles = None
    passed = 'PASSED' in output
    for line in output.split('\n'):
        if 'CYCLES:' in line:
            try:
                cycles = int(line.split('CYCLES:')[1].strip())
            except:
                pass
    
    results.append((list(perm), cycles, passed))
    if os.path.exists(tmpf):
        os.remove(tmpf)

results.sort(key=lambda x: x[1] if x[1] is not None else 99999)
print("\n=== HASH DESK ORDERING RESULTS ===")
print(f"{'Ordering':<16} {'Cycles':>8} {'Status'}")
print("-" * 35)
for perm, cycles, passed in results:
    status = "PASS" if passed else "FAIL"
    marker = ""
    if perm == [0,2,1,3]:
        marker = " <-- CURRENT"
    elif cycles is not None and cycles == results[0][1]:
        marker = " <-- BEST"
    c_str = str(cycles) if cycles is not None else "N/A"
    print(f"{str(perm):<16} {c_str:>8} {status}{marker}")
