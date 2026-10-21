#!/usr/bin/env python3
"""Stage table from eigen-bench output with --opts variants: per matrix, per variant, total and stage times (ms).

  python3 scripts/opt_table.py build/opt-small-fp64.txt [b]
"""
import re, sys
txt = open(sys.argv[1]).read().splitlines()
want = sys.argv[2] if len(sys.argv) > 2 else None
n = None
for l in txt:
    m = re.search(r'n = (\d+), (FP\d+)', l)
    if m:
        n = m.group(1); print(f"== n = {n} {m.group(2)}"); continue
    if 'total' not in l: continue
    name = l.split('total')[0].strip()
    if want and 'b=' in name and not name.endswith('b=' + want): continue
    tot = float(re.findall(r'total\s+([\d.]+) ms', l)[0])
    stages = re.findall(r'(\S+) ([\d.]+)', l.split('|')[1])
    print(f"  {name:20} {tot:8.2f} | " + ' '.join(f"{k} {float(v):.2f}" for k, v in stages))
