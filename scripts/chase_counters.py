#!/usr/bin/env python3
"""Hardware counters per run of the bulge chase's memory pattern (scripts/kernels/chase_trace.c), from an Instruments
trace recorded with build/memory-counters.tracetemplate (CPU Counters, events INST_ALL, L1D_CACHE_MISS_LD,
L1D_CACHE_MISS_ST, L1D_CACHE_WRITEBACK, MAP_STALL, L1D_TLB_MISS, L2_TLB_MISS_DATA).

  python3 scripts/chase_counters.py TRACE PROCESS_NAME

Exports the counters-profile and os-signpost tables (xctrace export), keeps the samples of PROCESS_NAME, assigns them
to the "pattern" signpost intervals by time, and prints per interval: duration, then each counter summed over the
interval's samples (the per-sample values are differences since the thread's previous sample), per run and per state
(runs slower or faster than the midpoint between the fastest and slowest run).
"""
import os, re, subprocess, sys, tempfile
import xml.etree.ElementTree as ET

EVENTS = ('INST_ALL', 'L1D_CACHE_MISS_LD', 'L1D_CACHE_MISS_ST', 'L1D_CACHE_WRITEBACK', 'MAP_STALL', 'L1D_TLB_MISS',
          'L2_TLB_MISS_DATA')


def export(trace, schema):
    out = tempfile.NamedTemporaryFile(suffix='.xml', delete=False).name
    xp = f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'
    with open(out, 'w') as f:
        subprocess.run(['xcrun', 'xctrace', 'export', '--input', trace, '--xpath', xp], stdout=f, check=True)
    return out


def rows(path):
    """Rows as lists of (tag, text, fmt), with id/ref references resolved."""
    seen = {}
    for _, row in ET.iterparse(path):
        if row.tag != 'row':
            continue
        cells = []
        for c in row:
            if 'ref' in c.attrib:
                cells.append(seen[c.attrib['ref']])
                continue
            cell = (c.tag, c.text, c.attrib.get('fmt', ''), [(g.tag, g.text, g.attrib.get('fmt', '')) for g in c])
            if 'id' in c.attrib:
                seen[c.attrib['id']] = cell
            for g in c.iter():
                if 'id' in g.attrib and g is not c:
                    seen[g.attrib['id']] = (g.tag, g.text, g.attrib.get('fmt', ''), [])
            cells.append(cell)
        yield cells
        row.clear()


def main():
    trace, name = sys.argv[1], sys.argv[2]
    # Signpost intervals: begin/end events of "pattern", matched by identifier.
    begins, intervals = {}, []
    sp = export(trace, 'os-signpost')
    for cells in rows(sp):
        d = {c[0]: c for c in cells}
        if name not in d.get('process', ('', '', '', []))[2]:
            continue
        t = int(d['event-time'][1]); kind = d['event-type'][2]; ident = d['os-signpost-identifier'][1] \
            if 'os-signpost-identifier' in d else d.get('uint64', ('', '', '', []))[1]
        msg = d.get('message', ('', '', '', []))[2]
        if kind == 'Begin':
            begins[ident] = (t, msg)
        elif kind == 'End' and ident in begins:
            t0, m = begins.pop(ident); intervals.append((t0, t, m))
    os.unlink(sp)
    intervals.sort()
    # Counter samples of the process.
    cp = export(trace, 'counters-profile')
    samples = []
    for cells in rows(cp):
        d = {c[0]: c for c in cells}
        if name not in d.get('process', ('', '', '', []))[2] or 'pmc-events' not in d:
            continue
        vals = [int(v.replace(',', '')) for v in re.findall(r'\(([\d,]+)\)', d['pmc-events'][2])][:len(EVENTS)]
        samples.append((int(d['sample-time'][1]), vals))
    os.unlink(cp)
    samples.sort()
    print(f"{len(intervals)} intervals, {len(samples)} samples of {name}")
    if len(samples) > 3:
        mono = all(samples[i + 1][1][0] >= samples[i][1][0] for i in range(len(samples) - 1))
        print(f"INST_ALL non-decreasing across samples (cumulative?): {mono}")
    # Sum per interval.
    res, j = [], 0
    for t0, t1, m in intervals:
        tot = [0] * len(EVENTS); cnt = 0
        while j < len(samples) and samples[j][0] < t0:
            j += 1
        k = j
        while k < len(samples) and samples[k][0] <= t1:
            tot = [a + b for a, b in zip(tot, samples[k][1])]; cnt += 1; k += 1
        res.append(((t1 - t0) * 1e-6, m, cnt, tot))
    durations = sorted(r[0] for r in res)
    mid = (durations[0] + durations[-1]) / 2
    print(f"durations {durations[0]:.1f}-{durations[-1]:.1f} ms; slow = above {mid:.1f} ms")
    hdr = ' '.join(f"{e[:14]:>14s}" for e in EVENTS)
    print(f"{'run':16s} {'ms':>6s} {'n':>4s} {hdr}")
    for state, keep in (('fast', lambda r: r[0] <= mid), ('slow', lambda r: r[0] > mid)):
        sel = [r for r in res if keep(r)]
        if not sel:
            continue
        avg = [sum(r[3][i] for r in sel) / len(sel) for i in range(len(EVENTS))]
        print(f"{state + ' mean':16s} {sum(r[0] for r in sel) / len(sel):6.1f} {sum(r[2] for r in sel) / len(sel):4.0f} "
              + ' '.join(f"{v:14.3g}" for v in avg) + f"   ({len(sel)} runs)")
    if '--runs' in sys.argv:
        for r in res:
            print(f"{r[1]:16s} {r[0]:6.1f} {r[2]:4d} " + ' '.join(f"{v:14.3g}" for v in r[3]))


if __name__ == '__main__':
    main()
