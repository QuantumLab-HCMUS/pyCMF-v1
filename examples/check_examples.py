#!/usr/bin/env python
'''
Test run: execute the example scripts and compare every number they print
with the reference output in ``reference_output/``.

    python check_examples.py            # stable examples (a few minutes)
    python check_examples.py --all      # also the experimental ones
'''

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STABLE = ['01-obmp2', '02-o2bmp2', '03-excited_state', '04-obdf', '05-obdh', '06-properties']
EXPERIMENTAL = ['experimental/ro_obmp2', 'experimental/embedding', 'experimental/kobmp2',
                'experimental/kobdf', 'experimental/transition_dipole', 'experimental/noci',
                'experimental/oodh']
NUMBER = re.compile(r'-?\d+\.\d+(?:[eE][-+]?\d+)?')


def numbers(text):
    return [float(x) for x in NUMBER.findall(text)]


def check(name, rtol=2e-6):
    script = os.path.join(HERE, name + '.py')
    ref = os.path.join(HERE, 'reference_output', name.replace('/', '-') + '.out')
    out = subprocess.run([sys.executable, '-W', 'ignore', script], cwd=HERE,
                         capture_output=True, text=True)
    if out.returncode != 0:
        return False, 'failed:\n' + out.stderr[-500:]
    new, old = numbers(out.stdout), numbers(open(ref).read())
    if len(new) != len(old):
        return False, 'printed %d numbers, reference has %d' % (len(new), len(old))
    worst = max((abs(a - b) / max(1., abs(b)) for a, b in zip(new, old)), default=0.)
    return worst <= rtol, '%d numbers, largest relative deviation %.1e' % (len(new), worst)


if __name__ == '__main__':
    names = STABLE + (EXPERIMENTAL if '--all' in sys.argv else [])
    failed = 0
    for name in names:
        ok, msg = check(name)
        failed += not ok
        print('%-28s %-6s %s' % (name, 'OK' if ok else 'FAILED', msg))
    for f in ('n2_obdf.fcidump',):
        if os.path.exists(os.path.join(HERE, f)):
            os.remove(os.path.join(HERE, f))
    sys.exit(1 if failed else 0)
