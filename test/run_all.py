"""Regression test gate: run every test_*.py and summarize. Exit nonzero on any
failure. Run after any numerical change:  .conda/bin/python test/run_all.py
"""
import os, sys, subprocess, glob

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
tests = sorted(glob.glob(os.path.join(HERE, "test_*.py")))
fails = []
for t in tests:
    name = os.path.basename(t)
    r = subprocess.run([PY, t], capture_output=True, text=True)
    ok = r.returncode == 0
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        fails.append(name)
        print("    " + (r.stdout + r.stderr).strip().replace("\n", "\n    ")[-1500:])
print(f"\n{len(tests)-len(fails)}/{len(tests)} suites passed")
sys.exit(1 if fails else 0)
