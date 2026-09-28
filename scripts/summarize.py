"""Mean ± std of the reported test accuracy over run directories.

    python scripts/summarize.py runs/gsc/fits_o1_w512_seed*
"""

import json
import os
import statistics
import sys


def main(dirs):
    accs = []
    for d in dirs:
        path = os.path.join(d, "summary.json")
        if not os.path.isfile(path):
            print(f"  (skipped {d}: no summary.json)")
            continue
        acc = json.load(open(path))["test_acc"] * 100
        accs.append(acc)
        print(f"  {d}: {acc:.2f}%")
    if accs:
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(f"test accuracy over {len(accs)} runs: {statistics.mean(accs):.2f} ± {std:.2f} %")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
