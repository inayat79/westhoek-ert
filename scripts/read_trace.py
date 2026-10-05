"""
read_trace.py — record every file a script opens for reading (provenance audit, see run_all.py).

Active only when the environment variable WESTHOEK_READ_TRACE names an output file: a Python audit hook collects the
paths of all read-mode open() calls (this includes np.load / json.load / pandas, which go through open()) and appends
"<tag>\\t<absolute path>" lines to that file at interpreter exit.
"""

import atexit
import os
import sys


def install(tag):
    out = os.environ.get("WESTHOEK_READ_TRACE")
    if not out:
        return lambda: None
    seen = set()

    def hook(event, args):
        if event != "open" or not args:
            return
        path, mode = args[0], (args[1] if len(args) > 1 else None)
        if isinstance(path, int) or path is None:
            return
        m = str(mode or "r")
        if any(c in m for c in "wax+"):
            return
        try:
            seen.add(os.path.abspath(os.fsdecode(path)))
        except (TypeError, ValueError):
            pass

    done = []

    def dump():
        """Append the collected paths once (called explicitly or at exit)."""
        if done:
            return
        done.append(True)
        import multiprocessing
        # worker processes of a pool write to their own file (concurrent appends to one file interleave on Windows);
        # run_all.audit reads provenance_reads.tsv and provenance_reads.tsv.<pid>
        target = out if multiprocessing.parent_process() is None else f"{out}.{os.getpid()}"
        with open(target, "a", encoding="utf-8") as f:
            f.writelines(f"{tag}\t{p}\n" for p in sorted(seen))

    sys.addaudithook(hook)
    atexit.register(dump)
    return dump
