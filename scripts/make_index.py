"""Regenerate docs/INDEX.md from the files on disk.

The index maps each experiment to its contract, memo, scripts and artifact
count, and lists every `docs/...` path a memo or script cites that the
repository does not contain.

IT IS GENERATED, NOT MAINTAINED. A hand-written index drifts from the tree the
moment an experiment is added, and a stale index is worse than none: it is a
claim about what exists, which is the one kind of claim this program does not
let stand unverified. Run this after adding an experiment:

    python scripts/make_index.py
"""

from __future__ import annotations

import glob
import os
import re


def key(name: str) -> str | None:
    m = re.match(r"(e\d+[a-z]?)", os.path.basename(name).lower())
    return m.group(1) if m else None


def sort_key(k: str):
    m = re.match(r"e(\d+)([a-z]?)", k)
    return (int(m.group(1)), m.group(2))


def main() -> int:
    memos = {os.path.basename(f)[5:-3].lower(): f for f in glob.glob("runs/MEMO_*.md")}
    docs = glob.glob("docs/*.md")
    scripts = sorted(os.path.basename(f) for f in glob.glob("scripts/*.py"))
    rundirs = sorted(os.path.basename(d.rstrip("/")) for d in glob.glob("runs/*/"))

    keys = {k for k in (key(m) for m in memos) if k}
    keys |= {k for k in (key(d) for d in docs) if k}
    keys |= {k for k in (key(s) for s in scripts) if k}

    # every docs/ path cited by a memo or a script, and whether it exists
    sources = glob.glob("runs/*.md") + [f"scripts/{s}" for s in scripts]
    cited: dict[str, list[str]] = {}
    for f in sources:
        try:
            text = open(f, errors="ignore").read()
        except OSError:
            continue
        for p in set(re.findall(r"docs/[A-Za-z0-9_]+\.(?:md|tex)", text)):
            cited.setdefault(p, []).append(os.path.basename(f))
    missing = sorted(p for p in cited if not os.path.exists(p))

    out = [
        "# Experiment index",
        "",
        "Generated from the files, not maintained by hand: run",
        "`python scripts/make_index.py` after adding an experiment.",
        "",
        "Each row is one experiment. **Contract** is the pre-registration signed",
        "before it ran; **memo** is the write-up; **artifacts** counts the run",
        "directories under `runs/` whose name begins with that key.",
        "",
        "| Experiment | Contract | Memo | Scripts | Artifacts |",
        "|---|---|---|---|---|",
    ]
    for k in sorted(keys, key=sort_key):
        pre = [os.path.basename(d) for d in docs
               if key(d) == k and "prereg" in d.lower()]
        oth = [os.path.basename(d) for d in docs
               if key(d) == k and "prereg" not in d.lower()]
        show = pre or oth
        c = ", ".join(f"`{x}`" for x in show) if show else "**—**"
        memo = memos.get(k)
        m = f"`{os.path.basename(memo)}`" if memo else "—"
        n_scr = sum(1 for s in scripts if key(s) == k)
        n_run = sum(1 for d in rundirs if d.lower().startswith(k))
        out.append(f"| **{k.upper()}** | {c} | {m} | {n_scr or '—'} | {n_run or '—'} |")

    out += ["", "## Dangling citations", "",
            "Paths cited by a memo or script that the repository does not contain.",
            "Listed so a reader meets them here rather than in a dead link.", ""]
    if missing:
        for p in missing:
            who = ", ".join(f"`{w}`" for w in sorted(set(cited[p]))[:4])
            out.append(f"- `{p}` — cited by {who}")
        out += ["",
                "These are provenance gaps, not disputed results: the predictions",
                "those contracts registered are scored in `docs/appendix.tex`. The",
                "contracts were signed in a working session and never committed."]
    else:
        out.append("None.")
    out.append("")

    open("docs/INDEX.md", "w").write("\n".join(out))
    print(f"wrote docs/INDEX.md — {len(keys)} experiments, {len(missing)} dangling citations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
