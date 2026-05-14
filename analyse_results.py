"""
analyse_results.py
------------------
Run this after main.py to get a full confidence score report
for all three models, broken down by language.

Usage:
    python analyse_results.py
"""

import re
import os
from collections import defaultdict


RESULT_FILES = {
    "MuRIL":       "outputs/muril_results.txt",
    "mT5":         "outputs/mt5_results.txt",
    "IndicTrans2": "outputs/indictrans_results.txt",
}


def parse_results(filepath):
    results = []
    if not os.path.exists(filepath):
        return results
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(
                r"Sentence (\d+) \| File: (\S+) \| (\w+) \| Confidence: ([\d.]+)",
                line
            )
            if m:
                results.append({
                    "sentence_number": int(m.group(1)),
                    "language_file":   m.group(2),
                    "status":          m.group(3),
                    "confidence":      float(m.group(4)),
                })
    return results


def report(model_name, results):
    bar = "=" * 55
    print(bar)
    print(f"  {model_name}")
    print(bar)

    if not results:
        print("  No results found — output file is empty or missing.\n")
        return

    total     = len(results)
    avg_conf  = sum(r["confidence"] for r in results) / total
    wrong     = sum(1 for r in results if r["status"] == "WRONG")
    correct   = total - wrong

    print(f"  Total sentences  : {total}")
    print(f"  Overall avg conf : {avg_conf:.4f}")
    print(f"  Correct          : {correct} ({correct/total*100:.1f}%)")
    print(f"  Flagged WRONG    : {wrong}  ({wrong/total*100:.1f}%)")
    print()
    print(f"  {'Language':<22} {'Avg Conf':>10} {'Correct':>10} {'Wrong':>8} {'Total':>7}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*8} {'-'*7}")

    by_lang = defaultdict(list)
    for r in results:
        by_lang[r["language_file"]].append(r)

    for lang in sorted(by_lang.keys()):
        rows      = by_lang[lang]
        lang_avg  = sum(r["confidence"] for r in rows) / len(rows)
        lang_wrong   = sum(1 for r in rows if r["status"] == "WRONG")
        lang_correct = len(rows) - lang_wrong
        print(
            f"  {lang:<22} {lang_avg:>10.4f} {lang_correct:>10} "
            f"{lang_wrong:>8} {len(rows):>7}"
        )

    print()


def summary(all_model_results):
    print("=" * 55)
    print("  OVERALL SUMMARY")
    print("=" * 55)
    print(f"  {'Model':<15} {'Avg Conf':>10} {'Wrong':>8} {'Total':>8}")
    print(f"  {'-'*15} {'-'*10} {'-'*8} {'-'*8}")
    for model_name, results in all_model_results.items():
        if not results:
            print(f"  {model_name:<15} {'—':>10} {'—':>8} {'—':>8}")
            continue
        total    = len(results)
        avg_conf = sum(r["confidence"] for r in results) / total
        wrong    = sum(1 for r in results if r["status"] == "WRONG")
        print(f"  {model_name:<15} {avg_conf:>10.4f} {wrong:>8} {total:>8}")
    print()


if __name__ == "__main__":
    all_model_results = {}
    for model_name, filepath in RESULT_FILES.items():
        results = parse_results(filepath)
        all_model_results[model_name] = results
        report(model_name, results)

    summary(all_model_results)