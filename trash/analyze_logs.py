"""
analyze_logs.py — Analyse des fichiers de logs d'entraînement MIL.

Chaque ligne a le format :
    [status, ] encoder, mil, marker, meilleur modèle — epoch N, val AUC: X.XXXX

Usage :
    python analyze_logs.py --stat mean
    python analyze_logs.py --stat max --encoder titan
    python analyze_logs.py --stat min --mil transmil --marker BCL2
    python analyze_logs.py --stat mean --status LDH --encoder prism
    python analyze_logs.py --file output_logs_concat.txt --stat mean --mil abmil
"""

import argparse
import re
import sys
from pathlib import Path


FIELDS = ("status", "encoder", "mil", "marker")

LINE_RE = re.compile(
    r"^(?P<prefix>.+?),\s*meilleur\s+mod[eè]le\s*[—-]+\s*epoch\s+\d+,\s*val\s+AUC:\s*(?P<auc>[\d.]+)",
    re.IGNORECASE,
)


def parse_line(line: str) -> dict | None:
    line = line.strip()
    m = LINE_RE.match(line)
    if not m:
        return None
    auc    = float(m.group("auc"))
    parts  = [p.strip() for p in m.group("prefix").split(",")]
    if len(parts) == 3:
        status, encoder, mil, marker = None, parts[0], parts[1], parts[2]
    elif len(parts) == 4:
        status, encoder, mil, marker = parts[0], parts[1], parts[2], parts[3]
    else:
        return None
    return dict(status=status, encoder=encoder, mil=mil, marker=marker, auc=auc)


def load_records(log_file: str) -> list[dict]:
    records = []
    for line in Path(log_file).read_text(encoding="utf-8").splitlines():
        r = parse_line(line)
        if r:
            records.append(r)
    return records


def filter_records(records: list[dict], filters: dict[str, str]) -> list[dict]:
    out = []
    for r in records:
        match = all(
            r.get(field) is not None and r[field].lower() == val.lower()
            for field, val in filters.items()
        )
        if match:
            out.append(r)
    return out


def compute_stat(records: list[dict], stat: str) -> float:
    aucs = [r["auc"] for r in records]
    if not aucs:
        raise ValueError("Aucun résultat après filtrage.")
    if stat == "mean":
        return sum(aucs) / len(aucs)
    if stat == "min":
        return min(aucs)
    if stat == "max":
        return max(aucs)
    raise ValueError(f"stat inconnu : {stat}  (attendu : mean | min | max)")


def main():
    parser = argparse.ArgumentParser(description="Analyse des logs MIL")
    parser.add_argument("--file",    default="output_logs.txt", help="Fichier de logs")
    parser.add_argument("--stat",    choices=["mean", "min", "max"], default="mean")
    parser.add_argument("--status",  default=None)
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--mil",     default=None)
    parser.add_argument("--marker",  default=None)
    parser.add_argument("--verbose", action="store_true", help="Affiche les lignes retenues")
    args = parser.parse_args()

    records = load_records(args.file)
    print(f"{len(records)} lignes chargées depuis {args.file}")

    filters = {
        field: getattr(args, field)
        for field in FIELDS
        if getattr(args, field) is not None
    }
    if filters:
        print(f"Filtres : {filters}")

    filtered = filter_records(records, filters)
    print(f"{len(filtered)} lignes après filtrage")

    if args.verbose:
        for r in filtered:
            status_str = f"{r['status']}, " if r["status"] else ""
            print(f"  {status_str}{r['encoder']}, {r['mil']}, {r['marker']} — AUC: {r['auc']:.4f}")

    if not filtered:
        sys.exit(1)

    result = compute_stat(filtered, args.stat)
    print(f"\n{args.stat.upper()} AUC : {result:.4f}")


if __name__ == "__main__":
    main()
