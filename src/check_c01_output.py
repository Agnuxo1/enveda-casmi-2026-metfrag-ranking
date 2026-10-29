"""Independent integrity gate for visible MetFrag outputs, not a quality metric."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def validate(folder):
    report_file = folder / 'metfrag_report.json'
    report = json.loads(report_file.read_text(encoding='utf-8'))
    if not report.get('status', '').startswith('applied'):
        raise ValueError('MetFrag must be applied, not a control-only/no-op output')
    if report.get('molecules_scored', 0) <= 0 or report.get('rows_reordered', 0) <= 0:
        raise ValueError('Need actual scored molecules and changed rankings')
    paths = [folder / 'submission_control_c472.csv', folder / 'submission.csv']
    tables = []
    for path in paths:
        with path.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ['molecule_id', 'smiles']:
                raise ValueError('Unexpected submission columns')
            tables.append(list(reader))
    control, candidate = tables
    ids = [r['molecule_id'] for r in control]
    if not ids or len(ids) != len(set(ids)) or ids != [r['molecule_id'] for r in candidate]:
        raise ValueError('Changed, missing or duplicate molecule IDs')
    changed = 0
    for a, b in zip(control, candidate, strict=True):
        old, new = a['smiles'].split(';'), b['smiles'].split(';')
        if not (1 <= len(new) <= 25 and all(new) and len(new) == len(set(new))):
            raise ValueError('Invalid candidate-list cardinality or duplicates')
        if old[0] != new[0] or sorted(old) != sorted(new):
            raise ValueError('Leader or candidate set changed')
        changed += old != new
    if changed != report['rows_reordered']:
        raise ValueError('Ranking-change count disagrees with report')
    sha = hashlib.sha256(paths[1].read_bytes()).hexdigest()
    if sha != report['sha256']:
        raise ValueError('Final submission hash disagrees with runtime report')
    return dict(status='integrity_passed', rows=len(ids), changed_rows=changed,
                unchanged_leaders=len(ids), unchanged_candidate_sets=len(ids),
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [report_file, *paths]}, metfrag_report=report,
                scope='Visible smoke-run integrity only; not official hidden-set quality or MRR improvement')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--receipt', required=True, type=Path)
    args = ap.parse_args()
    result = validate(args.output)
    if args.receipt.exists():
        raise FileExistsError('Preserve existing receipt; choose a new filename')
    args.receipt.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: result[k] for k in ['status', 'rows', 'changed_rows', 'scope']}))
