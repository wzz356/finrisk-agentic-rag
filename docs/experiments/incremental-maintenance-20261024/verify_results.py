"""Offline integrity/count verification of the public result package."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    hashes = read('checksums.json')['files']
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
              if p.is_file() and p.name != 'checksums.json' and '__pycache__' not in p.parts}
    require(actual == set(hashes), 'File inventory changed')
    for name, value in hashes.items():
        require(sha256((ROOT / name).read_bytes()).hexdigest() == value, f'File changed: {name}')
    results, metrics = read('paired-results.json'), read('metrics.json')
    require(len(results) == len({r['sample_id'] for r in results}) == 20, 'Development count')
    require(dict(Counter(r['B_action'] for r in results)) == metrics['action_counts']
            == {'add': 14, 'exclude_from_penalty': 5, 'hold': 1}, 'Action counts')
    samples = read('samples.json')['samples']
    require(dict(Counter(s['stage'] for s in samples)) == {'development': 20, 'formal': 80}, '20/80 split')
    for s in samples + read('old-sources-read.json'):
        require(sha256(s['text'].encode('utf-8')).hexdigest() == s['text_sha256'], 'Source text hash')
    jobs = [j for r in results for j in r['jobs']]
    require(len(jobs) == metrics['role_tasks_dispatched'] == 42, 'Role task count')
    require(len(read('evidence-calls.json')) == metrics['evidence_tool_calls'] == 143, 'Evidence count')
    require(len(list((ROOT / 'jobs').glob('*/output.json'))) == 42, 'Output count')
    for j in jobs:
        read(f"jobs/{j['job_id']}/output.json")
        require(read(f"jobs/{j['job_id']}/validation.json")['valid'], 'Role validation')
    require(len(read('old-sources-read.json')) == 6, 'Old source count')
    d18 = next(r for r in results if r['sample_id'] == 'D18')
    require(d18['A']['action'] == 'exclude_from_penalty' and d18['B_action'] == 'hold', 'D18 revision')
    for arm in ['A', 'B']:
        cards = [json.loads(x) for x in (ROOT / f'shadow/{arm}/cards.jsonl').read_text(encoding='utf-8').splitlines()]
        require(len(cards) == 35 and sum('penalties' in c for c in cards) == 14, '14 case / 21 behavior cards')
        require(all(c['human_review_status'] == 'pending' and c['publication_status'] == 'candidate'
                    for c in cards), 'Candidate boundary')
        overlay = read(f'shadow/{arm}/overlay.json')
        require(not overlay['production_admission'] and not overlay['human_approved'], 'Production boundary')
    require(not metrics['production_modified'] and not metrics['formal_evaluation_started'], 'Unrun/production state')
    require(read('scope-followup.json')['needs_additional_scope_verification'] == ['D14', 'D20'], 'Scope followup')
    print(json.dumps({'verified': True, 'files': len(hashes), 'development': 20, 'formal_unrun': 80,
                      'role_outputs': 42, 'evidence_calls': 143, 'production_enabled': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
