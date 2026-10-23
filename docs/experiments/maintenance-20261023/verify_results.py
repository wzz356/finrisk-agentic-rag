"""Verify published evidence consistency using only Python's standard library."""
from collections import Counter
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(value, message):
    if not value:
        raise ValueError(message)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if tag == 'a' and 'href' in attrs:
            self.links.append(attrs['href'])


def main():
    hashes = read(ROOT / 'checksums.json')['files']
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
              if p.is_file() and p.name != 'checksums.json' and '__pycache__' not in p.parts}
    require(actual == set(hashes), 'Public file inventory differs from manifest')
    for rel, digest in hashes.items():
        require(sha256((ROOT / rel).read_bytes()).hexdigest() == digest, f'File changed: {rel}')
    b, n = ROOT / 'deepseek-100', ROOT / 'codex-low'
    bm, nm = read(b / 'experiment-metrics.json'), read(n / 'experiment-metrics.json')
    bp, np = read(b / 'paired-results.json'), read(n / 'paired-results.json')
    require(len(bp) == bm['denominator'] == 100, 'Batch denominator')
    require(len(np) == nm['full_pipeline_denominator'] == 12, 'Codex denominator')
    require(dict(Counter(x['status'] for x in bp)) == bm['status_counts'], 'Batch statuses')
    require(dict(Counter(x['new_status'] for x in np)) == nm['full_pipeline_statuses'], 'Codex statuses')
    require(len({x['id'] for x in bp}) == 100 and len({x['id'] for x in np}) == 12, 'Duplicate sample IDs')
    source_checks = 0
    for folder, size in [(b, 100), (n, 12)]:
        samples = read(folder / 'samples.json')['samples']
        require(len(samples) == size, 'Source sample count')
        for s in samples:
            digest = sha256(s['text'].encode('utf-8')).hexdigest()
            require(digest == s.get('text_sha256_utf8', s['raw_sha256']), 'Extracted source text changed')
            source_checks += 1
    require(bm['usage']['prompt_tokens'] + bm['usage']['completion_tokens'] == 2013557, 'Known token total')
    require(bm['physical_requests_ledger'] == sum(bm['ledger_states'].values()) == 412, 'Request denominator')
    reviews = [x['codex_review_only'] for x in np if x['codex_review_only']]
    require(len(reviews) == 8, 'Review-only denominator')
    require(dict(Counter(x['decision'] for x in reviews)) == nm['review_only']['decisions'], 'Review decisions')
    outputs = list((n / 'jobs').glob('*/output.json'))
    require(len(outputs) == nm['role_tasks'] == 30, 'Role output count')
    for p in outputs:
        read(p)
        read(p.parent / 'validation.json')
    calls = read(n / 'evidence-calls.json')
    require(len(calls) == nm['evidence_tool_calls'] == 87, 'Evidence call count')
    for d in read(n / 'duplicate-readback-checks.json'):
        matches = [x for x in calls if x['public_job'] == f"C-{d['id']}-proposer-r1-a1"
                   and x['action'] == 'read' and x['args']['source_id'] == d['correct_known_old_source']]
        require(len(matches) == 1, 'Duplicate old source read')
        r = matches[0]['result']
        require(r['start'] == 0 and r['next_start'] is None, 'Duplicate partial read')
        require(next(x for x in np if x['id'] == d['id'])['new_status'] == 'proposed_no_change_unreviewed', 'Duplicate outcome')
    cards = [json.loads(x) for x in (n / 'shadow/CodexPR/cards.jsonl').read_text(encoding='utf-8').splitlines()]
    require(dict(Counter(x['record_kind'] for x in cards)) == {'case': 7, 'behavior': 13}, 'Card counts')
    require(all(x['human_approved'] is False and x['production_admission'] is False for x in cards), 'Admission boundary')
    approved = {x['id'] for x in np if x['new_status'] == 'experiment_protocol_approved_not_applied'}
    require({x['sample_id'] for x in cards} == approved, 'Accepted card source mismatch')
    link_count = 0
    for p in ROOT.rglob('*.html'):
        parser = Links()
        parser.feed(p.read_text(encoding='utf-8'))
        for href in parser.links:
            target = urlsplit(href)
            if target.scheme:
                require(target.scheme == 'https', 'Unexpected external link scheme')
                continue
            if not target.path:
                require(unquote(target.fragment) in parser.ids, 'Missing HTML anchor')
            else:
                require((p.parent / unquote(target.path)).is_file(), f'Missing report link: {href}')
            link_count += 1
    require(not nm['production_modified'] and not bm['production_modified'], 'Production boundary')
    print(json.dumps({'verified': True, 'files': len(hashes), 'samples': [100, 12],
                      'source_text_hash_checks': source_checks, 'role_outputs': len(outputs),
                      'evidence_calls': len(calls), 'cards': {'case': 7, 'behavior': 13},
                      'local_report_links': link_count, 'network_calls': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
