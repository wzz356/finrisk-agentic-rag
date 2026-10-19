"""Verify an extracted release bundle and prepare a machine-local V2 snapshot.

Standard library only. Does not download, extract archives, call a model, change
the current registry, or overwrite a prior installation.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(',', ':')), encoding='utf-8')


def checked_path(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError('BUNDLE_PATH_OUTSIDE_ROOT')
    return path


def verify(root):
    root = Path(root).resolve()
    manifest = load(root / 'bundle-manifest.json')
    for relative, expected in manifest['files'].items():
        path = checked_path(root, relative)
        if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            raise ValueError('BUNDLE_FILE_HASH_MISMATCH:' + relative)
    return manifest


def relocate_metadata(metadata, model_path):
    value = copy.deepcopy(metadata)
    value['binding']['model']['path'] = str(Path(model_path).resolve())
    value['key'] = hashlib.sha256(json.dumps(
        {'text_sha': value['ordered_unit_text_sha256'], 'config': value['binding']},
        sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
    return value


def install(root, destination, model_path, gpu_python):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    manifest = verify(root)
    if manifest.get('kind') != 'runtime':
        raise ValueError('RUNTIME_BUNDLE_REQUIRED')
    if destination.exists():
        raise ValueError('INSTALL_DESTINATION_ALREADY_EXISTS')
    model_path, gpu_python = Path(model_path).resolve(), Path(gpu_python).resolve()
    template = load(root / 'snapshot-template.json')
    dense = load(root / 'dense/dense-config.template.json')
    if not gpu_python.is_file() or not model_path.is_dir() or model_path.name != dense['model_revision']:
        raise ValueError('MODEL_REVISION_OR_PYTHON_NOT_FOUND')
    meta = relocate_metadata(load(root / 'dense/vectors.json'), model_path)
    # Match the original worker's complete file inventory, not just its weights.
    actual = [{'path':p.relative_to(model_path).as_posix(), 'size':p.stat().st_size,
               'sha256':sha(p)} for p in sorted(model_path.rglob('*')) if p.is_file()]
    if actual != meta['binding']['model']['files']:
        raise ValueError('MODEL_FILES_MISMATCH')
    if sha(root / 'dense/vectors.npy') != meta['npy_sha256']:
        raise ValueError('VECTOR_HASH_MISMATCH')
    for asset in template['assets'].values():
        path = checked_path(root, asset['path'])
        if sha(path) != asset['sha256']:
            raise ValueError('SNAPSHOT_ASSET_MISMATCH')
        asset['path'] = str(path)
    destination.mkdir(parents=True, exist_ok=False)
    dump(destination / 'vectors.json', meta)
    dense.update(python_executable=str(gpu_python),model_path=str(model_path),
                 index_path=str(root / 'structured.jsonl'),
                 vector_path=str(root / 'dense/vectors.npy'),
                 vector_meta_path=str(destination / 'vectors.json'),
                 vector_meta_sha256=sha(destination / 'vectors.json'))
    dump(destination / 'dense-config.json', dense)
    rules = template['backend']['assets']['rules']
    rules['path'] = str(checked_path(root, rules['path']))
    if sha(Path(rules['path'])) != rules['sha256']:
        raise ValueError('RULES_HASH_MISMATCH')
    template['backend']['assets']['dense_config'] = {
        'path':str(destination / 'dense-config.json'),
        'sha256':sha(destination / 'dense-config.json')}
    identity = {'origin':manifest['origin_snapshot_id'],
                'assets':template['assets'],'backend':template['backend']}
    template['snapshot_id'] = 'v2-' + hashlib.sha256(json.dumps(
        identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
    dump(destination / 'manifest.json', template)
    return {'snapshot_id':template['snapshot_id'],
            'origin_snapshot_id':manifest['origin_snapshot_id'],
            'source_count':template['source_count'],'chunk_count':template['chunk_count'],
            'snapshot_path':str(destination),'model_requests_sent':0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--snapshot-output', type=Path)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--gpu-python', type=Path)
    args = parser.parse_args()
    if args.verify_only:
        value = verify(args.bundle)
        print(json.dumps({'verified':True,'kind':value['kind'],
                          'files':len(value['files']),'model_requests_sent':0}))
    else:
        if not all((args.snapshot_output,args.model_path,args.gpu_python)):
            parser.error('installation requires --snapshot-output, --model-path, --gpu-python')
        print(json.dumps(install(args.bundle,args.snapshot_output,args.model_path,args.gpu_python),
                         ensure_ascii=False))
