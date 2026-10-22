"""Sealed, idempotent result import. Conflicting runs are preserved, never selected by score."""
import shutil
import time
from pathlib import Path
from .config import FORMAT, digest
from .experiment import suite_root
from .matrix import matrix
from .storage import read_json, write_json, load_checkpoint


def job_identity(job):
    return {k: v for k, v in job.items() if k != 'output'}


def artifact_files(folder):
    folder = Path(folder).resolve()
    files = {}
    for p in sorted(folder.rglob('*')):
        # Path.is_junction exists from Python 3.12; older interpreters have no junctions.
        if p.is_symlink() or getattr(p, 'is_junction', lambda: False)() or not p.resolve().is_relative_to(folder):
            raise ValueError('Result packages may not contain links or external paths')
        rel = p.relative_to(folder)
        if p.is_file() and rel.parts[0] not in ('final', 'curves') and rel.as_posix() != 'artifact.json':
            files[rel.as_posix()] = digest(p)
    return files


def seal(folder):
    folder = Path(folder)
    write_json(folder / 'artifact.json', dict(format=FORMAT, files=artifact_files(folder)))


def validate_training(options, job, folder, *, require_complete=False):
    from .train import signature
    from .config import source_manifest, scientific_options
    folder = Path(folder)
    record = read_json(folder / 'run.json')
    plan_hash = matrix(options, job['suite'])['protocol_hash']
    if record['format'] != FORMAT or record['protocol_hash'] != plan_hash or record['job'] != job_identity(job):
        raise ValueError('Run registration/source/scientific configuration/data/budget mismatch')
    snapshot = read_json(folder / 'config_snapshot.json')
    baselines = read_json(folder / 'baselines.json')
    expected = signature(options, job['method'], job['seed'], job['training'], job.get('validation', []), baselines)
    if snapshot['signature'] != expected or snapshot['source']['digest'] != source_manifest()['digest']:
        raise ValueError('Invalid portable checkpoint signature')
    if snapshot['target_epochs'] != job['budget_epochs'] or snapshot['instance_order'] != [r['instance_id'] for r in job['training']]:
        raise ValueError('Run budget/order differs from matrix')
    state = load_checkpoint(folder / 'latest.pt', expected)
    if state['scientific_options'] != scientific_options(options) or state['source_digest'] != snapshot['source']['digest']:
        raise ValueError('Checkpoint configuration/source mismatch')
    progress = read_json(folder / 'progress.json')
    if progress != state['progress'] or state['epoch'] != state['policy_version'] or progress['committed_episodes'] != state['epoch'] * options['environments']:
        raise ValueError('Checkpoint/publication counters disagree')
    if state['epoch'] > job['budget_epochs'] or (require_complete and state['epoch'] != job['budget_epochs']):
        raise ValueError('Incomplete or exceeded training budget')
    return state


def collect(options, source):
    source = Path(source).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    records = [source / 'run.json'] if (source / 'run.json').exists() else sorted(source.rglob('run.json'))
    result = dict(source=str(source), imported=[], unchanged=[], conflicts=[], rejected=[])
    for registration in records:
        if any(p in registration.relative_to(source).parts for p in ('_local', '_conflicts', '_reports')):
            continue
        incoming = registration.parent
        try:
            record = read_json(registration)
            if record['job']['kind'] != 'train':
                continue  # Final timing is produced only by the aggregation host.
            plan = matrix(options, record['job']['suite'])
            job = next((j for j in plan['jobs'] if j['id'] == record['job']['id']), None)
            if job is None or job['run_id'] != record['job']['run_id']:
                raise ValueError('Unregistered job/run ID; select the predeclared --run-id')
            manifest = read_json(incoming / 'artifact.json')
            actual = artifact_files(incoming)
            if manifest.get('format') != FORMAT or manifest['files'] != actual:
                raise ValueError('Unsealed, incomplete or modified result package')
            # Failed jobs without a checkpoint remain importable failure evidence.
            if (incoming / 'latest.pt').exists():
                validate_training(options, job, incoming)
            elif record['protocol_hash'] != plan['protocol_hash'] or record['job'] != job_identity(job):
                raise ValueError('Failure evidence has incompatible registration')
            target = Path(job['output'])
            if target.resolve() == incoming:
                result['unchanged'].append(job['id'])
                continue
            differences = [name for name, sha in actual.items() if (target / name).exists() and digest(target / name) != sha]
            if (target / 'artifact.json').exists():
                existing = artifact_files(target)
                differences = sorted(set(differences) | {name for name in set(existing) | set(actual) if existing.get(name) != actual.get(name)})
            if differences:
                quarantine = suite_root(options, job['suite']) / '_conflicts' / options['run_id'] / job['id'] / str(time.time_ns())
                quarantine.mkdir(parents=True)
                for name in list(actual) + ['artifact.json']:
                    dest = quarantine / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(incoming / name, dest)
                write_json(quarantine / 'conflict.json', dict(job=job['id'], different_files=differences,
                           action='blocked; retain both; no outcome-based selection'))
                result['conflicts'].append(dict(job=job['id'], path=str(quarantine), files=differences))
                continue
            copied = 0
            for name in list(actual) + ['artifact.json']:
                dest = target / name
                if dest.exists() and digest(dest) == digest(incoming / name):
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(incoming / name, dest)
                copied += 1
            result['imported' if copied else 'unchanged'].append(job['id'])
        except Exception as exc:
            result['rejected'].append(dict(path=str(incoming), error=str(exc)))
    if not records:
        result['rejected'].append(dict(path=str(source), error='No registered runs found'))
    from .experiment import local_root
    write_json(local_root(options) / 'imports' / f'{time.time_ns()}.json', result)
    return result
