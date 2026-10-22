"""Balanced case visits over seeded cycles, with a separate final-evaluation phase."""
from .config import LEARNERS, RULES, ALIASES
from .data import rows
from .experiment import suite_root, protocol_hash, token

PILOT_LOADS = (0.7, 0.9)
PILOT_RELOCATION = 1.25


def pilot_training_cases(train_rows):
    """Pilot v2 learns on twelve registered training cases: S/M x D1/D3/D4 x two loads."""
    chosen = [r for r in train_rows if r['parameters']['relocation_multiplier'] == PILOT_RELOCATION
              and r['parameters']['load_ratio'] in PILOT_LOADS]
    if len(chosen) != 12:
        raise ValueError('Pilot v2 expects exactly twelve registered training cases')
    return chosen


def matrix(options, suite):
    data = rows(options)
    by = {s: [r for r in data if r['split'] == s] for s in
          ('pilot', 'train', 'validation', 'main', 'generalization', 'tiny')}
    by['pilot'].sort(key=lambda r: ('S', 'M').index(r['parameters']['scale']))
    by['train'].sort(key=lambda r: (r['parameters']['load_ratio'], r['parameters']['relocation_multiplier'],
        ('D1', 'D3', 'D4').index(r['parameters']['scenario']), ('S', 'M').index(r['parameters']['scale'])))
    by['validation'].sort(key=lambda r: (('S', 'M').index(r['parameters']['scale']), r['instance_id']))
    root = suite_root(options, suite)
    run_id = token(options['run_id'])
    updates = options['formal_updates'] if suite == 'formal' else options['pilot_updates']
    k, n = options['instances_per_update'], options['episodes_per_instance']
    pilot_cases = pilot_training_cases(by['train'])
    if suite == 'pilot-single':
        configurations = [('full', [c for c in pilot_cases if c['parameters']['scale'] == scale],
                           [x for x in by['validation'] if x['parameters']['scale'] == scale],
                           [x for x in by['validation'] if x['parameters']['scale'] == scale], scale) for scale in ('S', 'M')]
    elif suite in ('pilot-joint', 'formal'):
        configurations = [(m, by['train'] if suite == 'formal' else pilot_cases, by['validation'],
            by['main'] + by['generalization'] if suite == 'formal' else by['validation'], 'joint') for m in LEARNERS]
    else:
        raise ValueError(suite)
    jobs = []
    for method, training, validation, evaluation, tag in configurations:
        for seed in options['seeds']:
            tid = f'train_{method}_{tag}_s{seed}'
            rid = run_id + ('_' + tag if suite == 'pilot-single' else '')
            folder = root / method / f'seed_{seed}' / rid
            base = dict(suite=suite, method=method, seed=seed, run_id=rid, events=None,
                        budget_epochs=updates, budget_episodes=updates * options['environments'],
                        environments_per_epoch=options['environments'])
            jobs.append(dict(base, id=tid, kind='train', phase='train', output=str(folder), depends=[],
                training=training, validation=validation,
                expected_visits={r['instance_id']: updates * k / len(training) for r in training}))
            for kind in ('latest', 'best'):
                for repeat in range(options['replays']):
                    jobs.append(dict(base, id=f'replay_{method}_{tag}_s{seed}_{kind}_{repeat}', kind='replay', phase='evaluate',
                        output=str(folder / 'final' / kind / f'repeat_{repeat}'), depends=[tid], training_job=tid,
                        checkpoint_kind=kind, repeat=repeat, evaluation=evaluation))
    if suite != 'pilot-single':
        evaluation = by['main'] + by['generalization'] if suite == 'formal' else by['validation']
        for method in RULES:
            for seed in options['seeds'] if method == 'RH-ALNS' else [0]:
                jobs.append(dict(id=f'rule_{method}_s{seed}', suite=suite, kind='rule', phase='evaluate',
                    method=method, seed=seed, run_id=run_id, depends=[], evaluation=evaluation,
                    output=str(root / method / f'seed_{seed}' / run_id), budget_epochs=0, budget_episodes=0))
    if suite == 'formal':
        jobs.append(dict(id='tiny_references', suite=suite, kind='reference', phase='evaluate', method='tiny-reference',
                    seed=0, run_id=run_id, depends=[], evaluation=by['tiny'], budget_epochs=0, budget_episodes=0,
                    output=str(root / 'tiny-reference' / 'seed_0' / run_id)))
    return dict(protocol=options['protocol'], protocol_hash=protocol_hash(options), run_id=run_id,
        epoch_semantics=options['epoch_semantics'], suite=suite, jobs=jobs,
        update=dict(instances_per_update=k, episodes_per_instance=n, updates=updates),
        aliases={'no_graph': 'mlp (same implementation and result; not independent evidence)'},
        selection_data='development split only: best.pt is selected on registered validation cases; latest.pt is the final model',
        source_selection='All registered methods retained')


def select_jobs(plan, *, phase, methods=None, seeds=None, job_ids=None):
    methods = {ALIASES.get(m, m) for m in methods} if methods else None
    if methods and not methods <= set(LEARNERS + RULES + ('tiny-reference',)):
        raise ValueError('Unknown method filter')
    if job_ids and not set(job_ids) <= {j['id'] for j in plan['jobs']}:
        raise ValueError('Unknown job ID')
    selected = [j for j in plan['jobs'] if (phase == 'all' or j['phase'] == phase)
        and (not methods or j['method'] in methods) and (seeds is None or j['seed'] in seeds)
        and (not job_ids or j['id'] in job_ids)]
    if not selected:
        raise ValueError('No jobs match selection')
    return selected
