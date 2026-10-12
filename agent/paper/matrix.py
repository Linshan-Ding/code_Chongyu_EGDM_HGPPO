"""Balanced case visits, with a separate final-evaluation phase."""
from .config import LEARNERS, RULES, ALIASES
from .data import rows
from .experiment import suite_root, protocol_hash, token


def matrix(options, suite):
    data = rows(options)
    by = {s: [r for r in data if r['split'] == s] for s in
          ('pilot', 'train', 'validation', 'main', 'generalization', 'tiny')}
    by['pilot'].sort(key=lambda r: ('S', 'M').index(r['parameters']['scale']))
    by['train'].sort(key=lambda r: (r['parameters']['load_ratio'], r['parameters']['relocation_multiplier'],
        ('D1', 'D3', 'D4').index(r['parameters']['scenario']), ('S', 'M').index(r['parameters']['scale'])))
    root = suite_root(options, suite)
    run_id = token(options['run_id'])
    visits = options['formal_epochs_per_instance'] if suite == 'formal' else options['pilot_epochs_per_instance']
    if suite == 'pilot-single':
        configurations = [('full', [r], [x for x in by['validation'] if x['parameters']['scale'] == r['parameters']['scale']], r['parameters']['scale']) for r in by['pilot']]
    elif suite in ('pilot-joint', 'formal'):
        configurations = [(m, by['train'] if suite == 'formal' else by['pilot'],
            by['main'] + by['generalization'] if suite == 'formal' else by['validation'], 'joint') for m in LEARNERS]
    else:
        raise ValueError(suite)
    jobs = []
    for method, training, evaluation, tag in configurations:
        for seed in options['seeds']:
            tid = f'train_{method}_{tag}_s{seed}'
            rid = run_id + ('_' + tag if suite == 'pilot-single' else '')
            folder = root / method / f'seed_{seed}' / rid
            budget = len(training) * visits
            base = dict(suite=suite, method=method, seed=seed, run_id=rid, events=None,
                        budget_epochs=budget, budget_episodes=budget * options['environments'],
                        environments_per_epoch=options['environments'])
            jobs.append(dict(base, id=tid, kind='train', phase='train', output=str(folder), depends=[],
                training=training, validation=[], case_batches={r['instance_id']: visits for r in training}))
            for repeat in range(options['replays']):
                jobs.append(dict(base, id=f'replay_{method}_{tag}_s{seed}_latest_{repeat}', kind='replay', phase='evaluate',
                    output=str(folder / 'final' / f'repeat_{repeat}'), depends=[tid], training_job=tid,
                    checkpoint_kind='latest', repeat=repeat, evaluation=evaluation))
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
        aliases={'no_graph': 'mlp (same implementation and result; not independent evidence)'},
        selection_data='none: final committed model only', source_selection='All registered methods retained')


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
