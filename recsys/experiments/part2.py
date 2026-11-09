"""Part 2 reproducible matched-budget comparisons, uncertainty and ablations.

Keeps Part 1's temporal cutoffs, feedback threshold, full-catalogue evaluator,
results template and random seed. Every one of five compared models is tuned
on the exact SAME validation split using EIGHT candidates (TWO in --quick mode).
Part 1's wider original baseline searches are preserved independently.
"""
from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data.dataset import make_train_data
from recsys.evaluation import evaluate, bootstrap_ci, paired_bootstrap_diff
from recsys.experiments.context import RunContext
from recsys.models import UserKNN, BiasedMF, ContentProfile, ItemKNN, MostPopular
from recsys.utils import save_table

log = logging.getLogger(__name__)
SHEET = 'Part 2 - Models'
DISSHEET = 'Part 2 - Disaggregated'
SPARSESHEET = 'Part 2 - Sparsity'
MODEL_ORDER = ('most_popular', 'item_knn', 'user_knn', 'biased_mf', 'content_profile')


def grids(quick: bool) -> dict[str, list[dict]]:
    """Pre-registered, equal-size per-model searches; each uses validation NDCG."""
    if quick:
        return {
            'most_popular': [{'window_days': None, 'min_rating': None},
                             {'window_days': 90, 'min_rating': 4}],
            'item_knn': [{'k': 20, 'similarity': 'cosine', 'min_rating': None},
                         {'k': 50, 'similarity': 'jaccard', 'min_rating': 4}],
            'user_knn': [{'k': 10, 'similarity': 'cosine', 'min_rating': 4},
                         {'k': 30, 'similarity': 'jaccard', 'min_rating': 4}],
            'biased_mf': [{'factors': 12, 'epochs': 3, 'lr': 0.10, 'reg': 0.02},
                          {'factors': 24, 'epochs': 3, 'lr': 0.10, 'reg': 0.05}],
            'content_profile': [{'min_rating': 4, 'year_weight': 0.},
                                {'min_rating': 4, 'year_weight': 0.75}],
        }
    return {
        'most_popular': [{'window_days': window, 'min_rating': minimum}
                         for window, minimum in itertools.product([None, 30, 90, 180], [None, 4])],
        'item_knn': [{'k': k, 'similarity': sim, 'min_rating': minimum}
                     for k in [50, 200]
                     for sim, minimum in [('cosine', None), ('cosine', 4),
                                          ('jaccard', 4), ('adjusted_cosine', None)]],
        'user_knn': [{'k': k, 'similarity': sim, 'min_rating': 4}
                     for k, sim in itertools.product([20, 40, 80, 160], ['cosine', 'jaccard'])],
        'biased_mf': [{'factors': factors, 'epochs': 16, 'lr': lr, 'reg': reg}
                      for factors, lr, reg in itertools.product([16, 48], [0.05, 0.15], [0.02, 0.1])],
        'content_profile': [{'min_rating': minimum, 'year_weight': weight}
                            for minimum, weight in itertools.product([4, 5], [0., 0.25, 0.5, 1.])],
    }


def make_model(key: str, params: dict, ctx: RunContext, seed: int):
    if key == 'user_knn':
        return UserKNN(**params)
    if key == 'biased_mf':
        return BiasedMF(**params, seed=seed)
    if key == 'content_profile':
        return ContentProfile(ctx.data.movies, **params)
    if key == 'most_popular':
        return MostPopular(**params)
    if key == 'item_knn':
        return ItemKNN(**params)
    raise KeyError(key)


def scored(ctx, model, train, held_out):
    return evaluate(model, train, held_out, ctx.config.evaluation,
                    ctx.config.filtering.min_user_history, seed=ctx.config.seed)


def fit_and_score(ctx, key, params, train, held_out, seed):
    model = make_model(key, params, ctx, seed).fit(train)
    return model, scored(ctx, model, train, held_out)


def scores_row(ctx, key, model, result, seed, validation_ndcg: float, budget: int):
    summary = result.summary()
    ci = bootstrap_ci(result.per_user['ndcg'].to_numpy(),
                      ctx.config.evaluation.bootstrap_samples,
                      ctx.config.evaluation.confidence, ctx.config.seed)
    return {
        'Model': model.name,
        'Paradigm': {'user_knn': 'Neighbourhood CF', 'biased_mf': 'Latent-factor CF',
                     'content_profile': 'Content-based', 'item_knn': 'Neighbourhood CF',
                     'most_popular': 'Non-personalised'}[key],
        'Self-implemented?': 'Yes (mini-batch SGD)' if key == 'biased_mf' else 'No',
        'Key hyperparameters': model.describe(),
        'Precision@10': summary['precision@10'], 'Recall@10': summary['recall@10'],
        'NDCG@10': summary['ndcg@10'], 'Catalogue coverage': summary['coverage'],
        'Variability / CI / test': (f'{ctx.config.evaluation.confidence:.0%} user-bootstrap '
                                    f'NDCG [{ci.low:.4f}, {ci.high:.4f}]'),
        'Seed(s)': str(seed),
        'Notes': (f'Part 2 equal-budget validation: {budget} candidates; selected '
                  f'NDCG {validation_ndcg:.4f}; test evaluated once; '
                  f'{len(result.per_user)} evaluated users'),
    }


def disaggregate(ctx, model, result, train, seed):
    """Two prespecified axes; cuts use train interactions, not held-out data.

    Item-group evaluation treats only positives in that popularity stratum as
    relevant while retaining all eligible movies in the ranked candidate set.
    Thus each group's evaluated-user population may differ. This is documented.
    """
    counts = train.history_sizes()
    users = result.per_user['user_id'].to_numpy()
    activity = counts[train.ids.user_index(users)]
    item_counts = np.asarray(train.seen.sum(axis=0)).ravel()
    eligible_counts = item_counts[train.catalogue]
    user_cut = float(np.median(counts[counts > 0])) if np.any(counts > 0) else 0.
    item_cut = float(np.median(eligible_counts)) if len(eligible_counts) else 0.

    for label, mask, condition in (
        ('Low activity', activity <= user_cut, '<='),
        ('High activity', activity > user_cut, '>'),
    ):
        frame = result.per_user.loc[mask]
        if frame.empty:
            continue
        coverage = (len(np.unique(result.recommendations[mask])) /
                    result.catalogue_size) if result.catalogue_size else 0.
        ctx.results.add(DISSHEET, {
            'Model': model.name, 'Breakdown dimension': 'User activity', 'Group': label,
            'Definition / threshold': f'train ratings {condition} {user_cut:g}',
            'Precision@10': float(frame['precision'].mean()),
            'Recall@10': float(frame['recall'].mean()),
            'NDCG@10': float(frame['ndcg'].mean()),
            'Catalogue coverage': coverage, 'Seed(s)': str(seed),
            'Notes': f'{len(frame)} evaluated users; group defined on training history',
        })

    held = ctx.split.test.copy()
    item_idx = train.ids.item_index(held['item_id'])
    held['item_group'] = np.where(item_counts[item_idx] <= item_cut,
                                  'Long tail', 'Popular')
    for label, condition in (('Long tail', '<='), ('Popular', '>')):
        group_test = held.loc[held.item_group == label].drop(columns='item_group')
        group_result = scored(ctx, model, train, group_test)
        if group_result.per_user.empty:
            continue
        m = group_result.summary()
        ctx.results.add(DISSHEET, {
            'Model': model.name, 'Breakdown dimension': 'Test item popularity',
            'Group': label,
            'Definition / threshold': f'train item ratings {condition} {item_cut:g}',
            'Precision@10': m['precision@10'], 'Recall@10': m['recall@10'],
            'NDCG@10': m['ndcg@10'], 'Catalogue coverage': m['coverage'],
            'Seed(s)': str(seed),
            'Notes': (f'{len(group_result.per_user)} users with group-positive truth; '
                      'full catalogue ranked; different test-user populations'),
        })


def _sparse_train(ctx: RunContext, fraction: float, seed: int):
    """User-stratified, seeded train+validation subsample; original test fixed."""
    full = ctx.split.train_val
    if fraction == 1.:
        return full
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(full), dtype=bool)
    for indices in full.groupby('user_id', sort=False).indices.values():
        n = max(1, int(round(fraction * len(indices))))
        keep[rng.choice(indices, size=n, replace=False)] = True
    return full.loc[keep].reset_index(drop=True)


def _write_analysis_helper(ctx, results, paired_rows, ablation_rows):
    """Write automatically grounded, checkable takeaways, not invented numbers."""
    lines = [
        '# Part 2 generated findings (copy and critically review for the report)', '',
        'These figures are generated from the current, non-quick experiment run.',
        'The paired CI is over shared evaluated users; zero in the interval means',
        'the data do not establish the direction of the difference.', '',
        '## Overall ranking', '',
    ]
    ranked = sorted(results.items(), key=lambda pair: pair[1].mean('ndcg'), reverse=True)
    for rank, (key, result) in enumerate(ranked, 1):
        m = result.summary()
        lines.append(f'{rank}. {key}: NDCG@10 = {m["ndcg@10"]:.5f}; '
                     f'P@10 = {m["precision@10"]:.5f}; '
                     f'R@10 = {m["recall@10"]:.5f}; '
                     f'coverage = {m["coverage"]:.4f}.')
    lines.extend(['', '## Paired NDCG differences vs Part 1 baselines', ''])
    for entry in paired_rows:
        if entry['model_b'] in ('most_popular', 'item_knn') and entry['model_a'] in ('user_knn', 'biased_mf', 'content_profile'):
            lines.append(f'- {entry["model_a"]} minus {entry["model_b"]}: '
                         f'{entry["mean_ndcg_difference"]:+.5f} '
                         f'[{entry["ci_lower"]:+.5f}, {entry["ci_upper"]:+.5f}] '
                         f'(n={entry["n_shared_users"]}).')
    lines.extend(['', '## Sparsity: model order by retained training fraction', ''])
    for frac in (1., 0.5, 0.2):
        subset = [r for r in ablation_rows if r['target_fraction'] == frac]
        ordered = sorted(subset, key=lambda r: r['NDCG@10'], reverse=True)
        lines.append(f'- {frac:.0%}: ' + ' > '.join(f'{r["Model"]} '
                    f'({r["NDCG@10"]:.5f})' for r in ordered))
    lines.extend(['', '## Interpretation to add manually', '',
                  'Discuss whether user/item group rankings differ and whether an',
                  'apparent gain is meaningful given the paired CI. Note that',
                  'the candidate catalogue and seen-item mask shrink with the training',
                  'sample, so sparsity comparisons also reflect reachability.', ''])
    target = ctx.out.root / 'part2_auto_analysis.md'
    target.write_text('\n'.join(lines), encoding='utf-8')


def run(ctx: RunContext) -> None:
    seed = ctx.config.seed
    split = ctx.split
    config = ctx.config
    search = grids(ctx.quick)
    budgets = {key: len(choices) for key, choices in search.items()}
    if len(set(budgets.values())) != 1:
        raise AssertionError(f'Part 2 comparisons require equal validation budgets: {budgets}')
    if set(search) != set(MODEL_ORDER):
        raise AssertionError(f'Expected exactly five models, got {list(search)}')

    val_train = make_train_data(split.train, ctx.ids, split.val_start,
                                config.filtering.min_item_train_ratings)
    tuned: dict[str, tuple[dict, float]] = {}
    grid_rows = []
    for key in MODEL_ORDER:
        for params in search[key]:
            model, result = fit_and_score(ctx, key, params, val_train, split.val, seed)
            row = {'model': key, 'params': str(params), 'ndcg@10': result.mean('ndcg'),
                   'precision@10': result.mean('precision'),
                   'recall@10': result.mean('recall'), 'coverage': result.coverage,
                   'budget_for_model': budgets[key]}
            grid_rows.append(row)
            if key not in tuned or row['ndcg@10'] > tuned[key][1]:
                tuned[key] = (params, row['ndcg@10'])
            log.info('Part 2 validation %s %s: NDCG %.5f', key, params, row['ndcg@10'])
    save_table(pd.DataFrame(grid_rows), ctx.out.tables / 'part2_validation_grid.csv')
    save_table(pd.DataFrame([
        {'model': key, 'selected_params': str(tuned[key][0]),
         'validation_ndcg@10': tuned[key][1], 'budget': budgets[key]}
        for key in MODEL_ORDER
    ]), ctx.out.tables / 'part2_selected_models.csv')

    full_train = make_train_data(split.train_val, ctx.ids, split.test_start,
                                 config.filtering.min_item_train_ratings)
    results = {}
    model_records = []
    for key in MODEL_ORDER:
        params, validation_ndcg = tuned[key]
        model, result = fit_and_score(ctx, key, params, full_train, split.test, seed)
        results[key] = result
        ctx.results.add(SHEET, scores_row(ctx, key, model, result, seed,
                                          validation_ndcg, budgets[key]))
        disaggregate(ctx, model, result, full_train, seed)
        model_records.append(result.per_user.assign(model=key))
    save_table(pd.concat(model_records, ignore_index=True),
               ctx.out.tables / 'part2_test_per_user.csv')

    paired_rows = []
    for pos, first_key in enumerate(MODEL_ORDER):
        for second_key in MODEL_ORDER[pos + 1:]:
            # All ten unordered pairs exactly once. When comparing a Part 2
            # model with a baseline, orient the difference as NEW - BASELINE.
            if (first_key in ('most_popular', 'item_knn') and
                    second_key in ('user_knn', 'biased_mf', 'content_profile')):
                a_key, b_key = second_key, first_key
            else:
                a_key, b_key = first_key, second_key
            a = results[a_key].per_user[['user_id', 'ndcg']]
            b = results[b_key].per_user[['user_id', 'ndcg']]
            common = a.merge(b, on='user_id', suffixes=('_a', '_b'))
            if common.empty:
                continue
            interval = paired_bootstrap_diff(common.ndcg_a.to_numpy(),
                                             common.ndcg_b.to_numpy(),
                                             config.evaluation.bootstrap_samples,
                                             config.evaluation.confidence, seed)
            paired_rows.append({'model_a': a_key, 'model_b': b_key,
                                'n_shared_users': len(common),
                                'mean_ndcg_difference': interval.mean,
                                'ci_lower': interval.low, 'ci_upper': interval.high,
                                'confidence': interval.confidence,
                                'seed': seed})
    save_table(pd.DataFrame(paired_rows),
               ctx.out.tables / 'part2_paired_comparisons.csv')

    ablation_rows = []
    for fraction in (1., 0.5, 0.2):
        sampled = _sparse_train(ctx, fraction, seed)
        reduced = make_train_data(sampled, ctx.ids, split.test_start,
                                  config.filtering.min_item_train_ratings)
        for key in MODEL_ORDER:
            model, result = fit_and_score(ctx, key, tuned[key][0],
                                          reduced, split.test, seed)
            metric = result.summary()
            row = {
                'Model': model.name,
                'Training interactions (%)': round(100 * len(sampled) / len(split.train_val), 2),
                'Precision@10': metric['precision@10'],
                'Recall@10': metric['recall@10'],
                'NDCG@10': metric['ndcg@10'],
                'Catalogue coverage': metric['coverage'], 'Seed(s)': str(seed),
                'Notes': (f'{len(sampled)} rows; {len(result.per_user)} evaluated users; '
                          'validation-selected params frozen; test unchanged; '
                          'user-stratified sampling'),
            }
            ctx.results.add(SPARSESHEET, row)
            ablation_rows.append({**row, 'target_fraction': fraction})
    save_table(pd.DataFrame(ablation_rows), ctx.out.tables / 'part2_sparsity_detail.csv')
    ranking = []
    for fraction in (1., 0.5, 0.2):
        group = sorted((r for r in ablation_rows if r['target_fraction'] == fraction),
                       key=lambda r: r['NDCG@10'], reverse=True)
        ranking.extend({'target_fraction': fraction, 'rank': rank,
                        'model': row['Model'], 'ndcg@10': row['NDCG@10']}
                       for rank, row in enumerate(group, 1))
    save_table(pd.DataFrame(ranking),
               ctx.out.tables / 'part2_sparsity_ranking.csv')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(MODEL_ORDER),
           [results[key].mean('ndcg') for key in MODEL_ORDER])
    ax.set_ylabel('Test NDCG@10 (full catalogue)')
    ax.set_title('Part 2: matched validation budgets, five models')
    ax.tick_params(axis='x', labelrotation=20)
    fig.tight_layout()
    fig.savefig(ctx.out.figures / 'fig3_part2_model_comparison.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for key in MODEL_ORDER:
        name = make_model(key, tuned[key][0], ctx, seed).name
        matched = sorted((r for r in ablation_rows if r['Model'] == name),
                         key=lambda r: r['Training interactions (%)'])
        ax.plot([r['Training interactions (%)'] for r in matched],
                [r['NDCG@10'] for r in matched], marker='o', label=key)
    ax.set_xlabel('Retained training interactions (%)')
    ax.set_ylabel('Test NDCG@10')
    ax.set_title('Sparsity ablation: five models, frozen settings')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ctx.out.figures / 'fig4_part2_sparsity.png', dpi=160)
    plt.close(fig)
    if not ctx.quick:
        _write_analysis_helper(ctx, results, paired_rows, ablation_rows)
    log.info('Part 2 finished: five matched-budget models, groups, CIs and ablations')
