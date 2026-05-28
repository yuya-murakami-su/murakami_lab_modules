"""Cross-validation and hyper-parameter search utilities.

The model-selection API uses a user-provided ``model_factory`` so every
fold/trial receives a fresh ``ModelHandler`` and freshly split data. This keeps
state leakage between folds explicit and avoidable.
"""

import itertools
import json
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import utils

__all__ = [
    'IndexSplit',
    'TrialContext',
    'TrialResult',
    'SearchResult',
    'Metric',
    'KFoldSplitter',
    'iter_parameter_grid',
    'sample_parameter_space',
    'results_to_dataframe',
    'save_trial_results',
    'CrossValidator',
    'GridSearch',
    'RandomSearch',
    'NestedCrossValidator',
]


@dataclass(frozen=True)
class IndexSplit:
    """Train/validation/test indices for one fold."""

    train_indices: np.ndarray
    valid_indices: np.ndarray | None = None
    test_indices: np.ndarray | None = None
    fold: int | None = None

    def config_dict(self) -> dict[str, object]:
        return {
            'fold': self.fold,
            'train_indices': self.train_indices.tolist(),
            'valid_indices': None if self.valid_indices is None else self.valid_indices.tolist(),
            'test_indices': None if self.test_indices is None else self.test_indices.tolist(),
        }


@dataclass(frozen=True)
class TrialContext:
    """Metadata passed to ``model_factory`` for one trial."""

    trial_id: str
    params: dict[str, object]
    split: IndexSplit
    param_index: int = 0
    fold: int | None = None
    outer_fold: int | None = None
    inner_fold: int | None = None
    seed: int | None = None
    output_dir: Path | None = None


@dataclass
class TrialResult:
    """Result from one fitted model in one fold or final evaluation."""

    trial_id: str
    params: dict[str, object]
    status: str
    fold: int | None = None
    outer_fold: int | None = None
    inner_fold: int | None = None
    seed: int | None = None
    best_loss: float | None = None
    train_loss: float | None = None
    validation_loss: float | None = None
    test_loss: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    model_path: str | None = None
    n_epochs: int | None = None
    best_epoch: int | None = None
    elapsed_time: float | None = None
    error: str | None = None
    traceback: str | None = None

    def objective(self, key: str = 'validation_loss') -> float | None:
        if key in self.metrics:
            return self.metrics[key]
        return getattr(self, key)

    def to_dict(self) -> dict[str, object]:
        return {
            'trial_id': self.trial_id,
            'status': self.status,
            'fold': self.fold,
            'outer_fold': self.outer_fold,
            'inner_fold': self.inner_fold,
            'seed': self.seed,
            'best_loss': self.best_loss,
            'train_loss': self.train_loss,
            'validation_loss': self.validation_loss,
            'test_loss': self.test_loss,
            'metrics': self.metrics,
            'params': utils.serialize_config_value(self.params),
            'model_path': self.model_path,
            'n_epochs': self.n_epochs,
            'best_epoch': self.best_epoch,
            'elapsed_time': self.elapsed_time,
            'error': self.error,
            'traceback': self.traceback,
        }


@dataclass
class SearchResult:
    """Aggregated score for one hyper-parameter setting across folds."""

    params: dict[str, object]
    param_index: int
    fold_results: list[TrialResult]
    score_key: str = 'validation_loss'
    greater_is_better: bool = False
    mean_score: float | None = None
    std_score: float | None = None
    status: str = 'ok'
    rank: int | None = None

    def __post_init__(self):
        scores = [
            result.objective(self.score_key)
            for result in self.fold_results
            if result.status == 'ok' and result.objective(self.score_key) is not None
        ]
        if not scores:
            self.status = 'failed'
            return
        self.mean_score = float(np.mean(scores))
        self.std_score = float(np.std(scores))
        if any(result.status != 'ok' for result in self.fold_results):
            self.status = 'partial'

    def to_dict(self) -> dict[str, object]:
        return {
            'param_index': self.param_index,
            'status': self.status,
            'rank': self.rank,
            'score_key': self.score_key,
            'greater_is_better': self.greater_is_better,
            'mean_score': self.mean_score,
            'std_score': self.std_score,
            'params': utils.serialize_config_value(self.params),
        }


@dataclass(frozen=True)
class Metric:
    """Metric definition used by CV/search evaluation."""

    name: str
    function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor | float]
    unnormalize: bool = True


class KFoldSplitter:
    """Create reproducible K-fold train/validation splits."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_seed: int = 2025):
        if type(n_splits) is not int or n_splits < 2:
            raise ValueError('n_splits must be an int >= 2.')
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_seed = random_seed

    def split(self, indices: int | Sequence[int] | np.ndarray) -> list[IndexSplit]:
        if type(indices) is int:
            if indices < self.n_splits:
                raise ValueError('n_samples must be >= n_splits.')
            indices = np.arange(indices)
        else:
            indices = np.asarray(indices)
            if len(indices) < self.n_splits:
                raise ValueError('len(indices) must be >= n_splits.')

        indices = indices.copy()
        if self.shuffle:
            rng = np.random.default_rng(self.random_seed)
            rng.shuffle(indices)

        folds = np.array_split(indices, self.n_splits)
        splits = []
        for fold, valid_indices in enumerate(folds):
            train_indices = np.concatenate([part for idx, part in enumerate(folds) if idx != fold])
            splits.append(IndexSplit(train_indices=train_indices, valid_indices=valid_indices, fold=fold))
        return splits


def iter_parameter_grid(param_grid: Mapping[str, Iterable[object]]) -> list[dict[str, object]]:
    """Return all combinations from a scikit-learn-style parameter grid."""

    keys = list(param_grid.keys())
    values = [list(param_grid[key]) for key in keys]
    if any(len(value) == 0 for value in values):
        raise ValueError('All parameter grid values must contain at least one candidate.')
    return [
        dict(zip(keys, combination))
        for combination in itertools.product(*values)
    ]


def sample_parameter_space(
        param_space: Mapping[str, Sequence[object] | Callable[[np.random.Generator], object] | object],
        n_iter: int,
        random_seed: int = 2025
) -> list[dict[str, object]]:
    """Sample random parameter dictionaries from sequences or callables."""

    if type(n_iter) is not int or n_iter < 1:
        raise ValueError('n_iter must be a positive int.')
    rng = np.random.default_rng(random_seed)
    samples = []
    for _ in range(n_iter):
        params = {}
        for key, value in param_space.items():
            if callable(value):
                params[key] = value(rng)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                if len(value) == 0:
                    raise ValueError(f'Parameter space for {key} is empty.')
                params[key] = value[int(rng.integers(0, len(value)))]
            else:
                params[key] = value
        samples.append(params)
    return samples


def _metric_value(metric: Metric, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    value = metric.function(y_true, y_pred)
    if torch.is_tensor(value):
        return utils.to_float(value)
    return float(value)


def evaluate_data_loss(model_handler, phase: str) -> float | None:
    """Evaluate average data loss for a trained ``ModelHandler`` split."""

    if not getattr(model_handler, 'has_data', False):
        return None
    data_handler = model_handler.data_fitting.data_handler
    if phase not in data_handler.n_data or data_handler.n_data[phase] == 0:
        return None

    model_handler.nn.eval()
    losses, batch_sizes = [], []
    with torch.no_grad():
        for x, y, label in data_handler(phase):
            loss_info = model_handler.data_fitting.compute_loss(
                nn=model_handler.nn,
                x=x,
                y=y,
                label=label,
                phase=phase,
                epoch=getattr(model_handler, 'epoch', None)
            )
            losses.append(utils.to_float(loss_info['total']))
            batch_sizes.append(len(x))
    return float(np.average(losses, weights=batch_sizes)) if losses else None


def _loss_phase_from_key(key: str) -> str | None:
    if not key.endswith('_loss'):
        return None
    phase = key[:-len('_loss')]
    if phase == 'validation':
        return 'valid'
    return phase if phase in {'train', 'valid', 'test'} else None


def _record_index(model_handler, prefer_best: bool) -> int | None:
    evolution = getattr(model_handler, 'evolution', None)
    if not evolution:
        return None
    if not prefer_best:
        return len(evolution) - 1
    best_epoch = getattr(model_handler, 'best_epoch', None)
    if best_epoch is None:
        return len(evolution) - 1
    for idx, record in enumerate(evolution):
        if record.get('epoch') == best_epoch:
            return idx
    return None


def _recorded_data_loss(model_handler, phase: str, prefer_best: bool = True) -> float | None:
    if phase == 'test':
        return None
    idx = _record_index(model_handler, prefer_best=prefer_best)
    if idx is None:
        return None

    record = model_handler.evolution[idx]
    if phase == 'train':
        total_key = 'train_loss'
        data_key = 'train_data_loss'
    elif phase == 'valid':
        total_key = 'validation_loss'
        data_key = 'validation_data_loss'
    else:
        total_key = f'{phase}_loss'
        data_key = f'{phase}_data_loss'
    key = data_key if getattr(model_handler, 'has_reg', False) and data_key in record else total_key
    if key not in record:
        return None
    return utils.to_float(record[key])


def collect_data_losses(model_handler, phases: Sequence[str]) -> dict[str, float | None]:
    """Collect train, validation, and/or test losses from records or evaluation."""

    losses = {'train_loss': None, 'validation_loss': None, 'test_loss': None}
    for phase in phases:
        if phase not in {'train', 'valid', 'test'}:
            raise ValueError(f"loss phase must be one of 'train', 'valid', or 'test'. {phase} was given.")
        loss = _recorded_data_loss(model_handler, phase=phase)
        if loss is None:
            loss = evaluate_data_loss(model_handler, phase)
        key = 'validation_loss' if phase == 'valid' else f'{phase}_loss'
        losses[key] = loss
    return losses


def evaluate_metrics(model_handler, metrics: Sequence[Metric], phase: str) -> dict[str, float]:
    """Evaluate custom metrics on a data split."""

    if not metrics or not getattr(model_handler, 'has_data', False):
        return {}
    data_handler = model_handler.data_fitting.data_handler
    if phase not in data_handler.n_data or data_handler.n_data[phase] == 0:
        return {}

    metric_values = {metric.name: [] for metric in metrics}
    batch_sizes = []
    model_handler.nn.eval()
    with torch.no_grad():
        for x, y, label in data_handler(phase):
            y_pred = model_handler.data_fitting.predict(
                nn=model_handler.nn,
                x=x,
                label=label,
                phase=phase,
                epoch=getattr(model_handler, 'epoch', None)
            )
            batch_sizes.append(len(x))
            for metric in metrics:
                if metric.unnormalize:
                    y_true_ = model_handler.data_fitting.to_observed_target(y)
                    y_pred_ = model_handler.data_fitting.to_observed_prediction(y_pred)
                else:
                    y_true_, y_pred_ = y, y_pred
                metric_values[metric.name].append(_metric_value(metric, y_true_, y_pred_))

    return {
        f'{phase}_{name}': float(np.average(values, weights=batch_sizes))
        for name, values in metric_values.items()
        if values
    }


def _result_from_model_handler(
        model_handler,
        context: TrialContext,
        elapsed_time: float,
        metrics: Sequence[Metric],
        metric_phases: Sequence[str],
        loss_phases: Sequence[str]
) -> TrialResult:
    metric_values = {}
    for phase in metric_phases:
        metric_values.update(evaluate_metrics(model_handler, metrics=metrics, phase=phase))

    losses = collect_data_losses(model_handler, phases=loss_phases)
    best_epoch = getattr(model_handler, 'best_epoch', None)
    if best_epoch is not None:
        best_epoch += 1

    return TrialResult(
        trial_id=context.trial_id,
        params=dict(context.params),
        status='ok',
        fold=context.fold,
        outer_fold=context.outer_fold,
        inner_fold=context.inner_fold,
        seed=context.seed,
        best_loss=utils.to_float(getattr(model_handler, 'best_loss', None)),
        train_loss=losses['train_loss'],
        validation_loss=losses['validation_loss'],
        test_loss=losses['test_loss'],
        metrics=metric_values,
        model_path=getattr(model_handler, 'model_path', None),
        n_epochs=getattr(model_handler, 'epoch', None),
        best_epoch=best_epoch,
        elapsed_time=elapsed_time,
    )


def _failed_result(context: TrialContext, elapsed_time: float, error: BaseException) -> TrialResult:
    return TrialResult(
        trial_id=context.trial_id,
        params=dict(context.params),
        status='failed',
        fold=context.fold,
        outer_fold=context.outer_fold,
        inner_fold=context.inner_fold,
        seed=context.seed,
        elapsed_time=elapsed_time,
        error=f'{error.__class__.__name__}: {error}',
        traceback=traceback.format_exc(),
    )


def results_to_dataframe(results: Sequence[TrialResult | SearchResult]) -> pd.DataFrame:
    """Convert trial or search results to a tabular DataFrame."""

    rows = []
    for result in results:
        row = result.to_dict()
        if isinstance(result, TrialResult):
            metrics = row.pop('metrics')
            row.update(metrics)
        row['params'] = json.dumps(row['params'], ensure_ascii=False)
        rows.append(row)
    return pd.DataFrame(rows)


def save_trial_results(results: Sequence[TrialResult | SearchResult], path: str | Path) -> None:
    """Save trial/search results as CSV or JSONL based on file suffix."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = results_to_dataframe(results)
    if path.suffix.lower() == '.jsonl':
        with open(path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + '\n')
    else:
        df.to_csv(path, index=False)


class CrossValidator:
    """Run one parameter setting over K-fold splits."""

    def __init__(
            self,
            model_factory: Callable[[dict[str, object], IndexSplit, TrialContext], object],
            splitter: KFoldSplitter,
            indices: int | Sequence[int] | np.ndarray,
            metrics: Sequence[Metric] = (),
            metric_phases: Sequence[str] = ('valid',),
            loss_phases: Sequence[str] = ('valid',),
            random_seed: int = 2025,
            raise_on_error: bool = True,
    ):
        self.model_factory = model_factory
        self.splitter = splitter
        self.indices = indices
        self.metrics = tuple(metrics)
        self.metric_phases = tuple(metric_phases)
        self.loss_phases = tuple(loss_phases)
        self.random_seed = random_seed
        self.raise_on_error = raise_on_error

    def run(
            self,
            params: Mapping[str, object] = None,
            param_index: int = 0,
            trial_prefix: str = 'cv',
            outer_fold: int | None = None,
            output_dir: str | Path = None,
            splits: Sequence[IndexSplit] = None,
    ) -> list[TrialResult]:
        """Fit one fresh model per fold and return fold-level results."""

        params = dict(params or {})
        output_dir = None if output_dir is None else Path(output_dir)
        splits = list(splits) if splits is not None else self.splitter.split(self.indices)
        results = []
        for fold_index, split in enumerate(splits):
            fold = split.fold if split.fold is not None else fold_index
            context = TrialContext(
                trial_id=f'{trial_prefix}_p{param_index:03d}_f{fold:03d}',
                params=params,
                split=split,
                param_index=param_index,
                fold=fold,
                outer_fold=outer_fold,
                inner_fold=fold,
                seed=self.random_seed + param_index * 10_000 + fold,
                output_dir=output_dir,
            )
            results.append(self._run_trial(context))
        return results

    def _run_trial(self, context: TrialContext) -> TrialResult:
        t0 = time.perf_counter()
        try:
            model_handler = self.model_factory(dict(context.params), context.split, context)
            model_handler()
            return _result_from_model_handler(
                model_handler=model_handler,
                context=context,
                elapsed_time=time.perf_counter() - t0,
                metrics=self.metrics,
                metric_phases=self.metric_phases,
                loss_phases=self.loss_phases,
            )
        except Exception as e:
            if self.raise_on_error:
                raise
            return _failed_result(context=context, elapsed_time=time.perf_counter() - t0, error=e)


class GridSearch:
    """Evaluate every parameter combination with cross-validation."""

    def __init__(
            self,
            model_factory: Callable[[dict[str, object], IndexSplit, TrialContext], object],
            splitter: KFoldSplitter,
            indices: int | Sequence[int] | np.ndarray,
            param_grid: Mapping[str, Iterable[object]],
            score_key: str = 'validation_loss',
            greater_is_better: bool = False,
            metrics: Sequence[Metric] = (),
            metric_phases: Sequence[str] = ('valid',),
            loss_phases: Sequence[str] = None,
            random_seed: int = 2025,
            raise_on_error: bool = True,
    ):
        self.model_factory = model_factory
        self.splitter = splitter
        self.indices = indices
        self.param_grid = param_grid
        self.score_key = score_key
        self.greater_is_better = greater_is_better
        self.metrics = tuple(metrics)
        self.metric_phases = tuple(metric_phases)
        score_phase = _loss_phase_from_key(score_key)
        if loss_phases is None:
            loss_phases = () if score_phase is None else (score_phase,)
        self.loss_phases = tuple(loss_phases)
        self.random_seed = random_seed
        self.raise_on_error = raise_on_error
        self.cv_results_: list[TrialResult] = []
        self.search_results_: list[SearchResult] = []
        self.best_result_: SearchResult | None = None

    def run(self, trial_prefix: str = 'grid', output_dir: str | Path = None) -> list[SearchResult]:
        """Run the grid search and return ranked search results."""

        return self._run_parameter_list(
            parameter_list=iter_parameter_grid(self.param_grid),
            trial_prefix=trial_prefix,
            output_dir=output_dir,
        )

    def _run_parameter_list(
            self,
            parameter_list: Sequence[dict[str, object]],
            trial_prefix: str,
            output_dir: str | Path = None,
            outer_fold: int | None = None,
            splits: Sequence[IndexSplit] = None,
    ) -> list[SearchResult]:
        self.cv_results_ = []
        self.search_results_ = []
        cv = CrossValidator(
            model_factory=self.model_factory,
            splitter=self.splitter,
            indices=self.indices,
            metrics=self.metrics,
            metric_phases=self.metric_phases,
            loss_phases=self.loss_phases,
            random_seed=self.random_seed,
            raise_on_error=self.raise_on_error,
        )
        for param_index, params in enumerate(parameter_list):
            fold_results = cv.run(
                params=params,
                param_index=param_index,
                trial_prefix=trial_prefix,
                outer_fold=outer_fold,
                output_dir=output_dir,
                splits=splits,
            )
            self.cv_results_.extend(fold_results)
            self.search_results_.append(SearchResult(
                params=dict(params),
                param_index=param_index,
                fold_results=fold_results,
                score_key=self.score_key,
                greater_is_better=self.greater_is_better,
            ))
        self._rank_results()
        return self.search_results_

    def _rank_results(self) -> None:
        scored = [
            result for result in self.search_results_
            if result.mean_score is not None
        ]
        scored.sort(key=lambda result: result.mean_score, reverse=self.greater_is_better)
        for rank, result in enumerate(scored, start=1):
            result.rank = rank
        self.best_result_ = scored[0] if scored else None


class RandomSearch(GridSearch):
    """Evaluate randomly sampled parameter settings with cross-validation."""

    def __init__(
            self,
            model_factory: Callable[[dict[str, object], IndexSplit, TrialContext], object],
            splitter: KFoldSplitter,
            indices: int | Sequence[int] | np.ndarray,
            param_space: Mapping[str, Sequence[object] | Callable[[np.random.Generator], object] | object],
            n_iter: int,
            score_key: str = 'validation_loss',
            greater_is_better: bool = False,
            metrics: Sequence[Metric] = (),
            metric_phases: Sequence[str] = ('valid',),
            loss_phases: Sequence[str] = None,
            random_seed: int = 2025,
            raise_on_error: bool = True,
    ):
        super().__init__(
            model_factory=model_factory,
            splitter=splitter,
            indices=indices,
            param_grid={},
            score_key=score_key,
            greater_is_better=greater_is_better,
            metrics=metrics,
            metric_phases=metric_phases,
            loss_phases=loss_phases,
            random_seed=random_seed,
            raise_on_error=raise_on_error,
        )
        self.param_space = param_space
        self.n_iter = n_iter

    def run(self, trial_prefix: str = 'random', output_dir: str | Path = None) -> list[SearchResult]:
        """Run random search and return ranked search results."""

        return self._run_parameter_list(
            parameter_list=sample_parameter_space(self.param_space, n_iter=self.n_iter, random_seed=self.random_seed),
            trial_prefix=trial_prefix,
            output_dir=output_dir,
        )


class NestedCrossValidator:
    """Run nested CV with inner hyper-parameter selection and outer evaluation."""

    def __init__(
            self,
            model_factory: Callable[[dict[str, object], IndexSplit, TrialContext], object],
            outer_splitter: KFoldSplitter,
            inner_splitter: KFoldSplitter,
            indices: int | Sequence[int] | np.ndarray,
            param_grid: Mapping[str, Iterable[object]],
            score_key: str = 'validation_loss',
            greater_is_better: bool = False,
            metrics: Sequence[Metric] = (),
            metric_phases: Sequence[str] = ('valid', 'test'),
            loss_phases: Sequence[str] = ('test',),
            random_seed: int = 2025,
            raise_on_error: bool = True,
    ):
        self.model_factory = model_factory
        self.outer_splitter = outer_splitter
        self.inner_splitter = inner_splitter
        self.indices = indices
        self.param_grid = param_grid
        self.score_key = score_key
        self.greater_is_better = greater_is_better
        self.metrics = tuple(metrics)
        self.metric_phases = tuple(metric_phases)
        self.loss_phases = tuple(loss_phases)
        self.random_seed = random_seed
        self.raise_on_error = raise_on_error
        self.inner_search_results_: list[SearchResult] = []
        self.inner_cv_results_: list[TrialResult] = []
        self.outer_results_: list[TrialResult] = []

    def run(self, trial_prefix: str = 'nested', output_dir: str | Path = None) -> list[TrialResult]:
        """Run nested CV and return the outer-fold evaluation results."""

        self.inner_search_results_ = []
        self.inner_cv_results_ = []
        self.outer_results_ = []
        output_dir = None if output_dir is None else Path(output_dir)

        for outer_fold, outer_split in enumerate(self.outer_splitter.split(self.indices)):
            inner_search = GridSearch(
                model_factory=self.model_factory,
                splitter=self.inner_splitter,
                indices=outer_split.train_indices,
                param_grid=self.param_grid,
                score_key=self.score_key,
                greater_is_better=self.greater_is_better,
                metrics=self.metrics,
                metric_phases=('valid',),
                loss_phases=None,
                random_seed=self.random_seed + outer_fold * 100_000,
                raise_on_error=self.raise_on_error,
            )
            inner_search._run_parameter_list(
                parameter_list=iter_parameter_grid(self.param_grid),
                trial_prefix=f'{trial_prefix}_outer{outer_fold:03d}_inner',
                output_dir=None if output_dir is None else output_dir / f'outer_{outer_fold:03d}' / 'inner',
                outer_fold=outer_fold,
            )
            self.inner_search_results_.extend(inner_search.search_results_)
            self.inner_cv_results_.extend(inner_search.cv_results_)
            if inner_search.best_result_ is None:
                continue

            final_split = IndexSplit(
                train_indices=outer_split.train_indices,
                valid_indices=None,
                test_indices=outer_split.valid_indices,
                fold=outer_fold,
            )
            context = TrialContext(
                trial_id=f'{trial_prefix}_outer{outer_fold:03d}_final',
                params=dict(inner_search.best_result_.params),
                split=final_split,
                param_index=inner_search.best_result_.param_index,
                fold=outer_fold,
                outer_fold=outer_fold,
                inner_fold=None,
                seed=self.random_seed + outer_fold,
                output_dir=None if output_dir is None else output_dir / f'outer_{outer_fold:03d}' / 'final',
            )
            self.outer_results_.append(self._run_final_trial(context))
        return self.outer_results_

    def _run_final_trial(self, context: TrialContext) -> TrialResult:
        t0 = time.perf_counter()
        try:
            model_handler = self.model_factory(dict(context.params), context.split, context)
            model_handler()
            return _result_from_model_handler(
                model_handler=model_handler,
                context=context,
                elapsed_time=time.perf_counter() - t0,
                metrics=self.metrics,
                metric_phases=self.metric_phases,
                loss_phases=self.loss_phases,
            )
        except Exception as e:
            if self.raise_on_error:
                raise
            return _failed_result(context=context, elapsed_time=time.perf_counter() - t0, error=e)
