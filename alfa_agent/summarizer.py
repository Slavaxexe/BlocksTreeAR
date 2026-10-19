"""Metric differences and architecture diffs are exact computations, not LLM guesses."""

from .contracts import Change, Comparison, Delta, Finish, Score, Start
from .storage import canonical


def _flatten(value: dict, prefix="") -> dict:
    flattened = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flattened.update(_flatten(item, path))
        else:
            flattened[path] = canonical(item)
    return flattened


def architecture_diff(left: Start, right: Start) -> list[Change]:
    before = _flatten(left.hypothesis.pipeline.model_dump(mode="json"))
    after = _flatten(right.hypothesis.pipeline.model_dump(mode="json"))
    return [Change(path=path, before=before.get(path, "<absent>"), after=after.get(path, "<absent>"))
            for path in sorted(before.keys() | after.keys()) if before.get(path) != after.get(path)]


def summarize(starts: dict[str, Start], finishes: dict[str, Finish], latest: str,
              previous_best: str | None) -> Delta:
    task = starts[latest].task
    valid = [f for f in finishes.values() if f.status == "completed"]
    # Incomparable measurements are never allowed onto one leaderboard.
    for finish in valid:
        start = starts[finish.node_id]
        if (finish.metric != task.metric or finish.split_sha256 != starts[latest].validation.split_sha256
                or start.dataset_sha256 != starts[latest].dataset_sha256 or start.task != task):
            raise ValueError("Cannot compare experiments with different data, metrics or splits")
    sign = 1 if task.metric.direction == "minimize" else -1
    valid.sort(key=lambda f: (sign * f.value, f.node_id))
    leaderboard = [Score(node_id=f.node_id, value=f.value,
                         family=starts[f.node_id].hypothesis.pipeline.model.family) for f in valid]
    comparisons = []
    candidate = finishes[latest]
    if candidate.status == "completed":
        references = {starts[latest].parent_id, previous_best, "node_000"} - {None, latest}
        for reference_id in sorted(references):
            reference = finishes.get(reference_id)
            if reference and reference.status == "completed":
                delta = candidate.value - reference.value
                comparisons.append(Comparison(reference_id=reference_id, candidate_id=latest,
                                              metric_delta=delta, improvement=-sign * delta,
                                              changes=architecture_diff(starts[reference_id], starts[latest])))
    return Delta(latest_node_id=latest, best_node_id=leaderboard[0].node_id if leaderboard else None,
                 metric=task.metric, leaderboard=leaderboard, comparisons=comparisons,
                 failed_node_ids=[f.node_id for f in finishes.values() if f.status != "completed"])
