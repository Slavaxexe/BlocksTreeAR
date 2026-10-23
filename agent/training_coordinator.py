"""Deterministic search admission, stopping and winner selection; no LLM calls."""
import hashlib
import json
import math
import statistics
import time


def candidate_fingerprint(candidate):
    return hashlib.sha256(json.dumps({'model_index': candidate['model_index'],
                                     'parameters': candidate['parameters']}, sort_keys=True).encode()).hexdigest()


class TrainingCoordinator:
    def __init__(self, *, started, time_budget, trial_timeout, max_trials,
                 patience_rounds=2, relative_min_improvement=1e-4):
        if patience_rounds < 1 or not math.isfinite(relative_min_improvement) or relative_min_improvement < 0:
            raise ValueError('Positive patience_rounds and finite nonnegative relative_min_improvement required')
        self.started, self.deadline = started, started+time_budget
        self.reserve = min(1200., time_budget*.25)
        self.search_deadline = self.deadline-self.reserve
        self.time_budget, self.trial_timeout, self.max_trials = time_budget, trial_timeout, max_trials
        self.patience, self.min_improvement = patience_rounds, relative_min_improvement
        self.best, self.best_candidate = None, None
        self.seen, self.skipped, self.durations, self.role_durations = set(), [], {}, []
        self.trial_count, self.rounds, self.stale_rounds = 0, 0, 0
        self.round_start_score, self.stop_reason = None, None
        self.direction = None

    def remaining_search(self):
        return max(0., self.search_deadline-time.monotonic())

    def stopping_reason(self):
        if self.trial_count >= self.max_trials:
            return 'max_trials'
        if self.remaining_search() <= 0:
            return 'search_time_limit'
        if self.stale_rounds >= self.patience:
            return 'no_improvement'
        return None

    def finish_round(self):
        current = self.best['score'] if self.best else None
        if self.round_start_score is not None:
            improvement = (current-self.round_start_score if self.direction == 'maximize'
                           else self.round_start_score-current) if current is not None else 0.
            threshold = self.min_improvement*max(abs(self.round_start_score), 1e-12)
            self.stale_rounds = 0 if improvement > threshold else self.stale_rounds+1
        self.round_start_score = current
        self.rounds += 1

    def planning_budget(self):
        # Leave room for at least a small actual trial after the two API calls.
        floor = min(self.trial_timeout, max(2., self.trial_timeout*.1))
        measured = max(self.role_durations[-6:], default=30.)
        requested = min(300., max(180., 4*measured))
        if self.remaining_search() < 2*measured+floor:
            return 0.
        # A real EDA handoff can require much longer than a tiny synthetic task.
        # Account for two calls and variance, while preserving the final reserve.
        # Error-repair calls are often much shorter than a full next-plan response.
        # A median of their mixed durations left the actual planner only 41s.
        return min(requested, self.remaining_search()-floor)

    def admit(self, candidate, family):
        frozen = candidate.model_dump(mode='json')
        fingerprint = candidate_fingerprint(frozen)
        if fingerprint in self.seen:
            self.skipped.append({'candidate': frozen, 'reason': 'duplicate_configuration'})
            return False
        self.seen.add(fingerprint)
        observed = self.durations.get(family, [])
        measured = statistics.median(observed[-3:])*1.25 if observed else min(self.trial_timeout, max(2., self.trial_timeout*.25))
        estimate = max(measured, candidate.estimated_seconds or 0.)
        estimate = min(self.trial_timeout, estimate)
        margin = min(2., self.time_budget*.01)
        if estimate+margin > self.remaining_search():
            self.skipped.append({'candidate': frozen, 'reason': 'insufficient_search_time',
                                 'estimated_seconds': estimate, 'remaining_search_seconds': self.remaining_search()})
            return False
        return True

    def observe(self, result, candidate, family, wall_seconds):
        self.trial_count += 1
        result['wall_duration_seconds'] = wall_seconds
        if result['status'] != 'completed':
            return False
        value = float(result['score'])
        if not math.isfinite(value):
            raise ValueError('Coordinator rejects non-finite primary score')
        self.durations.setdefault(family, []).append(wall_seconds)
        improved = self.best is None or (value > self.best['score'] if self.direction == 'maximize'
                                        else value < self.best['score'])
        if improved:
            self.best, self.best_candidate = result, candidate
        return improved

    def state(self):
        return {'role': 'deterministic_code', 'remaining_search_seconds': self.remaining_search(),
                'remaining_total_seconds': max(0., self.deadline-time.monotonic()),
                'final_reserved_seconds': self.reserve, 'trial_timeout_seconds': self.trial_timeout,
                'trials_remaining': max(0, self.max_trials-self.trial_count),
                'completed_rounds': self.rounds, 'rounds_without_improvement': self.stale_rounds,
                'patience_rounds': self.patience, 'relative_min_improvement': self.min_improvement,
                'stop_reason': self.stop_reason, 'skipped_candidates': self.skipped}
