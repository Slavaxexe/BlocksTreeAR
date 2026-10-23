"""Separate stateless Qwen roles. Neither role owns scoring or publication."""
import json

from .training_contracts import TrainingAnalysis, TrainingNextPlan, TrainingPlan, ValidationDecision


def trial_summaries(trials):
    keys = ('trial_id', 'status', 'candidate', 'family', 'score', 'primary_metric',
            'training_score', 'inner_cv_score', 'inner_fold_scores', 'actual_parameters',
            'actual_training_rows', 'outer_validation_rows', 'feature_columns', 'excluded_columns',
            'wall_duration_seconds', 'parameter_warnings', 'error', 'diagnostic')
    return [{key: result[key] for key in keys if key in result} for result in trials]


class TrainingAnalyst:
    def __init__(self, agents, rules):
        self.agents, self.rules = agents, rules

    async def analyze(self, context, trials, winner, *, failed_trial=None, final=False):
        instruction = (
            'You are the ANALYST, not the planner or coordinator. Analyze recorded measurements only. '
            'Put numerical facts in observations, possible explanations in hypotheses and limitations '
            'in risks. Never emit candidate parameters, executable code or a new model-selection plan. '
            'Use actual feature_columns and actual_parameters from worker results where available. '
            'A train/validation gap is a warning, not proof of causality or proof about hidden-test quality. '
            'recommendation is advisory: the coordinator controls the timer and selects the winner. ')
        if failed_trial is not None:
            instruction += 'Diagnose the supplied failure/traceback. Explain what is confirmed and whether parameter repair is possible. '
        if final:
            instruction += 'Final review only: summarize the measured winner and uncertainty; recommend stop. No further experiments. '
        payload = {**context, 'results': trial_summaries(trials), 'winner': winner,
                   'failed_trial': trial_summaries([failed_trial])[0] if failed_trial else None}
        return await self.agents.ask(TrainingAnalysis, self.rules+instruction+'\n'+
            json.dumps(payload, ensure_ascii=False),
            'training_analyst_final' if final else 'training_analyst_error' if failed_trial else 'training_analyst')


class TrainingPlanner:
    def __init__(self, agents, rules):
        self.agents, self.rules = agents, rules

    async def validation(self, context):
        return await self.agents.ask(ValidationDecision, self.rules+
            'You are the PLANNER. Justify the training/validation ratio and need for internal CV '
            'versus holdout. Upstream group/time constraints are mandatory. Account for CPU/time '
            'and adapter limitations in model_templates.\n'+json.dumps(context, ensure_ascii=False),
            'training_planner_validation')

    async def initial(self, context):
        return await self.agents.ask(TrainingPlan, self.rules+
            'You are the PLANNER. Propose a SMALL executable set of initial trials within the '
            'upstream-selected families. Start with ONE configuration per model_index; later parameter '
            'variations must use the analyst results. Use model_index into selected_models. Cover each upstream '
            'model or explain its numeric index in skipped_models. Justify regularization/optimizer '
            'and supply estimated_seconds for the complete trial including CV and process startup. '
            'Use actual template/estimator parameters. Estimates must fit the coordinator budget.\n'+
            json.dumps(context, ensure_ascii=False), 'training_planner_initial')

    async def next(self, context, analysis, trials, winner, *, failed_trial=None):
        instruction = (
            'You are the PLANNER, distinct from the analyst. Use the supplied analyst_report to '
            'propose a small set of testable next hypotheses and parameter configurations, or stop. '
            'Do not claim a proposed configuration has already improved a score. Supply estimated_seconds '
            'for each complete trial, including CV and startup. Keep metric, split, data and selected '
            'families fixed. Do not repeat previously attempted or skipped configurations. '
            'The analyst recommendation is advice; justify your decision using measured facts and time. ')
        if failed_trial is not None:
            instruction += (
                'Repair mode: return at most ONE corrected candidate for the SAME model_index as '
                'failed_trial, or stop=true if supported parameters cannot fix it. Do not edit backend code. ')
        payload = {**context, 'analyst_report': analysis.model_dump(mode='json'),
                   'results': trial_summaries(trials), 'winner': winner,
                   'failed_trial': trial_summaries([failed_trial])[0] if failed_trial else None}
        return await self.agents.ask(TrainingNextPlan, self.rules+instruction+'\n'+
            json.dumps(payload, ensure_ascii=False), 'training_planner_repair' if failed_trial else 'training_planner_next')
