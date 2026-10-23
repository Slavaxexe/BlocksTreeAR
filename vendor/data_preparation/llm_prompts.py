"""Prompt templates for Qwen LLM integration."""

SYSTEM_PROMPT = """You are a data science assistant analyzing tabular data for a regression task.
You will receive a data profile with measured statistics. Your role:
1. Interpret the profile (do NOT invent statistics — use only what is provided).
2. Suggest useful new features from supported operations.
3. Suggest columns to drop if clearly unhelpful.

Respond ONLY with valid JSON matching this schema:
{
  "summary": "1-2 sentence data summary",
  "key_observations": ["observation1", ...],
  "suggested_features": [
    {
      "name": "feature_name",
      "source_columns": ["col1", "col2"],
      "transform_type": "<one of: interaction, ratio, log_transform, sqrt_transform, polynomial, missing_indicator>",
      "definition": "how to compute it",
      "rationale": "why it helps",
      "missing_handling": "what to do with NaN",
      "requires_fit": false
    }
  ],
  "suggested_drops": [
    {"column": "col_name", "reason": "why drop"}
  ],
  "suggested_transforms": []
}

Rules:
- Use ONLY columns listed in the profile.
- transform_type must be one of: interaction, ratio, log_transform, sqrt_transform, polynomial, missing_indicator, passthrough, drop.
- Do NOT suggest target encoding or any transform using the target variable.
- Suggest at most 5 new features.
- Be concise. No markdown, no explanation outside JSON.
- Do NOT wrap JSON in code blocks."""


def build_profile_prompt(
    task_description: str | None,
    target_col: str,
    metric: str,
    column_summaries: list[dict],
    target_stats: dict,
    n_rows_train: int,
    n_rows_test: int,
) -> str:
    parts = []
    parts.append(f"Task: Regression, predict '{target_col}', metric: {metric}")
    parts.append(f"Train rows: {n_rows_train}, Test rows: {n_rows_test}")

    if task_description:
        parts.append(f"\nTask description (first 500 chars):\n{task_description[:500]}")

    parts.append(f"\nTarget '{target_col}' stats: {_format_stats(target_stats)}")

    parts.append(f"\nColumns ({len(column_summaries)} total):")
    for cs in column_summaries[:40]:
        parts.append(f"  - {cs['name']} ({cs['dtype']}): {_format_col_summary(cs)}")

    return "\n".join(parts)


def _format_stats(stats: dict) -> str:
    keys = ["min", "max", "mean", "median", "std", "n_missing", "n_zeros"]
    items = [f"{k}={stats.get(k, '?')}" for k in keys if stats.get(k) is not None]
    return ", ".join(items)


def _format_col_summary(cs: dict) -> str:
    parts = []
    if cs.get("n_missing", 0) > 0:
        parts.append(f"missing={cs['n_missing']}")
    if cs.get("n_unique") is not None:
        parts.append(f"unique={cs['n_unique']}")
    if cs.get("min") is not None:
        parts.append(f"range=[{cs['min']}, {cs['max']}]")
    if cs.get("top_values"):
        top = cs["top_values"][:3]
        parts.append(f"top={[v['value'] for v in top]}")
    return ", ".join(parts) if parts else "no details"
