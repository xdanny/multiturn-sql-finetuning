"""
RAGAS-based evaluation metrics for SQL generation (single-turn + multi-turn).

Uses:
  - DataCompyScore: execution-based DataFrame comparison (primary signal)
  - SQLSemanticEquivalence: LLM-based semantic equivalence (tiebreaker)
  - ToolCallAccuracy: multi-turn sequence correctness
  - AgentGoalAccuracyWithReference: dialog-level success

TODO: implement once data/prepare.py and train/finetune.py produce outputs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SqlEvalResult:
    datacompy_score: float          # 0-1, primary signal
    semantic_equivalent: bool       # tiebreaker
    syntax_valid: bool              # sanity check
    latency_ms: float
    input_tokens: int
    output_tokens: int


def score_single_turn(
    question: str,
    schema: str,
    reference_sql: str,
    generated_sql: str,
    database_path: str,
) -> SqlEvalResult:
    """Evaluate a single-turn SQL generation.

    1. Parse generated SQL (sqlparse/sqlglot) → syntax_valid
    2. Execute both queries → compare DataFrames via datacompy → datacompy_score
    3. If execution fails or results differ, run SQLSemanticEquivalence as tiebreaker
    """
    raise NotImplementedError("TODO")


def score_multi_turn(
    dialog_turns: list[dict],
    reference_tool_calls: list[dict],
    database_path: str,
) -> dict:
    """Evaluate a multi-turn SQL dialog.

    Returns:
        - tool_call_accuracy: RAGAS ToolCallAccuracy over the sequence
        - agent_goal_accuracy: RAGAS AgentGoalAccuracyWithReference
        - interaction_match: all turns correct (CoSQL-style)
        - per_turn_datacompy: list of per-turn execution accuracy scores
    """
    raise NotImplementedError("TODO")
