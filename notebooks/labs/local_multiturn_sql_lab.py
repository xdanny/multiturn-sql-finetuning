import marimo

__generated_with = "0.17.0"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    from notebooks.labs.local_multiturn_sql_lab_support import (
        lab_turns,
        run_multiturn_lab,
    )

    return lab_turns, mo, pd, run_multiturn_lab


@app.cell
def _(mo):
    runtime_choice = mo.ui.dropdown(
        options=["auto", "cpu", "cuda", "mps", "xpu"],
        value="auto",
        label="Runtime",
    )
    mo.vstack(
        [
            mo.md(
                """
                # Local multi-turn SQL lab

                This is the runnable companion to the blog post. It starts from
                the same question as the project: single-turn SQL benchmark
                performance is improving, but multi-turn data analysis still
                fails when state, semantic concepts, metric intent, and recovery
                matter.

                ## 1. Runtime

                The lab uses `auto` by default: auto-select CUDA, MPS, or XPU
                when PyTorch detects one, otherwise fall back to CPU. Force CPU
                when you want the most portable run.
                """
            ),
            runtime_choice,
        ]
    )
    return (runtime_choice,)


@app.cell
def _(run_multiturn_lab, runtime_choice):
    report = run_multiturn_lab(device_preference=runtime_choice.value)
    return (report,)


@app.cell
def _(mo, pd, report):
    device = report["device"]
    detected = report["detected_accelerator"]
    runtime_summary = pd.DataFrame(
        [
            {
                "requested_device": report["runtime_policy"]["device_preference"],
                "selected_device": device.kind,
                "selected_label": device.label,
                "detected_accelerator": detected.label,
                "fallback": report["runtime_policy"]["fallback"] or "none",
                "scenario_hash": report["scenario_contract"]["shared_input_sha256"],
            }
        ]
    )
    mo.vstack(
        [
            mo.ui.table(runtime_summary, label="Runtime selected by the lab"),
            mo.ui.table(
                report["accelerator_report"],
                label="CUDA/MPS/XPU visibility",
            ),
        ]
    )
    return (runtime_summary,)


@app.cell
def _(lab_turns, mo, pd):
    turns = pd.DataFrame(
        [
            {
                "turn_id": turn.turn_id,
                "question": turn.question,
                "context_note": turn.context_note,
                "requires_recovery": turn.requires_recovery,
                "reference_sql": turn.reference_sql,
            }
            for turn in lab_turns()
        ]
    )
    mo.vstack(
        [
            mo.md(
                """
                ## 2. Multi-turn task

                The toy warehouse has customers and orders. The four turns keep
                the metric alive, add a follow-up filter, change the grain, and
                ask the system to recover after an empty result. This is small on
                purpose: the behavior is visible without a model download.
                """
            ),
            mo.ui.table(turns, label="Conversation turns and reference SQL"),
        ]
    )
    return (turns,)


@app.cell
def _(mo, pd, report):
    method_matrix = pd.DataFrame(report["method_matrix"])
    mo.vstack(
        [
            mo.md(
                """
                ## 3. Candidate training targets

                The lab compares direct SQL with four richer targets:
                planner-first SQL, semantic value grounding, a `MEASURE()`-
                preserving DSL, and behavior/recovery tuning. The goal is to
                decide what a small specialized model should learn before
                spending larger GPU time.
                """
            ),
            mo.ui.table(method_matrix, label="Training targets compared in the lab"),
        ]
    )
    return (method_matrix,)


@app.cell
def _(mo, pd, report):
    scores = pd.DataFrame(
        [
            {"system": system, **metrics}
            for system, metrics in report["systems"].items()
        ]
    ).sort_values(["value_accuracy", "measure_preservation_rate"], ascending=False)
    mo.vstack(
        [
            mo.md(
                """
                ## 4. Lab scorecard

                `value_accuracy` says whether the returned rows match. The other
                columns separate why a system got there: context carryover, value
                grounding, metric preservation, and recovery.
                """
            ),
            mo.ui.table(scores, label="Value and subtask scores"),
        ]
    )
    return (scores,)


@app.cell
def _(mo, pd, report):
    trace_columns = [
        "turn_id",
        "question",
        "system",
        "value_match",
        "context_carryover",
        "value_grounded",
        "measure_preserved",
        "recovery_success",
        "failure_type",
        "actual_rows",
        "expected_rows",
    ]
    trace = pd.DataFrame(report["rows"])
    failures = trace.loc[~trace["value_match"], trace_columns]
    mo.vstack(
        [
            mo.md(
                """
                ## 5. Failure trace

                Multi-turn SQL failures should not be collapsed into "bad SQL."
                A miss can come from forgotten context, an ungrounded value, lost
                metric semantics, or a missing repair behavior.
                """
            ),
            mo.ui.table(trace[trace_columns], label="All turn-level outcomes"),
            mo.ui.table(failures, label="Rows that expose trainable failures"),
        ]
    )
    return failures, trace


@app.cell
def _(mo, trace):
    plans = trace[
        [
            "turn_id",
            "question",
            "system",
            "intermediate_plan",
            "sql",
        ]
    ]
    mo.vstack(
        [
            mo.md(
                """
                ## 6. Intermediate state

                This is the part a pure SQL target hides. Planner-first training
                makes filters and grain explicit. Semantic grounding maps display
                values to storage values. A DSL keeps `MEASURE()` intent alive
                until compilation. Recovery tuning uses feedback from the last
                turn instead of blindly retrying the same query.
                """
            ),
            mo.ui.table(plans, label="Intermediate plans and SQL"),
        ]
    )
    return (plans,)


@app.cell
def _(mo, pd, report):
    fixtures = pd.DataFrame(report["synthetic_fixture_table"])
    mo.vstack(
        [
            mo.md(
                """
                ## 7. Synthetic fixture pack

                The tiny lab shows the behavior. The synthetic fixture pack turns
                those behaviors into versioned rows the repo can use before a
                larger endpoint run: value normalization, entity resolution,
                grain/fanout, `MEASURE()` preservation, and recovery.
                """
            ),
            mo.ui.table(fixtures, label="Synthetic fixtures promoted from the lab"),
        ]
    )
    return (fixtures,)


@app.cell
def _(mo, pd):
    next_gates = pd.DataFrame(
        [
            {
                "target": "planner-first SQL",
                "next_gate": (
                    "Predict non-oracle plans, then compare generated SQL against "
                    "same-model direct SQL on identical rows."
                ),
            },
            {
                "target": "semantic layer concepts",
                "next_gate": (
                    "Build value/entity indexes and score value normalization, "
                    "grain, joins, and governed dimensions separately."
                ),
            },
            {
                "target": "DSL first, SQL after",
                "next_gate": (
                    "Train DSL-preserving outputs, compile them to SQL, then compare "
                    "compiled execution against direct SQL."
                ),
            },
            {
                "target": "MEASURE() preservation",
                "next_gate": (
                    "Track whether governed metrics survive generation before SQL "
                    "expansion, especially on metric-heavy tasks."
                ),
            },
            {
                "target": "behavior/recovery",
                "next_gate": (
                    "Evaluate generated-history rollouts so the model has to recover "
                    "from its own previous misses."
                ),
            },
        ]
    )
    mo.vstack(
        [
            mo.md(
                """
                ## 8. What this proves

                This lab is not a benchmark result and it is not a hosted-SOTA
                comparison. It is a small executable argument for what the repo
                should measure next: whether a small local model can learn the
                behavior and semantic concepts that multi-turn data analysis
                requires.
                """
            ),
            mo.ui.table(next_gates, label="Next evidence gates"),
        ]
    )
    return (next_gates,)


if __name__ == "__main__":
    app.run()
