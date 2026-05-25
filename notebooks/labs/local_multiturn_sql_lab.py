import marimo

__generated_with = "0.17.0"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo

    from notebooks.blog_support import (
        accuracy_scorecard,
        claim_table,
        data_engineering_gates,
        endpoint_run_scorecard,
        lab_failure_trace,
        lab_method_scorecard,
        metric_dsl_demo,
        metric_dsl_eval_contract,
        planner_scorecard,
        prompt_optimization_findings,
        target_comparison,
        target_evidence_matrix,
    )
    from notebooks.labs.local_multiturn_sql_lab_support import run_multiturn_lab

    return (
        accuracy_scorecard,
        claim_table,
        data_engineering_gates,
        endpoint_run_scorecard,
        lab_failure_trace,
        lab_method_scorecard,
        metric_dsl_demo,
        metric_dsl_eval_contract,
        mo,
        planner_scorecard,
        prompt_optimization_findings,
        run_multiturn_lab,
        target_comparison,
        target_evidence_matrix,
    )


@app.cell
def _(mo):
    runtime_choice = mo.ui.dropdown(
        options=["cpu", "auto"],
        value="cpu",
        label="Runtime",
    )
    mo.vstack(
        [
            mo.md(
                "Choose `cpu` for the portable default. Choose `auto` to report "
                "CUDA, MPS, or XPU availability when PyTorch detects one. The "
                "SQLite lab computation remains CPU-safe."
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
def _(mo, report):
    device = report["device"]
    detected = report["detected_accelerator"]
    contract = report["scenario_contract"]
    mo.vstack(
        [
            mo.md(
                f"""
                # Local multi-turn SQL lab

                This lab is the runnable notebook attached to the post's codebase. It
                uses a tiny in-memory SQLite warehouse so the experiment is CPU-safe by
                default. The runtime selector reports CUDA, MPS, or XPU availability
                when PyTorch can see an accelerator, but this lab does not require or
                use GPU compute.

                Lab runtime: `{device.label}`. Accelerator availability: `{detected.label}`.
                Shared scenario hash: `{contract["shared_input_sha256"]}`.
                """
            ),
            mo.ui.table(
                report["accelerator_report"],
                label="CUDA/MPS/XPU status reported only",
            ),
        ]
    )
    return


@app.cell
def _(mo, report):
    sections = report["walkthrough_sections"]
    mo.vstack(
        [
            mo.md(
                """
                ## 1. Research question

                Can a small specialized model learn behavior and semantic concepts
                well enough to challenge hosted SOTA on multi-turn data analysis?
                This lab keeps that question executable before spending more GPU
                time on a larger fine-tune.
                """
            ),
            mo.ui.table(sections, label="Reader flow through the post argument"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md(
                """
                ## 2. Why single-turn SQL fails here

                A strong zero-shot model can write a valid query for one complete
                question. Multi-turn analysis adds state: the user can keep the
                metric, change the filter, change the grain, and ask for a repair
                after an empty result. Appending chat history does not guarantee the
                model preserves that state.
                """
            )
        ]
    )
    return


@app.cell
def _(mo, report):
    contract = report["scenario_contract"]
    mo.vstack(
        [
            mo.md(
                f"""
                ## 3. The proxy slice

                The repository uses a fixed 100-turn CoSQL proxy slice for endpoint
                iteration. This notebook uses a tiny four-turn SQLite scenario with
                the same kind of follow-up pressure, so the method comparison is
                runnable without downloading a model.

                Shared scenario hash: `{contract["shared_input_sha256"]}`.
                """
            )
        ]
    )
    return


@app.cell
def _(mo, report):
    matrix = report["method_matrix"]
    mo.vstack(
        [
            mo.md("## 4. Candidate fine-tuning targets"),
            mo.ui.table(matrix, label="Fine-tuning targets compared by the lab"),
        ]
    )
    return


@app.cell
def _(lab_method_scorecard, mo, target_comparison, target_evidence_matrix):
    mo.vstack(
        [
            mo.md(
                """
                ## Target scorecard

                The toy lab is not a benchmark, but it gives each candidate target
                a concrete behavior to isolate. The target-comparison table connects
                that behavior to the current repo evidence and the next gate needed
                before it can become a real method claim.
                """
            ),
            mo.ui.table(lab_method_scorecard(), label="Toy lab scores by target"),
            mo.ui.table(target_comparison(), label="Method hypotheses and next gates"),
            mo.ui.table(
                target_evidence_matrix(),
                label="Manifest-backed target evidence matrix",
            ),
        ]
    )
    return


@app.cell
def _(mo, report):
    mo.md(
        """
        ## 5. Execution trace

        The trace below separates execution correctness from the intermediate
        behaviors we need to train and score: context carryover, value grounding,
        governed metric preservation, and recovery after a failed previous turn.
        """
    )
    return


@app.cell
def _(mo, report):
    summary = [
        {"system": system, **metrics}
        for system, metrics in report["systems"].items()
    ]
    mo.vstack(
        [
            mo.md("## Value and subtask scores"),
            mo.ui.table(summary, label="Value accuracy on the four-turn lab"),
        ]
    )
    return


@app.cell
def _(mo, report):
    rows = [
        {
            "turn_id": row["turn_id"],
            "question": row["question"],
            "system": row["system"],
            "value_match": row["value_match"],
            "context_carryover": row["context_carryover"],
            "value_grounded": row["value_grounded"],
            "measure_preserved": row["measure_preserved"],
            "recovery_success": row["recovery_success"],
            "failure_type": row["failure_type"] or "",
            "actual_rows": row["actual_rows"],
            "expected_rows": row["expected_rows"],
        }
        for row in report["rows"]
    ]
    mo.vstack(
        [
            mo.md("## Behavior trace"),
            mo.ui.table(rows, label="Per-turn execution results"),
        ]
    )
    return


@app.cell
def _(mo, report):
    plans = [
        {
            "turn_id": row["turn_id"],
            "question": row["question"],
            "system": row["system"],
            "intermediate_plan": row["intermediate_plan"],
            "sql": row["sql"],
        }
        for row in report["rows"]
    ]
    mo.vstack(
        [
            mo.md("## Plans and SQL"),
            mo.ui.table(plans, label="Generated intermediate plans and SQL"),
        ]
    )
    return


@app.cell
def _(lab_failure_trace, mo):
    mo.vstack(
        [
            mo.md(
                """
                ## Failure slice

                This filtered trace keeps the misses that matter for training data:
                value grounding, context carryover, and recovery after an empty
                result. It is the bridge between the toy scenario and the data
                artifacts the repository still needs.
                """
            ),
            mo.ui.table(lab_failure_trace(), label="Selected trainable failure modes"),
        ]
    )
    return


@app.cell
def _(accuracy_scorecard, claim_table, endpoint_run_scorecard, mo, planner_scorecard):
    mo.vstack(
        [
            mo.md(
                """
                ## Endpoint and planner evidence

                These tables are the checked-in evidence behind the article's
                larger claims. The non-oracle scores are production-style proxy
                results. The oracle rows are ceilings, not deployable evidence.
                The planner table shows why a separate planning target is still
                necessary.
                """
            ),
            mo.ui.table(endpoint_run_scorecard(), label="Endpoint runs on the fixed CoSQL slice"),
            mo.ui.table(accuracy_scorecard(), label="Accuracy ladder and oracle boundary"),
            mo.ui.table(planner_scorecard(), label="Non-oracle planner baseline"),
            mo.ui.table(claim_table(), label="Claim ledger boundary"),
        ]
    )
    return


@app.cell
def _(metric_dsl_demo, metric_dsl_eval_contract, mo):
    demo = metric_dsl_demo()
    mo.vstack(
        [
            mo.md(
                f"""
                ## Metric DSL checkpoint

                A raw SQL target can return rows while erasing governed metric
                intent. The DSL target keeps `MEASURE(revenue)` until a semantic
                model compiles it to SQL.

                Compiled SQL:

                ```sql
                {demo["compiled_sql"]}
                ```
                """
            ),
            mo.ui.table(metric_dsl_eval_contract(), label="Metric DSL evaluation contract"),
        ]
    )
    return


@app.cell
def _(mo, prompt_optimization_findings):
    mo.vstack(
        [
            mo.md(
                """
                ## DSPy boundary

                Prompt search is useful as a harness, but the current evidence says
                the next useful DSPy target is the planner program, not more wording
                search for final SQL.
                """
            ),
            mo.ui.table(prompt_optimization_findings(), label="Prompt optimization findings"),
        ]
    )
    return


@app.cell
def _(data_engineering_gates, mo):
    mo.vstack(
        [
            mo.md(
                """
                ## Data engineering gates

                These are the repo artifacts that have to exist before the lab's
                method comparison can support a stronger multi-turn SQL claim.
                """
            ),
            mo.ui.table(
                data_engineering_gates(),
                label="Data engineering gates for multi-turn SQL evaluation",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 6. Boundary and next gates

        This lab is not a benchmark result. It is a compact way to inspect which
        fine-tuning target deserves the next expensive endpoint run. The next
        evidence gates are a non-oracle planner, a metric-DSL manifest, generated
        history rollouts, hosted baselines on the same protocol, and transfer to
        BIRD-Interact-style tasks.
        """
    )
    return


if __name__ == "__main__":
    app.run()
