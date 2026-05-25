import marimo

__generated_with = "0.17.0"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo

    from notebooks.blog_support import (
        blog_notebook_series,
        lab_method_scorecard,
        metric_dsl_demo,
        metric_dsl_eval_contract,
        target_comparison,
        target_evidence_matrix,
    )

    return (
        blog_notebook_series,
        lab_method_scorecard,
        metric_dsl_demo,
        metric_dsl_eval_contract,
        mo,
        target_comparison,
        target_evidence_matrix,
    )


@app.cell
def _(blog_notebook_series, mo):
    series = blog_notebook_series()
    current = series[series["notebook"] == "notebooks/blog/03_method_targets.py"].iloc[0]
    mo.md(
        f"""
        # 3. Fine-tuning targets

        Direct SQL SFT is the control arm, not the full bet. A small model might
        need a better target: planner/DSL first then SQL, semantic-layer concepts,
        `MEASURE()` preservation, or behavior/recovery after failed turns.

        **Reader action:** {current["reader_action"]}

        **Claim boundary:** {current["claim_boundary"]}
        """
    )
    return (series,)


@app.cell
def _(lab_method_scorecard, mo, target_comparison):
    targets = target_comparison()
    lab_scores = lab_method_scorecard()
    mo.vstack(
        [
            mo.md(
                """
                ## Hypotheses before rankings

                The target table says what each method is supposed to learn. The
                lab scorecard only isolates behavior; it does not decide which
                method wins on the full benchmark.
                """
            ),
            mo.ui.table(targets, label="Fine-tuning target hypotheses"),
            mo.ui.table(lab_scores, label="Behavior isolated in the compact lab"),
        ]
    )
    return lab_scores, targets


@app.cell
def _(metric_dsl_demo, metric_dsl_eval_contract, mo):
    demo = metric_dsl_demo()
    contract = metric_dsl_eval_contract()
    mo.vstack(
        [
            mo.md(
                f"""
                ## Why `MEASURE()` matters

                A raw SQL string can be executable and still bypass governed metric
                intent. The DSL target keeps `MEASURE(revenue)` alive until a
                semantic model compiles it.

                ```sql
                {demo["compiled_sql"]}
                ```
                """
            ),
            mo.ui.table(contract, label="Metric-DSL evidence contract"),
        ]
    )
    return contract, demo


@app.cell
def _(mo, target_evidence_matrix):
    evidence = target_evidence_matrix()
    mo.vstack(
        [
            mo.md(
                """
                ## What is actually supported

                The evidence matrix connects the toy lab behavior to manifest-backed
                repo evidence. This is the guardrail against treating a notebook
                demo as a model-result leaderboard.
                """
            ),
            mo.ui.table(evidence, label="Target evidence matrix"),
        ]
    )
    return (evidence,)


if __name__ == "__main__":
    app.run()
