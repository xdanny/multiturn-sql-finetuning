import marimo

__generated_with = "0.17.0"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo

    from notebooks.blog_support import (
        blog_notebook_series,
        lab_method_scorecard,
        semantic_strategy_table,
        target_comparison,
        target_evidence_matrix,
    )

    return (
        blog_notebook_series,
        lab_method_scorecard,
        mo,
        semantic_strategy_table,
        target_comparison,
        target_evidence_matrix,
    )


@app.cell
def _(blog_notebook_series, mo):
    current = blog_notebook_series().iloc[2]
    mo.md(
        f"""
        # Fine-Tuning Targets

        The project should not assume that direct SQL is the right training
        target. This section compares direct SQL, planner-first SQL,
        semantic-layer state, `MEASURE()`-preserving DSL, and behavior/recovery
        as separate hypotheses.

        Run command: `{current["run_command"]}`.
        """
    )
    return


@app.cell
def _(
    lab_method_scorecard,
    mo,
    semantic_strategy_table,
    target_comparison,
    target_evidence_matrix,
):
    mo.vstack(
        [
            mo.ui.table(semantic_strategy_table(), label="Training target strategies"),
            mo.ui.table(lab_method_scorecard(), label="Toy lab scorecard"),
            mo.ui.table(target_comparison(), label="Method hypotheses"),
            mo.ui.table(target_evidence_matrix(), label="Manifest-backed evidence"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
