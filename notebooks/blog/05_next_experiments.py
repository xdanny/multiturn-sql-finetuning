import marimo

__generated_with = "0.17.0"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo

    from notebooks.blog_support import (
        blog_notebook_series,
        data_engineering_gates,
        metric_dsl_eval_contract,
        prompt_optimization_findings,
        target_comparison,
    )

    return (
        blog_notebook_series,
        data_engineering_gates,
        metric_dsl_eval_contract,
        mo,
        prompt_optimization_findings,
        target_comparison,
    )


@app.cell
def _(blog_notebook_series, mo):
    current = blog_notebook_series().iloc[4]
    mo.md(
        f"""
        # Next Experiments

        The next work is not "make the model better" in the abstract. It is to
        build artifacts that can decide whether planner-first, semantic-layer,
        `MEASURE()`-preserving DSL, or behavior/recovery training is actually
        better under the same rows and scorer.

        Run command: `{current["run_command"]}`.
        """
    )
    return


@app.cell
def _(
    data_engineering_gates,
    metric_dsl_eval_contract,
    mo,
    prompt_optimization_findings,
    target_comparison,
):
    mo.vstack(
        [
            mo.ui.table(target_comparison(), label="Target gates"),
            mo.ui.table(metric_dsl_eval_contract(), label="Metric DSL contract"),
            mo.ui.table(prompt_optimization_findings(), label="DSPy boundary"),
            mo.ui.table(data_engineering_gates(), label="Data artifacts to build next"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
