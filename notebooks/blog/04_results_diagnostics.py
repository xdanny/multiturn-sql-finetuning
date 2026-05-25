import marimo

__generated_with = "0.17.0"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo

    from notebooks.blog_support import (
        accuracy_scorecard,
        blog_notebook_series,
        endpoint_run_scorecard,
        planner_scorecard,
        prompt_optimization_findings,
    )

    return (
        accuracy_scorecard,
        blog_notebook_series,
        endpoint_run_scorecard,
        mo,
        planner_scorecard,
        prompt_optimization_findings,
    )


@app.cell
def _(blog_notebook_series, mo):
    series = blog_notebook_series()
    current = series[series["notebook"] == "notebooks/blog/04_results_diagnostics.py"].iloc[0]
    mo.md(
        f"""
        # 4. Results and diagnostics

        The current endpoint evidence says the loop can move a local model, not
        that it has beaten hosted SOTA. The key numbers are a non-oracle proxy
        result, an oracle diagnostic ceiling, and a weak but measurable planner
        baseline.

        **Reader action:** {current["reader_action"]}

        **Claim boundary:** {current["claim_boundary"]}
        """
    )
    return (series,)


@app.cell
def _(accuracy_scorecard, endpoint_run_scorecard, mo):
    ladder = accuracy_scorecard()
    endpoint_runs = endpoint_run_scorecard()
    mo.vstack(
        [
            mo.md(
                """
                ## Control arm and scorer

                Direct SQL SFT is the baseline target. Strict execution accuracy
                captures label-sensitive SQL equivalence; value-aware execution
                keeps duplicate and order sensitivity while ignoring harmless alias
                differences.
                """
            ),
            mo.ui.table(endpoint_runs, label="Endpoint run scorecard"),
            mo.ui.table(ladder, label="Proxy accuracy ladder"),
        ]
    )
    return endpoint_runs, ladder


@app.cell
def _(mo, planner_scorecard):
    planner = planner_scorecard()
    mo.vstack(
        [
            mo.md(
                """
                ## Oracle ceiling, non-oracle floor

                The oracle run shows that SQL generation improves sharply when
                planning labels are already correct. The lexical planner shows the
                non-oracle floor is still weak. The next result has to be predicted
                planner SQL execution, compared against direct SQL on the same rows.
                """
            ),
            mo.ui.table(planner, label="Lexical planner baseline"),
        ]
    )
    return (planner,)


@app.cell
def _(mo, prompt_optimization_findings):
    prompt_findings = prompt_optimization_findings()
    mo.vstack(
        [
            mo.md(
                """
                ## DSPy boundary

                Prompt search is useful as a harness, but the current evidence does
                not say longer SQL prompts are the next best bet. The more rigorous
                DSPy target is the planner or semantic program.
                """
            ),
            mo.ui.table(prompt_findings, label="Prompt optimization findings"),
        ]
    )
    return (prompt_findings,)


if __name__ == "__main__":
    app.run()
