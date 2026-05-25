import marimo

__generated_with = "0.17.0"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo

    from notebooks.blog_support import (
        accuracy_scorecard,
        blog_notebook_series,
        claim_table,
        endpoint_run_scorecard,
        planner_scorecard,
    )

    return (
        accuracy_scorecard,
        blog_notebook_series,
        claim_table,
        endpoint_run_scorecard,
        mo,
        planner_scorecard,
    )


@app.cell
def _(blog_notebook_series, mo):
    current = blog_notebook_series().iloc[3]
    mo.md(
        f"""
        # Results and Diagnostics

        The current numbers are useful only when the scorer and leakage boundary
        are visible. Non-oracle endpoint runs are proxy evidence; oracle-planner
        runs are ceilings that explain what better planning could unlock.

        Run command: `{current["run_command"]}`.
        """
    )
    return


@app.cell
def _(accuracy_scorecard, claim_table, endpoint_run_scorecard, mo, planner_scorecard):
    mo.vstack(
        [
            mo.ui.table(endpoint_run_scorecard(), label="Endpoint runs"),
            mo.ui.table(accuracy_scorecard(), label="Accuracy ladder"),
            mo.ui.table(planner_scorecard(), label="Planner baseline"),
            mo.ui.table(claim_table(), label="Claim boundary"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
