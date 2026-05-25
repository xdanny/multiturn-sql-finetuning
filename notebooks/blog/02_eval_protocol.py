import marimo

__generated_with = "0.17.0"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo

    from notebooks.blog_support import (
        blog_notebook_series,
        claim_table,
        data_engineering_gates,
    )

    return blog_notebook_series, claim_table, data_engineering_gates, mo


@app.cell
def _(blog_notebook_series, mo):
    current = blog_notebook_series().iloc[1]
    mo.md(
        f"""
        # Evaluation Protocol

        This section exists to keep the story honest before any result appears.
        CoSQL is the fast proxy slice, SParC is the related context benchmark,
        synthetic schema-rich SQL is for targeted fixtures, and BIRD-Interact is
        the final benchmark shape the project has not claimed yet.

        Run command: `{current["run_command"]}`.
        Attached lab: `{current["attached_lab"]}`.
        """
    )
    return


@app.cell
def _(claim_table, data_engineering_gates, mo):
    mo.vstack(
        [
            mo.md(
                """
                ## Claim Ledger

                The ledger separates supported proxy results from pending claims:
                hosted SOTA, BIRD-Interact transfer, generated-history rollouts,
                and metric-DSL superiority all need same-protocol manifests.
                """
            ),
            mo.ui.table(claim_table(), label="Public claim boundary"),
            mo.ui.table(data_engineering_gates(), label="Data engineering gates"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
