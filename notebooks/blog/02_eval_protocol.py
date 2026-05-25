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
    series = blog_notebook_series()
    current = series[series["notebook"] == "notebooks/blog/02_eval_protocol.py"].iloc[0]
    mo.md(
        f"""
        # 2. Evaluation protocol

        Before comparing models, the project has to define what the numbers are
        allowed to mean. CoSQL is the current proxy slice, SParC is related
        context-dependent SQL data, synthetic SQL stresses schema shape, BIRD
        mini-dev is single-turn harness work, and BIRD-Interact is the target
        benchmark for the final interactive claim.

        **Reader action:** {current["reader_action"]}

        **Claim boundary:** {current["claim_boundary"]}
        """
    )
    return (series,)


@app.cell
def _(claim_table, mo):
    claims = claim_table()
    mo.vstack(
        [
            mo.md(
                """
                ## Claim ledger first

                The claim ledger prevents the post from turning a proxy result into
                a hosted-SOTA claim. It separates production-style proxy runs,
                oracle diagnostics, metric-DSL pending claims, rollout pending
                claims, hosted baselines, and BIRD-Interact transfer.
                """
            ),
            mo.ui.table(claims, label="Claim ledger slice"),
        ]
    )
    return (claims,)


@app.cell
def _(data_engineering_gates, mo):
    gates = data_engineering_gates()
    mo.vstack(
        [
            mo.md(
                """
                ## Data engineering gates

                Multi-turn SQL evaluation is a data-engineering problem before it
                is a leaderboard problem. The same slice, scorer, oracle policy,
                value normalization, fanout fixtures, and hosted baseline protocol
                must hold across methods.
                """
            ),
            mo.ui.table(gates, label="Current gates and blockers"),
        ]
    )
    return (gates,)


if __name__ == "__main__":
    app.run()
