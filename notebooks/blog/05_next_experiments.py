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
        target_comparison,
        target_evidence_matrix,
    )

    return (
        blog_notebook_series,
        claim_table,
        data_engineering_gates,
        mo,
        target_comparison,
        target_evidence_matrix,
    )


@app.cell
def _(blog_notebook_series, mo):
    series = blog_notebook_series()
    current = series[series["notebook"] == "notebooks/blog/05_next_experiments.py"].iloc[0]
    mo.md(
        f"""
        # 5. Next experiments

        The next work is not "make the model better" in the abstract. It is to
        produce the artifacts that can rank the five targets on identical turns:
        predicted planner execution, metric-DSL comparison, generated-history
        rollout, hosted baselines, and BIRD-Interact transfer.

        **Reader action:** {current["reader_action"]}

        **Claim boundary:** {current["claim_boundary"]}
        """
    )
    return (series,)


@app.cell
def _(data_engineering_gates, mo):
    gates = data_engineering_gates()
    mo.vstack(
        [
            mo.md(
                """
                ## Blocking artifacts

                The data-engineering gates are the build list. They turn failure
                classes into artifacts: value indexes, entity-resolution labels,
                fanout fixtures, alias validation, semantic manifests, hosted
                baselines, and BIRD-Interact transfer.
                """
            ),
            mo.ui.table(gates, label="Build targets"),
        ]
    )
    return (gates,)


@app.cell
def _(claim_table, mo, target_comparison, target_evidence_matrix):
    claims = claim_table()
    targets = target_comparison()
    evidence = target_evidence_matrix()
    mo.vstack(
        [
            mo.md(
                """
                ## Decision rule

                A target becomes the best candidate only after it beats direct SQL
                on the same rows, with the same scorer, under the same oracle policy.
                Hosted and BIRD-Interact claims require their own same-protocol
                manifests.
                """
            ),
            mo.ui.table(targets, label="Target comparison"),
            mo.ui.table(evidence, label="Evidence matrix"),
            mo.ui.table(claims, label="Claim statuses"),
        ]
    )
    return claims, evidence, targets


if __name__ == "__main__":
    app.run()
