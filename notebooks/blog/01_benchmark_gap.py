import marimo

__generated_with = "0.17.0"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo

    from notebooks.blog_support import blog_notebook_series, lab_failure_trace
    from notebooks.labs.local_multiturn_sql_lab_support import run_multiturn_lab

    return blog_notebook_series, lab_failure_trace, mo, run_multiturn_lab


@app.cell
def _(blog_notebook_series, mo):
    current = blog_notebook_series().iloc[0]
    mo.md(
        f"""
        # Benchmark Gap

        The post starts with one question: if zero-shot SQL is getting strong on
        BIRD-style single-turn benchmarks, why does conversational analysis still
        fail? This section notebook is the runnable first stop before looking at
        model scores.

        Attached lab: `{current["attached_lab"]}`.
        """
    )
    return


@app.cell
def _(mo, run_multiturn_lab):
    report = run_multiturn_lab(device_preference="auto")
    mo.ui.table(report["walkthrough_sections"], label="Post-to-lab walkthrough")
    return (report,)


@app.cell
def _(lab_failure_trace, mo):
    mo.vstack(
        [
            mo.md(
                """
                ## The Small Failure

                The trace keeps the failure modes the article needs to explain:
                value grounding, context carryover, and recovery after an empty
                result. These are the behaviors that a specialized model may need
                to learn separately from raw SQL syntax.
                """
            ),
            mo.ui.table(lab_failure_trace(), label="Selected multi-turn failures"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
