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
    series = blog_notebook_series()
    current = series[series["notebook"] == "notebooks/blog/01_benchmark_gap.py"].iloc[0]
    mo.vstack(
        [
            mo.md(
                f"""
                # 1. Benchmark gap

                Single-turn BIRD-style SQL success does not prove multi-turn data
                analysis ability. A one-shot benchmark gives the model a complete
                question. A real analyst changes filters, grain, and repair intent
                across turns.

                **Reader action:** {current["reader_action"]}

                **Claim boundary:** {current["claim_boundary"]}
                """
            ),
            mo.ui.table(series, label="Notebook path through the post"),
        ]
    )
    return (series,)


@app.cell
def _(mo, run_multiturn_lab):
    report = run_multiturn_lab(device_preference="cpu")
    mo.vstack(
        [
            mo.md(
                """
                ## The failing move

                The compact SQLite lab starts with a query that can be answered in
                isolation, then asks follow-ups that depend on state. The direct-SQL
                baseline fails when it has to preserve the metric, normalize
                `France -> FR`, change grain, and recover from an empty result.
                """
            ),
            mo.ui.table(report["scenario_turns"], label="Four-turn analysis scenario"),
        ]
    )
    return (report,)


@app.cell
def _(lab_failure_trace, mo):
    trace = lab_failure_trace()
    mo.vstack(
        [
            mo.md(
                """
                ## What breaks first

                These are not syntax errors. They are data-analysis errors:
                value grounding, context carryover, and recovery. That is why the
                project tests behavior and semantic concepts instead of only asking
                whether a model can write valid SQL.
                """
            ),
            mo.ui.table(trace, label="Failure trace from the attached lab"),
        ]
    )
    return (trace,)


if __name__ == "__main__":
    app.run()
