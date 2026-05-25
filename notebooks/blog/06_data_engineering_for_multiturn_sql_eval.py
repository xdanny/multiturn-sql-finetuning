import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    from notebooks.blog_support import (
        metric_dsl_demo,
        metric_dsl_eval_contract,
        planner_scorecard,
        read_csv_artifact,
    )

    return metric_dsl_demo, metric_dsl_eval_contract, mo, pd, planner_scorecard, read_csv_artifact


@app.cell
def _(mo):
    mo.md(
        """
        # 06 - Data engineering for multi-turn SQL evals

        This notebook turns the vague "make SQL better" goal into typed artifacts:
        planner labels, semantic models, value indexes, MEASURE()-preserving metric
        intent, and adversarial eval fixtures.
        """
    )
    return


@app.cell
def _(mo, planner_scorecard):
    mo.ui.table(planner_scorecard(), label="Planner metrics that need to improve before SQL")
    return


@app.cell
def _(mo, pd):
    artifacts = pd.DataFrame(
        [
            {
                "artifact": "semantic_model",
                "fields": "entities, dimensions, measures, grain, joins",
                "why": "prevents the model from inventing business meaning from column names",
            },
            {
                "artifact": "metric_dsl",
                "fields": "MEASURE(name), dimensions, filters, time grain",
                "why": "lets the model preserve governed metrics before SQL compilation",
            },
            {
                "artifact": "resolved_question",
                "fields": "standalone version of each follow-up turn",
                "why": "separates context resolution from SQL generation",
            },
            {
                "artifact": "value_index",
                "fields": "aliases, casing, abbreviations, storage values",
                "why": "turns value grounding into a measurable retrieval problem",
            },
            {
                "artifact": "join_fanout_fixtures",
                "fields": "duplicating joins, bridge tables, grain traps",
                "why": "catches wrong answers that still execute successfully",
            },
        ]
    )
    mo.ui.table(artifacts, label="Next data artifacts")
    return


@app.cell
def _(metric_dsl_demo, mo):
    demo = metric_dsl_demo()
    mo.vstack(
        [
            mo.md("## MEASURE() preservation as an eval target"),
            mo.md(
                "The model should be able to emit `MEASURE(revenue)` as semantic intent, "
                "then let the semantic model expand that metric at the SQL boundary."
            ),
            mo.md(f"```sql\n{demo['compiled_sql']}\n```"),
        ]
    )
    return


@app.cell
def _(metric_dsl_eval_contract, mo):
    mo.ui.table(metric_dsl_eval_contract(), label="Metric-DSL evaluator outputs")
    return


@app.cell
def _(mo, read_csv_artifact):
    taxonomy = read_csv_artifact("plots/failure_taxonomy/schema_pruned_trained100.csv")
    mo.vstack(
        [
            mo.md("## Remaining failures after oracle-labelled training"),
            mo.ui.table(taxonomy, label="Failure taxonomy"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
