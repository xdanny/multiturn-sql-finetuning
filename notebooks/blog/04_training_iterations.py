import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    from notebooks.blog_support import (
        metric_dsl_demo,
        metric_dsl_eval_contract,
        read_csv_artifact,
        semantic_strategy_table,
    )

    return metric_dsl_demo, metric_dsl_eval_contract, mo, plt, read_csv_artifact, semantic_strategy_table


@app.cell
def _(mo):
    mo.md(
        """
        # 04 - Training iterations

        The key question is not "did LoRA move one number?" It is which training
        target is worth scaling: direct SQL, planner/DSL first, semantic-layer
        concepts, MEASURE()-preserving metric queries, or behavior/recovery tuning.
        """
    )
    return


@app.cell
def _(mo, semantic_strategy_table):
    mo.ui.table(semantic_strategy_table(), label="Candidate fine-tuning strategies")
    return


@app.cell
def _(metric_dsl_demo, mo):
    demo = metric_dsl_demo()
    mo.vstack(
        [
            mo.md("## DSL-first metric example"),
            mo.md(f"`{demo['gold_query']}`"),
            mo.md("Compiled SQL after semantic validation:"),
            mo.md(f"```sql\n{demo['compiled_sql']}\n```"),
            mo.md(
                "A raw `SUM(orders.amount)` prediction gets "
                f"`measure_preservation={demo['raw_sql_like_score']['measure_preservation']}` "
                "because it skipped the governed `MEASURE(revenue)` token."
            ),
        ]
    )
    return


@app.cell
def _(metric_dsl_eval_contract, mo):
    mo.ui.table(metric_dsl_eval_contract(), label="Metric-DSL manifest gate")
    return


@app.cell
def _(read_csv_artifact):
    strict_runs = read_csv_artifact("plots/vllm_iterations_100turns/summary.csv")
    value_runs = read_csv_artifact("plots/rescored_vllm_semantic_prompt_iteration_100turns/summary.csv")
    return strict_runs, value_runs


@app.cell
def _(mo, strict_runs, value_runs):
    mo.vstack(
        [
            mo.md("## Strict execution runs"),
            mo.ui.table(strict_runs, label="Strict-era 100-turn endpoint runs"),
            mo.md("## Value-aware rescoring"),
            mo.ui.table(value_runs, label="Strict and value-aware scores"),
        ]
    )
    return


@app.cell
def _(plt, value_runs):
    labels = value_runs["model_name"].where(value_runs["prompt_variant"].isna(), value_runs["model_name"] + "[" + value_runs["prompt_variant"].fillna("") + "]")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, value_runs["value_accuracy"], color="#444444")
    ax.set_ylim(0, 1)
    ax.set_ylabel("value accuracy")
    ax.set_title("Best current non-oracle score is a proxy, not the final claim")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig  # noqa: B018
    return


if __name__ == "__main__":
    app.run()
