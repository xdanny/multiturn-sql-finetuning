import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    from notebooks.blog_support import read_csv_artifact

    return mo, plt, read_csv_artifact


@app.cell
def _(mo):
    mo.md(
        """
        # 05 - vLLM and Blackwell serving

        Serving matters because the benchmark should compare behavior, not a pile
        of inconsistent inference paths. The same OpenAI-compatible endpoint path
        should serve base, LoRA, semantic, planner-first, and future adapters.
        """
    )
    return


@app.cell
def _(mo, read_csv_artifact):
    latency = read_csv_artifact("plots/rescored_vllm_semantic_prompt_iteration_100turns/summary.csv")[
        ["model_name", "prompt_variant", "value_accuracy", "mean_latency_ms", "samples"]
    ]
    mo.ui.table(latency, label="Accuracy and latency tradeoff")
    return (latency,)


@app.cell
def _(latency, plt):
    labels = latency["model_name"].where(latency["prompt_variant"].isna(), latency["model_name"] + "[" + latency["prompt_variant"].fillna("") + "]")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(latency["mean_latency_ms"], latency["value_accuracy"], color="#333333")
    for label, x, y in zip(labels, latency["mean_latency_ms"], latency["value_accuracy"], strict=False):
        ax.annotate(label, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("mean latency ms")
    ax.set_ylabel("value accuracy")
    ax.set_title("Semantic context is not free")
    fig.tight_layout()
    fig  # noqa: B018
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Serving invariant

        A future SOTA comparison should use the same endpoint contract for local
        and hosted models where possible: same prepared input, same prompt mode,
        same execution scorer, same result manifest, same latency accounting.
        """
    )
    return


if __name__ == "__main__":
    app.run()
