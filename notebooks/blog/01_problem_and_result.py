import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    from notebooks.blog_support import accuracy_scorecard, claim_table, planner_scorecard

    return accuracy_scorecard, claim_table, mo, planner_scorecard, plt


@app.cell
def _(mo):
    mo.md(
        """
        # 01 - Problem and result

        The series starts from one question:

        **Can a small specialized local model learn the behavior and semantic concepts
        needed to beat larger state-of-the-art systems on multi-turn analytical SQL?**

        This notebook keeps the first post honest. It separates the current local
        proxy evidence from the future BIRD-Interact and hosted-model claim.
        """
    )
    return


@app.cell
def _(accuracy_scorecard, mo):
    scores = accuracy_scorecard()
    mo.vstack(
        [
            mo.md("## Current accuracy ladder"),
            mo.ui.table(scores, label="Proxy and diagnostic scores"),
        ]
    )
    return (scores,)


@app.cell
def _(plt, scores):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#333333" if mode == "non_oracle_generation" else "#bdbdbd" for mode in scores["mode"]]
    ax.bar(scores["run"], scores["score"], color=colors)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title("The gap is the finding: non-oracle proxy vs oracle ceiling")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig  # noqa: B018
    return


@app.cell
def _(claim_table, mo):
    mo.vstack(
        [
            mo.md("## Claim ledger"),
            mo.ui.table(claim_table(), label="What the repo can and cannot claim today"),
        ]
    )
    return


@app.cell
def _(mo, planner_scorecard):
    planner = planner_scorecard()
    mo.vstack(
        [
            mo.md("## Planner baseline"),
            mo.ui.table(planner, label="Lexical planner scores"),
            mo.md(
                "Column F1 is the obvious weak spot. That is the first signal that direct SQL "
                "fine-tuning is not enough; the system needs an explicit planning target."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
