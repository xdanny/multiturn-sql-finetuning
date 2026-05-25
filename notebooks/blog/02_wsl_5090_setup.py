import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    return mo, pd


@app.cell
def _(mo):
    mo.md(
        """
        # 02 - Local training and serving setup

        The hardware post is not a detour. It defines whether the research loop
        can run cheaply and repeatedly enough to compare many fine-tuning methods:
        direct SQL, planner-first, semantic-layer-first, and behavior tuning.
        """
    )
    return


@app.cell
def _(mo, pd):
    setup = pd.DataFrame(
        [
            {
                "layer": "training",
                "tooling": "Unsloth + TRL SFTTrainer",
                "why it matters": "Cheap LoRA iteration on a local 9B model",
            },
            {
                "layer": "serving",
                "tooling": "vLLM OpenAI-compatible endpoint",
                "why it matters": "Same eval path for base, LoRA, and future adapters",
            },
            {
                "layer": "evaluation",
                "tooling": "SQLite execution plus strict/value scorers",
                "why it matters": "SQL answers can be checked instead of hand-waved",
            },
            {
                "layer": "observability",
                "tooling": "result manifests, hashes, latency, failure labels",
                "why it matters": "Prevents benchmark stories without rerunnable evidence",
            },
        ]
    )
    mo.ui.table(setup, label="Local loop contract")
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## What the setup has to preserve

        - One local endpoint path for base and adapters.
        - No hidden oracle planning hints in production-style runs.
        - Separate training and serving environments when Blackwell/vLLM constraints require it.
        - Command-level evidence for every number used in the blog series.
        """
    )
    return


if __name__ == "__main__":
    app.run()
