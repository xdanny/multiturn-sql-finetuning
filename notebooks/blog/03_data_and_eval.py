import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    from notebooks.blog_support import read_json_artifact

    return mo, pd, read_json_artifact


@app.cell
def _(mo):
    mo.md(
        """
        # 03 - Data and evaluation are the product

        The dataset cannot be one vague training pile. The evaluation needs to say
        which source each row came from, whether history is teacher-forced, and
        whether the row carries oracle labels.
        """
    )
    return


@app.cell
def _(mo, pd):
    datasets = pd.DataFrame(
        [
            ["BIRD-Interact", "target", "dynamic multi-turn benchmark for the final claim"],
            ["BIRD mini-dev", "harness", "single-turn BIRD-style SQLite debugging"],
            ["CoSQL", "proxy", "local multi-turn dialog slice used for current runs"],
            ["SParC", "training signal", "context-dependent SQL, but current mirror is flatter"],
            ["Synthetic schema-rich SQL", "training signal", "schema variety, not conversation evidence"],
        ],
        columns=["dataset", "role", "reason"],
    )
    mo.ui.table(datasets, label="Dataset roles")
    return


@app.cell
def _(mo, pd, read_json_artifact):
    manifest = read_json_artifact("data/processed/eval_cosql_dev_100.manifest.json")
    manifest_rows = pd.DataFrame(
        [
            {"field": "total_records", "value": manifest["total_records"]},
            {"field": "assistant_turns_total", "value": manifest["assistant_turns"]["total"]},
            {"field": "turn_formats", "value": manifest["turn_formats"]},
            {"field": "history_policies", "value": manifest["history_policies"]},
            {"field": "evaluation_modes", "value": manifest["evaluation_modes"]},
        ]
    )
    mo.ui.table(manifest_rows, label="Fixed CoSQL proxy manifest")
    return (manifest,)


@app.cell
def _(manifest, mo):
    teacher_forced = manifest["history_policies"].get("gold_sql_teacher_forced", 0)
    total = manifest["total_records"]
    mo.md(
        f"""
        ## Interpretation

        The current 100-turn proxy has **{teacher_forced} / {total}** rows with
        gold SQL teacher-forced history. That is useful for isolating whether the
        model can use clean context. It is not enough to prove recovery from its
        own previous mistakes.
        """
    )
    return


if __name__ == "__main__":
    app.run()
