import marimo

__generated_with = "0.17.0"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo

    from notebooks.labs.local_multiturn_sql_lab_support import run_multiturn_lab

    return mo, run_multiturn_lab


@app.cell
def _(run_multiturn_lab):
    report = run_multiturn_lab()
    return (report,)


@app.cell
def _(mo, report):
    device = report["device"]
    detected = report["detected_accelerator"]
    mo.md(
        f"""
        # Local multi-turn SQL lab

        This lab is the runnable companion to the post. It uses a tiny in-memory
        SQLite warehouse so the experiment defaults to CPU while reporting CUDA,
        MPS, or XPU when PyTorch can see an accelerator.

        Lab runtime: `{device.label}`. Detected accelerator: `{detected.label}`.
        """
    )
    return


@app.cell
def _(mo, report):
    summary = [
        {"system": system, **metrics}
        for system, metrics in report["systems"].items()
    ]
    mo.vstack(
        [
            mo.md("## Direct SQL versus semantic plan"),
            mo.ui.table(summary, label="Value accuracy on the three-turn lab"),
        ]
    )
    return


@app.cell
def _(mo, report):
    rows = [
        {
            "turn_id": row["turn_id"],
            "question": row["question"],
            "system": row["system"],
            "value_match": row["value_match"],
            "failure_type": row["failure_type"] or "",
            "actual_rows": row["actual_rows"],
            "expected_rows": row["expected_rows"],
        }
        for row in report["rows"]
    ]
    mo.vstack(
        [
            mo.md("## Behavior trace"),
            mo.ui.table(rows, label="Per-turn execution results"),
        ]
    )
    return


@app.cell
def _(mo, report):
    plans = [
        {
            "turn_id": row["turn_id"],
            "question": row["question"],
            "system": row["system"],
            "intermediate_plan": row["intermediate_plan"],
            "sql": row["sql"],
        }
        for row in report["rows"]
    ]
    mo.vstack(
        [
            mo.md("## Plans and SQL"),
            mo.ui.table(plans, label="Generated intermediate plans and SQL"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
