import marimo

__generated_with = "0.17.0"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo

    from notebooks.labs.local_multiturn_sql_lab_support import run_multiturn_lab

    return mo, run_multiturn_lab


@app.cell
def _(mo):
    runtime_choice = mo.ui.dropdown(
        options=["cpu", "auto"],
        value="cpu",
        label="Runtime",
    )
    mo.vstack(
        [
            mo.md(
                "Choose `cpu` for the portable default. Choose `auto` to report "
                "CUDA, MPS, or XPU availability when PyTorch detects one. The "
                "SQLite lab computation remains CPU-safe."
            ),
            runtime_choice,
        ]
    )
    return (runtime_choice,)


@app.cell
def _(run_multiturn_lab, runtime_choice):
    report = run_multiturn_lab(device_preference=runtime_choice.value)
    return (report,)


@app.cell
def _(mo, report):
    device = report["device"]
    detected = report["detected_accelerator"]
    contract = report["scenario_contract"]
    mo.md(
        f"""
        # Local multi-turn SQL lab

        This lab is the runnable companion to the post. It uses a tiny in-memory
        SQLite warehouse so the experiment is CPU-safe by default. The runtime
        selector reports CUDA, MPS, or XPU availability when PyTorch can see an
        accelerator, but this lab does not require or use GPU compute.

        Lab runtime: `{device.label}`. Detected accelerator: `{detected.label}`.
        Shared scenario hash: `{contract["shared_input_sha256"]}`.
        """
    )
    return


@app.cell
def _(mo, report):
    matrix = report["method_matrix"]
    mo.vstack(
        [
            mo.md("## Method matrix"),
            mo.ui.table(matrix, label="Fine-tuning targets compared by the lab"),
        ]
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
            mo.md("## Value and subtask scores"),
            mo.ui.table(summary, label="Value accuracy on the four-turn lab"),
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
            "context_carryover": row["context_carryover"],
            "value_grounded": row["value_grounded"],
            "measure_preserved": row["measure_preserved"],
            "recovery_success": row["recovery_success"],
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
