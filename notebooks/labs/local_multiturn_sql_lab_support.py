from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Accelerator:
    kind: str
    label: str
    torch_available: bool


@dataclass(frozen=True)
class LabTurn:
    turn_id: str
    question: str
    reference_sql: str
    context_note: str


def available_accelerator() -> Accelerator:
    """Return the best local accelerator without requiring torch or a GPU."""

    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return Accelerator(kind="cpu", label="cpu (torch not installed)", torch_available=False)

    try:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(index) for index in range(count)]
            return Accelerator(
                kind="cuda",
                label=f"cuda ({count} device{'s' if count != 1 else ''}: {', '.join(names)})",
                torch_available=True,
            )
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return Accelerator(kind="mps", label="mps", torch_available=True)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return Accelerator(kind="xpu", label="xpu", torch_available=True)
    except Exception as exc:
        return Accelerator(kind="cpu", label=f"cpu (accelerator probe failed: {exc})", torch_available=True)

    return Accelerator(kind="cpu", label="cpu", torch_available=True)


def lab_turns() -> list[LabTurn]:
    return [
        LabTurn(
            turn_id="turn_1",
            question="Show revenue by country.",
            reference_sql=(
                "SELECT customers.country, SUM(orders.amount) AS revenue "
                "FROM orders JOIN customers ON orders.customer_id = customers.id "
                "GROUP BY customers.country ORDER BY revenue DESC"
            ),
            context_note="Standalone metric request.",
        ),
        LabTurn(
            turn_id="turn_2",
            question="Only France.",
            reference_sql=(
                "SELECT customers.country, SUM(orders.amount) AS revenue "
                "FROM orders JOIN customers ON orders.customer_id = customers.id "
                "WHERE customers.country = 'FR' GROUP BY customers.country"
            ),
            context_note="Follow-up keeps the metric and grain, but grounds France to FR.",
        ),
        LabTurn(
            turn_id="turn_3",
            question="Which customer there spent the most?",
            reference_sql=(
                "SELECT customers.name, SUM(orders.amount) AS revenue "
                "FROM orders JOIN customers ON orders.customer_id = customers.id "
                "WHERE customers.country = 'FR' GROUP BY customers.name "
                "ORDER BY revenue DESC LIMIT 1"
            ),
            context_note="Follow-up changes grain and carries the France filter forward.",
        ),
    ]


def _connect_demo_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            country TEXT NOT NULL
        );

        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            order_date TEXT NOT NULL,
            FOREIGN KEY(customer_id) REFERENCES customers(id)
        );

        INSERT INTO customers (id, name, country) VALUES
            (1, 'Alice', 'FR'),
            (2, 'Bruno', 'FR'),
            (3, 'Cara', 'US');

        INSERT INTO orders (id, customer_id, amount, order_date) VALUES
            (1, 1, 100.0, '2026-01-01'),
            (2, 2, 25.0, '2026-01-02'),
            (3, 3, 200.0, '2026-01-03');
        """
    )
    return conn


def _rows_for_sql(conn: sqlite3.Connection, sql: str) -> list[tuple[Any, ...]]:
    return [tuple(row) for row in conn.execute(sql).fetchall()]


def _direct_sql_baseline(turn: LabTurn) -> tuple[str, str, str | None]:
    if turn.turn_id == "turn_1":
        sql = (
            "SELECT customers.country, SUM(orders.amount) AS revenue "
            "FROM orders JOIN customers ON orders.customer_id = customers.id "
            "GROUP BY customers.country ORDER BY revenue DESC"
        )
        return "direct SQL from the standalone question", sql, None
    if turn.turn_id == "turn_2":
        sql = (
            "SELECT customers.country, SUM(orders.amount) AS revenue "
            "FROM orders JOIN customers ON orders.customer_id = customers.id "
            "WHERE customers.country = 'France' GROUP BY customers.country"
        )
        return "carry metric, but copy display value France into storage SQL", sql, "value_grounding"
    sql = (
        "SELECT customers.name, SUM(orders.amount) AS revenue "
        "FROM orders JOIN customers ON orders.customer_id = customers.id "
        "GROUP BY customers.name ORDER BY revenue DESC LIMIT 1"
    )
    return "change grain, but forget the France filter", sql, "context_carryover"


def _semantic_dsl_planner(turn: LabTurn) -> tuple[str, str, str | None]:
    if turn.turn_id == "turn_1":
        plan = "MEASURE(revenue) BY customer_country ORDER BY MEASURE(revenue) DESC"
        sql = (
            "SELECT customers.country, SUM(orders.amount) AS revenue "
            "FROM orders JOIN customers ON orders.customer_id = customers.id "
            "GROUP BY customers.country ORDER BY revenue DESC"
        )
        return plan, sql, None
    if turn.turn_id == "turn_2":
        plan = "MEASURE(revenue) BY customer_country WHERE customer_country = 'FR'"
        sql = (
            "SELECT customers.country, SUM(orders.amount) AS revenue "
            "FROM orders JOIN customers ON orders.customer_id = customers.id "
            "WHERE customers.country = 'FR' GROUP BY customers.country"
        )
        return plan, sql, None
    plan = (
        "MEASURE(revenue) BY customer_name WHERE customer_country = 'FR' "
        "ORDER BY MEASURE(revenue) DESC LIMIT 1"
    )
    sql = (
        "SELECT customers.name, SUM(orders.amount) AS revenue "
        "FROM orders JOIN customers ON orders.customer_id = customers.id "
        "WHERE customers.country = 'FR' GROUP BY customers.name "
        "ORDER BY revenue DESC LIMIT 1"
    )
    return plan, sql, None


def _normalize_rows(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    normalized = []
    for row in rows:
        normalized.append(
            tuple(round(value, 8) if isinstance(value, float) else value for value in row)
        )
    return normalized


def run_multiturn_lab() -> dict[str, Any]:
    """Run a small executable multi-turn SQL lab with no external model download."""

    detected_accelerator = available_accelerator()
    conn = _connect_demo_db()
    turns = lab_turns()
    systems = {
        "direct_sql_baseline": _direct_sql_baseline,
        "semantic_dsl_planner": _semantic_dsl_planner,
    }
    rows: list[dict[str, Any]] = []

    try:
        for turn in turns:
            expected_rows = _normalize_rows(_rows_for_sql(conn, turn.reference_sql))
            for system_name, planner in systems.items():
                plan, sql, expected_failure = planner(turn)
                actual_rows = _normalize_rows(_rows_for_sql(conn, sql))
                value_match = actual_rows == expected_rows
                rows.append(
                    {
                        "turn_id": turn.turn_id,
                        "question": turn.question,
                        "context_note": turn.context_note,
                        "system": system_name,
                        "intermediate_plan": plan,
                        "sql": sql,
                        "actual_rows": actual_rows,
                        "expected_rows": expected_rows,
                        "value_match": value_match,
                        "failure_type": None if value_match else expected_failure,
                    }
                )
    finally:
        conn.close()

    summaries: dict[str, dict[str, Any]] = {}
    for system_name in systems:
        system_rows = [row for row in rows if row["system"] == system_name]
        correct = sum(1 for row in system_rows if row["value_match"])
        summaries[system_name] = {
            "correct": correct,
            "turns": len(system_rows),
            "value_accuracy": correct / len(system_rows),
        }

    return {
        "device": Accelerator(
            kind="cpu",
            label="cpu",
            torch_available=detected_accelerator.torch_available,
        ),
        "detected_accelerator": detected_accelerator,
        "turns": turns,
        "rows": rows,
        "systems": summaries,
    }
