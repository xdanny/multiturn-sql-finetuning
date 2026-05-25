from __future__ import annotations

import hashlib
import json
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
    requires_recovery: bool = False


@dataclass(frozen=True)
class MethodOutput:
    intermediate_plan: str
    sql: str
    expected_failure: str | None
    context_carryover: bool
    value_grounded: bool
    measure_preserved: bool
    recovery_success: bool = False


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
        LabTurn(
            turn_id="turn_4",
            question="That returned no rows. Repair it and show the top customer there.",
            reference_sql=(
                "SELECT customers.name, SUM(orders.amount) AS revenue "
                "FROM orders JOIN customers ON orders.customer_id = customers.id "
                "WHERE customers.country = 'FR' GROUP BY customers.name "
                "ORDER BY revenue DESC LIMIT 1"
            ),
            context_note=(
                "Recovery turn: previous SQL used the display value France, returned no "
                "rows, and needs a value-grounding repair."
            ),
            requires_recovery=True,
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


def _direct_sql_baseline(turn: LabTurn) -> MethodOutput:
    if turn.turn_id == "turn_1":
        sql = (
            "SELECT customers.country, SUM(orders.amount) AS revenue "
            "FROM orders JOIN customers ON orders.customer_id = customers.id "
            "GROUP BY customers.country ORDER BY revenue DESC"
        )
        return MethodOutput(
            intermediate_plan="direct SQL from the standalone question",
            sql=sql,
            expected_failure=None,
            context_carryover=True,
            value_grounded=True,
            measure_preserved=False,
        )
    if turn.turn_id == "turn_2":
        sql = (
            "SELECT customers.country, SUM(orders.amount) AS revenue "
            "FROM orders JOIN customers ON orders.customer_id = customers.id "
            "WHERE customers.country = 'France' GROUP BY customers.country"
        )
        return MethodOutput(
            intermediate_plan="carry metric, but copy display value France into storage SQL",
            sql=sql,
            expected_failure="value_grounding",
            context_carryover=True,
            value_grounded=False,
            measure_preserved=False,
        )
    if turn.turn_id == "turn_3":
        sql = (
            "SELECT customers.name, SUM(orders.amount) AS revenue "
            "FROM orders JOIN customers ON orders.customer_id = customers.id "
            "GROUP BY customers.name ORDER BY revenue DESC LIMIT 1"
        )
        return MethodOutput(
            intermediate_plan="change grain, but forget the France filter",
            sql=sql,
            expected_failure="context_carryover",
            context_carryover=False,
            value_grounded=True,
            measure_preserved=False,
        )
    sql = (
        "SELECT customers.name, SUM(orders.amount) AS revenue "
        "FROM orders JOIN customers ON orders.customer_id = customers.id "
        "WHERE customers.country = 'France' GROUP BY customers.name "
        "ORDER BY revenue DESC LIMIT 1"
    )
    return MethodOutput(
        intermediate_plan="retry failed SQL without reading the empty-result feedback",
        sql=sql,
        expected_failure="recovery",
        context_carryover=True,
        value_grounded=False,
        measure_preserved=False,
    )


def _planner_first_sql(turn: LabTurn) -> MethodOutput:
    if turn.turn_id == "turn_1":
        sql = (
            "SELECT customers.country, SUM(orders.amount) AS revenue "
            "FROM orders JOIN customers ON orders.customer_id = customers.id "
            "GROUP BY customers.country ORDER BY revenue DESC"
        )
        return MethodOutput(
            intermediate_plan="plan: metric=revenue, grain=country",
            sql=sql,
            expected_failure=None,
            context_carryover=True,
            value_grounded=True,
            measure_preserved=False,
        )
    if turn.turn_id == "turn_2":
        sql = (
            "SELECT customers.country, SUM(orders.amount) AS revenue "
            "FROM orders JOIN customers ON orders.customer_id = customers.id "
            "WHERE customers.country = 'France' GROUP BY customers.country"
        )
        return MethodOutput(
            intermediate_plan="plan: keep revenue by country; filter country=France",
            sql=sql,
            expected_failure="value_grounding",
            context_carryover=True,
            value_grounded=False,
            measure_preserved=False,
        )
    if turn.turn_id == "turn_3":
        sql = (
            "SELECT customers.name, SUM(orders.amount) AS revenue "
            "FROM orders JOIN customers ON orders.customer_id = customers.id "
            "WHERE customers.country = 'France' GROUP BY customers.name "
            "ORDER BY revenue DESC LIMIT 1"
        )
        return MethodOutput(
            intermediate_plan="plan: keep country=France; change grain to customer",
            sql=sql,
            expected_failure="value_grounding",
            context_carryover=True,
            value_grounded=False,
            measure_preserved=False,
        )
    sql = (
        "SELECT customers.name, SUM(orders.amount) AS revenue "
        "FROM orders JOIN customers ON orders.customer_id = customers.id "
        "WHERE customers.country = 'France' GROUP BY customers.name "
        "ORDER BY revenue DESC LIMIT 1"
    )
    return MethodOutput(
        intermediate_plan="plan: retry customer grain with country=France despite empty result",
        sql=sql,
        expected_failure="recovery",
        context_carryover=True,
        value_grounded=False,
        measure_preserved=False,
    )


def _semantic_value_sql(turn: LabTurn) -> MethodOutput:
    if turn.turn_id == "turn_1":
        sql = (
            "SELECT customers.country, SUM(orders.amount) AS revenue "
            "FROM orders JOIN customers ON orders.customer_id = customers.id "
            "GROUP BY customers.country ORDER BY revenue DESC"
        )
        return MethodOutput(
            intermediate_plan="semantic state: metric revenue maps to SUM(orders.amount)",
            sql=sql,
            expected_failure=None,
            context_carryover=True,
            value_grounded=True,
            measure_preserved=False,
        )
    if turn.turn_id == "turn_2":
        sql = (
            "SELECT customers.country, SUM(orders.amount) AS revenue "
            "FROM orders JOIN customers ON orders.customer_id = customers.id "
            "WHERE customers.country = 'FR' GROUP BY customers.country"
        )
        return MethodOutput(
            intermediate_plan="semantic state: France normalized to country code FR",
            sql=sql,
            expected_failure=None,
            context_carryover=True,
            value_grounded=True,
            measure_preserved=False,
        )
    sql = (
        "SELECT customers.name, SUM(orders.amount) AS revenue "
        "FROM orders JOIN customers ON orders.customer_id = customers.id "
        "WHERE customers.country = 'FR' GROUP BY customers.name "
        "ORDER BY revenue DESC LIMIT 1"
    )
    return MethodOutput(
        intermediate_plan=(
            "semantic state: keep country=FR; grain changes to customer"
            if turn.turn_id == "turn_3"
            else "semantic state: recompute with country=FR, but no explicit repair action"
        ),
        sql=sql,
        expected_failure=None,
        context_carryover=True,
        value_grounded=True,
        measure_preserved=False,
    )


def _semantic_dsl_planner(turn: LabTurn) -> MethodOutput:
    if turn.turn_id == "turn_1":
        plan = "MEASURE(revenue) BY customer_country ORDER BY MEASURE(revenue) DESC"
        sql = (
            "SELECT customers.country, SUM(orders.amount) AS revenue "
            "FROM orders JOIN customers ON orders.customer_id = customers.id "
            "GROUP BY customers.country ORDER BY revenue DESC"
        )
        return MethodOutput(
            intermediate_plan=plan,
            sql=sql,
            expected_failure=None,
            context_carryover=True,
            value_grounded=True,
            measure_preserved=True,
        )
    if turn.turn_id == "turn_2":
        plan = "MEASURE(revenue) BY customer_country WHERE customer_country = 'FR'"
        sql = (
            "SELECT customers.country, SUM(orders.amount) AS revenue "
            "FROM orders JOIN customers ON orders.customer_id = customers.id "
            "WHERE customers.country = 'FR' GROUP BY customers.country"
        )
        return MethodOutput(
            intermediate_plan=plan,
            sql=sql,
            expected_failure=None,
            context_carryover=True,
            value_grounded=True,
            measure_preserved=True,
        )
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
    return MethodOutput(
        intermediate_plan=plan
        if turn.turn_id == "turn_3"
        else f"{plan}; no explicit empty-result repair step",
        sql=sql,
        expected_failure=None,
        context_carryover=True,
        value_grounded=True,
        measure_preserved=True,
    )


def _behavior_recovery_sql(turn: LabTurn) -> MethodOutput:
    if turn.turn_id == "turn_1":
        output = _semantic_dsl_planner(turn)
        return MethodOutput(
            intermediate_plan="behavior policy: answer initial metric request with governed metric state",
            sql=output.sql,
            expected_failure=None,
            context_carryover=True,
            value_grounded=True,
            measure_preserved=True,
        )
    if turn.turn_id == "turn_2":
        output = _semantic_dsl_planner(turn)
        return MethodOutput(
            intermediate_plan="behavior policy: carry state and normalize France to FR before SQL",
            sql=output.sql,
            expected_failure=None,
            context_carryover=True,
            value_grounded=True,
            measure_preserved=True,
        )
    if turn.turn_id == "turn_3":
        sql = (
            "SELECT customers.name, SUM(orders.amount) AS revenue "
            "FROM orders JOIN customers ON orders.customer_id = customers.id "
            "WHERE customers.country = 'France' GROUP BY customers.name "
            "ORDER BY revenue DESC LIMIT 1"
        )
        return MethodOutput(
            intermediate_plan=(
                "behavior policy before recovery: carries the follow-up state but "
                "uses display value France, producing an empty result"
            ),
            sql=sql,
            expected_failure="value_grounding",
            context_carryover=True,
            value_grounded=False,
            measure_preserved=True,
        )

    sql = (
        "SELECT customers.name, SUM(orders.amount) AS revenue "
        "FROM orders JOIN customers ON orders.customer_id = customers.id "
        "WHERE customers.country = 'FR' GROUP BY customers.name "
        "ORDER BY revenue DESC LIMIT 1"
    )
    return MethodOutput(
        intermediate_plan=(
            "behavior policy: inspects previous empty result, identifies France/FR "
            "value mismatch, repairs empty result with country=FR"
        ),
        sql=sql,
        expected_failure=None,
        context_carryover=True,
        value_grounded=True,
        measure_preserved=True,
        recovery_success=True,
    )


def _normalize_rows(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    normalized = []
    for row in rows:
        normalized.append(
            tuple(round(value, 8) if isinstance(value, float) else value for value in row)
        )
    return normalized


def scenario_contract(turns: list[LabTurn]) -> dict[str, Any]:
    payload = [
        {
            "turn_id": turn.turn_id,
            "question": turn.question,
            "reference_sql": turn.reference_sql,
            "context_note": turn.context_note,
            "requires_recovery": turn.requires_recovery,
        }
        for turn in turns
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    recovery_turns = [turn.turn_id for turn in turns if turn.requires_recovery]
    return {
        "turn_count": len(turns),
        "recovery_turn_id": recovery_turns[0] if recovery_turns else None,
        "shared_input_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def run_multiturn_lab() -> dict[str, Any]:
    """Run a small executable multi-turn SQL lab with no external model download."""

    detected_accelerator = available_accelerator()
    conn = _connect_demo_db()
    turns = lab_turns()
    systems = {
        "direct_sql_baseline": _direct_sql_baseline,
        "planner_first_sql": _planner_first_sql,
        "semantic_value_sql": _semantic_value_sql,
        "semantic_dsl_planner": _semantic_dsl_planner,
        "behavior_recovery_sql": _behavior_recovery_sql,
    }
    rows: list[dict[str, Any]] = []

    try:
        for turn in turns:
            expected_rows = _normalize_rows(_rows_for_sql(conn, turn.reference_sql))
            for system_name, planner in systems.items():
                output = planner(turn)
                actual_rows = _normalize_rows(_rows_for_sql(conn, output.sql))
                value_match = actual_rows == expected_rows
                rows.append(
                    {
                        "turn_id": turn.turn_id,
                        "question": turn.question,
                        "context_note": turn.context_note,
                        "requires_recovery": turn.requires_recovery,
                        "system": system_name,
                        "intermediate_plan": output.intermediate_plan,
                        "sql": output.sql,
                        "actual_rows": actual_rows,
                        "expected_rows": expected_rows,
                        "value_match": value_match,
                        "context_carryover": output.context_carryover,
                        "value_grounded": output.value_grounded,
                        "measure_preserved": output.measure_preserved,
                        "recovery_success": output.recovery_success,
                        "failure_type": None if value_match else output.expected_failure,
                    }
                )
    finally:
        conn.close()

    summaries: dict[str, dict[str, Any]] = {}
    for system_name in systems:
        system_rows = [row for row in rows if row["system"] == system_name]
        correct = sum(1 for row in system_rows if row["value_match"])
        context_correct = sum(1 for row in system_rows if row["context_carryover"])
        value_grounded = sum(1 for row in system_rows if row["value_grounded"])
        measure_preserved = sum(1 for row in system_rows if row["measure_preserved"])
        recovery_rows = [row for row in system_rows if row["requires_recovery"]]
        recovery_success = sum(1 for row in recovery_rows if row["recovery_success"])
        summaries[system_name] = {
            "correct": correct,
            "turns": len(system_rows),
            "value_accuracy": correct / len(system_rows),
            "context_carryover_accuracy": context_correct / len(system_rows),
            "value_grounding_accuracy": value_grounded / len(system_rows),
            "measure_preservation_rate": measure_preserved / len(system_rows),
            "recovery_success_rate": (
                recovery_success / len(recovery_rows) if recovery_rows else 0.0
            ),
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
        "method_matrix": method_matrix(),
        "scenario_contract": scenario_contract(turns),
    }


def method_matrix() -> list[dict[str, str]]:
    return [
        {
            "system": "direct_sql_baseline",
            "fine_tuning_target": "assistant SQL",
            "training_signal": "question/history/schema to SQL",
            "what_it_isolates": "raw SQL behavior without explicit intermediate state",
        },
        {
            "system": "planner_first_sql",
            "fine_tuning_target": "query plan then SQL",
            "training_signal": "metric, grain, filters, and follow-up state before SQL",
            "what_it_isolates": "whether planning fixes context before value grounding",
        },
        {
            "system": "semantic_value_sql",
            "fine_tuning_target": "semantic state then SQL",
            "training_signal": "semantic values, entity normalization, grain, and joins",
            "what_it_isolates": "whether semantic grounding fixes execution values",
        },
        {
            "system": "semantic_dsl_planner",
            "fine_tuning_target": "MEASURE-preserving DSL then SQL",
            "training_signal": "governed metrics and dimensions before SQL compilation",
            "what_it_isolates": "whether metric intent survives before SQL expansion",
        },
        {
            "system": "behavior_recovery_sql",
            "fine_tuning_target": "execution feedback then repair",
            "training_signal": "failed result, error class, repair action, and corrected SQL",
            "what_it_isolates": "whether the model learns to recover after its own failed turn",
        },
    ]
