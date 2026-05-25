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
    available: bool = False


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


def _accelerator_status(
    *,
    kind: str,
    label: str,
    torch_available: bool,
    available: bool,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "label": label,
        "torch_available": torch_available,
        "available": available,
        "usage": "available_for_auto",
    }


def accelerator_statuses() -> list[dict[str, Any]]:
    """Report CUDA/MPS/XPU visibility without selecting an accelerator."""

    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return [
            _accelerator_status(
                kind=kind,
                label=f"{kind.upper()} unavailable (torch not installed)",
                torch_available=False,
                available=False,
            )
            for kind in ("cuda", "mps", "xpu")
        ]

    statuses: list[dict[str, Any]] = []

    try:
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            count = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(index) for index in range(count)]
            cuda_label = (
                f"CUDA available ({count} device{'s' if count != 1 else ''}: "
                f"{', '.join(names)})"
            )
        else:
            cuda_label = "CUDA unavailable"
    except Exception as exc:
        cuda_available = False
        cuda_label = f"CUDA probe failed: {exc}"
    statuses.append(
        _accelerator_status(
            kind="cuda",
            label=cuda_label,
            torch_available=True,
            available=cuda_available,
        )
    )

    try:
        mps_available = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        mps_label = "MPS available" if mps_available else "MPS unavailable"
    except Exception as exc:
        mps_available = False
        mps_label = f"MPS probe failed: {exc}"
    statuses.append(
        _accelerator_status(
            kind="mps",
            label=mps_label,
            torch_available=True,
            available=mps_available,
        )
    )

    try:
        xpu_available = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
        xpu_label = "XPU available" if xpu_available else "XPU unavailable"
    except Exception as exc:
        xpu_available = False
        xpu_label = f"XPU probe failed: {exc}"
    statuses.append(
        _accelerator_status(
            kind="xpu",
            label=xpu_label,
            torch_available=True,
            available=xpu_available,
        )
    )

    return statuses


def available_accelerator() -> Accelerator:
    """Return the best visible accelerator without requiring torch or GPU compute."""

    statuses = accelerator_statuses()
    for status in statuses:
        if status["available"]:
            return Accelerator(
                kind=str(status["kind"]),
                label=str(status["label"]),
                torch_available=bool(status["torch_available"]),
                available=True,
            )

    torch_available = any(bool(status["torch_available"]) for status in statuses)
    label = "no accelerator detected" if torch_available else "no accelerator detected (torch not installed)"
    return Accelerator(kind="none", label=label, torch_available=torch_available)


def _accelerator_from_status(status: dict[str, Any]) -> Accelerator:
    return Accelerator(
        kind=str(status["kind"]),
        label=str(status["label"]),
        torch_available=bool(status["torch_available"]),
        available=bool(status["available"]),
    )


def lab_walkthrough_sections() -> list[dict[str, str]]:
    return [
        {
            "section_id": "research_question",
            "title": "Research question",
            "reader_question": (
                "Can a small specialized model learn behavior and semantic "
                "concepts well enough to challenge hosted SOTA on multi-turn "
                "data analysis?"
            ),
            "takeaway": (
                "The lab starts from the same question as the post: model size is "
                "not the main variable if the target behavior is wrong."
            ),
            "next_artifact": "Compare target formats before making a larger training run.",
        },
        {
            "section_id": "single_turn_gap",
            "title": "single-turn gap",
            "reader_question": "Why is zero-shot benchmark SQL not enough for analysis?",
            "takeaway": (
                "single-turn SQL plus chat history is not enough when the user "
                "changes filters, grain, values, and recovery state across turns."
            ),
            "next_artifact": "Make follow-up state visible in the execution trace.",
        },
        {
            "section_id": "proxy_slice",
            "title": "proxy slice",
            "reader_question": "What fixed slice keeps iteration honest?",
            "takeaway": (
                "100 CoSQL turns are the current proxy for fast iteration; the "
                "tiny SQLite lab mirrors the same failure classes without a model download."
            ),
            "next_artifact": "Keep CoSQL as a proxy and move the contract to BIRD-Interact.",
        },
        {
            "section_id": "target_comparison",
            "title": "target comparison",
            "reader_question": "What should the small model learn?",
            "takeaway": (
                "The lab compares five fine-tuning targets: direct SQL, planner-first "
                "SQL, semantic value grounding, MEASURE()-preserving DSL, and "
                "behavior/recovery."
            ),
            "next_artifact": "Promote each target to a real endpoint evaluation manifest.",
        },
        {
            "section_id": "execution_trace",
            "title": "execution trace",
            "reader_question": "Where does each target fail?",
            "takeaway": (
                "The four-turn SQLite scenario exposes context carryover, value "
                "grounding, metric preservation, and recovery as separate behaviors."
            ),
            "next_artifact": "Score generated-history rollouts, not only teacher-forced turns.",
        },
        {
            "section_id": "claim_boundary",
            "title": "claim boundary",
            "reader_question": "What does this lab prove?",
            "takeaway": (
                "This is not a benchmark result; it is a runnable method comparison "
                "that explains what evidence the full repo still needs."
            ),
            "next_artifact": "Do not claim hosted-SOTA parity until same-protocol baselines exist.",
        },
        {
            "section_id": "next_gates",
            "title": "next gates",
            "reader_question": "What should be built next?",
            "takeaway": (
                "The next repo work should turn planning, semantic concepts, DSL "
                "preservation, and recovery into separately scored artifacts."
            ),
            "next_artifact": (
                "non-oracle planner, metric DSL manifest, generated-history rollout, "
                "hosted baseline, and BIRD-Interact transfer."
            ),
        },
    ]


def select_lab_device(
    device_preference: str = "auto",
) -> tuple[Accelerator, Accelerator, dict[str, Any], list[dict[str, Any]]]:
    """Select a portable lab runtime, falling back to CPU when needed."""

    allowed_devices = {"cpu", "auto", "cuda", "mps", "xpu"}
    if device_preference not in allowed_devices:
        raise ValueError("device_preference must be one of: auto, cpu, cuda, mps, xpu")

    accelerator_report = accelerator_statuses()
    statuses_by_kind = {
        str(status["kind"]): status for status in accelerator_report
    }
    detected_accelerator = next(
        (
            _accelerator_from_status(statuses_by_kind[kind])
            for kind in ("cuda", "mps", "xpu")
            if kind in statuses_by_kind and statuses_by_kind[kind]["available"]
        ),
        Accelerator(
            kind="none",
            label=(
                "no accelerator detected"
                if any(bool(status["torch_available"]) for status in accelerator_report)
                else "no accelerator detected (torch not installed)"
            ),
            torch_available=any(
                bool(status["torch_available"]) for status in accelerator_report
            ),
        ),
    )
    fallback = None
    fallback_reason = None
    if device_preference == "cpu":
        selected_kind = "cpu"
        accelerator_usage = "forced_cpu"
    elif device_preference == "auto":
        selected_kind = detected_accelerator.kind
        accelerator_usage = "selected_if_available"
        if selected_kind == "none":
            selected_kind = "cpu"
            fallback = "cpu"
            fallback_reason = "no_accelerator_detected"
    else:
        preferred = statuses_by_kind[device_preference]
        accelerator_usage = "selected_if_available"
        if preferred["available"]:
            selected_kind = device_preference
        else:
            selected_kind = "cpu"
            fallback = "cpu"
            fallback_reason = f"{device_preference}_unavailable"

    if selected_kind == "cpu":
        torch_available = detected_accelerator.torch_available
        selected_label = "cpu"
    else:
        selected = _accelerator_from_status(statuses_by_kind[selected_kind])
        torch_available = selected.torch_available
        selected_label = selected.label

    device = Accelerator(
        kind=selected_kind,
        label=selected_label,
        torch_available=torch_available,
        available=True,
    )

    return (
        device,
        detected_accelerator,
        {
            "device_preference": device_preference,
            "fallback": fallback,
            "fallback_reason": fallback_reason,
            "accelerator_usage": accelerator_usage,
            "selected_device": selected_kind,
            "reported_accelerators": "CUDA, MPS, XPU",
        },
        accelerator_report,
    )


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


def run_multiturn_lab(device_preference: str = "auto") -> dict[str, Any]:
    """Run a small executable multi-turn SQL lab with no external model download."""

    device, detected_accelerator, runtime_policy, accelerator_report = select_lab_device(
        device_preference
    )
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
        "device": device,
        "detected_accelerator": detected_accelerator,
        "accelerator_report": accelerator_report,
        "runtime_policy": runtime_policy,
        "turns": turns,
        "rows": rows,
        "systems": summaries,
        "method_matrix": method_matrix(),
        "scenario_contract": scenario_contract(turns),
        "walkthrough_sections": lab_walkthrough_sections(),
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
