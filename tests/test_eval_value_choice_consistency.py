from __future__ import annotations

import json

from eval.value_choice_consistency import score_value_choice_consistency


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_score_value_choice_consistency_detects_wrong_storage_value(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    rollout_path = tmp_path / "rollout.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "dialog_id": "d1",
                "expected_value_choice": {
                    "table": "customers",
                    "column": "country_code",
                    "source_mention": "France",
                    "storage_value": "FR",
                },
            }
        ],
    )
    _write_jsonl(
        rollout_path,
        [
            {
                "id": "d1:1",
                "dialog_id": "d1",
                "generated_sql": "SELECT * FROM customers WHERE customers.country_code = 'US'",
            }
        ],
    )

    payload = score_value_choice_consistency(
        input_path=input_path,
        rollout_output_path=rollout_path,
    )

    assert payload["scored_row_count"] == 1
    assert payload["value_choice_accuracy"] == 0.0
    assert payload["rows"][0]["selected_storage_value"] == "US"


def test_score_value_choice_consistency_accepts_expected_storage_value(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    rollout_path = tmp_path / "rollout.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "dialog_id": "d1",
                "expected_value_choice": {
                    "table": "customers",
                    "column": "country_code",
                    "source_mention": "France",
                    "storage_value": "FR",
                },
            }
        ],
    )
    _write_jsonl(
        rollout_path,
        [
            {
                "id": "d1:1",
                "dialog_id": "d1",
                "generated_sql": "SELECT * FROM customers WHERE customers.country_code = 'FR'",
            }
        ],
    )

    payload = score_value_choice_consistency(
        input_path=input_path,
        rollout_output_path=rollout_path,
    )

    assert payload["value_choice_accuracy"] == 1.0
    assert payload["rows"][0]["value_choice_match"] is True
