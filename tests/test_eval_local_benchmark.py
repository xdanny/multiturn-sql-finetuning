from __future__ import annotations

from eval.local_benchmark import (
    SQL_ONLY_INSTRUCTION,
    enforce_sql_only_instruction,
    prompt_from_messages,
)


def test_prompt_from_messages_uses_chat_template() -> None:
    class Tokenizer:
        chat_template = "template"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
            assert tokenize is False
            assert add_generation_prompt is True
            assert kwargs["enable_thinking"] is False
            return messages[-1]["content"] + "\nassistant:"

    prompt = prompt_from_messages(Tokenizer(), [{"role": "user", "content": "SQL?"}])

    assert prompt == "SQL?\nassistant:"


def test_prompt_from_messages_fallback() -> None:
    class Tokenizer:
        chat_template = ""

    prompt = prompt_from_messages(Tokenizer(), [{"role": "user", "content": "SQL?"}])

    assert prompt == f"system: {SQL_ONLY_INSTRUCTION}\nuser: SQL?\nassistant:"


def test_enforce_sql_only_instruction_adds_system_message() -> None:
    messages = enforce_sql_only_instruction([{"role": "user", "content": "SQL?"}])

    assert messages[0]["role"] == "system"
    assert SQL_ONLY_INSTRUCTION in messages[0]["content"]
