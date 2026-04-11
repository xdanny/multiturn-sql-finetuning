"""
Data preparation for multi-turn SQL fine-tuning.

Downloads CoSQL, SParC, and gretelai/synthetic_text_to_sql, formats each as
Qwen 3.5 chat messages, and writes a combined JSONL ready for SFTTrainer.

Usage:
    python -m data.prepare --output data/processed/train.jsonl

TODO: implement dataset-specific formatters. Each dataset has different
column names and multi-turn conventions that must be normalized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


SYSTEM_PROMPT = """You are a SQL expert. Given a database schema and a user question, generate an accurate SQL query. If the question is ambiguous, ask a clarifying question before generating SQL."""


def format_cosql(example: dict) -> list[dict]:
    """Format a CoSQL dialog as Qwen 3.5 chat messages.

    CoSQL structure (from HF dataset `alpineai/cosql`):
      - interaction: list of turns, each with {'utterance', 'query'}
      - database schema for context

    TODO: verify column names by inspecting the dataset.
    """
    raise NotImplementedError("CoSQL formatter TODO")


def format_sparc(example: dict) -> list[dict]:
    """Format a SParC question sequence as Qwen 3.5 chat messages.

    SParC structure (from HF dataset `jellyChiru/SParC`):
      - interaction: list of turns, each with {'utterance', 'query'}

    TODO: verify column names by inspecting the dataset.
    """
    raise NotImplementedError("SParC formatter TODO")


def format_gretelai(example: dict) -> list[dict]:
    """Format a gretelai/synthetic_text_to_sql example as chat messages.

    Expected columns: sql_prompt, sql_context, sql
    Single-turn: user asks with schema, assistant answers with SQL.
    """
    raise NotImplementedError("gretelai formatter TODO")


def build_conversation(messages: list[dict]) -> dict:
    """Wrap a list of chat messages into the format expected by SFTTrainer."""
    return {"messages": messages}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per dataset (debug)")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}

    with args.output.open("w") as f:
        # CoSQL
        ds = load_dataset("alpineai/cosql", split="train")
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        n = 0
        for example in ds:
            try:
                messages = format_cosql(example)
                f.write(json.dumps(build_conversation(messages)) + "\n")
                n += 1
            except NotImplementedError:
                break
        counts["cosql"] = n

        # SParC
        ds = load_dataset("jellyChiru/SParC", split="train")
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        n = 0
        for example in ds:
            try:
                messages = format_sparc(example)
                f.write(json.dumps(build_conversation(messages)) + "\n")
                n += 1
            except NotImplementedError:
                break
        counts["sparc"] = n

        # gretelai synthetic
        ds = load_dataset("gretelai/synthetic_text_to_sql", split="train")
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        n = 0
        for example in ds:
            try:
                messages = format_gretelai(example)
                f.write(json.dumps(build_conversation(messages)) + "\n")
                n += 1
            except NotImplementedError:
                break
        counts["gretelai"] = n

    print(f"Wrote {args.output}")
    for name, n in counts.items():
        print(f"  {name}: {n} examples")


if __name__ == "__main__":
    main()
