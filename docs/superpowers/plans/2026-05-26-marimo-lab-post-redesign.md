# Marimo Lab Post Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the local multi-turn SQL post and repository revolve around a primary Marimo lab that tests whether a small specialized local model can learn behavior and semantic concepts for multi-turn data analysis.

**Architecture:** The SQL repo owns the runnable lab and evidence artifacts. The blog repo consumes those artifacts and uses the article to walk through the same lab checkpoints. The Marimo notebook must be a clean executable walkthrough, while generated tables remain supporting evidence rather than the lead narrative.

**Tech Stack:** Python, Marimo, pandas, SQLite, pytest, ruff, Next.js markdown evidence includes.

---

### Task 1: Make The Marimo Lab The Primary Clean Walkthrough

**Files:**
- Modify: `notebooks/labs/local_multiturn_sql_lab.py`
- Modify: `tests/test_local_multiturn_sql_lab.py`
- Modify: `tests/test_blog_notebooks.py`

- [ ] **Step 1: Update tests to reject blog evidence imports in the Marimo lab**

Require `notebooks/labs/local_multiturn_sql_lab.py` to import only `notebooks.labs.local_multiturn_sql_lab_support` plus presentation libraries. Assert it contains these notebook sections:

```python
for heading in [
    "## 1. Runtime",
    "## 2. Multi-turn task",
    "## 3. Candidate training targets",
    "## 4. Lab scorecard",
    "## 5. Failure trace",
    "## 6. Intermediate state",
    "## 7. What this proves",
]:
    assert heading in source
assert "notebooks.blog_support" not in source
assert "endpoint_run_scorecard" not in source
```

- [ ] **Step 2: Run the focused tests and confirm they fail before implementation**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_local_multiturn_sql_lab.py::test_shareable_lab_notebook_is_plain_python_marimo_app tests/test_blog_notebooks.py::test_public_blog_artifacts_are_one_shareable_lab_notebook
```

Expected: fail because the current Marimo app still imports `notebooks.blog_support`.

- [ ] **Step 3: Rewrite the Marimo app as a lab-first walkthrough**

Replace blog evidence helper imports with:

```python
import pandas as pd
import marimo as mo
from notebooks.labs.local_multiturn_sql_lab_support import lab_turns, run_multiturn_lab
```

The app should:
- expose a runtime dropdown with `auto`, `cpu`, `cuda`, `mps`, `xpu`;
- call `run_multiturn_lab(device_preference=runtime_choice.value)`;
- show the scenario turns from `lab_turns()`;
- show `report["method_matrix"]`;
- show value/subtask scores from `report["systems"]`;
- show failure rows from `report["rows"]`;
- show intermediate plans and SQL;
- end with next gates for planner-first, semantic grounding, `MEASURE()` DSL, and behavior/recovery.

- [ ] **Step 4: Run focused tests again**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_local_multiturn_sql_lab.py::test_shareable_lab_notebook_is_plain_python_marimo_app tests/test_blog_notebooks.py::test_public_blog_artifacts_are_one_shareable_lab_notebook
```

Expected: pass.

### Task 2: Update The SQL Repo Contract Around Marimo-First Labs

**Files:**
- Modify: `README.md`
- Modify: `docs/research_goal.md`
- Modify: `docs/blog/README.md`
- Modify: `notebooks/blog_support.py`
- Modify: generated files under `docs/blog/generated/`
- Modify: `tests/test_blog_notebooks.py`
- Modify: `tests/test_local_multiturn_sql_lab.py`

- [ ] **Step 1: Update tests for Marimo-primary run commands**

The shareable lab asset should use:

```python
assert lab_row["run_command"] == "marimo edit notebooks/labs/local_multiturn_sql_lab.py"
assert lab_row["alternate_command"] == "jupyter lab notebooks/labs/local_multiturn_sql_lab.ipynb"
```

The docs should say Marimo is the primary walkthrough and Jupyter is the portable export.

- [ ] **Step 2: Update `notebooks/blog_support.py`**

Change `shareable_lab_attachment()` and `lab_reader_flow()` so `run_command` points to Marimo first and `alternate_command` points to Jupyter.

- [ ] **Step 3: Update repo docs**

Make these docs state the actual research program:
- zero-shot BIRD-style SQL is strong in single turns;
- multi-turn data analysis is weaker because it needs behavior, state, semantic concepts, metric preservation, and recovery;
- the repo compares direct SQL SFT, planner/DSL-first, semantic-layer tuning, `MEASURE()` preservation, and behavior/recovery tuning.

- [ ] **Step 4: Regenerate evidence**

Run:

```bash
.venv/bin/python -m notebooks.blog_support --output-dir docs/blog/generated
```

- [ ] **Step 5: Verify generated evidence is current**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_blog_notebooks.py::test_checked_in_blog_evidence_assets_are_current
```

Expected: pass.

### Task 3: Make The Blog Post Follow The Lab

**Files:**
- Modify in blog repo: `_posts/local-multiturn-sql-finetuning.md`
- Modify in blog repo: `scripts/verify-sql-evidence.js`
- Modify copied evidence under `public/assets/blog/local-multiturn-sql-finetuning/evidence/`

- [ ] **Step 1: Update verifier expectations**

The verifier should require:

```javascript
'marimo edit notebooks/labs/local_multiturn_sql_lab.py'
'jupyter lab notebooks/labs/local_multiturn_sql_lab.ipynb'
'DSL first'
'semantic layer'
'MEASURE()'
'behavior/recovery'
```

- [ ] **Step 2: Rewrite the top half of the post**

The post should not dump all generated tables before the argument. It should proceed in this order:

1. basic question;
2. why single-turn BIRD-style strength is insufficient;
3. open the Marimo lab;
4. run the toy multi-turn scenario;
5. compare fine-tuning targets;
6. only then map the lab to CoSQL proxy evidence and claim boundaries.

- [ ] **Step 3: Copy regenerated SQL evidence into the blog repo**

Run:

```bash
npm run prebuild
```

- [ ] **Step 4: Verify the blog repo**

Run:

```bash
npm run verify:sql-evidence
npm run lint
npm run typecheck
npm run build
```

Expected: all pass. If build fails because Google Fonts are blocked by sandbox networking, rerun `npm run build` with escalation.

### Task 4: Commit Scoped Changes

**Files:**
- SQL repo commit should exclude untracked `uv.lock`.
- Blog repo commit should include only post/verifier/public evidence changes.

- [ ] **Step 1: Run SQL verification**

Run:

```bash
.venv/bin/ruff check .
.venv/bin/python -m pytest -q
```

- [ ] **Step 2: Commit SQL repo**

Run:

```bash
git add README.md docs/research_goal.md docs/blog/README.md docs/blog/generated notebooks tests
git commit -m "Make Marimo the primary SQL lab walkthrough"
```

- [ ] **Step 3: Commit blog repo**

Run:

```bash
git add _posts/local-multiturn-sql-finetuning.md public/assets/blog/local-multiturn-sql-finetuning/evidence scripts/verify-sql-evidence.js
git commit -m "Rewrite SQL post around Marimo lab walkthrough"
```

---

## Self-Review

- Spec coverage: The plan covers a primary Marimo notebook, blog-led lab walkthrough, SQL repo goal update, generated evidence refresh, verifier updates, and verification.
- Placeholder scan: No placeholders or open TODOs remain.
- Type consistency: The plan uses existing file names and existing lab support functions: `lab_turns()` and `run_multiturn_lab()`.
