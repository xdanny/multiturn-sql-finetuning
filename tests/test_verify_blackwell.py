from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_verify_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "verify_blackwell.py"
    spec = importlib.util.spec_from_file_location("verify_blackwell", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_check_result_phase_requirements() -> None:
    verify = load_verify_module()

    training = verify.CheckResult("training", True, "ok")
    serving = verify.CheckResult("serving", False, "missing", phases={"serving"})

    assert training.required_for("training") is True
    assert training.required_for("serving") is False
    assert serving.required_for("training") is False
    assert serving.required_for("serving") is True
    assert serving.required_for("all") is True
