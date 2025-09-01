"""Utilities to evaluate HumanEval-style problems using the official library."""

from __future__ import annotations

import ast
from typing import Any, Dict, Tuple

from human_eval.execution import check_correctness


def evaluate(problem: Dict[str, Any], solution: str, timeout: float = 1.0) -> Tuple[int, int]:
    """Return number of passed tests and total tests for a dataset problem.

    The implementation delegates execution to ``human_eval.execution.check_correctness``
    for each individual assertion in the problem's test suite.
    ``timeout`` controls how long each individual test is allowed to run.
    """

    module_ast = ast.parse(problem["test"])
    check_func = next(
        node for node in module_ast.body if isinstance(node, ast.FunctionDef)
    )
    passed = 0
    total = 0
    for stmt in check_func.body:
        if not isinstance(stmt, ast.Assert):
            continue
        total += 1
        test_func = ast.FunctionDef(
            name=check_func.name,
            args=check_func.args,
            body=[stmt],
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        test_module = ast.Module(body=[test_func], type_ignores=[])
        test_src = ast.unparse(ast.fix_missing_locations(test_module))
        single_problem = {**problem, "test": test_src}
        result = check_correctness(single_problem, solution, timeout)
        if result.get("passed"):
            passed += 1
    return passed, total
