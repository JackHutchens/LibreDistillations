#!/usr/bin/env python3
"""
Interactive Scientific Worksheet Engine: Hidden-Test Code Runner
(`engine/code_runner.py`).

Pure Python + NumPy execution engine compatible with both desktop CPython
(`JupyterLab`) and browser WebAssembly (`Pyodide` / `JupyterLite`).

Core Guarantees:
1. AST Pre-Flight Gate (`ast_preflight_check`):
   Blocks casual introspection, frame-snooping, garbage-collector traversal,
   filesystem/network I/O, and dynamic code evaluation (`import inspect`,
   `sys._getframe`, `gc.get_objects`, `open(`, `__subclasses__`, `globals()`,
   `locals()`, traceback frame attributes) with clear pedagogical error cards,
   and injects a deterministic loop-step guard preventing infinite-loop freezes
   on the single-threaded Pyodide/WASM main thread.
2. Opaque Sealed Test Bundle Format (`compile_hidden_suite` / `execute_hidden_suite`):
   Compiles authoring test suites into base85 + zlib + SHA-256-CTR/HMAC sealed
   test bundles so distributed notebooks and WASM static assets contain zero
   plaintext test source or expected answers at rest (an anti-spoiler pedagogical
   curtain for open self-study).
3. 3-Tier Scientific Test Architecture:
   - Tier 1 (`CONTRACT_AND_UNITS`): Return type, shape, finite-value (`not isnan/isinf`),
     and dimensional sanity checks.
   - Tier 2 (`RANDOMIZED_PHYSICAL_INVARIANTS`): Deterministically replayable (`t2_rng_seed_hex`)
     randomized physical property checks that cannot be hardcoded (Newton's 3rd Law,
     translational invariance, PBC minimum-image wrap symmetry across `+/- L/2`, Velocity
     Verlet time-reversibility & `O(dt^2)` shadow-energy scaling, Metropolis detailed balance).
   - Tier 3 (`SEALED_REFERENCE_VECTORS`): Edge-case physical regimes (`r -> r_cutoff`,
     `T -> 0`, `|dx| = L/2` tie-breaking, odd/even lattice boundaries) evaluated in an
     isolated scope whose locals, stdout/stderr, return-value spy hooks, and exception
     tracebacks are scrubbed before returning.
4. Diagnostic-Without-Disclosure Output Contract (`SuiteExecutionReport`):
   Emits public test IDs/names, pass/fail status, `t2_rng_seed_hex` for deterministic
   SHA-256 event-chain replay, zero-leak physical failure diagnostics, and
   `conceptual_gap_tag` identifiers feeding directly into the V2 Tree-of-Knowledge
   remediation engine.
5. Explicit Open-Source Client-Side Threat Model:
   Because execution occurs client-side inside an open-source `JupyterLab` or
   `Pyodide`/`JupyterLite` runtime where bundle keys and Python bytecode are shipped to
   the learner's machine, `base85 + zlib + SHA256-CTR + HMAC` sealing and AST blocklists
   serve strictly as an **anti-spoiler pedagogical curtain** deterring accidental or
   casual peeking during asynchronous self-study -- NOT cryptographic secrecy,
   unforgeable receipt integrity, or OS-kernel sandbox isolation against deliberate local
   reverse-engineering. **Tier-2 Randomized Physical Invariants** provide the substantive
   anti-hardcoding defense (extracting static Tier-3 vectors still fails dynamic
   phase-space symmetry/conservation laws unless true physics is implemented), while
   **verified summative credit requires server-side or CI container replay**
   against an unshipped private grading bundle.
"""

from __future__ import annotations

import argparse
import ast
import base64
import builtins
import cmath
import collections
import copy
import dataclasses
import functools
import hashlib
import hmac
import inspect as _host_inspect
import io
import textwrap
import itertools
import json
import math
import numbers
import os
import random
import statistics
import sys
import time
import traceback
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore


BUNDLE_SCHEMA_VERSION = "V2-HIDDEN-TEST-BUNDLE-v1.0"
DEFAULT_MAX_LOOP_TICKS = 250_000

TIER_CONTRACT_AND_UNITS = "CONTRACT_AND_UNITS"
TIER_RANDOMIZED_INVARIANTS = "RANDOMIZED_PHYSICAL_INVARIANTS"
TIER_SEALED_VECTORS = "SEALED_REFERENCE_VECTORS"

VALID_TEST_TIERS: Set[str] = {
    TIER_CONTRACT_AND_UNITS,
    TIER_RANDOMIZED_INVARIANTS,
    TIER_SEALED_VECTORS,
}

OPEN_SOURCE_CLIENT_THREAT_MODEL: Dict[str, Any] = {
    "policy_id": "OPEN_SOURCE_ANTI_SPOILER_CURTAIN",
    "client_bundle_role": (
        "Anti-spoiler pedagogical curtain at rest: base85+zlib+SHA256-CTR+HMAC and "
        "AST pre-flight deter accidental or casual peeking during self-study, NOT "
        "cryptographic secrecy or OS sandbox isolation against deliberate local "
        "reverse-engineering of shipped keys or runtime frames."
    ),
    "substantive_anti_hardcoding_layer": (
        "Tier-2 RANDOMIZED_PHYSICAL_INVARIANTS: dynamically/session-seeded physical "
        "conservation, symmetry, and detailed-balance laws cannot be satisfied by "
        "hardcoded lookup tables even if static Tier-3 vectors were inspected."
    ),
    "summative_credit_requirement": (
        "Verified summative or for-credit assessment requires CI/server container "
        "re-grading against unshipped private test suites and independent seeds."
    ),
}

# Safe scientific modules allowed inside student exercise code
ALLOWED_STUDENT_IMPORTS: Set[str] = {
    "math",
    "cmath",
    "random",
    "statistics",
    "itertools",
    "functools",
    "collections",
    "numbers",
    "typing",
    "numpy",
}

# Explicitly blocked introspection, I/O, and sandbox-escape modules
FORBIDDEN_IMPORT_PEDAGOGY: Dict[str, str] = {
    "inspect": (
        "Importing 'inspect' is disallowed in worksheet coding cells because live frame "
        "inspection is reserved by the assessment runtime. Implement the physical "
        "algorithm directly using standard arithmetic, 'math', or 'numpy'."
    ),
    "sys": (
        "Importing 'sys' (including 'sys._getframe' or stream hooks) is disallowed in "
        "student solutions. Pass all physical parameters explicitly through function arguments."
    ),
    "gc": (
        "Importing 'gc' (garbage-collector object traversal) is disallowed in student "
        "solutions. Compute return values purely from the provided inputs."
    ),
    "os": (
        "Importing 'os' is disallowed because worksheet coding exercises are pure "
        "in-memory scientific functions requiring zero host filesystem or environment access."
    ),
    "pathlib": (
        "Filesystem modules ('pathlib') are disallowed in pure scientific functions."
    ),
    "subprocess": (
        "Spawning external processes ('subprocess') is disallowed in worksheet cells."
    ),
    "ctypes": (
        "Low-level memory access ('ctypes') is disallowed in worksheet cells."
    ),
    "traceback": (
        "Importing 'traceback' is disallowed; handle numerical edge cases directly via "
        "explicit conditionals (e.g., checking r < r_cutoff or T > 0)."
    ),
    "builtins": (
        "Importing 'builtins' directly is disallowed in isolated worksheet functions."
    ),
    "importlib": (
        "Dynamic module loading ('importlib') is disallowed in worksheet functions."
    ),
    "io": (
        "Stream manipulation ('io') is disallowed in pure numerical worksheet functions."
    ),
    "pickle": (
        "Binary object serialization ('pickle') is disallowed in worksheet functions."
    ),
    "base64": (
        "Encoding utilities ('base64') are unneeded and disallowed in physics exercises."
    ),
    "zlib": (
        "Compression utilities ('zlib') are unneeded and disallowed in physics exercises."
    ),
}

# Forbidden function calls in student AST
FORBIDDEN_CALL_NAMES: Dict[str, str] = {
    "open": (
        "Calling 'open()' is disallowed: coding exercises operate purely on in-memory "
        "numbers and arrays without reading or writing files."
    ),
    "eval": (
        "Dynamic code evaluation via 'eval()' is disallowed; write explicit mathematical "
        "expressions instead."
    ),
    "exec": (
        "Dynamic code execution via 'exec()' is disallowed in worksheet solutions."
    ),
    "compile": (
        "Dynamic bytecode compilation via 'compile()' is disallowed in worksheet solutions."
    ),
    "__import__": (
        "Direct calls to '__import__()' are disallowed; use standard 'import numpy as np' "
        "or 'import math' statements."
    ),
    "globals": (
        "Calling 'globals()' is disallowed to preserve execution-scope isolation. "
        "Reference your function parameters and local variables directly."
    ),
    "locals": (
        "Calling 'locals()' is disallowed to preserve execution-scope isolation. "
        "Reference your function parameters and local variables directly."
    ),
    "vars": (
        "Calling 'vars()' is disallowed in isolated worksheet functions."
    ),
    "dir": (
        "Introspection via 'dir()' is disallowed inside submitted worksheet solutions."
    ),
    "getattr": (
        "Dynamic attribute lookup via 'getattr()' is disallowed in worksheet solutions; "
        "use direct attribute syntax (e.g., 'arr.shape' or 'np.exp')."
    ),
    "setattr": (
        "Dynamic attribute mutation via 'setattr()' is disallowed in worksheet solutions."
    ),
    "delattr": (
        "Dynamic attribute deletion via 'delattr()' is disallowed in worksheet solutions."
    ),
    "input": (
        "Interactive 'input()' prompts block automated grading and WebAssembly execution; "
        "read inputs strictly from your function arguments."
    ),
    "breakpoint": (
        "Interactive 'breakpoint()' calls are disallowed during automated test execution."
    ),
}

# Forbidden attribute accesses (frame/traceback/class-hierarchy/GC snooping)
FORBIDDEN_ATTRIBUTES: Dict[str, str] = {
    "_getframe": "Frame access via '_getframe' is disallowed by the assessment sandbox.",
    "currentframe": "Frame inspection via 'currentframe' is disallowed by the sandbox.",
    "get_objects": "Heap traversal via 'get_objects' is disallowed by the sandbox.",
    "get_referrers": "Reference graph traversal via 'get_referrers' is disallowed.",
    "get_referents": "Reference graph traversal via 'get_referents' is disallowed.",
    "__subclasses__": "Class-hierarchy traversal via '__subclasses__' is disallowed.",
    "__bases__": "Class-hierarchy introspection via '__bases__' is disallowed.",
    "__mro__": "Method-resolution introspection via '__mro__' is disallowed.",
    "__class__": "Dunder '__class__' introspection is disallowed; use standard array/math ops.",
    "__globals__": "Function '__globals__' snooping is disallowed by the sandbox.",
    "__builtins__": "Dunder '__builtins__' access is disallowed by the sandbox.",
    "__closure__": "Closure cell introspection via '__closure__' is disallowed.",
    "__code__": "Code-object introspection via '__code__' is disallowed.",
    "__dict__": "Object dictionary introspection via '__dict__' is disallowed.",
    "f_back": "Caller stack-frame traversal ('f_back') is disallowed.",
    "f_locals": "Frame locals snooping ('f_locals') is disallowed.",
    "f_globals": "Frame globals snooping ('f_globals') is disallowed.",
    "f_code": "Frame code-object snooping ('f_code') is disallowed.",
    "f_builtins": "Frame builtins snooping ('f_builtins') is disallowed.",
    "tb_frame": "Traceback frame snooping ('tb_frame') is disallowed.",
    "tb_next": "Traceback chain traversal ('tb_next') is disallowed.",
    "gi_frame": "Generator frame snooping ('gi_frame') is disallowed.",
    "gi_code": "Generator code snooping ('gi_code') is disallowed.",
    "cr_frame": "Coroutine frame snooping ('cr_frame') is disallowed.",
    "ag_frame": "Async-generator frame snooping ('ag_frame') is disallowed.",
    "co_consts": "Bytecode constant snooping ('co_consts') is disallowed.",
    "co_code": "Raw bytecode access ('co_code') is disallowed.",
}

# Standard built-in exception names safe to name in student diagnostics
_SAFE_STD_EXCEPTION_NAMES: Set[str] = {
    "ArithmeticError",
    "FloatingPointError",
    "IndexError",
    "KeyError",
    "NameError",
    "NotImplementedError",
    "OverflowError",
    "RecursionError",
    "TypeError",
    "UnboundLocalError",
    "ValueError",
    "ZeroDivisionError",
}


class BundleIntegrityError(ValueError):
    """Raised when a sealed hidden-test bundle fails HMAC/SHA-256 verification."""


class _OracleAssertionFailed(Exception):
    """Internal sentinel raised when a hidden scientific test check fails."""


class _StepBudgetExceededError(Exception):
    """Internal sentinel raised when student loops exceed the WASM tick budget."""


class _ReturnContractViolation(Exception):
    """Internal sentinel raised when student code returns an unsafe custom spy object."""


# ==============================================================================
# Data Contracts
# ==============================================================================


@dataclass(frozen=True)
class PreflightViolation:
    rule_id: str
    lineno: int
    col_offset: int
    construct: str
    pedagogical_message: str
    conceptual_gap_tag: str = "HARNESS_SANDBOX_POLICY_VIOLATION"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class PreflightCheckResult:
    passed: bool
    entry_function_found: bool
    violations: List[PreflightViolation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "entry_function_found": self.entry_function_found,
            "violations": [v.to_dict() for v in self.violations],
        }


@dataclass(frozen=True)
class HiddenTestSpec:
    test_id: str
    public_test_name: str
    tier: str
    conceptual_gap_tag: str
    failure_diagnostic: str
    oracle_code: str
    hint_nudge: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class AuthoringSuiteSpec:
    suite_id: str
    lesson_id: str
    exercise_id: str
    entry_function: str
    curriculum_slice_ref: Dict[str, Any]
    tests: List[HiddenTestSpec]
    max_loop_ticks: int = DEFAULT_MAX_LOOP_TICKS
    default_rtol: float = 1e-6
    default_atol: float = 1e-9

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "lesson_id": self.lesson_id,
            "exercise_id": self.exercise_id,
            "entry_function": self.entry_function,
            "curriculum_slice_ref": copy.deepcopy(self.curriculum_slice_ref),
            "tests": [t.to_dict() for t in self.tests],
            "max_loop_ticks": self.max_loop_ticks,
            "default_rtol": self.default_rtol,
            "default_atol": self.default_atol,
        }


@dataclass(frozen=True)
class SealedTestBundle:
    schema_version: str
    suite_id: str
    lesson_id: str
    exercise_id: str
    entry_function: str
    curriculum_slice_ref: Dict[str, Any]
    max_loop_ticks: int
    test_count: int
    public_manifest: List[Dict[str, str]]
    key_mode: str
    payload_sha256: str
    auth_sha256: str
    sealed_payload_b85: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SealedTestBundle":
        return cls(
            schema_version=str(data["schema_version"]),
            suite_id=str(data["suite_id"]),
            lesson_id=str(data["lesson_id"]),
            exercise_id=str(data["exercise_id"]),
            entry_function=str(data["entry_function"]),
            curriculum_slice_ref=dict(data.get("curriculum_slice_ref", {})),
            max_loop_ticks=int(data.get("max_loop_ticks", DEFAULT_MAX_LOOP_TICKS)),
            test_count=int(data["test_count"]),
            public_manifest=list(data.get("public_manifest", [])),
            key_mode=str(data["key_mode"]),
            payload_sha256=str(data["payload_sha256"]),
            auth_sha256=str(data["auth_sha256"]),
            sealed_payload_b85=str(data["sealed_payload_b85"]),
        )

    @classmethod
    def from_json(cls, raw_json: str) -> "SealedTestBundle":
        return cls.from_dict(json.loads(raw_json))


@dataclass(frozen=True)
class TestCaseReport:
    test_id: str
    public_test_name: str
    tier: str
    passed: bool
    failure_diagnostic: str
    conceptual_gap_tag: Optional[str]
    hint_nudge: str = ""
    execution_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SuiteExecutionReport:
    suite_id: str
    lesson_id: str
    exercise_id: str
    entry_function: str
    preflight_passed: bool
    preflight_violations: List[PreflightViolation]
    all_passed: bool
    passed_count: int
    total_count: int
    test_reports: List[TestCaseReport]
    triggered_conceptual_gap_tags: List[str]
    curriculum_slice_ref: Dict[str, Any]
    report_sha256: str
    t2_rng_seed_hex: str = "0000000000000000"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "lesson_id": self.lesson_id,
            "exercise_id": self.exercise_id,
            "entry_function": self.entry_function,
            "preflight_passed": self.preflight_passed,
            "preflight_violations": [v.to_dict() for v in self.preflight_violations],
            "all_passed": self.all_passed,
            "passed_count": self.passed_count,
            "total_count": self.total_count,
            "test_reports": [r.to_dict() for r in self.test_reports],
            "triggered_conceptual_gap_tags": list(self.triggered_conceptual_gap_tags),
            "curriculum_slice_ref": copy.deepcopy(self.curriculum_slice_ref),
            "t2_rng_seed_hex": self.t2_rng_seed_hex,
            "report_sha256": self.report_sha256,
        }

    def format_student_card(self) -> str:
        """Render a clean 7-bit ASCII diagnostic report card with zero secret disclosure."""
        lines: List[str] = [
            f"=== CODE VERIFICATION REPORT: {self.exercise_id} ({self.entry_function}) ===",
        ]
        if not self.preflight_passed:
            lines.append("STATUS: BLOCKED BY PRE-FLIGHT SANDBOX GATE")
            for v in self.preflight_violations:
                lines.append(
                    f"  [Line {v.lineno}:{v.col_offset}] ({v.rule_id}) {v.pedagogical_message}"
                )
            if self.triggered_conceptual_gap_tags:
                lines.append(
                    "  Remediation Tags: " + ", ".join(self.triggered_conceptual_gap_tags)
                )
            return "\n".join(lines)

        status_txt = "ALL TESTS PASSED" if self.all_passed else "DIAGNOSTIC GAPS DETECTED"
        lines.append(
            f"STATUS: {status_txt} ({self.passed_count}/{self.total_count} passed | "
            f"t2_seed={self.t2_rng_seed_hex})"
        )
        for tr in self.test_reports:
            mark = "[PASS]" if tr.passed else "[FAIL]"
            lines.append(f"  {mark} [{tr.tier}] {tr.public_test_name}")
            if not tr.passed:
                lines.append(f"         Diagnostic: {tr.failure_diagnostic}")
                if tr.conceptual_gap_tag:
                    lines.append(f"         Concept Tag: {tr.conceptual_gap_tag}")
                if tr.hint_nudge:
                    lines.append(f"         Hint: {tr.hint_nudge}")
        return "\n".join(lines)


# ==============================================================================
# 1. AST Pre-Flight Gate & WASM Loop-Budget Instrumentation
# ==============================================================================


class _StudentASTInspector(ast.NodeVisitor):
    """Static AST analyzer blocking introspection, I/O, and frame-escape constructs."""

    def __init__(self, entry_function: Optional[str] = None) -> None:
        self.entry_function = entry_function
        self.top_level_functions: Set[str] = set()
        self.violations: List[PreflightViolation] = []

    def inspect_module(self, tree: ast.Module) -> None:
        for stmt in tree.body:
            if isinstance(stmt, ast.FunctionDef):
                self.top_level_functions.add(stmt.name)
        self.visit(tree)

    def _record(
        self,
        node: ast.AST,
        rule_id: str,
        construct: str,
        message: str,
        gap_tag: str = "HARNESS_SANDBOX_POLICY_VIOLATION",
    ) -> None:
        self.violations.append(
            PreflightViolation(
                rule_id=rule_id,
                lineno=int(getattr(node, "lineno", 1)),
                col_offset=int(getattr(node, "col_offset", 0)),
                construct=construct,
                pedagogical_message=message,
                conceptual_gap_tag=gap_tag,
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root_mod = alias.name.split(".")[0]
            if root_mod in FORBIDDEN_IMPORT_PEDAGOGY:
                self._record(
                    node,
                    "AST_FORBIDDEN_IMPORT",
                    f"import {alias.name}",
                    FORBIDDEN_IMPORT_PEDAGOGY[root_mod],
                )
            elif root_mod not in ALLOWED_STUDENT_IMPORTS:
                self._record(
                    node,
                    "AST_UNAPPROVED_IMPORT",
                    f"import {alias.name}",
                    (
                        f"Module '{alias.name}' is outside the scientific standard subset "
                        f"({', '.join(sorted(ALLOWED_STUDENT_IMPORTS))})."
                    ),
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level and node.level > 0:
            self._record(
                node,
                "AST_RELATIVE_IMPORT",
                "relative import",
                "Relative package imports are disallowed in standalone worksheet cells.",
            )
        mod_name = node.module or ""
        root_mod = mod_name.split(".")[0] if mod_name else ""
        if root_mod in FORBIDDEN_IMPORT_PEDAGOGY:
            self._record(
                node,
                "AST_FORBIDDEN_IMPORT",
                f"from {mod_name} import ...",
                FORBIDDEN_IMPORT_PEDAGOGY[root_mod],
            )
        elif root_mod not in ALLOWED_STUDENT_IMPORTS:
            self._record(
                node,
                "AST_UNAPPROVED_IMPORT",
                f"from {mod_name} import ...",
                (
                    f"Module '{mod_name}' is outside the scientific standard subset "
                    f"({', '.join(sorted(ALLOWED_STUDENT_IMPORTS))})."
                ),
            )
        for alias in node.names:
            if alias.name == "*":
                self._record(
                    node,
                    "AST_WILDCARD_IMPORT",
                    f"from {mod_name} import *",
                    "Wildcard 'import *' is disallowed; import explicit mathematical symbols.",
                )
            elif alias.name in FORBIDDEN_ATTRIBUTES or alias.name in FORBIDDEN_CALL_NAMES:
                self._record(
                    node,
                    "AST_FORBIDDEN_SYMBOL_IMPORT",
                    alias.name,
                    f"Importing restricted symbol '{alias.name}' is disallowed.",
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            fn_name = node.func.id
            if fn_name in FORBIDDEN_CALL_NAMES:
                self._record(
                    node,
                    "AST_FORBIDDEN_BUILTIN_CALL",
                    f"{fn_name}(...)",
                    FORBIDDEN_CALL_NAMES[fn_name],
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr = node.attr
        if attr in FORBIDDEN_ATTRIBUTES:
            self._record(
                node,
                "AST_FORBIDDEN_ATTRIBUTE_SNOOP",
                f".{attr}",
                FORBIDDEN_ATTRIBUTES[attr],
            )
        elif attr.startswith("__") and attr.endswith("__"):
            # Block all dunder attribute accesses in student code
            self._record(
                node,
                "AST_DUNDER_ATTRIBUTE_ACCESS",
                f".{attr}",
                (
                    f"Accessing dunder attribute '.{attr}' is disallowed in student "
                    "scientific functions; use standard NumPy/Python operators and properties."
                ),
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        ident = node.id
        if ident in {"__builtins__", "__loader__", "__spec__", "__package__", "_wasm_loop_guard"}:
            self._record(
                node,
                "AST_RESERVED_IDENTIFIER",
                ident,
                f"Identifier '{ident}' is reserved by the worksheet execution runtime.",
            )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Prevent custom spy classes overriding __eq__ / __float__ / __repr__
        self._record(
            node,
            "AST_CLASS_DEF_DISALLOWED",
            f"class {node.name}",
            (
                "Defining custom classes inside functional physics exercises is disallowed; "
                "implement pure functions operating on numbers, tuples, dicts, and NumPy arrays."
            ),
        )
        self.generic_visit(node)


class _LoopTickInjector(ast.NodeTransformer):
    """Injects `_wasm_loop_guard.tick()` at the top of `for` and `while` loops for WASM safety."""

    def _make_tick_stmt(self, ref_node: ast.AST) -> ast.stmt:
        call = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="_wasm_loop_guard", ctx=ast.Load()),
                    attr="tick",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )
        )
        return ast.copy_location(call, ref_node)

    def visit_While(self, node: ast.While) -> ast.AST:
        self.generic_visit(node)
        node.body.insert(0, self._make_tick_stmt(node))
        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        self.generic_visit(node)
        node.body.insert(0, self._make_tick_stmt(node))
        return node


class _LoopBudgetGuard:
    """Lightweight deterministic iteration counter preventing WASM main-thread hangs."""

    __slots__ = ("max_ticks", "ticks")

    def __init__(self, max_ticks: int = DEFAULT_MAX_LOOP_TICKS) -> None:
        self.max_ticks = max(100, int(max_ticks))
        self.ticks = 0

    def reset(self) -> None:
        self.ticks = 0

    def tick(self) -> None:
        self.ticks += 1
        if self.ticks > self.max_ticks:
            raise _StepBudgetExceededError()


def ast_preflight_check(
    student_source: str,
    entry_function: Optional[str] = None,
) -> PreflightCheckResult:
    """
    Validate student source code against the V2 AST Pre-Flight Gate.

    Rejects syntax errors, forbidden imports (`inspect`, `sys`, `gc`, `os`, ...),
    forbidden calls (`open`, `globals`, `locals`, `eval`, `exec`, ...), frame/traceback
    attribute snooping (`_getframe`, `f_back`, `tb_frame`, `__subclasses__`, ...),
    and missing target entry function signatures.
    """
    if not isinstance(student_source, str) or not student_source.strip():
        v = PreflightViolation(
            rule_id="AST_EMPTY_SUBMISSION",
            lineno=1,
            col_offset=0,
            construct="<empty>",
            pedagogical_message=(
                "Submission is empty. Define the requested Python function before running checks."
            ),
            conceptual_gap_tag="EMPTY_OR_INCOMPLETE_FUNCTION",
        )
        return PreflightCheckResult(passed=False, entry_function_found=False, violations=[v])

    try:
        tree = ast.parse(student_source, filename="<student_submission>", mode="exec")
    except SyntaxError as exc:
        v = PreflightViolation(
            rule_id="AST_SYNTAX_ERROR",
            lineno=int(exc.lineno or 1),
            col_offset=int(exc.offset or 0),
            construct=(exc.text or "").strip()[:60],
            pedagogical_message=(
                f"Python syntax error on line {exc.lineno or 1}: {exc.msg}. "
                "Check parentheses, colons, and indentation."
            ),
            conceptual_gap_tag="PYTHON_SYNTAX_ERROR",
        )
        return PreflightCheckResult(passed=False, entry_function_found=False, violations=[v])

    inspector = _StudentASTInspector(entry_function=entry_function)
    inspector.inspect_module(tree)

    found_entry = True
    if entry_function:
        found_entry = entry_function in inspector.top_level_functions
        if not found_entry:
            inspector.violations.append(
                PreflightViolation(
                    rule_id="AST_MISSING_ENTRY_FUNCTION",
                    lineno=1,
                    col_offset=0,
                    construct=entry_function,
                    pedagogical_message=(
                        f"Expected top-level function 'def {entry_function}(...):' was not "
                        f"found (defined functions: {sorted(inspector.top_level_functions)})."
                    ),
                    conceptual_gap_tag="MISSING_FUNCTION_SIGNATURE",
                )
            )

    return PreflightCheckResult(
        passed=(len(inspector.violations) == 0),
        entry_function_found=found_entry,
        violations=inspector.violations,
    )


# ==============================================================================
# 2. Opaque Sealed Test Bundle Compiler & Verifier
# ==============================================================================


def _derive_bundle_key(obfuscation_key: Optional[str]) -> Tuple[str, bytes]:
    if obfuscation_key is not None and len(obfuscation_key) > 0:
        material = hashlib.sha256(
            b"V2_KEYED_STREAM_v1::" + obfuscation_key.encode("utf-8")
        ).digest()
        return "KEYED_SHA256_STREAM", material
    material = hashlib.sha256(b"V2_DEFAULT_SEALED_STREAM_v1::LIBREDISTILLATIONS").digest()
    return "DEFAULT_OBFUSCATED_STREAM", material


def _xor_sha256_stream(data: bytes, key_material: bytes) -> bytes:
    """Apply deterministic SHA-256 counter-mode XOR stream mask (100% stdlib, WASM-safe)."""
    out = bytearray(len(data))
    block_idx = 0
    offset = 0
    n = len(data)
    while offset < n:
        keystream = hashlib.sha256(
            key_material + block_idx.to_bytes(8, byteorder="big")
        ).digest()
        chunk_len = min(32, n - offset)
        for i in range(chunk_len):
            out[offset + i] = data[offset + i] ^ keystream[i]
        offset += chunk_len
        block_idx += 1
    return bytes(out)


def _compute_bundle_hmac(
    key_material: bytes,
    suite_id: str,
    payload_sha256: str,
    sealed_payload_b85: str,
) -> str:
    msg = f"{BUNDLE_SCHEMA_VERSION}::{suite_id}::{payload_sha256}::{sealed_payload_b85}".encode(
        "ascii"
    )
    return hmac.new(key_material, msg, hashlib.sha256).hexdigest()


def compile_hidden_suite(
    suite_spec: Union[AuthoringSuiteSpec, Dict[str, Any]],
    obfuscation_key: Optional[str] = None,
) -> SealedTestBundle:
    """
    Compile an authoring test suite into an opaque, base85 + zlib + SHA-256/HMAC
    authenticated `SealedTestBundle` containing ZERO plaintext test source or
    expected numerical answers.
    """
    if isinstance(suite_spec, dict):
        tests_list = [
            t if isinstance(t, HiddenTestSpec) else HiddenTestSpec(**t)
            for t in suite_spec["tests"]
        ]
        spec = AuthoringSuiteSpec(
            suite_id=str(suite_spec["suite_id"]),
            lesson_id=str(suite_spec["lesson_id"]),
            exercise_id=str(suite_spec["exercise_id"]),
            entry_function=str(suite_spec["entry_function"]),
            curriculum_slice_ref=dict(suite_spec.get("curriculum_slice_ref", {})),
            tests=tests_list,
            max_loop_ticks=int(suite_spec.get("max_loop_ticks", DEFAULT_MAX_LOOP_TICKS)),
            default_rtol=float(suite_spec.get("default_rtol", 1e-6)),
            default_atol=float(suite_spec.get("default_atol", 1e-9)),
        )
    else:
        spec = suite_spec

    if not spec.tests:
        raise ValueError(f"Suite '{spec.suite_id}' must contain at least 1 HiddenTestSpec.")

    seen_ids: Set[str] = set()
    public_manifest: List[Dict[str, str]] = []
    for t in spec.tests:
        if t.tier not in VALID_TEST_TIERS:
            raise ValueError(
                f"Invalid tier '{t.tier}' in test '{t.test_id}'. Expected one of {sorted(VALID_TEST_TIERS)}."
            )
        if t.test_id in seen_ids:
            raise ValueError(f"Duplicate test_id '{t.test_id}' in suite '{spec.suite_id}'.")
        seen_ids.add(t.test_id)
        # Validate oracle_code syntax at compile time and ensure `_run_oracle` is defined
        oracle_ast = ast.parse(t.oracle_code, filename=f"<oracle:{t.test_id}>", mode="exec")
        fn_names = {
            n.name for n in oracle_ast.body if isinstance(n, ast.FunctionDef)
        }
        if "_run_oracle" not in fn_names:
            raise ValueError(
                f"Test '{t.test_id}' oracle_code must define 'def _run_oracle(student_fn, ctx):'."
            )
        public_manifest.append(
            {
                "test_id": t.test_id,
                "public_test_name": t.public_test_name,
                "tier": t.tier,
            }
        )

    canonical_json = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    raw_bytes = canonical_json.encode("ascii")
    payload_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    compressed = zlib.compress(raw_bytes, level=9)
    key_mode, key_material = _derive_bundle_key(obfuscation_key)
    masked = _xor_sha256_stream(compressed, key_material)
    sealed_b85 = base64.b85encode(masked).decode("ascii")
    auth_sha256 = _compute_bundle_hmac(key_material, spec.suite_id, payload_sha256, sealed_b85)

    return SealedTestBundle(
        schema_version=BUNDLE_SCHEMA_VERSION,
        suite_id=spec.suite_id,
        lesson_id=spec.lesson_id,
        exercise_id=spec.exercise_id,
        entry_function=spec.entry_function,
        curriculum_slice_ref=copy.deepcopy(spec.curriculum_slice_ref),
        max_loop_ticks=spec.max_loop_ticks,
        test_count=len(spec.tests),
        public_manifest=public_manifest,
        key_mode=key_mode,
        payload_sha256=payload_sha256,
        auth_sha256=auth_sha256,
        sealed_payload_b85=sealed_b85,
    )


def _unseal_bundle_internal(
    bundle: SealedTestBundle,
    obfuscation_key: Optional[str] = None,
) -> AuthoringSuiteSpec:
    """Authenticate and decode a `SealedTestBundle` entirely in private memory."""
    expected_mode = (
        "KEYED_SHA256_STREAM"
        if (obfuscation_key is not None and len(obfuscation_key) > 0)
        else "DEFAULT_OBFUSCATED_STREAM"
    )
    if bundle.key_mode != expected_mode:
        raise BundleIntegrityError(
            f"Bundle key_mode mismatch for suite '{bundle.suite_id}': "
            f"bundle requires {bundle.key_mode}."
        )
    _, key_material = _derive_bundle_key(obfuscation_key)
    expected_hmac = _compute_bundle_hmac(
        key_material, bundle.suite_id, bundle.payload_sha256, bundle.sealed_payload_b85
    )
    if not hmac.compare_digest(expected_hmac, bundle.auth_sha256):
        raise BundleIntegrityError(
            f"Sealed bundle authentication failed for suite '{bundle.suite_id}'."
        )
    try:
        masked = base64.b85decode(bundle.sealed_payload_b85.encode("ascii"))
        compressed = _xor_sha256_stream(masked, key_material)
        raw_bytes = zlib.decompress(compressed)
    except Exception as exc:
        raise BundleIntegrityError(
            f"Corrupted sealed payload in suite '{bundle.suite_id}'."
        ) from exc

    if hashlib.sha256(raw_bytes).hexdigest() != bundle.payload_sha256:
        raise BundleIntegrityError(
            f"Decompressed SHA-256 digest mismatch for suite '{bundle.suite_id}'."
        )

    payload = json.loads(raw_bytes.decode("ascii"))
    return AuthoringSuiteSpec(
        suite_id=str(payload["suite_id"]),
        lesson_id=str(payload["lesson_id"]),
        exercise_id=str(payload["exercise_id"]),
        entry_function=str(payload["entry_function"]),
        curriculum_slice_ref=dict(payload.get("curriculum_slice_ref", {})),
        tests=[HiddenTestSpec(**t) for t in payload["tests"]],
        max_loop_ticks=int(payload.get("max_loop_ticks", DEFAULT_MAX_LOOP_TICKS)),
        default_rtol=float(payload.get("default_rtol", 1e-6)),
        default_atol=float(payload.get("default_atol", 1e-9)),
    )


# ==============================================================================
# 3. Hardened Sandbox Runtime & Zero-Disclosure Oracle Helpers
# ==============================================================================


_ALLOWED_MODULE_OBJECTS: Dict[str, Any] = {
    "math": math,
    "cmath": cmath,
    "random": random,
    "statistics": statistics,
    "itertools": itertools,
    "functools": functools,
    "collections": collections,
    "numbers": numbers,
}
if np is not None:
    _ALLOWED_MODULE_OBJECTS["numpy"] = np


def _safe_student_import(
    name: str,
    globals_dict: Optional[Dict[str, Any]] = None,
    locals_dict: Optional[Dict[str, Any]] = None,
    fromlist: Sequence[str] = (),
    level: int = 0,
) -> Any:
    if level != 0:
        raise ImportError("Relative imports are disallowed in worksheet cells.")
    root_mod = name.split(".")[0]
    if root_mod not in _ALLOWED_MODULE_OBJECTS or name not in _ALLOWED_MODULE_OBJECTS:
        raise ImportError(f"Import of '{name}' is disallowed by the worksheet sandbox.")
    return _ALLOWED_MODULE_OBJECTS[name]


def _noop_print(*_args: Any, **_kwargs: Any) -> None:
    """Swallows student `print(...)` calls during hidden oracle execution so secret inputs cannot leak."""
    return None


def _build_restricted_builtins() -> Dict[str, Any]:
    safe_names = (
        "abs",
        "all",
        "any",
        "bool",
        "callable",
        "chr",
        "complex",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "frozenset",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        # Safe standard exceptions
        "ArithmeticError",
        "AssertionError",
        "Exception",
        "FloatingPointError",
        "IndexError",
        "KeyError",
        "LookupError",
        "NameError",
        "NotImplementedError",
        "OverflowError",
        "RuntimeError",
        "StopIteration",
        "TypeError",
        "ValueError",
        "ZeroDivisionError",
    )
    env: Dict[str, Any] = {k: getattr(builtins, k) for k in safe_names}
    env["__import__"] = _safe_student_import
    env["print"] = _noop_print
    return env


def _clone_input_arg(val: Any) -> Any:
    if np is not None and isinstance(val, np.ndarray):
        return np.array(val, copy=True)
    if isinstance(val, (int, float, complex, bool, str, type(None))):
        return val
    if isinstance(val, tuple):
        return tuple(_clone_input_arg(x) for x in val)
    if isinstance(val, list):
        return [_clone_input_arg(x) for x in val]
    if isinstance(val, dict):
        return {_clone_input_arg(k): _clone_input_arg(v) for k, v in val.items()}
    if callable(val):
        return val
    return copy.deepcopy(val)


def _sanitize_student_return(val: Any, _depth: int = 0) -> Any:
    """
    Coerce student return values into pure built-in scalars/containers or base
    `numpy.ndarray` copies so student code cannot return custom proxy objects whose
    `__eq__` or `__float__` methods spy on secret expected reference vectors.
    """
    if _depth > 12:
        raise _ReturnContractViolation("Return nesting depth exceeded.")
    t = type(val)
    if t in (int, float, complex, bool, str, type(None)):
        return val
    if np is not None:
        if t is np.ndarray:
            if val.dtype == object:
                raise _ReturnContractViolation("Object-dtype arrays are disallowed.")
            return np.array(val, copy=True)
        if isinstance(val, np.generic):
            return val.item()
    if t is tuple:
        return tuple(_sanitize_student_return(x, _depth + 1) for x in val)
    if t is list:
        return [_sanitize_student_return(x, _depth + 1) for x in val]
    if t is dict:
        return {
            _sanitize_student_return(k, _depth + 1): _sanitize_student_return(v, _depth + 1)
            for k, v in val.items()
        }
    raise _ReturnContractViolation(f"Unsupported return type: {t.__name__}")


def _scrub_traceback_frames(exc: BaseException) -> None:
    tb = exc.__traceback__
    while tb is not None:
        try:
            traceback.clear_frames(tb)
        except RuntimeError:
            pass
        tb = tb.tb_next


class _OracleContext:
    """Zero-leak assertion and RNG helper context exposed ONLY inside sealed oracle code."""

    def __init__(self, seed: int, default_rtol: float, default_atol: float) -> None:
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.np = np
        self.math = math
        self.np_rng = np.random.default_rng(self.seed) if np is not None else None
        self.default_rtol = default_rtol
        self.default_atol = default_atol

    def fail(self) -> None:
        raise _OracleAssertionFailed()

    def assert_true(self, condition: bool) -> None:
        if not bool(condition):
            raise _OracleAssertionFailed()

    def assert_finite(self, val: Any) -> None:
        if np is not None and isinstance(val, np.ndarray):
            if val.size == 0 or not bool(np.all(np.isfinite(val))):
                raise _OracleAssertionFailed()
            return
        if isinstance(val, (int, float)):
            if not math.isfinite(float(val)):
                raise _OracleAssertionFailed()
            return
        if isinstance(val, (list, tuple)):
            if len(val) == 0:
                raise _OracleAssertionFailed()
            for item in val:
                self.assert_finite(item)
            return
        raise _OracleAssertionFailed()

    def assert_array_contract(
        self,
        val: Any,
        expected_shape: Optional[Tuple[int, ...]] = None,
        require_ndarray: bool = False,
    ) -> Any:
        if np is None:
            raise RuntimeError("NumPy is required for array contract verification.")
        if require_ndarray and type(val) is not np.ndarray:
            raise _OracleAssertionFailed()
        try:
            arr = np.asarray(val, dtype=float)
        except (TypeError, ValueError) as exc:
            raise _OracleAssertionFailed() from exc
        if not bool(np.all(np.isfinite(arr))):
            raise _OracleAssertionFailed()
        if expected_shape is not None and arr.shape != expected_shape:
            raise _OracleAssertionFailed()
        return arr

    def assert_close(
        self,
        actual: Any,
        expected: Any,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ) -> None:
        rt = self.default_rtol if rtol is None else float(rtol)
        at = self.default_atol if atol is None else float(atol)

        if np is not None and (
            isinstance(actual, (np.ndarray, list, tuple))
            or isinstance(expected, (np.ndarray, list, tuple))
        ):
            try:
                a_arr = np.asarray(actual, dtype=float)
                e_arr = np.asarray(expected, dtype=float)
            except (TypeError, ValueError) as exc:
                raise _OracleAssertionFailed() from exc
            if a_arr.shape != e_arr.shape:
                raise _OracleAssertionFailed()
            if not bool(np.all(np.isfinite(a_arr))):
                raise _OracleAssertionFailed()
            if not bool(np.allclose(a_arr, e_arr, rtol=rt, atol=at)):
                raise _OracleAssertionFailed()
            return

        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            raise _OracleAssertionFailed()
        a_flt = float(actual)
        e_flt = float(expected)
        if not math.isfinite(a_flt):
            raise _OracleAssertionFailed()
        if not math.isclose(a_flt, e_flt, rel_tol=rt, abs_tol=at):
            raise _OracleAssertionFailed()


def _compile_student_callable(
    student_source: str,
    entry_function: str,
    max_loop_ticks: int,
) -> Tuple[Callable[..., Any], _LoopBudgetGuard]:
    tree = ast.parse(student_source, filename="<student_submission>", mode="exec")
    instrumented = _LoopTickInjector().visit(tree)
    ast.fix_missing_locations(instrumented)
    code_obj = compile(instrumented, filename="<student_submission>", mode="exec")

    loop_guard = _LoopBudgetGuard(max_ticks=max_loop_ticks)
    student_globals: Dict[str, Any] = {
        "__builtins__": _build_restricted_builtins(),
        "__name__": "__student_cell__",
        "_wasm_loop_guard": loop_guard,
        "math": math,
    }
    if np is not None:
        student_globals["np"] = np
        student_globals["numpy"] = np

    sink_out = io.StringIO()
    sink_err = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = sink_out
        sys.stderr = sink_err
        exec(code_obj, student_globals, student_globals)
    finally:
        sys.stdout = old_out
        sys.stderr = old_err

    raw_fn = student_globals.get(entry_function)
    if not callable(raw_fn):
        raise TypeError(f"Symbol '{entry_function}' is not callable.")

    def _wrapped_student_fn(*args: Any, **kwargs: Any) -> Any:
        cloned_args = tuple(_clone_input_arg(a) for a in args)
        cloned_kwargs = {k: _clone_input_arg(v) for k, v in kwargs.items()}
        s_out = io.StringIO()
        s_err = io.StringIO()
        prev_out, prev_err = sys.stdout, sys.stderr
        try:
            sys.stdout = s_out
            sys.stderr = s_err
            result = raw_fn(*cloned_args, **cloned_kwargs)
            return _sanitize_student_return(result)
        finally:
            sys.stdout = prev_out
            sys.stderr = prev_err

    return _wrapped_student_fn, loop_guard


def _classify_safe_exception_name(exc: BaseException) -> str:
    cls_name = type(exc).__name__
    if cls_name in _SAFE_STD_EXCEPTION_NAMES:
        return cls_name
    return "StudentRuntimeError"


def derive_t2_rng_seed(
    session_id: Optional[str] = None,
    exercise_id: str = "",
    attempt_index: int = 0,
    rng_seed: Optional[Union[int, str]] = None,
) -> Tuple[int, str]:
    """
    Derive a 64-bit unsigned integer seed and canonical 16-hex `t2_rng_seed_hex`
    for deterministic Tier-2 (`RANDOMIZED_PHYSICAL_INVARIANTS`) replay and
    Tree-of-Knowledge SHA-256 event-chain logging.
    """
    if rng_seed is not None:
        if isinstance(rng_seed, str):
            cleaned = rng_seed.strip().lower()
            if cleaned.startswith("0x"):
                cleaned = cleaned[2:]
            seed_u64 = int(cleaned, 16) & 0xFFFFFFFFFFFFFFFF
        else:
            seed_u64 = int(rng_seed) & 0xFFFFFFFFFFFFFFFF
        return seed_u64, f"{seed_u64:016x}"

    if session_id is not None and str(session_id).strip():
        material = (
            f"V2_T2_RNG_SEED_v1::{session_id}::{exercise_id}::{int(attempt_index)}"
        ).encode("utf-8")
        digest_hex = hashlib.sha256(material).hexdigest()[:16]
        return int(digest_hex, 16), digest_hex

    seed_u64 = (
        int.from_bytes(os.urandom(8), "big") ^ (time.perf_counter_ns() & 0xFFFFFFFFFFFFFFFF)
    ) & 0xFFFFFFFFFFFFFFFF
    return seed_u64, f"{seed_u64:016x}"


def execute_hidden_suite(
    bundle: Union[SealedTestBundle, Dict[str, Any]],
    student_source: str,
    obfuscation_key: Optional[str] = None,
    verification_seed: Optional[Union[int, str]] = None,
    rng_seed: Optional[Union[int, str]] = None,
    session_id: Optional[str] = None,
    attempt_index: int = 0,
) -> SuiteExecutionReport:
    """
    Execute student source code against a sealed hidden-test bundle.

    Enforces:
    1. Bundle HMAC/SHA-256 integrity verification (anti-spoiler corruption check).
    2. AST Pre-Flight Gate prior to unsealing or running student code.
    3. Isolated execution with loop-tick budget, muted stdout/stderr side-channels,
       deep-copied inputs, sanitized return primitives/arrays, and scrubbed frames.
    4. Deterministic `t2_rng_seed_hex` generation & replay from `(session_id,
       exercise_id, attempt_index)` or caller `rng_seed` / `verification_seed`.
    5. Diagnostic-Without-Disclosure `SuiteExecutionReport` output contract.
    """
    sealed = bundle if isinstance(bundle, SealedTestBundle) else SealedTestBundle.from_dict(bundle)

    # Authenticate bundle integrity up-front
    spec = _unseal_bundle_internal(sealed, obfuscation_key=obfuscation_key)

    effective_seed_arg = rng_seed if rng_seed is not None else verification_seed
    base_entropy, t2_rng_seed_hex = derive_t2_rng_seed(
        session_id=session_id,
        exercise_id=spec.exercise_id,
        attempt_index=attempt_index,
        rng_seed=effective_seed_arg,
    )

    # Step 1: AST Pre-Flight Gate
    preflight = ast_preflight_check(student_source, entry_function=spec.entry_function)
    if not preflight.passed:
        gap_tags: List[str] = []
        for v in preflight.violations:
            if v.conceptual_gap_tag and v.conceptual_gap_tag not in gap_tags:
                gap_tags.append(v.conceptual_gap_tag)
        payload_digest = hashlib.sha256(
            json.dumps(
                {
                    "suite_id": spec.suite_id,
                    "preflight_passed": False,
                    "t2_rng_seed_hex": t2_rng_seed_hex,
                    "violations": [v.to_dict() for v in preflight.violations],
                },
                sort_keys=True,
                ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest()
        return SuiteExecutionReport(
            suite_id=spec.suite_id,
            lesson_id=spec.lesson_id,
            exercise_id=spec.exercise_id,
            entry_function=spec.entry_function,
            preflight_passed=False,
            preflight_violations=preflight.violations,
            all_passed=False,
            passed_count=0,
            total_count=len(spec.tests),
            test_reports=[],
            triggered_conceptual_gap_tags=gap_tags,
            curriculum_slice_ref=copy.deepcopy(spec.curriculum_slice_ref),
            report_sha256=payload_digest,
            t2_rng_seed_hex=t2_rng_seed_hex,
        )

    # Step 2: Compile instrumented student module once in isolated scope
    try:
        wrapped_fn, loop_guard = _compile_student_callable(
            student_source=student_source,
            entry_function=spec.entry_function,
            max_loop_ticks=spec.max_loop_ticks,
        )
    except _StepBudgetExceededError:
        diag = (
            f"Top-level module initialization exceeded loop budget "
            f"({spec.max_loop_ticks} ticks); remove top-level infinite loops."
        )
        t_reports = [
            TestCaseReport(
                test_id=t.test_id,
                public_test_name=t.public_test_name,
                tier=t.tier,
                passed=False,
                failure_diagnostic=diag,
                conceptual_gap_tag="INFINITE_LOOP_OR_COMPLEXITY_GUARD",
                hint_nudge=t.hint_nudge,
                execution_ms=0.0,
            )
            for t in spec.tests
        ]
        digest = hashlib.sha256(
            f"{spec.suite_id}::{t2_rng_seed_hex}".encode("ascii")
        ).hexdigest()
        return SuiteExecutionReport(
            suite_id=spec.suite_id,
            lesson_id=spec.lesson_id,
            exercise_id=spec.exercise_id,
            entry_function=spec.entry_function,
            preflight_passed=True,
            preflight_violations=[],
            all_passed=False,
            passed_count=0,
            total_count=len(spec.tests),
            test_reports=t_reports,
            triggered_conceptual_gap_tags=["INFINITE_LOOP_OR_COMPLEXITY_GUARD"],
            curriculum_slice_ref=copy.deepcopy(spec.curriculum_slice_ref),
            report_sha256=digest,
            t2_rng_seed_hex=t2_rng_seed_hex,
        )
    except BaseException as exc:
        safe_exc = _classify_safe_exception_name(exc)
        _scrub_traceback_frames(exc)
        diag = (
            f"Module initialization raised {safe_exc} before tests could invoke "
            f"'{spec.entry_function}'. Keep top-level code limited to imports and 'def'."
        )
        t_reports = [
            TestCaseReport(
                test_id=t.test_id,
                public_test_name=t.public_test_name,
                tier=t.tier,
                passed=False,
                failure_diagnostic=diag,
                conceptual_gap_tag="MODULE_INITIALIZATION_ERROR",
                hint_nudge=t.hint_nudge,
                execution_ms=0.0,
            )
            for t in spec.tests
        ]
        digest = hashlib.sha256(
            f"{spec.suite_id}::{safe_exc}::{t2_rng_seed_hex}".encode("ascii")
        ).hexdigest()
        return SuiteExecutionReport(
            suite_id=spec.suite_id,
            lesson_id=spec.lesson_id,
            exercise_id=spec.exercise_id,
            entry_function=spec.entry_function,
            preflight_passed=True,
            preflight_violations=[],
            all_passed=False,
            passed_count=0,
            total_count=len(spec.tests),
            test_reports=t_reports,
            triggered_conceptual_gap_tags=["MODULE_INITIALIZATION_ERROR"],
            curriculum_slice_ref=copy.deepcopy(spec.curriculum_slice_ref),
            report_sha256=digest,
            t2_rng_seed_hex=t2_rng_seed_hex,
        )

    # Step 3: Execute 3-tier tests in isolated scrubbed oracle scopes
    test_reports: List[TestCaseReport] = []
    triggered_gaps: List[str] = []
    passed_count = 0

    for idx, t_spec in enumerate(spec.tests):
        loop_guard.reset()
        test_seed = (base_entropy + 0x9E3779B97F4A7C15 * (idx + 1)) & 0x7FFFFFFFFFFFFFFF
        ctx = _OracleContext(
            seed=test_seed,
            default_rtol=spec.default_rtol,
            default_atol=spec.default_atol,
        )
        oracle_ns: Dict[str, Any] = {
            "__builtins__": builtins.__dict__,
            "math": math,
            "np": np,
        }

        t0 = time.perf_counter()
        passed = False
        diag_out = ""
        gap_out: Optional[str] = None
        try:
            oracle_code_obj = compile(
                t_spec.oracle_code, filename=f"<sealed_oracle:{t_spec.test_id}>", mode="exec"
            )
            exec(oracle_code_obj, oracle_ns, oracle_ns)
            oracle_fn = oracle_ns["_run_oracle"]
            oracle_fn(wrapped_fn, ctx)
            passed = True
        except (_OracleAssertionFailed, AssertionError):
            passed = False
            diag_out = t_spec.failure_diagnostic
            gap_out = t_spec.conceptual_gap_tag
        except _ReturnContractViolation:
            passed = False
            diag_out = (
                f"{t_spec.failure_diagnostic} [Return contract violation: return standard "
                "Python floats/tuples or NumPy float arrays.]"
            )
            gap_out = t_spec.conceptual_gap_tag
        except _StepBudgetExceededError as exc:
            _scrub_traceback_frames(exc)
            passed = False
            diag_out = (
                f"Execution exceeded loop step budget ({spec.max_loop_ticks} iterations) "
                f"during '{t_spec.public_test_name}'. Check while/for termination conditions."
            )
            gap_out = "INFINITE_LOOP_OR_COMPLEXITY_GUARD"
        except BaseException as exc:
            safe_exc = _classify_safe_exception_name(exc)
            _scrub_traceback_frames(exc)
            passed = False
            diag_out = f"{t_spec.failure_diagnostic} [Runtime exception: {safe_exc}]"
            gap_out = t_spec.conceptual_gap_tag
        finally:
            oracle_ns.clear()

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        if passed:
            passed_count += 1
        elif gap_out and gap_out not in triggered_gaps:
            triggered_gaps.append(gap_out)

        test_reports.append(
            TestCaseReport(
                test_id=t_spec.test_id,
                public_test_name=t_spec.public_test_name,
                tier=t_spec.tier,
                passed=passed,
                failure_diagnostic=diag_out,
                conceptual_gap_tag=gap_out,
                hint_nudge=t_spec.hint_nudge if not passed else "",
                execution_ms=elapsed_ms,
            )
        )

    # Clear spec reference promptly
    total_count = len(test_reports)
    all_passed = passed_count == total_count
    digest_material = json.dumps(
        {
            "suite_id": sealed.suite_id,
            "t2_rng_seed_hex": t2_rng_seed_hex,
            "passed_count": passed_count,
            "total_count": total_count,
            "results": [
                {"test_id": r.test_id, "passed": r.passed, "gap": r.conceptual_gap_tag}
                for r in test_reports
            ],
        },
        sort_keys=True,
        ensure_ascii=True,
    ).encode("ascii")

    return SuiteExecutionReport(
        suite_id=sealed.suite_id,
        lesson_id=sealed.lesson_id,
        exercise_id=sealed.exercise_id,
        entry_function=sealed.entry_function,
        preflight_passed=True,
        preflight_violations=[],
        all_passed=all_passed,
        passed_count=passed_count,
        total_count=total_count,
        test_reports=test_reports,
        triggered_conceptual_gap_tags=triggered_gaps,
        curriculum_slice_ref=copy.deepcopy(sealed.curriculum_slice_ref),
        report_sha256=hashlib.sha256(digest_material).hexdigest(),
        t2_rng_seed_hex=t2_rng_seed_hex,
    )


# ==============================================================================
# 4. Canonical Atomic Scale Simulations Reference Suites (L05 PBC, L04 Verlet, L15 Metropolis)
# ==============================================================================


def build_reference_pbc_minimum_image_suite() -> AuthoringSuiteSpec:
    """
    Canonical 5-test 3-tier suite for Lesson 05 `minimum_image_displacement(dr, box_length)`.
    Binds directly to `lesson_05_pbc_and_neighbor_lists.yaml#L276-L313`.
    """
    slice_ref = {
        "entity_id": "L05::sec_mic_and_cutoffs",
        "section_id": "sec_mic_and_cutoffs",
        "rel_path": (
            "courses/"
            "Atomic Scale Simulations/worksheets/lesson_05_pbc_and_neighbor_lists.yaml"
        ),
        "start_line": 276,
        "end_line": 313,
        "ckv_id": "CKV-1",
        "sha256_64": "0ab8e85cac3ed3f8675e943689876fcc4d3e9e1718d8bbb76f2a176152af4e38",
        "sha256_16": "0ab8e85cac3ed3f8",
        "canonical_key": (
            "courses/"
            "Atomic Scale Simulations/worksheets/lesson_05_pbc_and_neighbor_lists.yaml"
            "#L276-L313[CKV-1:0ab8e85cac3ed3f8]"
        ),
    }

    tests = [
        HiddenTestSpec(
            test_id="L05_EX01_T1_CONTRACT",
            public_test_name=(
                "Test 1/5: 3D Displacement Array Shape, Finite Values & Dimensional Sanity"
            ),
            tier=TIER_CONTRACT_AND_UNITS,
            conceptual_gap_tag="ARRAY_SHAPE_OR_FINITE_CONTRACT_GAP",
            failure_diagnostic=(
                "Output failed 3D array shape or finiteness check: expected a finite 1D "
                "3-component float array matching input displacement dimensionality."
            ),
            hint_nudge="Convert `dr` with `np.asarray(dr, dtype=float)` and preserve its shape.",
            oracle_code="""
def _run_oracle(student_fn, ctx):
    out_3d = student_fn([1.2, -0.8, 0.4], 10.0)
    arr = ctx.assert_array_contract(out_3d, expected_shape=(3,))
    ctx.assert_close(arr, [1.2, -0.8, 0.4], rtol=1e-7, atol=1e-9)
    out_batch = student_fn([[0.5, -1.5, 2.0], [-0.2, 0.3, -0.9]], 8.0)
    arr_b = ctx.assert_array_contract(out_batch, expected_shape=(2, 3))
    ctx.assert_close(arr_b, [[0.5, -1.5, 2.0], [-0.2, 0.3, -0.9]], rtol=1e-7, atol=1e-9)
""",
        ),
        HiddenTestSpec(
            test_id="L05_EX01_T2_ANTISYMMETRY",
            public_test_name=(
                "Test 2/5: Randomized Pairwise Antisymmetry & Boundedness (|dr_mic| <= L/2)"
            ),
            tier=TIER_RANDOMIZED_INVARIANTS,
            conceptual_gap_tag="PBC_PAIR_ANTISYMMETRY_OR_BOUND_GAP",
            failure_diagnostic=(
                "Randomized physical invariant failed: wrapped displacement must satisfy "
                "|dr_mic| <= L/2 in every Cartesian component and antisymmetry "
                "mic(-dr, L) == -mic(dr, L) away from +/- L/2."
            ),
            hint_nudge="Use `dr - L * np.round(dr / L)` so positive and negative displacements wrap symmetrically.",
            oracle_code="""
def _run_oracle(student_fn, ctx):
    for _ in range(25):
        L = ctx.rng.uniform(5.5, 24.0)
        # Sample strictly away from half-integer multiples of L so tie-breaking cannot break antisymmetry
        vec = []
        for _d in range(3):
            frac = ctx.rng.uniform(-3.4, 3.4)
            if abs(abs(frac) - round(abs(frac) - 0.5) - 0.5) < 0.03:
                frac += 0.07
            vec.append(frac * L)
        pos = ctx.assert_array_contract(student_fn(vec, L), expected_shape=(3,))
        neg = ctx.assert_array_contract(student_fn([-x for x in vec], L), expected_shape=(3,))
        ctx.assert_true(bool(ctx.np.all(ctx.np.abs(pos) <= 0.5 * L + 1e-9)))
        ctx.assert_close(pos, -neg, rtol=1e-7, atol=1e-9)
""",
        ),
        HiddenTestSpec(
            test_id="L05_EX01_T3_HALF_BOX_WRAP",
            public_test_name=(
                "Test 3/5: Minimum-Image Periodic Wrap Across Box Half-Length (|dx| > L/2)"
            ),
            tier=TIER_RANDOMIZED_INVARIANTS,
            conceptual_gap_tag="PBC_MINIMUM_IMAGE_WRAP_GAP",
            failure_diagnostic=(
                "Separations exceeding half the simulation box (|dx| > L/2) were not wrapped "
                "to the nearest periodic image inside [-L/2, L/2]."
            ),
            hint_nudge="Subtract `box_length * np.round(dr / box_length)` rather than leaving raw coordinates unwrapped.",
            oracle_code="""
def _run_oracle(student_fn, ctx):
    for _ in range(25):
        L = ctx.rng.uniform(6.0, 18.0)
        base = [ctx.rng.uniform(-0.44 * L, 0.44 * L) for _ in range(3)]
        shifts = [ctx.rng.choice([-2, -1, 1, 2]) for _ in range(3)]
        displaced = [base[k] + shifts[k] * L for k in range(3)]
        got = ctx.assert_array_contract(student_fn(displaced, L), expected_shape=(3,))
        ctx.assert_close(got, base, rtol=1e-7, atol=1e-9)
""",
        ),
        HiddenTestSpec(
            test_id="L05_EX01_T4_LATTICE_INVARIANCE",
            public_test_name=(
                "Test 4/5: Multi-Cell Bravais Lattice Shift Invariance (dr + n*L -> dr_mic)"
            ),
            tier=TIER_RANDOMIZED_INVARIANTS,
            conceptual_gap_tag="PBC_MULTI_IMAGE_WRAP_GAP",
            failure_diagnostic=(
                "Multi-box periodic shifts (|dx| > 1.5*L) failed Bravais lattice invariance: "
                "a single `if dx > L/2: dx -= L` branch fails when atoms diffuse across "
                "multiple periodic cells; use nearest-integer rounding instead."
            ),
            hint_nudge="Avoid single-pass `if/elif` shifts; `dr - L * np.round(dr / L)` handles arbitrary integer cells `n * L`.",
            oracle_code="""
def _run_oracle(student_fn, ctx):
    for _ in range(20):
        L = ctx.rng.uniform(4.5, 15.0)
        core = [ctx.rng.uniform(-0.42 * L, 0.42 * L) for _ in range(3)]
        big_n = [ctx.rng.choice([-7, -5, -3, 3, 5, 8]) for _ in range(3)]
        far_vec = [core[i] + big_n[i] * L for i in range(3)]
        got = ctx.assert_array_contract(student_fn(far_vec, L), expected_shape=(3,))
        ctx.assert_close(got, core, rtol=1e-7, atol=1e-9)
""",
        ),
        HiddenTestSpec(
            test_id="L05_EX01_T5_SEALED_EDGES",
            public_test_name=(
                "Test 5/5: Sealed Edge Regimes (Corner Crossings, Orthorhombic Box & |dx| = L/2)"
            ),
            tier=TIER_SEALED_VECTORS,
            conceptual_gap_tag="PBC_BOUNDARY_CORNER_OR_ORTHORHOMBIC_GAP",
            failure_diagnostic=(
                "Edge-case regime check failed under corner crossings (`L - eps`), "
                "half-box boundary distance norm (`|dx| = L/2`), or per-axis orthorhombic "
                "box-length vectors `[Lx, Ly, Lz]`."
            ),
            hint_nudge="Ensure `box_length` works as both a scalar `L` and a 3-element array `[Lx, Ly, Lz]`, and `|dr_mic|` at `L/2` has magnitude `0.5 * L`.",
            oracle_code="""
def _run_oracle(student_fn, ctx):
    # 1. Corner-to-corner across cubic box L = 12.0
    got_corner = ctx.assert_array_contract(
        student_fn([11.875, -11.75, 6.25], 12.0), expected_shape=(3,)
    )
    ctx.assert_close(got_corner, [-0.125, 0.25, -5.75], rtol=1e-7, atol=1e-9)
    # 2. Exact half-box magnitude check (accepts either -L/2 or +L/2 convention)
    got_half = ctx.assert_array_contract(student_fn([5.0, -5.0, 0.0], 10.0), expected_shape=(3,))
    ctx.assert_close(ctx.np.abs(got_half), [5.0, 5.0, 0.0], rtol=1e-7, atol=1e-9)
    # 3. Orthorhombic box vector [Lx, Ly, Lz]
    got_ortho = ctx.assert_array_contract(
        student_fn([9.2, -13.5, 4.9], [10.0, 14.0, 8.0]), expected_shape=(3,)
    )
    ctx.assert_close(got_ortho, [-0.8, 0.5, -3.1], rtol=1e-7, atol=1e-9)
""",
        ),
    ]

    return AuthoringSuiteSpec(
        suite_id="suite_l05_ex01_minimum_image_displacement",
        lesson_id="lesson_05",
        exercise_id="L05_EX01_minimum_image_displacement",
        entry_function="minimum_image_displacement",
        curriculum_slice_ref=slice_ref,
        tests=tests,
    )


def build_reference_velocity_verlet_suite() -> AuthoringSuiteSpec:
    """
    Canonical 5-test 3-tier suite for Lesson 04 `velocity_verlet_step(r, v, mass, dt, force_fn)`.
    Tests output contract, exact time-reversibility, `O(dt^2)` shadow-Hamiltonian energy
    fluctuation scaling, single-force-evaluation efficiency, and sealed harmonic/anharmonic steps.
    """
    slice_ref = {
        "entity_id": "L04::sec_integrators_and_symplectic_geometry",
        "section_id": "sec_integrators_and_symplectic_geometry",
        "rel_path": (
            "courses/"
            "Atomic Scale Simulations/worksheets/lesson_04_molecular_dynamics.yaml"
        ),
        "start_line": 44,
        "end_line": 74,
        "ckv_id": "CKV-1",
        "sha256_64": "0db93a83c72cb7c5e915f4c8adc2faba2173f566f677cba53f5b8d1a1671bc82",
        "sha256_16": "0db93a83c72cb7c5",
        "canonical_key": (
            "courses/"
            "Atomic Scale Simulations/worksheets/lesson_04_molecular_dynamics.yaml"
            "#L44-L74[CKV-1:0db93a83c72cb7c5]"
        ),
    }

    tests = [
        HiddenTestSpec(
            test_id="L04_EX01_T1_CONTRACT",
            public_test_name="Test 1/5: Synchronized (r_new, v_new) Tuple Contract & Array Shapes",
            tier=TIER_CONTRACT_AND_UNITS,
            conceptual_gap_tag="VERLET_RETURN_CONTRACT_GAP",
            failure_diagnostic=(
                "Expected `velocity_verlet_step` to return a 2-tuple `(r_new, v_new)` of "
                "finite NumPy float arrays matching input shape."
            ),
            hint_nudge="Return `(r_next, v_next)` as float arrays with the exact shape of `r` and `v`.",
            oracle_code="""
def _run_oracle(student_fn, ctx):
    res = student_fn([1.0, 0.0], [0.0, 0.5], 2.0, 0.01, lambda x: -ctx.np.asarray(x, dtype=float))
    ctx.assert_true(isinstance(res, tuple) and len(res) == 2)
    r_n = ctx.assert_array_contract(res[0], expected_shape=(2,))
    v_n = ctx.assert_array_contract(res[1], expected_shape=(2,))
    ctx.assert_true(r_n.shape == v_n.shape)
""",
        ),
        HiddenTestSpec(
            test_id="L04_EX01_T2_TIME_REVERSIBILITY",
            public_test_name=(
                "Test 2/5: Algebraic Symplectic Time-Reversibility Under Velocity Inversion"
            ),
            tier=TIER_RANDOMIZED_INVARIANTS,
            conceptual_gap_tag="VERLET_SYMPLECTIC_REVERSIBILITY_GAP",
            failure_diagnostic=(
                "Trajectory failed algebraic time-reversibility: propagating forward N steps, "
                "negating velocity `v -> -v`, and propagating N steps did not recover `(r_0, -v_0)`. "
                "Forward Euler (`v_new = v + a(r_old)*dt`) is NOT time-reversible; Velocity Verlet "
                "requires the half-step average `0.5 * (a_old + a_new) * dt`."
            ),
            hint_nudge="Compute `r_new = r + v*dt + 0.5*a_old*dt**2`, then `a_new = force_fn(r_new)/mass`, then `v_new = v + 0.5*(a_old + a_new)*dt`.",
            oracle_code="""
def _run_oracle(student_fn, ctx):
    def anharmonic_force(pos):
        p = ctx.np.asarray(pos, dtype=float)
        return -1.7 * p - 0.45 * (p ** 3)
    r0 = ctx.np.array([ctx.rng.uniform(0.4, 1.3), ctx.rng.uniform(-1.2, -0.3)])
    v0 = ctx.np.array([ctx.rng.uniform(-0.8, 0.8), ctx.rng.uniform(-0.8, 0.8)])
    mass = ctx.rng.uniform(0.8, 2.5)
    dt = 0.025
    r, v = r0.copy(), v0.copy()
    for _ in range(12):
        r, v = student_fn(r, v, mass, dt, anharmonic_force)
    v = -ctx.np.asarray(v, dtype=float)
    for _ in range(12):
        r, v = student_fn(r, v, mass, dt, anharmonic_force)
    ctx.assert_close(r, r0, rtol=1e-9, atol=1e-11)
    ctx.assert_close(v, -v0, rtol=1e-9, atol=1e-11)
""",
        ),
        HiddenTestSpec(
            test_id="L04_EX01_T3_SHADOW_SCALING",
            public_test_name=(
                "Test 3/5: Global O(dt^2) Shadow-Hamiltonian Energy Fluctuation Scaling"
            ),
            tier=TIER_RANDOMIZED_INVARIANTS,
            conceptual_gap_tag="VERLET_SECOND_ORDER_ENERGY_SCALING_GAP",
            failure_diagnostic=(
                "Halving timestep `dt -> dt/2` did not reduce total-energy oscillation "
                "amplitude by ~4x (`O(dt^2)` global order) or exhibited secular Euler energy drift."
            ),
            hint_nudge="Ensure both the `0.5 * a_old * dt**2` position term and symmetric `0.5 * (a_old + a_new) * dt` velocity update are present.",
            oracle_code="""
def _run_oracle(student_fn, ctx):
    k_sp = ctx.rng.uniform(1.5, 3.0)
    mass = ctx.rng.uniform(0.9, 1.8)
    f_fn = lambda pos: -k_sp * ctx.np.asarray(pos, dtype=float)
    def energy_span(dt_val, n_steps):
        r = ctx.np.array([1.15])
        v = ctx.np.array([0.35])
        es = []
        for _ in range(n_steps):
            r, v = student_fn(r, v, mass, dt_val, f_fn)
            r_a = ctx.np.asarray(r, dtype=float)
            v_a = ctx.np.asarray(v, dtype=float)
            es.append(0.5 * mass * float(ctx.np.sum(v_a ** 2)) + 0.5 * k_sp * float(ctx.np.sum(r_a ** 2)))
        return max(es) - min(es)
    span_coarse = energy_span(0.08, 80)
    span_fine = energy_span(0.04, 160)
    ctx.assert_true(span_coarse > 0.0 and span_fine > 0.0)
    ratio = span_fine / span_coarse
    ctx.assert_true(0.20 <= ratio <= 0.30)
""",
        ),
        HiddenTestSpec(
            test_id="L04_EX01_T4_FREE_PARTICLE_GALILEAN",
            public_test_name="Test 4/5: Zero-Force Ballistic Motion & Constant-Force Exactness",
            tier=TIER_RANDOMIZED_INVARIANTS,
            conceptual_gap_tag="VERLET_CONSTANT_ACCELERATION_EXACTNESS_GAP",
            failure_diagnostic=(
                "Constant-force trajectory disagreed with exact parabola `r0 + v0*dt + 0.5*(F/m)*dt^2`; "
                "Velocity Verlet must integrate uniform acceleration with zero truncation error."
            ),
            hint_nudge="Check mass division `a = F / mass` and the `0.5` factor on `a * dt**2`.",
            oracle_code="""
def _run_oracle(student_fn, ctx):
    for _ in range(10):
        r0 = ctx.np.array([ctx.rng.uniform(-5.0, 5.0), ctx.rng.uniform(-5.0, 5.0)])
        v0 = ctx.np.array([ctx.rng.uniform(-2.0, 2.0), ctx.rng.uniform(-2.0, 2.0)])
        f_const = ctx.np.array([ctx.rng.uniform(-3.0, 3.0), ctx.rng.uniform(-3.0, 3.0)])
        mass = ctx.rng.uniform(0.5, 4.0)
        dt = ctx.rng.uniform(0.01, 0.25)
        r1, v1 = student_fn(r0, v0, mass, dt, lambda _x: f_const)
        ctx.assert_close(r1, r0 + v0 * dt + 0.5 * (f_const / mass) * (dt ** 2), rtol=1e-9, atol=1e-11)
        ctx.assert_close(v1, v0 + (f_const / mass) * dt, rtol=1e-9, atol=1e-11)
""",
        ),
        HiddenTestSpec(
            test_id="L04_EX01_T5_SEALED_REFERENCE",
            public_test_name="Test 5/5: Sealed Quartic Oscillator Step & Multi-Particle Array Vector",
            tier=TIER_SEALED_VECTORS,
            conceptual_gap_tag="VERLET_MULTIPARTICLE_VECTORIZATION_GAP",
            failure_diagnostic=(
                "Sealed multi-particle `(N, 3)` quartic force step diverged from exact "
                "Velocity Verlet coordinates or velocities."
            ),
            hint_nudge="Avoid flattening `(N, 3)` position arrays; broadcast scalar `mass` and `dt` cleanly.",
            oracle_code="""
def _run_oracle(student_fn, ctx):
    r0 = ctx.np.array([[1.0, -0.5, 0.25], [-0.4, 0.8, -1.2]], dtype=float)
    v0 = ctx.np.array([[0.2, 0.1, -0.3], [-0.5, 0.4, 0.0]], dtype=float)
    mass = 2.0
    dt = 0.05
    f_fn = lambda pos: -2.0 * ctx.np.asarray(pos, dtype=float) ** 3
    a0 = f_fn(r0) / mass
    r_ref = r0 + v0 * dt + 0.5 * a0 * (dt ** 2)
    a1 = f_fn(r_ref) / mass
    v_ref = v0 + 0.5 * (a0 + a1) * dt
    r_got, v_got = student_fn(r0, v0, mass, dt, f_fn)
    ctx.assert_close(r_got, r_ref, rtol=1e-9, atol=1e-11)
    ctx.assert_close(v_got, v_ref, rtol=1e-9, atol=1e-11)
""",
        ),
    ]

    return AuthoringSuiteSpec(
        suite_id="suite_l04_ex01_velocity_verlet_step",
        lesson_id="lesson_04",
        exercise_id="L04_EX01_velocity_verlet_step",
        entry_function="velocity_verlet_step",
        curriculum_slice_ref=slice_ref,
        tests=tests,
    )


def build_reference_metropolis_suite() -> AuthoringSuiteSpec:
    """
    Canonical 5-test 3-tier suite for Lesson 15
    `metropolis_acceptance_step(current_state, proposed_state, delta_energy, beta, u_rand)`.
    """
    slice_ref = {
        "entity_id": "L15::sec_metropolis_hastings",
        "section_id": "sec_metropolis_hastings",
        "rel_path": (
            "courses/"
            "Atomic Scale Simulations/worksheets/lesson_15_markov_chains_and_metropolis.yaml"
        ),
        "start_line": 763,
        "end_line": 795,
        "ckv_id": "CKV-6",
        "sha256_64": "a8ee39d0f89b679d0aba4b0c2681add85a322359d50601bcb749fcab2e3ba756",
        "sha256_16": "a8ee39d0f89b679d",
        "canonical_key": (
            "courses/"
            "Atomic Scale Simulations/worksheets/lesson_15_markov_chains_and_metropolis.yaml"
            "#L763-L795[CKV-6:a8ee39d0f89b679d]"
        ),
    }

    tests = [
        HiddenTestSpec(
            test_id="L15_EX01_T1_CONTRACT",
            public_test_name="Test 1/5: Return Contract (next_state, accepted_bool, acc_prob)",
            tier=TIER_CONTRACT_AND_UNITS,
            conceptual_gap_tag="METROPOLIS_RETURN_CONTRACT_GAP",
            failure_diagnostic=(
                "Expected a 3-tuple `(next_state, accepted, acc_prob)` with `accepted` as `bool` "
                "and `acc_prob` finite in `[0.0, 1.0]`."
            ),
            hint_nudge="Return `(next_state, bool(accepted), float(acc_prob))`.",
            oracle_code="""
def _run_oracle(student_fn, ctx):
    res = student_fn([0.0, 1.0], [0.2, 0.9], -0.5, 1.0, 0.7)
    ctx.assert_true(isinstance(res, tuple) and len(res) == 3)
    st, acc, p = res
    ctx.assert_array_contract(st, expected_shape=(2,))
    ctx.assert_true(isinstance(acc, bool))
    ctx.assert_true(0.0 <= float(p) <= 1.0)
""",
        ),
        HiddenTestSpec(
            test_id="L15_EX01_T2_DOWNHILL_ACCEPT",
            public_test_name="Test 2/5: Unconditional Acceptance for Downhill Moves (dE <= 0)",
            tier=TIER_RANDOMIZED_INVARIANTS,
            conceptual_gap_tag="METROPOLIS_DOWNHILL_OVERFLOW_OR_CLAMP_GAP",
            failure_diagnostic=(
                "Downhill energy transitions (`delta_energy <= 0`, including very large "
                "negative `delta_energy` at low T) must clamp `acc_prob = 1.0` without "
                "`math.exp` overflow and accept unconditionally."
            ),
            hint_nudge="Check `if delta_energy <= 0: acc_prob = 1.0` BEFORE calling `math.exp(-beta * delta_energy)`!",
            oracle_code="""
def _run_oracle(student_fn, ctx):
    for dE in [-0.0, -0.25, -15.0, -5000.0]:
        beta = ctx.rng.uniform(0.5, 200.0)
        u = ctx.rng.uniform(0.0, 0.999999)
        st, acc, p = student_fn([1.0], [2.0], dE, beta, u)
        ctx.assert_true(acc is True)
        ctx.assert_close(p, 1.0, rtol=1e-9, atol=1e-12)
        ctx.assert_close(st, [2.0], rtol=1e-9, atol=1e-12)
""",
        ),
        HiddenTestSpec(
            test_id="L15_EX01_T3_DETAILED_BALANCE",
            public_test_name=(
                "Test 3/5: Randomized Detailed Balance Ratio pi_i P(i->j) == pi_j P(j->i)"
            ),
            tier=TIER_RANDOMIZED_INVARIANTS,
            conceptual_gap_tag="METROPOLIS_DETAILED_BALANCE_RATIO_GAP",
            failure_diagnostic=(
                "Acceptance probabilities failed microscopic reversibility: "
                "`exp(-beta * E_i) * P_acc(i -> j)` must equal `exp(-beta * E_j) * P_acc(j -> i)`."
            ),
            hint_nudge="Use Metropolis `min(1.0, math.exp(-beta * delta_energy))` rather than uncalibrated sigmoid or linear rules.",
            oracle_code="""
def _run_oracle(student_fn, ctx):
    for _ in range(25):
        e_i = ctx.rng.uniform(-3.0, 3.0)
        e_j = ctx.rng.uniform(-3.0, 3.0)
        beta = ctx.rng.uniform(0.2, 4.0)
        _, _, p_ij = student_fn([0.0], [1.0], e_j - e_i, beta, 0.5)
        _, _, p_ji = student_fn([1.0], [0.0], e_i - e_j, beta, 0.5)
        lhs = math.exp(-beta * e_i) * float(p_ij)
        rhs = math.exp(-beta * e_j) * float(p_ji)
        ctx.assert_close(lhs, rhs, rtol=1e-8, atol=1e-11)
""",
        ),
        HiddenTestSpec(
            test_id="L15_EX01_T4_REJECTED_STATE_RETENTION",
            public_test_name=(
                "Test 4/5: Rejected-Move State Retention (Preventing Omitted-Rejection Bug)"
            ),
            tier=TIER_RANDOMIZED_INVARIANTS,
            conceptual_gap_tag="METROPOLIS_REJECTED_STATE_RETENTION_GAP",
            failure_diagnostic=(
                "On move rejection (`u_rand >= acc_prob`), the Markov chain failed to return "
                "`current_state` unchanged (returning `None` or `proposed_state` destroys the "
                "Boltzmann stationary distribution)."
            ),
            hint_nudge="When `u_rand >= acc_prob`, return `(current_state, False, acc_prob)`.",
            oracle_code="""
def _run_oracle(student_fn, ctx):
    for _ in range(20):
        dE = ctx.rng.uniform(0.4, 3.5)
        beta = ctx.rng.uniform(0.5, 2.5)
        p_true = math.exp(-beta * dE)
        u_hi = p_true + (1.0 - p_true) * ctx.rng.uniform(0.05, 0.95)
        u_lo = p_true * ctx.rng.uniform(0.05, 0.95)
        cur = [ctx.rng.uniform(-4.0, 4.0), ctx.rng.uniform(-4.0, 4.0)]
        prop = [cur[0] + 0.7, cur[1] - 0.9]
        st_rej, acc_rej, p_rej = student_fn(cur, prop, dE, beta, u_hi)
        ctx.assert_true(acc_rej is False)
        ctx.assert_close(st_rej, cur, rtol=1e-9, atol=1e-12)
        ctx.assert_close(p_rej, p_true, rtol=1e-8, atol=1e-11)
        st_acc, acc_acc, _ = student_fn(cur, prop, dE, beta, u_lo)
        ctx.assert_true(acc_acc is True)
        ctx.assert_close(st_acc, prop, rtol=1e-9, atol=1e-12)
""",
        ),
        HiddenTestSpec(
            test_id="L15_EX01_T5_ZERO_TEMP_GROUND_STATE",
            public_test_name="Test 5/5: Sealed Zero-Temperature Quench Limit (T -> 0, beta -> inf)",
            tier=TIER_SEALED_VECTORS,
            conceptual_gap_tag="METROPOLIS_EXTREME_BETA_UNDERFLOW_GAP",
            failure_diagnostic=(
                "Extreme low-temperature quench (`beta * delta_energy > 800`) failed "
                "underflow safety or rejected/accepted inconsistently at `u_rand -> 0`."
            ),
            hint_nudge="Guard large positive `beta * delta_energy > 700` to `acc_prob = 0.0` cleanly.",
            oracle_code="""
def _run_oracle(student_fn, ctx):
    st, acc, p = student_fn([1, -1, 1], [-1, -1, 1], 4.0, 500.0, 1e-12)
    ctx.assert_true(acc is False)
    ctx.assert_close(p, 0.0, rtol=1e-9, atol=1e-15)
    ctx.assert_close(st, [1, -1, 1], rtol=1e-9, atol=1e-12)
""",
        ),
    ]

    return AuthoringSuiteSpec(
        suite_id="suite_l15_ex01_metropolis_acceptance",
        lesson_id="lesson_15",
        exercise_id="L15_EX01_metropolis_acceptance_step",
        entry_function="metropolis_acceptance_step",
        curriculum_slice_ref=slice_ref,
        tests=tests,
    )




# ==============================================================================
# 5. Hybrid Native Notebook Submission Helper (Callable OR Source String)
# ==============================================================================


def _preflight_callable_code_object(fn: Callable[..., Any]) -> List[PreflightViolation]:
    """
    Bytecode-level pre-flight check used when `inspect.getsource(fn)` is unavailable
    (e.g., interactive `<ipython-input>`, Pyodide REPL cells, or lambdas).
    """
    violations: List[PreflightViolation] = []
    code_obj = getattr(fn, "__code__", None)
    if code_obj is None:
        return violations

    co_names = set(getattr(code_obj, "co_names", ()) or ())
    for mod_name, msg in FORBIDDEN_IMPORT_PEDAGOGY.items():
        if mod_name in co_names:
            violations.append(
                PreflightViolation(
                    rule_id="BYTECODE_FORBIDDEN_MODULE_REF",
                    lineno=int(getattr(code_obj, "co_firstlineno", 1)),
                    col_offset=0,
                    construct=mod_name,
                    pedagogical_message=msg,
                )
            )
    for call_name, msg in FORBIDDEN_CALL_NAMES.items():
        if call_name in co_names:
            violations.append(
                PreflightViolation(
                    rule_id="BYTECODE_FORBIDDEN_BUILTIN_REF",
                    lineno=int(getattr(code_obj, "co_firstlineno", 1)),
                    col_offset=0,
                    construct=call_name,
                    pedagogical_message=msg,
                )
            )
    for attr_name, msg in FORBIDDEN_ATTRIBUTES.items():
        if attr_name in co_names:
            violations.append(
                PreflightViolation(
                    rule_id="BYTECODE_FORBIDDEN_ATTRIBUTE_SNOOP",
                    lineno=int(getattr(code_obj, "co_firstlineno", 1)),
                    col_offset=0,
                    construct=f".{attr_name}",
                    pedagogical_message=msg,
                )
            )
    return violations


def _execute_live_callable_fallback(
    sealed: SealedTestBundle,
    spec: AuthoringSuiteSpec,
    student_fn: Callable[..., Any],
    base_entropy: int,
    t2_rng_seed_hex: str,
) -> SuiteExecutionReport:
    """
    Execute a live Python Callable against an unsealed suite when source text is
    unavailable (interactive REPL / Pyodide / lambda), preserving input cloning,
    muted stdout/stderr, return-type sanitization, and traceback scrubbing.
    """
    bc_violations = _preflight_callable_code_object(student_fn)
    if bc_violations:
        gap_tags: List[str] = []
        for v in bc_violations:
            if v.conceptual_gap_tag and v.conceptual_gap_tag not in gap_tags:
                gap_tags.append(v.conceptual_gap_tag)
        digest = hashlib.sha256(
            json.dumps(
                {
                    "suite_id": spec.suite_id,
                    "preflight_passed": False,
                    "t2_rng_seed_hex": t2_rng_seed_hex,
                    "violations": [v.to_dict() for v in bc_violations],
                },
                sort_keys=True,
                ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest()
        return SuiteExecutionReport(
            suite_id=spec.suite_id,
            lesson_id=spec.lesson_id,
            exercise_id=spec.exercise_id,
            entry_function=spec.entry_function,
            preflight_passed=False,
            preflight_violations=bc_violations,
            all_passed=False,
            passed_count=0,
            total_count=len(spec.tests),
            test_reports=[],
            triggered_conceptual_gap_tags=gap_tags,
            curriculum_slice_ref=copy.deepcopy(spec.curriculum_slice_ref),
            report_sha256=digest,
            t2_rng_seed_hex=t2_rng_seed_hex,
        )

    def _wrapped_live_fn(*args: Any, **kwargs: Any) -> Any:
        cloned_args = tuple(_clone_input_arg(a) for a in args)
        cloned_kwargs = {k: _clone_input_arg(v) for k, v in kwargs.items()}
        s_out = io.StringIO()
        s_err = io.StringIO()
        prev_out, prev_err = sys.stdout, sys.stderr
        try:
            sys.stdout = s_out
            sys.stderr = s_err
            result = student_fn(*cloned_args, **cloned_kwargs)
            return _sanitize_student_return(result)
        finally:
            sys.stdout = prev_out
            sys.stderr = prev_err

    test_reports: List[TestCaseReport] = []
    triggered_gaps: List[str] = []
    passed_count = 0

    for idx, t_spec in enumerate(spec.tests):
        test_seed = (base_entropy + 0x9E3779B97F4A7C15 * (idx + 1)) & 0x7FFFFFFFFFFFFFFF
        ctx = _OracleContext(
            seed=test_seed,
            default_rtol=spec.default_rtol,
            default_atol=spec.default_atol,
        )
        oracle_ns: Dict[str, Any] = {
            "__builtins__": builtins.__dict__,
            "math": math,
            "np": np,
        }
        t0 = time.perf_counter()
        passed = False
        diag_out = ""
        gap_out: Optional[str] = None
        try:
            oracle_code_obj = compile(
                t_spec.oracle_code, filename=f"<sealed_oracle:{t_spec.test_id}>", mode="exec"
            )
            exec(oracle_code_obj, oracle_ns, oracle_ns)
            oracle_fn = oracle_ns["_run_oracle"]
            oracle_fn(_wrapped_live_fn, ctx)
            passed = True
        except (_OracleAssertionFailed, AssertionError):
            passed = False
            diag_out = t_spec.failure_diagnostic
            gap_out = t_spec.conceptual_gap_tag
        except _ReturnContractViolation:
            passed = False
            diag_out = (
                f"{t_spec.failure_diagnostic} [Return contract violation: return standard "
                "Python floats/tuples or NumPy float arrays.]"
            )
            gap_out = t_spec.conceptual_gap_tag
        except BaseException as exc:
            safe_exc = _classify_safe_exception_name(exc)
            _scrub_traceback_frames(exc)
            passed = False
            diag_out = f"{t_spec.failure_diagnostic} [Runtime exception: {safe_exc}]"
            gap_out = t_spec.conceptual_gap_tag
        finally:
            oracle_ns.clear()

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        if passed:
            passed_count += 1
        elif gap_out and gap_out not in triggered_gaps:
            triggered_gaps.append(gap_out)

        test_reports.append(
            TestCaseReport(
                test_id=t_spec.test_id,
                public_test_name=t_spec.public_test_name,
                tier=t_spec.tier,
                passed=passed,
                failure_diagnostic=diag_out,
                conceptual_gap_tag=gap_out,
                hint_nudge=t_spec.hint_nudge if not passed else "",
                execution_ms=elapsed_ms,
            )
        )

    total_count = len(test_reports)
    all_passed = passed_count == total_count
    digest_material = json.dumps(
        {
            "suite_id": sealed.suite_id,
            "t2_rng_seed_hex": t2_rng_seed_hex,
            "passed_count": passed_count,
            "total_count": total_count,
            "results": [
                {"test_id": r.test_id, "passed": r.passed, "gap": r.conceptual_gap_tag}
                for r in test_reports
            ],
        },
        sort_keys=True,
        ensure_ascii=True,
    ).encode("ascii")

    return SuiteExecutionReport(
        suite_id=sealed.suite_id,
        lesson_id=sealed.lesson_id,
        exercise_id=sealed.exercise_id,
        entry_function=sealed.entry_function,
        preflight_passed=True,
        preflight_violations=[],
        all_passed=all_passed,
        passed_count=passed_count,
        total_count=total_count,
        test_reports=test_reports,
        triggered_conceptual_gap_tags=triggered_gaps,
        curriculum_slice_ref=copy.deepcopy(sealed.curriculum_slice_ref),
        report_sha256=hashlib.sha256(digest_material).hexdigest(),
        t2_rng_seed_hex=t2_rng_seed_hex,
    )


def execute_student_submission(
    bundle: Union[SealedTestBundle, Dict[str, Any]],
    student_fn_or_source: Union[str, Callable[..., Any]],
    obfuscation_key: Optional[str] = None,
    verification_seed: Optional[Union[int, str]] = None,
    rng_seed: Optional[Union[int, str]] = None,
    session_id: Optional[str] = None,
    attempt_index: int = 0,
) -> SuiteExecutionReport:
    """
    Hybrid execution entrypoint for `ws.check_code(exercise_id, student_fn_or_source)`.

    Accepts EITHER:
    - a Python source string `str` -> runs full `execute_hidden_suite` (AST + loop-tick + sandbox), OR
    - a live Python `Callable` `fn` -> attempts `inspect.getsource(fn)` (`textwrap.dedent`)
      to run the full AST + loop-tick sandbox path, and falls back gracefully when
      `inspect.getsource` raises `OSError`/`TypeError` (e.g., `<ipython-input>`, Pyodide
      REPL, `exec`-created callables, or lambdas) to isolated unsealed execution with
      bytecode pre-flight screening, cloned inputs, muted stdout/stderr, sanitized return
      values, scrubbed tracebacks, and deterministic `t2_rng_seed_hex` replay.
    """
    if isinstance(student_fn_or_source, str):
        return execute_hidden_suite(
            bundle=bundle,
            student_source=student_fn_or_source,
            obfuscation_key=obfuscation_key,
            verification_seed=verification_seed,
            rng_seed=rng_seed,
            session_id=session_id,
            attempt_index=attempt_index,
        )

    if not callable(student_fn_or_source):
        return execute_hidden_suite(
            bundle=bundle,
            student_source="",
            obfuscation_key=obfuscation_key,
            verification_seed=verification_seed,
            rng_seed=rng_seed,
            session_id=session_id,
            attempt_index=attempt_index,
        )

    sealed = bundle if isinstance(bundle, SealedTestBundle) else SealedTestBundle.from_dict(bundle)
    spec = _unseal_bundle_internal(sealed, obfuscation_key=obfuscation_key)
    effective_seed_arg = rng_seed if rng_seed is not None else verification_seed
    base_entropy, t2_rng_seed_hex = derive_t2_rng_seed(
        session_id=session_id,
        exercise_id=spec.exercise_id,
        attempt_index=attempt_index,
        rng_seed=effective_seed_arg,
    )

    extracted_src: Optional[str] = None
    try:
        raw_src = _host_inspect.getsource(student_fn_or_source)
        extracted_src = textwrap.dedent(raw_src)
    except (OSError, TypeError):
        extracted_src = None

    if extracted_src:
        # First check AST pre-flight without requiring matching def name (in case of lambda/alias)
        pf_body = ast_preflight_check(extracted_src, entry_function=None)
        if not pf_body.passed and all(
            v.rule_id != "AST_SYNTAX_ERROR" for v in pf_body.violations
        ):
            # Genuine sandbox policy violation inside the callable source!
            return execute_hidden_suite(
                bundle=sealed,
                student_source=extracted_src,
                obfuscation_key=obfuscation_key,
                rng_seed=t2_rng_seed_hex,
            )

        # Check if extracted_src defines top-level `def <entry_function>` and has no closure freevars
        freevars = getattr(getattr(student_fn_or_source, "__code__", None), "co_freevars", ()) or ()
        pf_full = ast_preflight_check(extracted_src, entry_function=spec.entry_function)
        if pf_full.passed and not freevars:
            return execute_hidden_suite(
                bundle=sealed,
                student_source=extracted_src,
                obfuscation_key=obfuscation_key,
                rng_seed=t2_rng_seed_hex,
            )

    return _execute_live_callable_fallback(
        sealed=sealed,
        spec=spec,
        student_fn=student_fn_or_source,
        base_entropy=base_entropy,
        t2_rng_seed_hex=t2_rng_seed_hex,
    )


# ==============================================================================
# CLI Self-Test Entrypoint
# ==============================================================================


def run_self_test() -> int:
    suite = build_reference_pbc_minimum_image_suite()
    bundle = compile_hidden_suite(suite, obfuscation_key="libredistillations-build-key")
    good_code = (
        "import numpy as np\n"
        "def minimum_image_displacement(dr, box_length):\n"
        "    dr_arr = np.asarray(dr, dtype=float)\n"
        "    box_arr = np.asarray(box_length, dtype=float)\n"
        "    return dr_arr - box_arr * np.round(dr_arr / box_arr)\n"
    )
    rep = execute_hidden_suite(bundle, good_code, obfuscation_key="libredistillations-build-key")
    assert rep.all_passed and rep.passed_count == 5, rep.format_student_card()
    print(f"[SELF-TEST PASS] {bundle.suite_id}: {rep.passed_count}/{rep.total_count} passed.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="V2 Bulletproof Hidden-Test Code Runner"
    )
    parser.add_argument("--self-test", action="store_true", help="Run built-in verification")
    args = parser.parse_args(argv)
    if args.self_test or not argv:
        return run_self_test()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
