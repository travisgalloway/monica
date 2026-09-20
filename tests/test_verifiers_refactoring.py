"""#347 -- Unit tests for Behavioral-Invariance Refactoring & Interface Decoupling Verifiers.

Comprehensive CI-safe tests covering:
1. Strategy Pattern Refactoring & Behavioral Invariance (100% tests pass -> reward 1.0)
2. Behavioral Regression Detection (test failures penalize reward < 1.0)
3. Factory Pattern Refactoring
4. Repository Pattern Refactoring & Data Decoupling
5. Dependency Injection & Mockability with Mock Objects (zero real sockets/db connections)
6. Leaky I/O Detection: Real Network Socket Access Blocked
7. Leaky I/O Detection: Real Database Connection Blocked
8. Sandboxed Test Execution Gate: Timeout Guarding Infinite Loops
9. Resource Limit Guarding
10. Anti-Goodhart Escape-Hatch Guards (refactor-ignore, @unittest.skip, eval bypass)
11. Degenerate Output Guards (empty, whitespace, comments)
12. Syntax Error Handling & Diagnostic Mapping
13. Multi-file Parsing Formats (Dict, JSON, XML, Markdown Fences)
14. Injectable Oracle Seam & Telemetry Tracking
15. Code Suite Benchmark Integration (evaluate_refactoring)
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import Mock, patch
import pytest

from src.eval.code_suite import RECORD_FIELDS, REFACTORING_BUCKETS, evaluate_refactoring
from src.lsp.diagnostics import Diagnostic
from src.train.verifiers import (
    RefactoringAnalysisResult,
    RefactoringAstSummary,
    RefactoringInvariantVerifier,
    RefactoringOracle,
    RefactoringPattern,
    RefactoringTask,
    SandboxedExecutionGate,
    TestExecutionResult,
    analyze_refactoring_ast,
    find_refactoring_escape_hatches,
    parse_refactoring_source,
)


# --------------------------------------------------------------------------- #
# 1. Strategy Pattern Refactoring & Behavioral Invariance
# --------------------------------------------------------------------------- #

def test_strategy_pattern_clean_refactoring_achieves_perfect_reward():
    """Verify clean Strategy pattern refactoring achieves 1.0 reward with 100% tests passing."""
    # Refactored multi-file strategy pattern
    completion = {
        "pricing/strategies.py": """
from abc import ABC, abstractmethod

class PricingStrategy(ABC):
    @abstractmethod
    def calculate_price(self, base_price: float, quantity: int) -> float:
        pass

class StandardPricing(PricingStrategy):
    def calculate_price(self, base_price: float, quantity: int) -> float:
        return base_price * quantity

class BulkPricing(PricingStrategy):
    def calculate_price(self, base_price: float, quantity: int) -> float:
        total = base_price * quantity
        if quantity >= 10:
            return total * 0.90
        return total

class VipPricing(PricingStrategy):
    def calculate_price(self, base_price: float, quantity: int) -> float:
        return (base_price * quantity) * 0.80
""",
        "pricing/service.py": """
from pricing.strategies import PricingStrategy, StandardPricing

class OrderPricingService:
    def __init__(self, strategy: PricingStrategy = None):
        self.strategy = strategy or StandardPricing()

    def set_strategy(self, strategy: PricingStrategy) -> None:
        self.strategy = strategy

    def compute_total(self, base_price: float, quantity: int) -> float:
        return self.strategy.calculate_price(base_price, quantity)
""",
    }

    # Invariant unit + property-style test suite
    test_suite = """
import unittest
from pricing.service import OrderPricingService
from pricing.strategies import StandardPricing, BulkPricing, VipPricing

class TestPricingBehavior(unittest.TestCase):
    def test_standard_pricing(self):
        svc = OrderPricingService(StandardPricing())
        self.assertAlmostEqual(svc.compute_total(100.0, 2), 200.0)

    def test_bulk_pricing_threshold(self):
        svc = OrderPricingService(BulkPricing())
        self.assertAlmostEqual(svc.compute_total(10.0, 5), 50.0)
        self.assertAlmostEqual(svc.compute_total(10.0, 10), 90.0)
        self.assertAlmostEqual(svc.compute_total(10.0, 20), 180.0)

    def test_vip_pricing(self):
        svc = OrderPricingService(VipPricing())
        self.assertAlmostEqual(svc.compute_total(100.0, 1), 80.0)

    def test_property_invariance_monotonicity(self):
        # Property invariant: bulk total <= standard total for all quantities
        standard = OrderPricingService(StandardPricing())
        bulk = OrderPricingService(BulkPricing())
        for q in range(1, 50):
            self.assertLessEqual(bulk.compute_total(25.0, q), standard.compute_total(25.0, q))
"""

    verifier = RefactoringInvariantVerifier(timeout_s=5.0)
    res = verifier.evaluate(
        completion,
        tests=test_suite,
        pattern="strategy",
        target_interfaces=["PricingStrategy"],
    )

    assert res["is_clean"] is True
    assert res["reward"] == 1.0
    assert res["all_tests_passed"] is True
    assert res["functional_passed"] is True
    assert res["no_socket_opened"] is True
    assert res["no_db_opened"] is True
    assert "strategy" in res["patterns_detected"]
    assert len(res["diagnostics"]) == 0


def test_strategy_pattern_behavioral_regression_is_penalized():
    """Verify that a behavioral bug in a refactored strategy drops reward < 1.0."""
    buggy_completion = {
        "pricing/strategies.py": """
from abc import ABC, abstractmethod

class PricingStrategy(ABC):
    @abstractmethod
    def calculate_price(self, base_price: float, quantity: int) -> float:
        pass

class StandardPricing(PricingStrategy):
    def calculate_price(self, base_price: float, quantity: int) -> float:
        return base_price * quantity

class BulkPricing(PricingStrategy):
    def calculate_price(self, base_price: float, quantity: int) -> float:
        # REGRESSION BUG: wrong threshold discount formula
        total = base_price * quantity
        if quantity >= 10:
            return total * 0.50  # Buggy 50% instead of 10%
        return total

class VipPricing(PricingStrategy):
    def calculate_price(self, base_price: float, quantity: int) -> float:
        return (base_price * quantity) * 0.80
""",
        "pricing/service.py": """
from pricing.strategies import PricingStrategy, StandardPricing

class OrderPricingService:
    def __init__(self, strategy: PricingStrategy = None):
        self.strategy = strategy or StandardPricing()

    def compute_total(self, base_price: float, quantity: int) -> float:
        return self.strategy.calculate_price(base_price, quantity)
""",
    }

    test_suite = """
import unittest
from pricing.service import OrderPricingService
from pricing.strategies import BulkPricing

class TestBulkRegression(unittest.TestCase):
    def test_bulk_exact_discount(self):
        svc = OrderPricingService(BulkPricing())
        # Expected 90.0 (10% off 100), buggy returns 50.0
        self.assertAlmostEqual(svc.compute_total(10.0, 10), 90.0)
"""

    verifier = RefactoringInvariantVerifier()
    res = verifier.evaluate(buggy_completion, tests=test_suite, pattern="strategy")

    assert res["is_clean"] is False
    assert res["reward"] < 1.0
    assert res["all_tests_passed"] is False
    assert any(d.code == "REFACTOR_TEST_FAILURE" for d in res["diagnostics"])


# --------------------------------------------------------------------------- #
# 2. Factory Pattern Refactoring
# --------------------------------------------------------------------------- #

def test_factory_pattern_clean_refactoring():
    """Verify Factory pattern refactoring separates instantiation from business logic."""
    completion = {
        "notifications/base.py": """
from abc import ABC, abstractmethod

class Notifier(ABC):
    @abstractmethod
    def notify(self, message: str) -> str:
        pass
""",
        "notifications/impl.py": """
from notifications.base import Notifier

class EmailNotifier(Notifier):
    def notify(self, message: str) -> str:
        return f"EMAIL: {message}"

class SmsNotifier(Notifier):
    def notify(self, message: str) -> str:
        return f"SMS: {message}"

class SlackNotifier(Notifier):
    def notify(self, message: str) -> str:
        return f"SLACK: {message}"
""",
        "notifications/factory.py": """
from notifications.base import Notifier
from notifications.impl import EmailNotifier, SmsNotifier, SlackNotifier

class NotifierFactory:
    @staticmethod
    def create_notifier(channel: str) -> Notifier:
        c = (channel or "").lower().strip()
        if c == "email":
            return EmailNotifier()
        elif c == "sms":
            return SmsNotifier()
        elif c == "slack":
            return SlackNotifier()
        raise ValueError(f"Unknown channel: {channel}")
""",
    }

    test_suite = """
import unittest
from notifications.factory import NotifierFactory

class TestNotifierFactory(unittest.TestCase):
    def test_email_notifier(self):
        n = NotifierFactory.create_notifier("email")
        self.assertEqual(n.notify("Hello"), "EMAIL: Hello")

    def test_sms_notifier(self):
        n = NotifierFactory.create_notifier("sms")
        self.assertEqual(n.notify("Alert"), "SMS: Alert")

    def test_slack_notifier(self):
        n = NotifierFactory.create_notifier("slack")
        self.assertEqual(n.notify("Ping"), "SLACK: Ping")

    def test_unknown_channel_raises(self):
        with self.assertRaises(ValueError):
            NotifierFactory.create_notifier("carrier_pigeon")
"""

    verifier = RefactoringInvariantVerifier()
    res = verifier.evaluate(completion, tests=test_suite, pattern="factory")

    assert res["is_clean"] is True
    assert res["reward"] == 1.0
    assert "factory" in res["patterns_detected"]


# --------------------------------------------------------------------------- #
# 3. Repository Pattern Refactoring
# --------------------------------------------------------------------------- #

def test_repository_pattern_clean_refactoring():
    """Verify Repository pattern decouples domain service from storage mechanism."""
    completion = {
        "domain/user.py": """
from dataclasses import dataclass

@dataclass
class User:
    id: str
    name: str
    email: str
""",
        "repositories/base.py": """
from abc import ABC, abstractmethod
from typing import Optional, List
from domain.user import User

class UserRepository(ABC):
    @abstractmethod
    def get_by_id(self, user_id: str) -> Optional[User]:
        pass

    @abstractmethod
    def save(self, user: User) -> None:
        pass

    @abstractmethod
    def list(self) -> List[User]:
        pass
""",
        "repositories/in_memory.py": """
from typing import Optional, List, Dict
from domain.user import User
from repositories.base import UserRepository

class InMemoryUserRepository(UserRepository):
    def __init__(self):
        self._store: Dict[str, User] = {}

    def get_by_id(self, user_id: str) -> Optional[User]:
        return self._store.get(user_id)

    def save(self, user: User) -> None:
        self._store[user.id] = user

    def list(self) -> List[User]:
        return list(self._store.values())
""",
        "services/user_service.py": """
from domain.user import User
from repositories.base import UserRepository

class UserService:
    def __init__(self, repo: UserRepository):
        self.repo = repo

    def register_user(self, id: str, name: str, email: str) -> User:
        user = User(id=id, name=name, email=email)
        self.repo.save(user)
        return user

    def find_user(self, id: str) -> User:
        user = self.repo.get_by_id(id)
        if not user:
            raise KeyError(f"User {id} not found")
        return user
""",
    }

    test_suite = """
import unittest
from domain.user import User
from repositories.in_memory import InMemoryUserRepository
from services.user_service import UserService

class TestUserServiceWithRepo(unittest.TestCase):
    def setUp(self):
        self.repo = InMemoryUserRepository()
        self.svc = UserService(self.repo)

    def test_register_and_find(self):
        u = self.svc.register_user("u1", "Alice", "alice@example.com")
        found = self.svc.find_user("u1")
        self.assertEqual(found.name, "Alice")
        self.assertEqual(len(self.repo.list()), 1)

    def test_missing_user_raises(self):
        with self.assertRaises(KeyError):
            self.svc.find_user("nonexistent")
"""

    verifier = RefactoringInvariantVerifier()
    res = verifier.evaluate(completion, tests=test_suite, pattern="repository")

    assert res["is_clean"] is True
    assert res["reward"] == 1.0
    assert "repository" in res["patterns_detected"]


# --------------------------------------------------------------------------- #
# 4. Dependency Injection & Mockability Verifier (Zero Real Sockets/DB)
# --------------------------------------------------------------------------- #

def test_dependency_injection_with_mocks_clean():
    """Verify class can be instantiated with mocked interface and executed without real I/O."""
    completion = {
        "clients/network.py": """
from abc import ABC, abstractmethod

class NetworkClient(ABC):
    @abstractmethod
    def fetch_data(self, endpoint: str) -> dict:
        pass
""",
        "services/sync_service.py": """
from clients.network import NetworkClient

class SyncService:
    def __init__(self, client: NetworkClient):
        self.client = client

    def sync_records(self, endpoint: str) -> int:
        data = self.client.fetch_data(endpoint)
        items = data.get("items", [])
        return len(items)
""",
    }

    # Unit tests using pure unittest.mock.Mock without real network
    mock_test_suite = """
import unittest
from unittest.mock import Mock
from clients.network import NetworkClient
from services.sync_service import SyncService

class TestSyncServiceMockability(unittest.TestCase):
    def test_sync_with_mocked_network_client(self):
        mock_client = Mock(spec=NetworkClient)
        mock_client.fetch_data.return_value = {"items": [{"id": 1}, {"id": 2}]}

        service = SyncService(client=mock_client)
        count = service.sync_records("https://api.example.com/v1/items")

        self.assertEqual(count, 2)
        mock_client.fetch_data.assert_called_once_with("https://api.example.com/v1/items")
"""

    verifier = RefactoringInvariantVerifier(block_network=True, block_database=True)
    res = verifier.evaluate(
        completion,
        mock_tests=mock_test_suite,
        pattern="dependency_injection",
        target_interfaces=["NetworkClient"],
    )

    assert res["is_clean"] is True
    assert res["reward"] == 1.0
    assert res["mockability_passed"] is True
    assert res["no_socket_opened"] is True
    assert res["no_db_opened"] is True


def test_dependency_injection_blocks_real_network_socket_access():
    """Assert sandboxed gate blocks real socket access and flags REFACTOR_LEAKY_IO."""
    leaky_completion = {
        "services/leaky_service.py": """
import socket

class LeakyNetworkService:
    def __init__(self, client=None):
        self.client = client

    def run(self):
        # VIOLATION: Creates a real network socket instead of using mock
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        return "connected"
"""
    }

    mock_test = """
import unittest
from services.leaky_service import LeakyNetworkService

class TestLeakySocket(unittest.TestCase):
    def test_run_attempts_socket(self):
        svc = LeakyNetworkService()
        svc.run()
"""

    verifier = RefactoringInvariantVerifier(block_network=True)
    res = verifier.evaluate(leaky_completion, mock_tests=mock_test)

    assert res["is_clean"] is False
    assert res["no_socket_opened"] is False
    assert res["reward"] < 1.0
    assert any(d.code == "REFACTOR_LEAKY_IO" for d in res["diagnostics"])


def test_dependency_injection_blocks_real_database_connection():
    """Assert sandboxed gate blocks real database connection and flags REFACTOR_LEAKY_IO."""
    leaky_db_completion = {
        "services/leaky_db_service.py": """
import sqlite3

class LeakyDbService:
    def __init__(self, repo=None):
        self.repo = repo

    def get_user_count(self):
        # VIOLATION: Opens real sqlite database file
        conn = sqlite3.connect("production.db")
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        return 1
"""
    }

    mock_test = """
import unittest
from services.leaky_db_service import LeakyDbService

class TestLeakyDb(unittest.TestCase):
    def test_get_user_count(self):
        svc = LeakyDbService()
        svc.get_user_count()
"""

    verifier = RefactoringInvariantVerifier(block_database=True)
    res = verifier.evaluate(leaky_db_completion, mock_tests=mock_test)

    assert res["is_clean"] is False
    assert res["no_db_opened"] is False
    assert res["reward"] < 1.0
    assert any(d.code == "REFACTOR_LEAKY_IO" for d in res["diagnostics"])


# --------------------------------------------------------------------------- #
# 5. Sandboxed Test Execution Gate & Safety Guards
# --------------------------------------------------------------------------- #

def test_sandboxed_gate_timeout_guard():
    """Verify that infinite loops in candidate code are halted by timeout guard."""
    hang_completion = {
        "hang.py": """
class HangService:
    def process(self):
        while True:
            pass
"""
    }

    hang_test = """
import unittest
from hang import HangService

class TestHang(unittest.TestCase):
    def test_infinite_loop(self):
        svc = HangService()
        svc.process()
"""

    verifier = RefactoringInvariantVerifier(timeout_s=0.5)
    t0 = time.monotonic()
    res = verifier.evaluate(hang_completion, tests=hang_test)
    elapsed = time.monotonic() - t0

    assert elapsed < 3.0  # Must not hang
    assert res["is_clean"] is False
    assert res["reward"] < 1.0
    assert any(d.code == "REFACTOR_TEST_TIMEOUT" for d in res["diagnostics"])


def test_anti_goodhart_escape_hatches():
    """Verify that escape hatches (@unittest.skip, refactor-ignore, eval) short-circuit to -1.0."""
    hatch_cases = [
        ("test_skip", "from unittest import skip\n@unittest.skip('skip test')\ndef test_fn(): pass"),
        ("refactor_ignore", "class Svc:\n    # refactor-ignore\n    pass"),
        ("eval_bypass", "def bypass():\n    return eval('2 + 2')"),
    ]

    verifier = RefactoringInvariantVerifier()
    for name, code in hatch_cases:
        res = verifier.evaluate(code)
        assert res["is_hacked"] is True
        assert res["is_clean"] is False
        assert res["reward"] == -1.0
        assert any(d.code == "REFACTOR_ESCAPE_HATCH" for d in res["diagnostics"])


def test_degenerate_output_guards():
    """Verify empty, whitespace-only, and comment-only completions short-circuit to -1.0."""
    verifier = RefactoringInvariantVerifier()

    assert verifier.reward("") == -1.0
    assert verifier.reward("   \n\t  ") == -1.0
    assert verifier.reward("# Just a comment explaining the refactoring\n# Nothing else") == -1.0

    eval_res = verifier.evaluate("# Comment only")
    assert eval_res["is_degenerate"] is True
    assert eval_res["reward"] == -1.0
    assert any(d.code == "REFACTOR_DEGENERATE_OUTPUT" for d in eval_res["diagnostics"])


def test_syntax_error_in_completion():
    """Verify syntax error in code bundle generates REFACTOR_SYNTAX_ERROR diagnostic."""
    broken_code = "def invalid_syntax(:"
    verifier = RefactoringInvariantVerifier()
    res = verifier.evaluate(broken_code)

    assert res["is_clean"] is False
    assert res["reward"] < 1.0
    assert any(d.code == "REFACTOR_SYNTAX_ERROR" for d in res["diagnostics"])


# --------------------------------------------------------------------------- #
# 6. Multi-File Parsing Formats
# --------------------------------------------------------------------------- #

def test_parse_multi_file_formats():
    """Verify multi-file parser handles XML, markdown headers, and JSON strings."""
    # 1. XML format
    xml_text = """
<file path="domain/model.py">
class Model:
    pass
</file>
<file path="service/runner.py">
from domain.model import Model
class Runner:
    pass
</file>
"""
    files_xml = parse_refactoring_source(xml_text)
    assert set(files_xml.keys()) == {"domain/model.py", "service/runner.py"}

    # 2. Markdown headers format
    md_text = """
### pkg/adapter.py
```python
class Adapter:
    pass
```

### pkg/port.py
```python
class Port:
    pass
```
"""
    files_md = parse_refactoring_source(md_text)
    assert set(files_md.keys()) == {"pkg/adapter.py", "pkg/port.py"}

    # 3. JSON format
    json_text = '{"src/a.py": "x = 1", "src/b.py": "y = 2"}'
    files_json = parse_refactoring_source(json_text)
    assert set(files_json.keys()) == {"src/a.py", "src/b.py"}


# --------------------------------------------------------------------------- #
# 7. Injectable Oracle Seam & Telemetry
# --------------------------------------------------------------------------- #

class FakeGate:
    def __init__(self, passed=True, real_socket=False):
        self.passed = passed
        self.real_socket = real_socket
        self.calls = 0

    def run_tests(self, files, **kwargs):
        self.calls += 1
        return TestExecutionResult(
            passed=self.passed,
            total_tests=2,
            passed_tests=2 if self.passed else 0,
            failed_tests=0 if self.passed else 2,
            real_socket_detected=self.real_socket,
        )


def test_injectable_oracle_seam_and_telemetry():
    """Verify oracle seam allows mocking gate without subprocess and tracks telemetry."""
    fake_gate = FakeGate(passed=True)
    oracle = RefactoringOracle(gate=fake_gate)
    verifier = RefactoringInvariantVerifier(oracle=oracle)

    code = "class Decoupled:\n    def __init__(self, dep):\n        self.dep = dep"
    res = verifier.evaluate(code, tests="test")

    assert res["is_clean"] is True
    assert res["reward"] == 1.0
    assert fake_gate.calls == 1

    telemetry = verifier.telemetry()
    assert telemetry["verifier"] == "RefactoringInvariantVerifier"
    assert telemetry["n_samples"] == 1
    assert telemetry["n_clean"] == 1
    assert telemetry["mean_reward"] == 1.0


# --------------------------------------------------------------------------- #
# 8. Code Suite Benchmark Integration (evaluate_refactoring)
# --------------------------------------------------------------------------- #

def test_code_suite_evaluate_refactoring():
    """Verify evaluate_refactoring runs benchmark cases and produces schema-compliant records."""
    test_cases = [
        # Strategy clean
        {
            "id": "strategy-clean-1",
            "pattern": "strategy",
            "code": {
                "strat.py": "class Strat: pass\nclass A(Strat): pass\nclass B(Strat): pass"
            },
            "tests": "import unittest\nclass T(unittest.TestCase):\n    def test_ok(self): self.assertTrue(True)",
            "expected_clean": True,
        },
        # Factory clean
        {
            "id": "factory-clean-1",
            "pattern": "factory",
            "code": {
                "factory.py": "def create_worker(kind): return kind.upper()"
            },
            "tests": "import unittest\nfrom factory import create_worker\nclass T(unittest.TestCase):\n    def test_f(self): self.assertEqual(create_worker('a'), 'A')",
            "expected_clean": True,
        },
        # Repository faulty
        {
            "id": "repo-faulty-1",
            "pattern": "repository",
            "code": {
                "repo.py": "class BrokenRepo:\n    def get(self, id): raise RuntimeError('Broken')"
            },
            "tests": "import unittest\nfrom repo import BrokenRepo\nclass T(unittest.TestCase):\n    def test_r(self): BrokenRepo().get('1')",
            "expected_clean": False,
        },
        # Dependency Injection clean
        {
            "id": "di-clean-1",
            "pattern": "dependency_injection",
            "code": {
                "svc.py": "class Svc:\n    def __init__(self, dep):\n        self.dep = dep\n    def run(self): return self.dep.call()",
            },
            "mock_tests": "import unittest\nfrom unittest.mock import Mock\nfrom svc import Svc\nclass T(unittest.TestCase):\n    def test_m(self):\n        m = Mock()\n        m.call.return_value = 'ok'\n        self.assertEqual(Svc(m).run(), 'ok')",
            "expected_clean": True,
        },
        # Mockability leaky
        {
            "id": "mock-leaky-1",
            "pattern": "mockability",
            "code": {
                "leaky.py": "import socket\nclass Leaky:\n    def run(self): socket.socket()"
            },
            "mock_tests": "import unittest\nfrom leaky import Leaky\nclass T(unittest.TestCase):\n    def test_l(self): Leaky().run()",
            "expected_clean": False,
        },
    ]

    res = evaluate_refactoring(test_cases)

    assert "records" in res
    assert len(res["records"]) == 5
    assert res["n_cases"] == 5
    assert res["n_passed"] == 5  # all matches expected_clean
    assert res["accuracy"] == 1.0

    for rec in res["records"]:
        assert tuple(sorted(rec)) == tuple(sorted(RECORD_FIELDS))
        assert rec["suite"] == "refactoring"
        assert rec["bucket"] in REFACTORING_BUCKETS
        assert rec["rank_top1"] is True

    for bucket in REFACTORING_BUCKETS:
        assert bucket in res["by_bucket"]
        assert res["by_bucket"][bucket]["n_instances"] == 1
        assert res["by_bucket"][bucket]["rank_top1_rate"] == 1.0
