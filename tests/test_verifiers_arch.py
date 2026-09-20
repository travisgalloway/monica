"""#346 -- Unit tests for Clean Architecture boundary linters & dependency DAG verifiers.

Comprehensive CI-safe unit tests covering:
1. Fast in-process AST execution (<15ms)
2. Valid hexagonal / clean architecture multi-file fixtures (Python & TypeScript)
3. Domain forbidden framework / infrastructure imports (ORMs, database drivers, web frameworks, clients)
4. Application forbidden framework imports
5. Inward dependency rule & layer inversion breaches
6. Circular import cycle detection (2-node, 3-node cycles)
7. Exact coupling metrics (Ca, Ce, Fan-Out, Instability)
8. Excessive coupling and fan-out penalties
9. Fine-grained RLVR penalty scoring for breaches and cycles
10. Anti-Goodhart escape hatches (arch-ignore, @ts-ignore, dynamic import bypasses)
11. Degenerate output guards (empty, whitespace, comment-only)
12. Multi-file serialization format parsing (JSON, Markdown code blocks, XML tags, Dict)
13. Telemetry tracking & injectable oracle seam
"""

from __future__ import annotations

import time
import pytest

from src.lsp.diagnostics import Diagnostic
from src.train.verifiers import (
    ArchAnalysisResult,
    ArchRuleOracle,
    ArchRuleVerifier,
    Layer,
    classify_file_layer,
    detect_cycles_tarjan,
    find_arch_rule_escape_hatches,
    parse_multi_file_source,
    parse_python_imports,
    parse_typescript_imports,
)


# --------------------------------------------------------------------------- #
# 1. Performance & Fast In-Process AST Execution (<15ms)
# --------------------------------------------------------------------------- #

def test_in_process_execution_latency_under_15ms():
    """Verify that multi-file AST parsing, graph construction, and evaluation complete in <15ms."""
    repo = {
        "src/domain/entities/order.py": """
from dataclasses import dataclass

@dataclass(frozen=True)
class Order:
    id: str
    amount: float
""",
        "src/application/use_cases/create_order.py": """
from src.domain.entities.order import Order

class CreateOrderUseCase:
    def execute(self, id: str, amount: float) -> Order:
        return Order(id=id, amount=amount)
""",
        "src/infrastructure/repositories/order_repo.py": """
from src.domain.entities.order import Order
from src.application.use_cases.create_order import CreateOrderUseCase
import sqlalchemy

class SqlOrderRepo:
    pass
""",
        "src/presentation/api/order_controller.py": """
from src.application.use_cases.create_order import CreateOrderUseCase
import fastapi

class OrderController:
    pass
""",
    }

    verifier = ArchRuleVerifier()

    t0 = time.monotonic()
    eval_res = verifier.evaluate(repo)
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    assert elapsed_ms < 15.0, f"Expected <15ms, got {elapsed_ms:.2f}ms"
    assert eval_res["elapsed_ms"] < 15.0
    assert eval_res["is_clean"] is True
    assert eval_res["reward"] == 1.0


# --------------------------------------------------------------------------- #
# 2. Valid Hexagonal Architecture Fixtures
# --------------------------------------------------------------------------- #

def test_valid_hexagonal_architecture_python():
    """Valid Python hexagonal architecture: Infrastructure and Presentation depend on Application and Domain."""
    files = {
        "src/domain/order.py": """
from dataclasses import dataclass

@dataclass
class Order:
    id: str
    total: float
""",
        "src/application/ports.py": """
from typing import Protocol
from src.domain.order import Order

class OrderRepository(Protocol):
    def save(self, order: Order) -> None: ...
""",
        "src/application/order_service.py": """
from src.domain.order import Order
from src.application.ports import OrderRepository

class OrderService:
    def __init__(self, repo: OrderRepository):
        self.repo = repo
""",
        "src/infrastructure/db/order_repo.py": """
from src.domain.order import Order
from src.application.ports import OrderRepository
import sqlalchemy

class SqlAlchemyOrderRepository(OrderRepository):
    def save(self, order: Order) -> None:
        pass
""",
        "src/presentation/api/order_router.py": """
from src.application.order_service import OrderService
import fastapi

router = fastapi.APIRouter()
""",
    }

    verifier = ArchRuleVerifier()
    res = verifier.evaluate(files)

    assert res["is_clean"] is True
    assert res["is_dag"] is True
    assert len(res["cycles"]) == 0
    assert len(res["diagnostics"]) == 0
    assert res["reward"] == 1.0


def test_valid_hexagonal_architecture_typescript():
    """Valid TypeScript hexagonal architecture: Inward dependencies only."""
    files = {
        "src/domain/user.ts": """
export interface User {
    id: string;
    email: string;
}
""",
        "src/application/ports.ts": """
import { User } from "../domain/user";

export interface UserRepository {
    findById(id: string): Promise<User | null>;
}
""",
        "src/application/user_service.ts": """
import { User } from "../domain/user";
import { UserRepository } from "./ports";

export class UserService {
    constructor(private repo: UserRepository) {}
}
""",
        "src/infrastructure/db/user_repo.ts": """
import { User } from "../../domain/user";
import { UserRepository } from "../../application/ports";
import { PrismaClient } from "@prisma/client";

export class PrismaUserRepository implements UserRepository {
    private prisma = new PrismaClient();
    async findById(id: string): Promise<User | null> {
        return null;
    }
}
""",
        "src/presentation/controllers/user_controller.ts": """
import { UserService } from "../../application/user_service";
import express from "express";

export class UserController {
    constructor(private service: UserService) {}
}
""",
    }

    verifier = ArchRuleVerifier()
    res = verifier.evaluate(files)

    assert res["is_clean"] is True
    assert res["is_dag"] is True
    assert len(res["cycles"]) == 0
    assert res["reward"] == 1.0


# --------------------------------------------------------------------------- #
# 3. Domain Forbidden Infrastructure Imports
# --------------------------------------------------------------------------- #

def test_domain_imports_database_orm_forbidden():
    """Domain must NEVER import ORMs or database drivers."""
    files = {
        "src/domain/user.py": """
from dataclasses import dataclass
import sqlalchemy

@dataclass
class User:
    id: int
""",
        "src/application/service.py": """
from src.domain.user import User
""",
    }

    verifier = ArchRuleVerifier()
    res = verifier.evaluate(files)

    assert res["is_clean"] is False
    assert any(d.code == "ARCH_DOMAIN_INFRASTRUCTURE_IMPORT" for d in res["diagnostics"])
    assert res["reward"] < 1.0
    assert res["reward"] == pytest.approx(1.0 - 0.35)


def test_domain_imports_web_framework_forbidden():
    """Domain must NEVER import web frameworks."""
    files = {
        "src/domain/order.py": """
from fastapi import HTTPException

class OrderError(HTTPException):
    pass
""",
        "src/application/service.py": """
from src.domain.order import OrderError
""",
    }

    verifier = ArchRuleVerifier()
    res = verifier.evaluate(files)

    assert res["is_clean"] is False
    assert any(d.code == "ARCH_DOMAIN_INFRASTRUCTURE_IMPORT" for d in res["diagnostics"])
    assert res["reward"] == pytest.approx(1.0 - 0.35)


def test_domain_imports_network_client_forbidden_ts():
    """TypeScript Domain importing network clients (axios, etc.) is forbidden."""
    files = {
        "src/domain/payment.ts": """
import axios from "axios";

export interface PaymentGateway {
    charge(): void;
}
""",
        "src/application/service.ts": """
import { PaymentGateway } from "../domain/payment";
""",
    }

    verifier = ArchRuleVerifier()
    res = verifier.evaluate(files)

    assert res["is_clean"] is False
    assert any(d.code == "ARCH_DOMAIN_INFRASTRUCTURE_IMPORT" for d in res["diagnostics"])
    assert res["reward"] < 1.0


# --------------------------------------------------------------------------- #
# 4. Inward Dependency Flow & Layer Inversion Breaches
# --------------------------------------------------------------------------- #

def test_domain_imports_application_breach():
    """Domain layer must not depend on Application layer."""
    files = {
        "src/domain/user.py": """
from src.application.service import UserService

class User:
    pass
""",
        "src/application/service.py": """
class UserService:
    pass
""",
    }

    verifier = ArchRuleVerifier()
    res = verifier.evaluate(files)

    assert res["is_clean"] is False
    breach = [d for d in res["diagnostics"] if d.code == "ARCH_LAYER_BREACH"]
    assert len(breach) >= 1
    assert "Domain layer" in breach[0].message and "Application" in breach[0].message
    assert res["reward"] == pytest.approx(1.0 - 0.35)


def test_domain_imports_infrastructure_breach():
    """Domain layer must not depend on Infrastructure layer."""
    files = {
        "src/domain/user.py": """
from src.infrastructure.db import get_connection

class User:
    pass
""",
        "src/infrastructure/db.py": """
def get_connection():
    pass
""",
    }

    verifier = ArchRuleVerifier()
    res = verifier.evaluate(files)

    assert res["is_clean"] is False
    breach = [d for d in res["diagnostics"] if d.code == "ARCH_LAYER_BREACH"]
    assert len(breach) >= 1
    assert "Domain layer" in breach[0].message and "Infrastructure" in breach[0].message


def test_application_imports_infrastructure_breach():
    """Application layer must not depend on Infrastructure layer (Dependency Inversion violation)."""
    files = {
        "src/domain/user.py": """
class User: pass
""",
        "src/application/user_service.py": """
from src.domain.user import User
from src.infrastructure.postgres_repo import PostgresUserRepo

class UserService:
    def __init__(self):
        self.repo = PostgresUserRepo()
""",
        "src/infrastructure/postgres_repo.py": """
from src.domain.user import User

class PostgresUserRepo:
    pass
""",
    }

    verifier = ArchRuleVerifier()
    res = verifier.evaluate(files)

    assert res["is_clean"] is False
    breach = [d for d in res["diagnostics"] if d.code == "ARCH_LAYER_BREACH"]
    assert any("Application layer" in d.message and "Infrastructure" in d.message for d in breach)


def test_infrastructure_imports_presentation_breach():
    """Infrastructure layer must not depend on Presentation layer."""
    files = {
        "src/domain/user.py": "class User: pass",
        "src/application/service.py": "from src.domain.user import User",
        "src/infrastructure/adapter.py": """
from src.presentation.controllers import UserController
""",
        "src/presentation/controllers.py": """
from src.application.service import User
""",
    }

    verifier = ArchRuleVerifier()
    res = verifier.evaluate(files)

    assert res["is_clean"] is False
    breach = [d for d in res["diagnostics"] if d.code == "ARCH_LAYER_BREACH"]
    assert any("Infrastructure layer" in d.message and "Presentation" in d.message for d in breach)


def test_presentation_imports_infrastructure_in_strict_mode():
    """Presentation importing Infrastructure triggers warning when strict mode is active."""
    files = {
        "src/domain/user.py": "class User: pass",
        "src/application/service.py": "from src.domain.user import User",
        "src/infrastructure/repo.py": "from src.domain.user import User",
        "src/presentation/api.py": """
from src.application.service import User
from src.infrastructure.repo import User as RepoUser
""",
    }

    v_default = ArchRuleVerifier(strict_presentation_to_infra=False)
    res_default = v_default.evaluate(files)
    assert not any("Presentation layer" in d.message and "Infrastructure" in d.message for d in res_default["diagnostics"])

    v_strict = ArchRuleVerifier(strict_presentation_to_infra=True)
    res_strict = v_strict.evaluate(files)
    assert any("Presentation layer" in d.message and "Infrastructure" in d.message for d in res_strict["diagnostics"])


# --------------------------------------------------------------------------- #
# 5. Circular Dependency Detection & DAG Verification
# --------------------------------------------------------------------------- #

def test_circular_dependency_2_nodes():
    """Detect circular dependency cycle A <-> B."""
    files = {
        "src/domain/user.py": """
from src.domain.account import Account

class User:
    account: Account
""",
        "src/domain/account.py": """
from src.domain.user import User

class Account:
    user: User
""",
    }

    verifier = ArchRuleVerifier()
    res = verifier.evaluate(files)

    assert res["is_dag"] is False
    assert len(res["cycles"]) == 1
    cycle = res["cycles"][0]
    assert set(cycle) == {"src/domain/account.py", "src/domain/user.py"}

    circ_diags = [d for d in res["diagnostics"] if d.code == "ARCH_CIRCULAR_DEPENDENCY"]
    assert len(circ_diags) == 1
    # Penalty: 1.0 - 0.40 = 0.60
    assert res["reward"] == pytest.approx(0.60)


def test_circular_dependency_3_nodes_ts():
    """Detect circular dependency cycle A -> B -> C -> A in TypeScript."""
    files = {
        "src/application/service_a.ts": """
import { ServiceB } from "./service_b";
export class ServiceA {}
""",
        "src/application/service_b.ts": """
import { ServiceC } from "./service_c";
export class ServiceB {}
""",
        "src/application/service_c.ts": """
import { ServiceA } from "./service_a";
export class ServiceC {}
""",
    }

    verifier = ArchRuleVerifier()
    res = verifier.evaluate(files)

    assert res["is_dag"] is False
    assert len(res["cycles"]) == 1
    cycle = res["cycles"][0]
    assert len(cycle) == 3
    assert set(cycle) == {
        "src/application/service_a.ts",
        "src/application/service_b.ts",
        "src/application/service_c.ts",
    }
    assert res["reward"] == pytest.approx(0.60)


def test_tarjan_cycle_detection_direct():
    """Test cycle detection algorithm on synthetic graph."""
    nodes = ["A", "B", "C", "D"]
    edges = {
        "A": {"B"},
        "B": {"C"},
        "C": {"A", "D"},
        "D": set(),
    }
    cycles = detect_cycles_tarjan(nodes, edges)
    assert len(cycles) == 1
    assert cycles[0] == ["A", "B", "C"]


# --------------------------------------------------------------------------- #
# 6. Fine-Grained Penalty Scoring
# --------------------------------------------------------------------------- #

def test_fine_grained_penalty_combination():
    """Penalties for layer breaches and circular dependencies combine predictably."""
    files = {
        "src/domain/user.py": """
from src.infrastructure.db import Database
class User: pass
""",
        "src/infrastructure/db.py": """
from src.domain.user import User
class Database: pass
""",
    }

    verifier = ArchRuleVerifier()
    res = verifier.evaluate(files)

    # 1.0 - 0.35 (breach) - 0.40 (cycle) = 0.25
    assert res["reward"] == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# 7. Coupling Metrics (Ca, Ce, Fan-Out, Instability)
# --------------------------------------------------------------------------- #

def test_coupling_metrics_exact_values():
    """Verify Ca, Ce, Fan-Out, and Instability calculation across modules."""
    files = {
        "src/domain/entity.py": """
class Entity: pass
""",
        "src/application/service.py": """
from src.domain.entity import Entity
class Service: pass
""",
        "src/infrastructure/repo.py": """
from src.domain.entity import Entity
from src.application.service import Service
import sqlalchemy
import redis
""",
    }

    verifier = ArchRuleVerifier()
    res = verifier.evaluate(files)
    coupling = res["coupling"]

    entity_m = coupling["src/domain/entity.py"]
    assert entity_m["ca"] == 2
    assert entity_m["ce"] == 0
    assert entity_m["fan_out"] == 0
    assert entity_m["instability"] == 0.0

    service_m = coupling["src/application/service.py"]
    assert service_m["ca"] == 1
    assert service_m["ce"] == 1
    assert service_m["fan_out"] == 1
    assert service_m["instability"] == 0.5

    repo_m = coupling["src/infrastructure/repo.py"]
    assert repo_m["ca"] == 0
    assert repo_m["ce"] == 2
    assert repo_m["fan_out"] == 4
    assert repo_m["instability"] == 1.0


def test_excessive_coupling_penalties():
    """Penalize modules with excessive efferent coupling (Ce) and fan-out."""
    files = {
        "src/domain/m1.py": "class M1: pass",
        "src/domain/m2.py": "class M2: pass",
        "src/domain/m3.py": "class M3: pass",
        "src/application/orchestrator.py": """
from src.domain.m1 import M1
from src.domain.m2 import M2
from src.domain.m3 import M3
import httpx
import boto3
""",
    }

    verifier = ArchRuleVerifier(max_efferent_coupling=2, max_fan_out=4)
    res = verifier.evaluate(files)

    assert any(d.code == "ARCH_EXCESSIVE_COUPLING" for d in res["diagnostics"])
    assert any(d.code == "ARCH_EXCESSIVE_FAN_OUT" for d in res["diagnostics"])
    # 1.0 - 0.15 - 0.15 = 0.70
    assert res["reward"] == pytest.approx(0.70)


# --------------------------------------------------------------------------- #
# 8. Anti-Goodhart Escape-Hatch Guards
# --------------------------------------------------------------------------- #

def test_anti_goodhart_arch_ignore_short_circuits():
    """Directive # arch-ignore triggers hacked short-circuit to -1.0."""
    code = """
# arch-ignore
from src.infrastructure.db import db
class User: pass
"""
    v = ArchRuleVerifier()
    res = v.evaluate(code)
    assert res["is_hacked"] is True
    assert res["reward"] == -1.0
    assert "arch_suppress_directive" in res["hatches"]


def test_anti_goodhart_ts_ignore_short_circuits():
    """Directive @ts-ignore triggers hacked short-circuit to -1.0."""
    code = """
// @ts-ignore
import { db } from "../infrastructure/db";
"""
    v = ArchRuleVerifier()
    res = v.evaluate(code)
    assert res["is_hacked"] is True
    assert res["reward"] == -1.0
    assert "ts_ignore_directive" in res["hatches"]


def test_anti_goodhart_dynamic_import_bypass_short_circuits():
    """Bypassing via __import__ or importlib short-circuits to -1.0."""
    v = ArchRuleVerifier()

    assert v.reward('db = __import__("sqlalchemy")') == -1.0
    assert v.reward('mod = importlib.import_module("fastapi")') == -1.0
    assert v.reward('app = sys.modules["fastapi"]') == -1.0
    assert v.reward('eval("import sqlalchemy")') == -1.0


# --------------------------------------------------------------------------- #
# 9. Degenerate Output Guards
# --------------------------------------------------------------------------- #

def test_degenerate_output_guards():
    """Empty, whitespace, or comment-only outputs short-circuit to -1.0."""
    v = ArchRuleVerifier()

    assert v.reward("") == -1.0
    assert v.reward("   \n\t  ") == -1.0
    assert v.reward("# Only a clean architecture comment\n# nothing else\n") == -1.0
    assert v.reward("// Just typescript comments\n") == -1.0


# --------------------------------------------------------------------------- #
# 10. Multi-File Format Parsing
# --------------------------------------------------------------------------- #

def test_multi_file_parsing_formats():
    """Verify parsing of Markdown headers, titles, and XML tags."""
    md_header = """
Here is the clean architecture implementation:

### src/domain/user.py
```python
class User: pass
```

### src/application/service.py
```python
from src.domain.user import User
class UserService: pass
```
"""
    parsed_header = parse_multi_file_source(md_header)
    assert "src/domain/user.py" in parsed_header
    assert "src/application/service.py" in parsed_header

    md_title = """
```ts title="src/domain/user.ts"
export interface User {}
```

```ts filename="src/application/service.ts"
import { User } from "../domain/user";
```
"""
    parsed_title = parse_multi_file_source(md_title)
    assert "src/domain/user.ts" in parsed_title
    assert "src/application/service.ts" in parsed_title

    xml_text = """
<file path="src/domain/user.py">
class User: pass
</file>
<file path="src/application/service.py">
from src.domain.user import User
</file>
"""
    parsed_xml = parse_multi_file_source(xml_text)
    assert "src/domain/user.py" in parsed_xml
    assert "src/application/service.py" in parsed_xml


# --------------------------------------------------------------------------- #
# 11. Telemetry Tracking & Injectable Oracle Seam
# --------------------------------------------------------------------------- #

def test_telemetry_tracking():
    """Verify telemetry aggregates counts and rewards accurately."""
    v = ArchRuleVerifier()

    # 1. Clean
    v.evaluate({"src/domain/user.py": "class User: pass"})
    # 2. Degenerate
    v.evaluate("")
    # 3. Hacked
    v.evaluate("# arch-ignore\nclass User: pass")

    t = v.telemetry()
    assert t["n_samples"] == 3
    assert t["n_clean"] == 1
    assert t["n_degenerate"] == 1
    assert t["n_hacked"] == 1
    assert t["verifier"] == "ArchRuleVerifier"


def test_injectable_oracle_seam():
    """Verify that a mock or custom oracle can be injected cleanly."""
    class MockOracle(ArchRuleOracle):
        def analyze(self, source, **kwargs):
            self.n_calls += 1
            return ArchAnalysisResult(
                is_clean=True,
                is_dag=True,
                cycles=[],
                coupling={},
                diagnostics=[],
                elapsed_ms=0.5,
                file_layers={},
                records=[],
            )

    custom_oracle = MockOracle()
    verifier = ArchRuleVerifier(oracle=custom_oracle)

    res = verifier.evaluate({"src/domain/test.py": "class Test: pass"})
    assert res["is_clean"] is True
    assert res["reward"] == 1.0
    assert custom_oracle.n_calls == 1
