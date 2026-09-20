"""#342 -- Unit tests for Data Engineering, Contracts & Query Verifiers.

Comprehensive CI-safe tests for:
1. Relational SQL (Postgres, MySQL, SQLite, DuckDB)
2. Data & ML Pipelines (Pandas, Polars, PyTorch)
3. API Contracts (OpenAPI v3.1 & JSON Schema)
4. GraphQL Schemas & Operations
5. Protocol Buffers & gRPC
6. Clean Architecture Boundary Linter
7. Unified Data Contracts Verifier & Auto-routing
"""

from __future__ import annotations

import pytest

from src.lsp.diagnostics import Diagnostic
from src.train.verifiers import (
    CleanArchitectureVerifier,
    DataContractsVerifier,
    DataPipelineVerifier,
    GraphQLVerifier,
    OpenApiVerifier,
    ProtobufVerifier,
    SqlVerifier,
    find_architecture_escape_hatches,
    find_data_pipeline_escape_hatches,
    find_graphql_escape_hatches,
    find_openapi_escape_hatches,
    find_protobuf_escape_hatches,
    find_sql_escape_hatches,
    resolve_graphql_toolchain,
    resolve_openapi_toolchain,
    resolve_protobuf_toolchain,
    resolve_spectral_toolchain,
    resolve_sql_toolchain,
)
from src.train.verifiers.data_contracts import (
    CleanArchitectureOracle,
    DataPipelineOracle,
    GraphQLContractOracle,
    OpenApiContractOracle,
    ProtobufContractOracle,
    SqliteMemoryOracle,
)


# --------------------------------------------------------------------------- #
# Fake Oracle for Injectable Seam Testing
# --------------------------------------------------------------------------- #

class FakeOracle:
    def __init__(self, diags=None, raises: bool = False):
        self._diags = [] if diags is None else diags
        self._raises = raises
        self.calls = []
        self.n_calls = 0
        self.wall_s = 0.0
        self.close_count = 0

    def diagnostics(self, source: str, **kwargs) -> list[Diagnostic]:
        self.calls.append(source)
        self.n_calls += 1
        self.wall_s += 0.001
        if self._raises:
            raise RuntimeError("simulated oracle failure")
        return list(self._diags(source)) if callable(self._diags) else list(self._diags)

    def close(self) -> None:
        self.close_count += 1


# --------------------------------------------------------------------------- #
# 1. Relational SQL Verifier Tests
# --------------------------------------------------------------------------- #

def test_sql_toolchain_probe():
    tools = resolve_sql_toolchain()
    assert tools["sqlite3"] is True
    assert isinstance(tools["sqlglot"], bool)


def test_sql_clean_query_reward():
    v = SqlVerifier()
    reward = v.reward("SELECT id, username, email FROM users WHERE active = 1;")
    assert reward == 1.0


def test_sql_anti_goodhart_select_star():
    v = SqlVerifier()
    assert v.reward("SELECT * FROM users;") == -1.0
    assert v.reward("select DISTINCT * from accounts;") == -1.0
    hatches = find_sql_escape_hatches("SELECT * FROM users;")
    assert "select_star" in hatches


def test_sql_anti_goodhart_cartesian_join():
    v = SqlVerifier()
    # Cross join
    assert v.reward("SELECT id FROM users CROSS JOIN orders;") == -1.0
    # Join without ON / USING
    assert v.reward("SELECT id FROM users JOIN orders;") == -1.0
    assert v.reward("SELECT id FROM users JOIN orders WHERE orders.id = 1;") == -1.0
    # Valid join with ON must not be penalized
    assert v.reward("SELECT u.id FROM users u JOIN orders o ON u.id = o.user_id;") == 1.0
    # Valid join with USING must not be penalized
    assert v.reward("SELECT id FROM users JOIN orders USING (user_id);") == 1.0


def test_sql_in_memory_execution_with_schema():
    ddl = "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL);"
    v = SqlVerifier(schema_ddl=ddl)
    # Valid column resolution
    assert v.reward("SELECT id, name FROM users WHERE id = 1;") == 1.0
    # Unknown column resolution
    assert v.reward("SELECT id, non_existent_column FROM users;") < 1.0
    # Unknown table
    assert v.reward("SELECT id FROM missing_table;") < 1.0


def test_sql_index_usability():
    ddl = """
    CREATE TABLE audit (id INTEGER PRIMARY KEY, ts INTEGER, data TEXT);
    CREATE INDEX idx_audit_ts ON audit(ts);
    """
    oracle = SqliteMemoryOracle(schema_ddl=ddl, check_indexes=True)
    # Table scan on table with index
    diags = oracle.diagnostics("SELECT data FROM audit WHERE data = 'test';")
    assert any(d.code == "SQL_UNINDEXED_SCAN" for d in diags)


def test_sql_equivalence_check():
    ddl = """
    CREATE TABLE metrics (name TEXT, value INTEGER);
    INSERT INTO metrics VALUES ('cpu', 80), ('mem', 60);
    """
    oracle = SqliteMemoryOracle(schema_ddl=ddl)
    query = "SELECT name, value FROM metrics WHERE value > 50;"
    ref_same = "SELECT name, value FROM metrics WHERE value >= 60;"
    ref_diff = "SELECT name, value FROM metrics WHERE value > 70;"

    # Same result set -> clean
    assert not oracle.diagnostics(query, reference=ref_same)
    # Different result set -> mismatch diagnostic
    diags_diff = oracle.diagnostics(query, reference=ref_diff)
    assert any(d.code == "SQL_EQUIVALENCE_MISMATCH" for d in diags_diff)


# --------------------------------------------------------------------------- #
# 2. Data & ML Pipelines Verifier Tests
# --------------------------------------------------------------------------- #

def test_data_pipeline_clean():
    v = DataPipelineVerifier(known_columns=["features", "target"])
    code = """
import torch
x = df.select(["features", "target"])
tensor = torch.tensor(x["features"].to_numpy())
y = tensor.permute(0, 2, 1)
"""
    assert v.reward(code) == 1.0


def test_data_pipeline_unknown_columns():
    v = DataPipelineVerifier(known_columns=["features", "target"])
    code = 'df.select(["features", "invalid_col"])'
    assert v.reward(code) < 1.0


def test_data_pipeline_anti_goodhart_dynamic_shape():
    v = DataPipelineVerifier()
    # view(-1) dynamic shape escape
    assert v.reward("x = tensor.view(-1)") == -1.0
    # reshape(-1)
    assert v.reward("x = tensor.reshape(-1)") == -1.0
    # unconstrained Any type annotation
    assert v.reward("def forward(x: Any): pass") == -1.0


def test_data_pipeline_anti_goodhart_bare_object():
    v = DataPipelineVerifier()
    assert v.reward('df["col"] = df["col"].astype("object")') == -1.0
    assert v.reward('s = pd.Series([1, 2], dtype=object)') == -1.0
    assert v.reward("col = pl.col('a').cast(pl.Object)") == -1.0


def test_data_pipeline_anti_goodhart_row_iteration():
    v = DataPipelineVerifier()
    assert v.reward("for idx, row in df.iterrows(): pass") == -1.0
    assert v.reward("for row in df.itertuples(): pass") == -1.0
    assert v.reward("df['res'] = df.apply(lambda r: r['a'] + 1, axis=1)") == -1.0


def test_data_pipeline_permute_duplicate_dims():
    oracle = DataPipelineOracle()
    code = "tensor = x.permute(0, 1, 1, 2)"
    diags = oracle.diagnostics(code)
    assert any(d.code == "TENSOR_INVALID_PERMUTE" for d in diags)


# --------------------------------------------------------------------------- #
# 3. API Contracts (OpenAPI & JSON Schema) Verifier Tests
# --------------------------------------------------------------------------- #

def test_openapi_toolchain_probe():
    tools = resolve_openapi_toolchain()
    assert "openapi_spec_validator" in tools
    assert "jsonschema" in tools
    assert "spectral" in tools


def test_openapi_valid_contract():
    v = OpenApiVerifier()
    spec = """
openapi: "3.1.0"
info:
  title: Users Service
  version: "1.0.0"
paths:
  /users/{userId}:
    get:
      summary: Get user
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
            pattern: "^usr_[a-zA-Z0-9]+$"
      responses:
        "200":
          description: Success
          content:
            application/json:
              schema:
                type: object
                properties:
                  id:
                    type: string
                    format: uuid
"""
    assert v.reward(spec) == 1.0


def test_openapi_unbound_path_parameter():
    v = OpenApiVerifier()
    spec = """
openapi: "3.1.0"
info:
  title: Test
  version: "1.0.0"
paths:
  /orders/{orderId}/items:
    get:
      responses:
        "200":
          description: OK
"""
    # orderId is in path template but not declared in parameters
    assert v.reward(spec) < 1.0


def test_openapi_missing_2xx_response():
    v = OpenApiVerifier()
    spec = """
openapi: "3.1.0"
info:
  title: Test
  version: "1.0.0"
paths:
  /ping:
    get:
      responses:
        "400":
          description: Bad Request
"""
    assert v.reward(spec) < 1.0


def test_openapi_anti_goodhart_untyped_payload():
    v = OpenApiVerifier()
    spec = """
openapi: "3.1.0"
info:
  title: Test
  version: "1.0.0"
paths:
  /data:
    post:
      requestBody:
        content:
          application/json:
            schema: {}
      responses:
        "200":
          description: OK
"""
    assert v.reward(spec) == -1.0


def test_openapi_anti_goodhart_unconstrained_string():
    v = OpenApiVerifier()
    spec = """
openapi: "3.1.0"
info:
  title: Test
  version: "1.0.0"
paths:
  /data:
    post:
      parameters:
        - name: query
          in: query
          schema:
            type: string
      responses:
        "200":
          description: OK
"""
    # Unconstrained string schema (no format, pattern, minLength, maxLength, or enum)
    assert v.reward(spec) == -1.0


# --------------------------------------------------------------------------- #
# 4. GraphQL Schemas & Operations Verifier Tests
# --------------------------------------------------------------------------- #

def test_graphql_toolchain_probe():
    tools = resolve_graphql_toolchain()
    assert "graphql_core" in tools


def test_graphql_valid_sdl():
    v = GraphQLVerifier()
    sdl = """
type Query {
  user(id: ID!): User
}

type User {
  id: ID!
  username: String!
  email: String
}
"""
    assert v.reward(sdl) == 1.0


def test_graphql_syntax_error():
    v = GraphQLVerifier()
    bad_sdl = "type Query { user: User"
    assert v.reward(bad_sdl) < 1.0


def test_graphql_anti_goodhart_circular_query_depth():
    v = GraphQLVerifier(max_query_depth=4)
    # Deeply nested recursive query exceeding threshold
    query = """
query GetUserFriends {
  user(id: "1") {
    friends {
      friends {
        friends {
          friends {
            friends {
              id
            }
          }
        }
      }
    }
  }
}
"""
    oracle = GraphQLContractOracle(max_query_depth=4)
    diags = oracle.diagnostics(query)
    assert any(d.code == "GRAPHQL_CIRCULAR_QUERY_DEPTH" for d in diags)


def test_graphql_anti_goodhart_unconstrained_scalar():
    v = GraphQLVerifier()
    assert v.reward("scalar Any") == -1.0
    assert v.reward("scalar JSON") == -1.0
    assert v.reward("scalar Object") == -1.0


# --------------------------------------------------------------------------- #
# 5. Protocol Buffers & gRPC Verifier Tests
# --------------------------------------------------------------------------- #

def test_protobuf_toolchain_probe():
    # May be None or a path, function must return safely
    path = resolve_protobuf_toolchain()
    assert path is None or isinstance(path, str)


def test_protobuf_valid_proto():
    v = ProtobufVerifier()
    proto = """
syntax = "proto3";
package telemetry.v1;

message MetricPayload {
  string metric_name = 1;
  double value = 2;
  int64 timestamp_ms = 3;
}
"""
    assert v.reward(proto) == 1.0


def test_protobuf_duplicate_tag():
    v = ProtobufVerifier()
    proto = """
syntax = "proto3";
package test;

message Record {
  string first_name = 1;
  string last_name = 1;
}
"""
    assert v.reward(proto) < 1.0


def test_protobuf_reserved_tag():
    v = ProtobufVerifier()
    proto = """
syntax = "proto3";
package test;

message Record {
  reserved 2, 10 to 15;
  string name = 1;
  string title = 2;
}
"""
    assert v.reward(proto) < 1.0


def test_protobuf_invalid_tag_range():
    oracle = ProtobufContractOracle()
    proto = """
syntax = "proto3";
package test;

message Record {
  string invalid_high = 19005;
}
"""
    diags = oracle.diagnostics(proto)
    assert any(d.code == "PROTO_INVALID_TAG_NUMBER" for d in diags)


def test_protobuf_breaking_mutation_against_reference():
    v = ProtobufVerifier()
    ref = """
syntax = "proto3";
package test;

message User {
  int64 id = 1;
  string email = 2;
}
"""
    # Breaking mutation: changed tag for email from 2 to 3
    candidate = """
syntax = "proto3";
package test;

message User {
  int64 id = 1;
  string email = 3;
}
"""
    oracle = ProtobufContractOracle()
    diags = oracle.diagnostics(candidate, reference=ref)
    assert any(d.code == "PROTO_BREAKING_TAG_MUTATION" for d in diags)


def test_protobuf_anti_goodhart_missing_field_number():
    v = ProtobufVerifier()
    proto = """
syntax = "proto3";
package test;

message User {
  string name;
  int64 id = 2;
}
"""
    assert v.reward(proto) == -1.0


# --------------------------------------------------------------------------- #
# 6. Clean Architecture Boundary Linter Tests
# --------------------------------------------------------------------------- #

def test_clean_architecture_domain_clean():
    v = CleanArchitectureVerifier(layer="domain")
    code = """
from dataclasses import dataclass
from typing import Optional

@dataclass
class Order:
    order_id: str
    amount: float
    status: str = "pending"

    def cancel(self) -> None:
        self.status = "cancelled"
"""
    assert v.reward(code) == 1.0


def test_clean_architecture_domain_forbidden_framework():
    v = CleanArchitectureVerifier(layer="domain")
    code = """
from fastapi import HTTPException
from sqlalchemy import Column, Integer

class UserEntity:
    pass
"""
    assert v.reward(code) < 1.0


def test_clean_architecture_layer_inversion():
    v = CleanArchitectureVerifier(layer="domain")
    code = """
from ..infrastructure.database import get_db
from ..api.routers import router

class UserEntity:
    pass
"""
    oracle = CleanArchitectureOracle(layer="domain")
    diags = oracle.diagnostics(code)
    assert any(d.code == "ARCH_LAYER_INVERSION" for d in diags)


def test_clean_architecture_anti_goodhart_dynamic_import_bypass():
    v = CleanArchitectureVerifier(layer="domain")
    assert v.reward('db = __import__("sqlalchemy")') == -1.0
    assert v.reward('mod = importlib.import_module("fastapi")') == -1.0
    assert v.reward('app = sys.modules["fastapi"]') == -1.0
    assert v.reward('eval("import fastapi")') == -1.0


# --------------------------------------------------------------------------- #
# 7. Unified Data Contracts Verifier & Auto-Routing Tests
# --------------------------------------------------------------------------- #

def test_unified_data_contracts_verifier_routing():
    v = DataContractsVerifier()
    # SQL routing
    sql = "SELECT id, name FROM users WHERE id = 1;"
    assert v.reward(sql) == 1.0
    assert v.reward("SELECT * FROM users;") == -1.0

    # Protobuf routing
    proto = 'syntax = "proto3"; message User { string name = 1; }'
    assert v.reward(proto) == 1.0
    assert v.reward('syntax = "proto3"; message User { string name; }') == -1.0

    # GraphQL routing
    gql = "type Query { user: User } type User { id: ID! }"
    assert v.reward(gql) == 1.0

    # Telemetry aggregation
    t = v.telemetry()
    assert "sub_verifiers" in t
    assert "sql" in t["sub_verifiers"]
    assert "protobuf" in t["sub_verifiers"]


def test_contract_verifier_fake_oracle_injection():
    fake = FakeOracle(raises=True)
    v = SqlVerifier(oracle=fake, on_error="skip")
    assert v.reward("SELECT 1;") is None
    assert fake.n_calls == 1


def test_contract_verifier_fail_fast_on_degenerate():
    fake = FakeOracle()
    v = SqlVerifier(oracle=fake, fail_fast=True)
    # Degenerate empty completion must not invoke oracle
    r = v.reward("")
    assert r == -1.0
    assert fake.n_calls == 0
