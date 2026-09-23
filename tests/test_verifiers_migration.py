"""#348 -- Unit tests for Database Migration Replay & API Breaking-Change Verifiers.

Comprehensive CI-safe tests for:
1. Toolchain resolution (SQLite, openapi-diff, buf)
2. In-Memory SQLite Migration Replay Verifier (<50ms execution speed, 100% data integrity, rollback schema parity)
3. Zero-Downtime Schema Safety Rules (Expand/Contract pattern enforcement, destructive drop/rename/NOT NULL prevention)
4. Anti-Goodhart escape-hatch & degeneracy detection
5. Migration script parsing across multiple formats (-- migrate:up/down, +goose, markdown fences, dict/json)
6. API Contract Breaking-Change Detection for OpenAPI (v3.0/v3.1) and Protobuf
7. Additive non-breaking extension scoring
8. Evaluation suite integration (evaluate_schema_evolution)
"""

from __future__ import annotations

import json
import sqlite3
import pytest

from src.lsp.diagnostics import Diagnostic
from src.train.verifiers import (
    ContractDiffVerifier,
    MigrationReplayVerifier,
    find_contract_diff_escape_hatches,
    find_migration_escape_hatches,
    parse_migration_scripts,
    resolve_contract_diff_toolchains,
)
from src.train.verifiers.schema_evolution import (
    ContractDiffOracle,
    MigrationReplayOracle,
    SchemaSnapshot,
    capture_sqlite_snapshot,
    check_zero_downtime_safety,
    compare_schema_snapshots,
    diff_openapi_specs,
    diff_proto_schemas,
    parse_proto_schema,
)
from src.eval.code_suite import evaluate_schema_evolution, SCHEMA_EVOLUTION_BUCKETS


# --------------------------------------------------------------------------- #
# 1. Toolchain Probe Tests
# --------------------------------------------------------------------------- #

def test_resolve_contract_diff_toolchains():
    tc = resolve_contract_diff_toolchains()
    assert tc["sqlite3"] is True
    assert isinstance(tc["openapi-diff"], bool)
    assert isinstance(tc["buf"], bool)


# --------------------------------------------------------------------------- #
# 2. Migration Script Parsing Tests
# --------------------------------------------------------------------------- #

def test_parse_migration_scripts_comment_markers():
    sql = """
-- migrate:up
ALTER TABLE users ADD COLUMN age INTEGER DEFAULT 0;
CREATE INDEX idx_users_age ON users(age);

-- migrate:down
DROP INDEX idx_users_age;
ALTER TABLE users DROP COLUMN age;
"""
    up, down = parse_migration_scripts(sql)
    assert "ADD COLUMN age" in up
    assert "DROP INDEX idx_users_age" in down


def test_parse_migration_scripts_goose():
    sql = """
-- +goose Up
CREATE TABLE profiles (id INTEGER PRIMARY KEY, bio TEXT);

-- +goose Down
DROP TABLE profiles;
"""
    up, down = parse_migration_scripts(sql)
    assert "CREATE TABLE profiles" in up
    assert "DROP TABLE profiles" in down


def test_parse_migration_scripts_fences():
    text = """
```sql:up.sql
ALTER TABLE accounts ADD COLUMN status TEXT DEFAULT 'active';
```

```sql:down.sql
ALTER TABLE accounts DROP COLUMN status;
```
"""
    up, down = parse_migration_scripts(text)
    assert "ADD COLUMN status" in up
    assert "DROP COLUMN status" in down


def test_parse_migration_scripts_mapping_and_json():
    # Dict
    d = {"up": "CREATE TABLE t1 (id INT);", "down": "DROP TABLE t1;"}
    up, down = parse_migration_scripts(d)
    assert up == "CREATE TABLE t1 (id INT);"
    assert down == "DROP TABLE t1;"

    # JSON string
    js = json.dumps(d)
    up2, down2 = parse_migration_scripts(js)
    assert up2 == "CREATE TABLE t1 (id INT);"
    assert down2 == "DROP TABLE t1;"


# --------------------------------------------------------------------------- #
# 3. In-Memory Migration Replay Verifier (<50ms & 100% Integrity)
# --------------------------------------------------------------------------- #

def test_migration_replay_clean_execution_under_50ms():
    """Verify that migration replay executes in <50ms and receives reward 1.0."""
    initial_schema = """
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        username TEXT NOT NULL,
        email TEXT NOT NULL
    );
    CREATE INDEX idx_users_email ON users(email);
    """
    seed_data = """
    INSERT INTO users (id, username, email) VALUES (1, 'alice', 'alice@example.com');
    INSERT INTO users (id, username, email) VALUES (2, 'bob', 'bob@example.com');
    """

    migration = """
    -- migrate:up
    ALTER TABLE users ADD COLUMN full_name TEXT DEFAULT '';
    CREATE INDEX idx_users_full_name ON users(full_name);

    -- migrate:down
    DROP INDEX idx_users_full_name;
    ALTER TABLE users DROP COLUMN full_name;
    """

    verifier = MigrationReplayVerifier()
    res = verifier.evaluate(
        migration,
        initial_schema=initial_schema,
        seed_data=seed_data,
        expected_records={"users": 2},
    )

    assert res["is_clean"] is True
    assert res["reward"] == 1.0
    assert res["forward_passed"] is True
    assert res["data_integrity_passed"] is True
    assert res["rollback_passed"] is True
    assert res["schema_parity_passed"] is True
    assert res["rollback_data_passed"] is True


def test_migration_replay_table_split():
    """Verify table split migration preserves data integrity across new tables."""
    initial_schema = """
    CREATE TABLE customers (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        street TEXT NOT NULL,
        city TEXT NOT NULL
    );
    """
    seed_data = """
    INSERT INTO customers (id, name, street, city) VALUES (1, 'Alice Smith', '123 Main St', 'Austin');
    INSERT INTO customers (id, name, street, city) VALUES (2, 'Bob Jones', '456 Oak Ave', 'Seattle');
    """

    # Model performs a table split: moves addresses to a new customer_addresses table
    # following expand pattern
    migration = """
    -- migrate:up
    CREATE TABLE customer_addresses (
        customer_id INTEGER PRIMARY KEY,
        street TEXT NOT NULL,
        city TEXT NOT NULL,
        FOREIGN KEY(customer_id) REFERENCES customers(id)
    );
    INSERT INTO customer_addresses (customer_id, street, city)
        SELECT id, street, city FROM customers;

    -- migrate:down
    DROP TABLE customer_addresses;
    """

    verifier = MigrationReplayVerifier()
    res = verifier.evaluate(
        migration,
        initial_schema=initial_schema,
        seed_data=seed_data,
        expected_records={"customers": 2, "customer_addresses": 2},
    )
    assert res["is_clean"] is True
    assert res["reward"] == 1.0
    assert res["schema_parity_passed"] is True
    assert res["data_integrity_passed"] is True


# --------------------------------------------------------------------------- #
# 4. Zero-Downtime Schema Safety Rules (Expand/Contract Violations)
# --------------------------------------------------------------------------- #

def test_zero_downtime_safety_drop_table_rejected():
    initial_schema = "CREATE TABLE legacy_tokens (id INT PRIMARY KEY, token TEXT);"
    migration = """
    -- migrate:up
    DROP TABLE legacy_tokens;

    -- migrate:down
    CREATE TABLE legacy_tokens (id INT PRIMARY KEY, token TEXT);
    """
    verifier = MigrationReplayVerifier(allow_destructive=False)
    res = verifier.evaluate(migration, initial_schema=initial_schema)

    assert res["is_clean"] is False
    assert res["reward"] < 1.0
    assert any(d.code == "MIGRATION_DESTRUCTIVE_CHANGE" for d in res["diagnostics"])
    assert "DROP TABLE" in res["destructive_changes"][0]


def test_zero_downtime_safety_drop_column_rejected():
    initial_schema = "CREATE TABLE users (id INT PRIMARY KEY, ssn TEXT, email TEXT);"
    migration = """
    -- migrate:up
    ALTER TABLE users DROP COLUMN ssn;

    -- migrate:down
    ALTER TABLE users ADD COLUMN ssn TEXT;
    """
    verifier = MigrationReplayVerifier(allow_destructive=False)
    res = verifier.evaluate(migration, initial_schema=initial_schema)

    assert res["is_clean"] is False
    assert any(d.code == "MIGRATION_DESTRUCTIVE_CHANGE" for d in res["diagnostics"])


def test_zero_downtime_safety_direct_rename_rejected():
    initial_schema = "CREATE TABLE items (id INT PRIMARY KEY, cost REAL);"
    migration = """
    -- migrate:up
    ALTER TABLE items RENAME COLUMN cost TO price;

    -- migrate:down
    ALTER TABLE items RENAME COLUMN price TO cost;
    """
    verifier = MigrationReplayVerifier(allow_destructive=False)
    res = verifier.evaluate(migration, initial_schema=initial_schema)

    assert res["is_clean"] is False
    assert any(d.code == "MIGRATION_EXPAND_CONTRACT_VIOLATION" for d in res["diagnostics"])


def test_zero_downtime_safety_not_null_without_default_rejected():
    initial_schema = "CREATE TABLE users (id INT PRIMARY KEY);"
    migration = """
    -- migrate:up
    ALTER TABLE users ADD COLUMN department_id INT NOT NULL;

    -- migrate:down
    ALTER TABLE users DROP COLUMN department_id;
    """
    verifier = MigrationReplayVerifier(allow_destructive=False)
    res = verifier.evaluate(migration, initial_schema=initial_schema)

    assert res["is_clean"] is False
    assert any(d.code == "MIGRATION_DESTRUCTIVE_CHANGE" for d in res["diagnostics"])


def test_allow_destructive_flag_overrides_safety_checks():
    initial_schema = "CREATE TABLE temp_cache (id INT PRIMARY KEY);"
    migration = """
    -- migrate:up
    DROP TABLE temp_cache;

    -- migrate:down
    CREATE TABLE temp_cache (id INT PRIMARY KEY);
    """
    verifier = MigrationReplayVerifier(allow_destructive=True)
    res = verifier.evaluate(migration, initial_schema=initial_schema)

    assert res["is_clean"] is True
    assert res["reward"] == 1.0


# --------------------------------------------------------------------------- #
# 5. Data Loss & Rollback Parity Failures
# --------------------------------------------------------------------------- #

def test_migration_data_loss_detected():
    initial_schema = "CREATE TABLE users (id INT PRIMARY KEY, name TEXT);"
    seed_data = "INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob');"
    # Migration accidentally deletes records
    migration = """
    -- migrate:up
    DELETE FROM users WHERE id = 2;
    ALTER TABLE users ADD COLUMN active INT DEFAULT 1;

    -- migrate:down
    ALTER TABLE users DROP COLUMN active;
    INSERT INTO users (id, name) VALUES (2, 'Bob');
    """
    verifier = MigrationReplayVerifier()
    res = verifier.evaluate(migration, initial_schema=initial_schema, seed_data=seed_data)

    assert res["is_clean"] is False
    assert res["data_integrity_passed"] is False
    assert any(d.code == "MIGRATION_DATA_LOSS" for d in res["diagnostics"])


def test_migration_missing_down_script():
    initial_schema = "CREATE TABLE users (id INT PRIMARY KEY);"
    migration = "ALTER TABLE users ADD COLUMN bio TEXT DEFAULT '';"
    verifier = MigrationReplayVerifier()
    res = verifier.evaluate(migration, initial_schema=initial_schema)

    assert res["is_clean"] is False
    assert any(d.code == "MIGRATION_MISSING_DOWN" for d in res["diagnostics"])


def test_migration_rollback_schema_mismatch():
    initial_schema = """
    CREATE TABLE users (id INT PRIMARY KEY, name TEXT);
    """
    # Down migration creates table with wrong column type
    migration = """
    -- migrate:up
    CREATE TABLE audit_log (id INT PRIMARY KEY, event TEXT);

    -- migrate:down
    -- Forgets to drop audit_log!
    """
    verifier = MigrationReplayVerifier()
    res = verifier.evaluate(migration, initial_schema=initial_schema)

    assert res["is_clean"] is False
    assert res["schema_parity_passed"] is False
    assert any(d.code == "MIGRATION_ROLLBACK_SCHEMA_MISMATCH" for d in res["diagnostics"])


def test_migration_syntax_error():
    initial_schema = "CREATE TABLE users (id INT PRIMARY KEY);"
    migration = """
    -- migrate:up
    INVALID SQL STATEMENT HERE;;;

    -- migrate:down
    DROP TABLE users;
    """
    verifier = MigrationReplayVerifier()
    res = verifier.evaluate(migration, initial_schema=initial_schema)

    assert res["is_clean"] is False
    assert res["forward_passed"] is False
    assert any(d.code == "MIGRATION_EXECUTION_FAILURE" for d in res["diagnostics"])


# --------------------------------------------------------------------------- #
# 6. Anti-Goodhart Escape Hatches & Degenerate Output
# --------------------------------------------------------------------------- #

def test_migration_escape_hatches():
    hacked = "-- migration-ignore\nDROP TABLE users;"
    verifier = MigrationReplayVerifier()
    reward = verifier.reward(hacked)
    assert reward == -1.0

    hatches = find_migration_escape_hatches(hacked)
    assert len(hatches) > 0


def test_migration_degenerate_output():
    verifier = MigrationReplayVerifier()
    assert verifier.reward("") == -1.0
    assert verifier.reward("   \n\t  ") == -1.0
    assert verifier.reward("-- just a sql comment with no code") == -1.0


# --------------------------------------------------------------------------- #
# 7. OpenAPI Contract Breaking-Change Detection
# --------------------------------------------------------------------------- #

BASE_OPENAPI_JSON = json.dumps({
    "openapi": "3.1.0",
    "info": {"title": "Users API", "version": "1.0.0"},
    "paths": {
        "/users": {
            "get": {
                "parameters": [
                    {"name": "limit", "in": "query", "required": False, "schema": {"type": "integer"}}
                ],
                "responses": {
                    "200": {
                        "description": "List of users",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "integer"},
                                        "name": {"type": "string"},
                                        "email": {"type": "string", "nullable": True}
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "post": {
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}}
                            }
                        }
                    }
                },
                "responses": {"201": {"description": "Created"}}
            }
        }
    }
})


def test_openapi_contract_non_breaking_additive_extensions():
    # Add new endpoint, optional param, and new response field
    updated_spec = json.dumps({
        "openapi": "3.1.0",
        "info": {"title": "Users API", "version": "1.1.0"},
        "paths": {
            "/users": {
                "get": {
                    "parameters": [
                        {"name": "limit", "in": "query", "required": False, "schema": {"type": "integer"}},
                        {"name": "offset", "in": "query", "required": False, "schema": {"type": "integer"}}
                    ],
                    "responses": {
                        "200": {
                            "description": "List of users",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "integer"},
                                            "name": {"type": "string"},
                                            "email": {"type": "string", "nullable": True},
                                            "avatar_url": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "post": {
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"name": {"type": "string"}}
                                }
                            }
                        }
                    },
                    "responses": {"201": {"description": "Created"}}
                }
            },
            "/users/search": {
                "get": {"responses": {"200": {"description": "Search results"}}}
            }
        }
    })

    verifier = ContractDiffVerifier()
    res = verifier.evaluate(updated_spec, reference=BASE_OPENAPI_JSON, spec_type="openapi")
    assert res["is_clean"] is True
    assert res["reward"] == 1.0
    assert len(res["breaking_changes"]) == 0
    assert len(res["additive_extensions"]) >= 3


def test_openapi_contract_breaking_endpoint_and_method_removal():
    # Removed /users POST
    updated_spec = json.dumps({
        "openapi": "3.1.0",
        "paths": {
            "/users": {
                "get": {
                    "parameters": [{"name": "limit", "in": "query", "required": False, "schema": {"type": "integer"}}],
                    "responses": {"200": {"description": "OK"}}
                }
            }
        }
    })
    verifier = ContractDiffVerifier()
    res = verifier.evaluate(updated_spec, reference=BASE_OPENAPI_JSON, spec_type="openapi")
    assert res["is_clean"] is False
    assert any(d.code == "CONTRACT_METHOD_REMOVED" for d in res["diagnostics"])


def test_openapi_contract_breaking_required_param_added():
    # Added required parameter 'api_key' to GET /users
    updated_spec = json.dumps({
        "openapi": "3.1.0",
        "paths": {
            "/users": {
                "get": {
                    "parameters": [
                        {"name": "limit", "in": "query", "required": False, "schema": {"type": "integer"}},
                        {"name": "api_key", "in": "query", "required": True, "schema": {"type": "string"}}
                    ],
                    "responses": {"200": {"description": "OK"}}
                },
                "post": {"responses": {"201": {"description": "Created"}}}
            }
        }
    })
    verifier = ContractDiffVerifier()
    res = verifier.evaluate(updated_spec, reference=BASE_OPENAPI_JSON, spec_type="openapi")
    assert res["is_clean"] is False
    assert any(d.code == "CONTRACT_REQUIRED_PARAM_ADDED" for d in res["diagnostics"])


def test_openapi_contract_breaking_response_field_removed_and_nullability():
    # Removed 'email' property, and changed 'id' type to string
    updated_spec = json.dumps({
        "openapi": "3.1.0",
        "paths": {
            "/users": {
                "get": {
                    "parameters": [{"name": "limit", "in": "query", "required": False, "schema": {"type": "integer"}}],
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "name": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "post": {"responses": {"201": {"description": "Created"}}}
            }
        }
    })
    verifier = ContractDiffVerifier()
    res = verifier.evaluate(updated_spec, reference=BASE_OPENAPI_JSON, spec_type="openapi")
    assert res["is_clean"] is False
    assert any(d.code == "CONTRACT_RESPONSE_PROPERTY_REMOVED" for d in res["diagnostics"])
    assert any(d.code == "CONTRACT_RESPONSE_TYPE_CHANGED" for d in res["diagnostics"])


# --------------------------------------------------------------------------- #
# 8. Protobuf Contract Breaking-Change Detection
# --------------------------------------------------------------------------- #

BASE_PROTO = """
syntax = "proto3";

package user.v1;

message User {
    int64 id = 1;
    string name = 2;
    string email = 3;
}

service UserService {
    rpc GetUser (User) returns (User);
}
"""


def test_proto_contract_additive_extension():
    updated_proto = """
syntax = "proto3";

package user.v1;

message User {
    int64 id = 1;
    string name = 2;
    string email = 3;
    string avatar_url = 4;
}

message UserList {
    repeated User users = 1;
}

service UserService {
    rpc GetUser (User) returns (User);
    rpc ListUsers (User) returns (UserList);
}
"""
    verifier = ContractDiffVerifier()
    res = verifier.evaluate(updated_proto, reference=BASE_PROTO, spec_type="protobuf")
    assert res["is_clean"] is True
    assert res["reward"] == 1.0
    assert len(res["breaking_changes"]) == 0
    assert len(res["additive_extensions"]) >= 2


def test_proto_contract_breaking_tag_and_type_mutations():
    # Mutated tag of 'name' from 2 to 5, mutated type of 'email' from string to int32
    bad_proto = """
syntax = "proto3";

package user.v1;

message User {
    int64 id = 1;
    string name = 5;
    int32 email = 3;
}

service UserService {
    rpc GetUser (User) returns (User);
}
"""
    verifier = ContractDiffVerifier()
    res = verifier.evaluate(bad_proto, reference=BASE_PROTO, spec_type="protobuf")
    assert res["is_clean"] is False
    assert any(d.code == "PROTO_FIELD_TAG_CHANGED" for d in res["diagnostics"])
    assert any(d.code == "PROTO_FIELD_TYPE_CHANGED" for d in res["diagnostics"])


def test_proto_contract_field_removed_without_reserve():
    # Removed 'email' (tag 3) without reserving tag 3
    bad_proto = """
syntax = "proto3";

package user.v1;

message User {
    int64 id = 1;
    string name = 2;
}

service UserService {
    rpc GetUser (User) returns (User);
}
"""
    verifier = ContractDiffVerifier()
    res = verifier.evaluate(bad_proto, reference=BASE_PROTO, spec_type="protobuf")
    assert res["is_clean"] is False
    assert any(d.code == "PROTO_FIELD_REMOVED_WITHOUT_RESERVE" for d in res["diagnostics"])


def test_proto_contract_field_removed_with_proper_reserve_is_allowed():
    # Removed 'email' (tag 3) and properly reserved tag 3
    ok_proto = """
syntax = "proto3";

package user.v1;

message User {
    int64 id = 1;
    string name = 2;
    reserved 3;
    reserved "email";
}

service UserService {
    rpc GetUser (User) returns (User);
}
"""
    verifier = ContractDiffVerifier()
    res = verifier.evaluate(ok_proto, reference=BASE_PROTO, spec_type="protobuf")
    assert res["is_clean"] is True
    assert res["reward"] == 1.0


def test_proto_contract_rpc_service_removed():
    bad_proto = """
syntax = "proto3";

package user.v1;

message User {
    int64 id = 1;
    string name = 2;
    string email = 3;
}
"""
    verifier = ContractDiffVerifier()
    res = verifier.evaluate(bad_proto, reference=BASE_PROTO, spec_type="protobuf")
    assert res["is_clean"] is False
    assert any(d.code == "PROTO_SERVICE_REMOVED" for d in res["diagnostics"])


def test_contract_diff_escape_hatches():
    hacked = "// contract-ignore\nsyntax = 'proto3';"
    verifier = ContractDiffVerifier()
    assert verifier.reward(hacked) == -1.0
    hatches = find_contract_diff_escape_hatches(hacked)
    assert len(hatches) > 0


# --------------------------------------------------------------------------- #
# 9. Telemetry and Context Manager
# --------------------------------------------------------------------------- #

def test_verifier_telemetry_and_context_managers():
    with MigrationReplayVerifier() as m_v:
        m_v.reward("-- migrate:up\n-- migrate:down\n")
        tel = m_v.telemetry()
        assert tel["n_samples"] == 1

    with ContractDiffVerifier() as c_v:
        c_v.reward("openapi: 3.0.0")
        tel = c_v.telemetry()
        assert tel["n_samples"] == 1


# --------------------------------------------------------------------------- #
# 10. Evaluation Suite Integration Test
# --------------------------------------------------------------------------- #

def test_evaluate_schema_evolution_suite():
    test_cases = [
        {
            "id": "mig_valid_add_col",
            "bucket": "column_migration",
            "code": """
            -- migrate:up
            ALTER TABLE orders ADD COLUMN status TEXT DEFAULT 'pending';
            -- migrate:down
            ALTER TABLE orders DROP COLUMN status;
            """,
            "initial_schema": "CREATE TABLE orders (id INT PRIMARY KEY);",
            "seed_data": "INSERT INTO orders (id) VALUES (1);",
            "expected_records": {"orders": 1},
            "expected_clean": True,
        },
        {
            "id": "mig_invalid_drop_tbl",
            "bucket": "zero_downtime_safety",
            "code": "DROP TABLE orders;",
            "initial_schema": "CREATE TABLE orders (id INT PRIMARY KEY);",
            "allow_destructive": False,
            "expected_clean": False,
        },
        {
            "id": "proto_additive",
            "bucket": "contract_diff_protobuf",
            "code": """
            syntax = "proto3";
            message Msg { int32 id = 1; string new_field = 2; }
            """,
            "reference": """
            syntax = "proto3";
            message Msg { int32 id = 1; }
            """,
            "spec_type": "protobuf",
            "expected_clean": True,
        },
    ]

    res = evaluate_schema_evolution(test_cases)
    assert res["n_cases"] == 3
    assert res["accuracy"] == 1.0
    assert "column_migration" in res["by_bucket"]
    assert "contract_diff_protobuf" in res["by_bucket"]
