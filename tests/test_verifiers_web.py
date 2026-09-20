"""#343 -- Unit tests for Web & Backend Application Stack Verifiers.

Comprehensive CI-safe tests for:
1. TypeScript / React / Next.js / Node / NestJS
2. Python Web (FastAPI / Pydantic & Django ORM)
3. Go Web & Microservices
4. Java / Spring Boot 3
5. C# / .NET & ASP.NET Core
6. PHP / Modern Web & Laravel
7. Ruby / Rails & Modern Ruby
8. Unified WebBackendVerifier auto-routing & telemetry
9. Code suite evaluation harness (evaluate_web_backend)
"""

from __future__ import annotations

import pytest

from src.lsp.diagnostics import Diagnostic
from src.eval.code_suite import WEB_BACKEND_BUCKETS, evaluate_web_backend
from src.train.verifiers import (
    CSharpWebVerifier,
    GoWebVerifier,
    JavaWebVerifier,
    PhpWebVerifier,
    PythonWebVerifier,
    RubyWebVerifier,
    TypeScriptWebVerifier,
    WebBackendVerifier,
    find_csharp_web_escape_hatches,
    find_go_web_escape_hatches,
    find_java_web_escape_hatches,
    find_php_web_escape_hatches,
    find_python_web_escape_hatches,
    find_ruby_web_escape_hatches,
    find_ts_web_escape_hatches,
    parse_dotnet_diagnostics,
    parse_go_web_diagnostics,
    parse_javac_diagnostics,
    parse_php_web_diagnostics,
    parse_pyright_diagnostics,
    parse_ruby_web_diagnostics,
    parse_tsc_web_diagnostics,
    resolve_csharp_web_toolchain,
    resolve_go_web_toolchain,
    resolve_java_web_toolchain,
    resolve_php_web_toolchain,
    resolve_python_web_toolchain,
    resolve_ruby_web_toolchain,
    resolve_ts_web_toolchain,
)
from src.train.verifiers.web_backend import (
    CSharpWebOracle,
    GoWebOracle,
    JavaWebOracle,
    PhpWebOracle,
    PythonWebOracle,
    RubyWebOracle,
    TypeScriptWebOracle,
)


# --------------------------------------------------------------------------- #
# Fake Oracle for Injectable Seam Testing
# --------------------------------------------------------------------------- #

class FakeOracle:
    """Stands in for static oracles without invoking host toolchains."""

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
# Toolchain Resolvers Tests
# --------------------------------------------------------------------------- #

def test_toolchain_resolvers():
    ts_tools = resolve_ts_web_toolchain()
    assert isinstance(ts_tools, dict)
    assert "tsc" in ts_tools and "node" in ts_tools

    py_tools = resolve_python_web_toolchain()
    assert py_tools["ast"] is True
    assert isinstance(py_tools["pyright"], bool)

    go_tool = resolve_go_web_toolchain()
    assert go_tool is None or isinstance(go_tool, str)

    java_tool = resolve_java_web_toolchain()
    assert java_tool is None or isinstance(java_tool, str)

    csharp_tool = resolve_csharp_web_toolchain()
    assert csharp_tool is None or isinstance(csharp_tool, str)

    php_tools = resolve_php_web_toolchain()
    assert isinstance(php_tools, dict)
    assert "php" in php_tools

    ruby_tools = resolve_ruby_web_toolchain()
    assert isinstance(ruby_tools, dict)
    assert "ruby" in ruby_tools


# --------------------------------------------------------------------------- #
# Diagnostic Parsers Tests
# --------------------------------------------------------------------------- #

def test_parse_tsc_web_diagnostics():
    raw = "src/app.tsx(10,5): error TS2322: Type 'string' is not assignable to type 'number'."
    diags = parse_tsc_web_diagnostics(raw)
    assert len(diags) == 1
    assert diags[0].code == "TS2322"
    assert diags[0].line == 10
    assert diags[0].col == 5
    assert diags[0].severity == 1


def test_parse_pyright_diagnostics():
    raw = "app/main.py:15:8 - error: Cannot assign member 'foo' (reportGeneralTypeIssues)"
    diags = parse_pyright_diagnostics(raw)
    assert len(diags) == 1
    assert diags[0].code == "reportGeneralTypeIssues"
    assert diags[0].line == 15
    assert diags[0].col == 8
    assert diags[0].severity == 1


def test_parse_go_web_diagnostics():
    raw = "main.go:20:5: error: unhandled error from db.Query"
    diags = parse_go_web_diagnostics(raw)
    assert len(diags) == 1
    assert diags[0].line == 20
    assert diags[0].severity == 1


def test_parse_javac_diagnostics():
    raw = "UserService.java:42: error: cannot find symbol"
    diags = parse_javac_diagnostics(raw)
    assert len(diags) == 1
    assert diags[0].line == 42
    assert diags[0].severity == 1


def test_parse_dotnet_diagnostics():
    raw = "Program.cs(30,12): error CS0103: The name 'builder' does not exist in the current context"
    diags = parse_dotnet_diagnostics(raw)
    assert len(diags) == 1
    assert diags[0].code == "CS0103"
    assert diags[0].line == 30
    assert diags[0].severity == 1


def test_parse_php_web_diagnostics():
    raw = "14 | Method App\\Models\\User::posts() should return App\\Models\\HasMany"
    diags = parse_php_web_diagnostics(raw)
    assert len(diags) == 1
    assert diags[0].line == 14
    assert diags[0].severity == 1


def test_parse_ruby_web_diagnostics():
    raw = "app/models/user.rb:25:3: E: Lint/Syntax: unexpected token kEND"
    diags = parse_ruby_web_diagnostics(raw)
    assert len(diags) == 1
    assert diags[0].line == 25
    assert diags[0].severity == 1


# --------------------------------------------------------------------------- #
# 1. TypeScript / React / Next.js / NestJS Verifier Tests
# --------------------------------------------------------------------------- #

def test_ts_web_clean_completion():
    code = """'use client';
import React, { useState, useEffect } from 'react';

export function UserProfile({ userId }: { userId: string }) {
    const [name, setName] = useState<string>('');
    useEffect(() => {
        setName(`User-${userId}`);
    }, [userId]);
    return <div>Hello {name}</div>;
}
"""
    v = TypeScriptWebVerifier()
    r = v.reward(code)
    assert r == 1.0


def test_ts_web_missing_use_client():
    code = """import React, { useState } from 'react';

export function Counter() {
    const [count, setCount] = useState<number>(0);
    return <div>{count}</div>;
}
"""
    v = TypeScriptWebVerifier()
    r = v.reward(code)
    assert r is not None and r < 1.0


def test_ts_web_hook_missing_deps():
    code = """'use client';
import React, { useEffect, useState } from 'react';

export function Tracker() {
    const [val, setVal] = useState(0);
    useEffect(() => { setVal(1); });
    return <div>{val}</div>;
}
"""
    v = TypeScriptWebVerifier()
    r = v.reward(code)
    assert r is not None and r < 1.0


def test_ts_web_hook_in_conditional():
    code = """'use client';
import React, { useState } from 'react';

export function BadComponent({ flag }: { flag: boolean }) {
    if (flag) {
        const [x] = useState(1);
    }
    return <div />;
}
"""
    v = TypeScriptWebVerifier()
    r = v.reward(code)
    assert r is not None and r < 1.0


def test_ts_web_nestjs_untyped_param():
    code = """import { Injectable } from '@nestjs/common';

@Injectable()
export class AuthService {
    constructor(private readonly authRepo) {}
}
"""
    v = TypeScriptWebVerifier()
    r = v.reward(code)
    assert r is not None and r < 1.0


def test_ts_web_anti_goodhart_hatches():
    v = TypeScriptWebVerifier()
    # as any
    assert v.reward("const x = value as any;") == -1.0
    # @ts-ignore
    assert v.reward("// @ts-ignore\nconst y = 1;") == -1.0
    # @ts-expect-error
    assert v.reward("/* @ts-expect-error */ const z = 2;") == -1.0
    # empty JSX fragment
    assert v.reward("export const F = () => <></>;") == -1.0
    assert v.reward("export const F = () => <React.Fragment></React.Fragment>;") == -1.0
    # empty handler body
    assert v.reward("<button onClick={() => {}} />") == -1.0


# --------------------------------------------------------------------------- #
# 2. Python Web (FastAPI / Pydantic / Django) Verifier Tests
# --------------------------------------------------------------------------- #

def test_python_web_clean_completion():
    code = """from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class UserItem(BaseModel):
    id: int
    name: str

@app.get("/items/{item_id}")
async def get_item(item_id: int) -> UserItem:
    return UserItem(id=item_id, name="widget")
"""
    v = PythonWebVerifier()
    assert v.reward(code) == 1.0


def test_python_web_pydantic_untyped_field():
    code = """from pydantic import BaseModel

class Product(BaseModel):
    id = 1
    name: str = "prod"
"""
    v = PythonWebVerifier()
    r = v.reward(code)
    assert r is not None and r < 1.0


def test_python_web_fastapi_unbound_path_param():
    code = """from fastapi import FastAPI

app = FastAPI()

@app.get("/users/{user_id}/items/{item_id}")
def get_user_item(user_id: int) -> str:
    return "ok"
"""
    v = PythonWebVerifier()
    r = v.reward(code)
    assert r is not None and r < 1.0


def test_python_web_django_missing_on_delete():
    code = """from django.db import models

class Comment(models.Model):
    post = models.ForeignKey("Post")
"""
    v = PythonWebVerifier()
    r = v.reward(code)
    assert r is not None and r < 1.0


def test_python_web_anti_goodhart_hatches():
    v = PythonWebVerifier()
    # type: ignore
    assert v.reward("x = 1 # type: ignore") == -1.0
    # bare dict return
    assert v.reward("def get_data() -> dict:\n    return {'a': 1}") == -1.0
    # empty pass body
    assert v.reward("def handle_request():\n    pass") == -1.0


# --------------------------------------------------------------------------- #
# 3. Go Web & Microservices Verifier Tests
# --------------------------------------------------------------------------- #

def test_go_web_clean_completion():
    code = """package main

import (
    "fmt"
    "net/http"
)

type UserRequest struct {
    ID   int    `json:"id"`
    Name string `json:"name"`
}

func handleUser(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "OK")
}
"""
    v = GoWebVerifier()
    assert v.reward(code) == 1.0


def test_go_web_malformed_struct_tag():
    code = """package main

type BadUser struct {
    ID int `json:id`
}
"""
    v = GoWebVerifier()
    r = v.reward(code)
    assert r is not None and r < 1.0


def test_go_web_unhandled_error():
    code = """package main

func process() error {
    res, err := doSomething()
    return nil
}
"""
    v = GoWebVerifier()
    r = v.reward(code)
    assert r is not None and r < 1.0


def test_go_web_anti_goodhart_hatches():
    v = GoWebVerifier()
    # _ = err
    assert v.reward("func run() { _ = err }") == -1.0
    # empty select{}
    assert v.reward("func blockForever() { select{} }") == -1.0
    # panic only function
    assert v.reward("func notImplemented() { panic(\"TODO\") }") == -1.0


# --------------------------------------------------------------------------- #
# 4. Java / Spring Boot 3 Verifier Tests
# --------------------------------------------------------------------------- #

def test_java_web_clean_completion():
    code = """package com.example.demo;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import org.springframework.stereotype.Service;

@Entity
class Account {
    @Id
    private Long id;
}

@Service
class AccountService {
    private final AccountRepo repo;
    public AccountService(AccountRepo repo) {
        this.repo = repo;
    }
}
"""
    v = JavaWebVerifier()
    assert v.reward(code) == 1.0


def test_java_web_entity_missing_id():
    code = """package com.example.demo;

import jakarta.persistence.Entity;

@Entity
class Person {
    private String name;
}
"""
    v = JavaWebVerifier()
    r = v.reward(code)
    assert r is not None and r < 1.0


def test_java_web_field_injection():
    code = """package com.example.demo;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
class PaymentService {
    @Autowired
    private PaymentGateway gateway;
}
"""
    v = JavaWebVerifier()
    r = v.reward(code)
    assert r is not None and r < 1.0


def test_java_web_anti_goodhart_hatches():
    v = JavaWebVerifier()
    # empty catch
    assert v.reward("try { doWork(); } catch (Exception e) {}") == -1.0
    # raw types
    assert v.reward("List items = new ArrayList();") == -1.0
    # unmapped injection
    assert v.reward("@Autowired private Object rawBean;") == -1.0


# --------------------------------------------------------------------------- #
# 5. C# / .NET & ASP.NET Core Verifier Tests
# --------------------------------------------------------------------------- #

def test_csharp_web_clean_completion():
    code = """using Microsoft.AspNetCore.Builder;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.MapGet("/users/{id}", (int id) => Results.Ok(new { Id = id }));

public class Order
{
    public int Id { get; set; }
}
"""
    v = CSharpWebVerifier()
    assert v.reward(code) == 1.0


def test_csharp_web_unbound_route_param():
    code = """app.MapGet("/users/{id}", () => Results.Ok());
"""
    v = CSharpWebVerifier()
    r = v.reward(code)
    assert r is not None and r < 1.0


def test_csharp_web_anti_goodhart_hatches():
    v = CSharpWebVerifier()
    # pragma warning disable
    assert v.reward("#pragma warning disable CS8600") == -1.0
    # dynamic
    assert v.reward("dynamic item = GetDynamicObject();") == -1.0
    # null forgiving abuse
    assert v.reward("var name = user!.Name;") == -1.0


# --------------------------------------------------------------------------- #
# 6. PHP / Modern Web & Laravel Verifier Tests
# --------------------------------------------------------------------------- #

def test_php_web_clean_completion():
    code = """<?php

declare(strict_types=1);

namespace App\\Models;

use Illuminate\\Database\\Eloquent\\Model;
use Illuminate\\Database\\Eloquent\\Relations\\HasMany;

class Post extends Model
{
    public function comments(): HasMany
    {
        return $this->hasMany(Comment::class);
    }
}
"""
    v = PhpWebVerifier()
    assert v.reward(code) == 1.0


def test_php_web_missing_strict_types():
    code = """<?php

namespace App\\Services;

class CalcService {}
"""
    v = PhpWebVerifier()
    r = v.reward(code)
    assert r is not None and r < 1.0


def test_php_web_untyped_eloquent_relationship():
    code = """<?php

declare(strict_types=1);

namespace App\\Models;

use Illuminate\\Database\\Eloquent\\Model;

class Post extends Model
{
    public function comments()
    {
        return $this->hasMany(Comment::class);
    }
}
"""
    v = PhpWebVerifier()
    r = v.reward(code)
    assert r is not None and r < 1.0


def test_php_web_anti_goodhart_hatches():
    v = PhpWebVerifier()
    # phpstan-ignore
    assert v.reward("// @phpstan-ignore\n$x = 1;") == -1.0
    # untyped docblock
    assert v.reward("/** @param mixed $data */") == -1.0
    # untyped param
    assert v.reward("function compute($value) { return $value; }") == -1.0


# --------------------------------------------------------------------------- #
# 7. Ruby / Rails & Modern Ruby Verifier Tests
# --------------------------------------------------------------------------- #

def test_ruby_web_clean_completion():
    code = """# typed: strict
require 'sorbet-runtime'

class User < ApplicationRecord
  extend T::Sig
  validates :email, presence: true

  sig { params(name: String).returns(String) }
  def greet(name)
    "Hello #{name}"
  end
end
"""
    v = RubyWebVerifier()
    assert v.reward(code) == 1.0


def test_ruby_web_syntax_error():
    code = """def broken_method
  if true
    x = 1
end
"""
    v = RubyWebVerifier()
    r = v.reward(code)
    assert r is not None and r < 1.0


def test_ruby_web_sorbet_missing_sig():
    code = """# typed: strict

class Service
  def perform
    1
  end
end
"""
    v = RubyWebVerifier()
    r = v.reward(code)
    assert r is not None and r < 1.0


def test_ruby_web_anti_goodhart_hatches():
    v = RubyWebVerifier()
    # rubocop:disable
    assert v.reward("# rubocop:disable Metrics/MethodLength\ndef foo; end") == -1.0
    # T.untyped
    assert v.reward("sig { params(x: T.untyped).returns(T.untyped) }") == -1.0


# --------------------------------------------------------------------------- #
# 8. Unified WebBackendVerifier Auto-Routing & Telemetry Tests
# --------------------------------------------------------------------------- #

def test_unified_web_backend_verifier_routing():
    v = WebBackendVerifier()

    # Routing to php_web
    php_code = "<?php\ndeclare(strict_types=1);\nclass S extends ServiceProvider { public function register() {} }"
    assert v.reward(php_code) == 1.0

    # Routing to ruby_web
    ruby_code = "# typed: true\nclass Book < ApplicationRecord\n  validates :title, presence: true\nend"
    assert v.reward(ruby_code) == 1.0

    # Routing to go_web
    go_code = "package api\nimport \"net/http\"\nfunc Ping(w http.ResponseWriter, r *http.Request) {}"
    assert v.reward(go_code) == 1.0

    # Routing to python_web
    py_code = "from pydantic import BaseModel\nclass Item(BaseModel):\n    id: int\n"
    assert v.reward(py_code) == 1.0

    # Routing to java_web
    java_code = "@RestController\npublic class PingCtrl {\n    @GetMapping(\"/ping\")\n    public String ping() { return \"pong\"; }\n}"
    assert v.reward(java_code) == 1.0

    # Check telemetry
    tel = v.telemetry()
    assert tel["n_samples"] >= 5
    assert "sub_verifiers" in tel


# --------------------------------------------------------------------------- #
# 9. Code Suite Evaluation Harness Integration (evaluate_web_backend)
# --------------------------------------------------------------------------- #

def test_evaluate_web_backend_records():
    cases = [
        {
            "id": "ts_clean",
            "stack": "typescript_web",
            "code": "'use client';\nexport function View() { return <div>ok</div>; }",
            "expected_clean": True,
        },
        {
            "id": "py_hacked",
            "stack": "python_web",
            "code": "x = 1 # type: ignore",
            "expected_clean": False,
        },
    ]

    res = evaluate_web_backend(cases)
    assert res["n_cases"] == 2
    assert res["accuracy"] == 1.0
    assert len(res["records"]) == 2
    assert res["records"][0]["suite"] == "web_backend"
    assert res["records"][0]["bucket"] == "typescript_web"
    assert res["records"][1]["bucket"] == "python_web"


# --------------------------------------------------------------------------- #
# 10. Degenerate Input and Injectable Oracle Seam Tests
# --------------------------------------------------------------------------- #

def test_web_degenerate_inputs():
    v = TypeScriptWebVerifier()
    assert v.reward("") == -1.0
    assert v.reward("   \n\t  ") == -1.0
    assert v.reward("// only comments here") == -1.0


def test_web_injectable_oracle_seam():
    synthetic_diag = Diagnostic(
        code="TEST_DIAG",
        line=1,
        col=1,
        message="synthetic violation",
        offset=0,
        source="test",
        severity=1,
    )
    fake_oracle = FakeOracle(diags=[synthetic_diag])
    v = TypeScriptWebVerifier(oracle=fake_oracle)
    r = v.reward("export function Test() { return 1; }")
    assert fake_oracle.n_calls == 1
    assert r is not None and r < 1.0
    # Formula check: base_clean (1.0) - weight(sev 1 = 0.3) = 0.7
    assert r == pytest.approx(0.7)
