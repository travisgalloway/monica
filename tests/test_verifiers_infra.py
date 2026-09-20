"""#344 -- Unit tests for Cloud, Infra, Containers & Automation Verifiers.

Comprehensive CI-safe tests for:
1. Cloud IaC (Terraform / OpenTofu & HCL)
2. Containers & Packaging (Docker & Containerfiles)
3. Orchestration (Kubernetes YAML & Helm)
4. Shell Scripting & POSIX Automation (Bash / POSIX sh)
5. Web Core & UI Styling (HTML5, CSS3, Tailwind CSS)
6. Unified CloudInfraVerifier & Auto-Routing
7. Telemetry Reporting (syntax vs. linter vs. escape-hatch penalties)
8. Code Suite Benchmark Evaluator (evaluate_cloud_infra)
9. Safety Invariants (no container launch, no cloud calls, zero untrusted code execution)
"""

from __future__ import annotations

import pytest

from src.lsp.diagnostics import Diagnostic
from src.train.verifiers import (
    BaseCloudInfraVerifier,
    CloudInfraVerifier,
    DockerfileOracle,
    DockerfileVerifier,
    HtmlTailwindOracle,
    HtmlTailwindVerifier,
    KubernetesOracle,
    KubernetesVerifier,
    ShellOracle,
    ShellVerifier,
    TerraformOracle,
    TerraformVerifier,
    find_docker_escape_hatches,
    find_html_tailwind_escape_hatches,
    find_kubernetes_escape_hatches,
    find_shell_escape_hatches,
    find_terraform_escape_hatches,
    resolve_docker_toolchain,
    resolve_html_tailwind_toolchain,
    resolve_kubernetes_toolchain,
    resolve_shell_toolchain,
    resolve_terraform_toolchain,
)
from src.eval.code_suite import evaluate_cloud_infra, CLOUD_INFRA_BUCKETS


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
# 1. Toolchain Probe Tests
# --------------------------------------------------------------------------- #

def test_toolchain_probes():
    tf = resolve_terraform_toolchain()
    assert isinstance(tf["hcl2"], bool)
    assert isinstance(tf["terraform"], bool)

    docker = resolve_docker_toolchain()
    assert isinstance(docker["dockerfile_parse"], bool)
    assert isinstance(docker["hadolint"], bool)

    k8s = resolve_kubernetes_toolchain()
    assert k8s["pyyaml"] is True
    assert isinstance(k8s["yamllint"], bool)

    shell = resolve_shell_toolchain()
    assert isinstance(shell["bashlex"], bool)
    assert isinstance(shell["shellcheck"], bool)

    html = resolve_html_tailwind_toolchain()
    assert isinstance(html["html5lib"], bool)
    assert isinstance(html["lightningcss"], bool)


# --------------------------------------------------------------------------- #
# 2. Cloud IaC (Terraform / OpenTofu & HCL) Tests
# --------------------------------------------------------------------------- #

def test_terraform_clean_config():
    code = """
variable "environment" {
  type        = string
  description = "Deployment target"
  default     = "production"
}

locals {
  tier = "frontend"
}

resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.micro"
  tags = {
    Env  = var.environment
    Tier = local.tier
  }
}
"""
    v = TerraformVerifier()
    reward = v.reward(code)
    assert reward == 1.0


def test_terraform_syntax_error_unbalanced_braces():
    code = """
resource "aws_instance" "web" {
  ami = "ami-12345"
"""
    v = TerraformVerifier()
    reward = v.reward(code)
    assert reward < 1.0
    oracle = TerraformOracle()
    diags = oracle.diagnostics(code)
    assert any(d.code == "HCL_SYNTAX_UNBALANCED_BRACES" for d in diags)


def test_terraform_invalid_variable_type():
    code = """
variable "instance_count" {
  type = completely_invalid_type
}
"""
    v = TerraformVerifier()
    reward = v.reward(code)
    assert reward < 1.0
    oracle = TerraformOracle()
    diags = oracle.diagnostics(code)
    assert any(d.code == "TF_INVALID_VARIABLE_TYPE" for d in diags)


def test_terraform_dag_unresolved_reference():
    code = """
resource "aws_instance" "web" {
  subnet_id = aws_subnet.missing_subnet.id
}
"""
    v = TerraformVerifier()
    reward = v.reward(code)
    assert reward < 1.0
    oracle = TerraformOracle()
    diags = oracle.diagnostics(code)
    assert any(d.code == "TF_UNRESOLVED_REFERENCE" for d in diags)


def test_terraform_dag_circular_dependency():
    code = """
resource "aws_security_group" "a" {
  name = "sg_a"
  description = aws_security_group.b.id
}

resource "aws_security_group" "b" {
  name = "sg_b"
  description = aws_security_group.a.id
}
"""
    v = TerraformVerifier()
    reward = v.reward(code)
    assert reward < 1.0
    oracle = TerraformOracle()
    diags = oracle.diagnostics(code)
    assert any(d.code == "TF_CIRCULAR_DEPENDENCY" for d in diags)


def test_terraform_anti_goodhart_hardcoded_secrets():
    code = """
resource "aws_db_instance" "default" {
  allocated_storage = 10
  username          = "foo"
  password          = "SuperSecretPassword123"
}
"""
    v = TerraformVerifier()
    assert v.reward(code) == -1.0
    hatches = find_terraform_escape_hatches(code)
    assert "hardcoded_secrets" in hatches


def test_terraform_anti_goodhart_ignore_changes_all():
    code = """
resource "aws_instance" "web" {
  ami = "ami-123"
  lifecycle {
    ignore_changes = all
  }
}
"""
    v = TerraformVerifier()
    assert v.reward(code) == -1.0
    hatches = find_terraform_escape_hatches(code)
    assert "ignore_changes_all" in hatches


def test_terraform_anti_goodhart_directive_suppression():
    code = """
# tflint:ignore
resource "aws_instance" "web" {
  ami = "ami-123"
}
"""
    v = TerraformVerifier()
    assert v.reward(code) == -1.0


# --------------------------------------------------------------------------- #
# 3. Containers & Packaging (Docker & Containerfiles) Tests
# --------------------------------------------------------------------------- #

def test_dockerfile_clean_multistage():
    code = """
FROM golang:1.22-alpine AS builder
WORKDIR /src
COPY . .
RUN go build -o server .

FROM alpine:3.19
WORKDIR /app
COPY --from=builder /src/server .
USER appuser
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://localhost:8080/health || exit 1
CMD ["./server"]
"""
    v = DockerfileVerifier()
    reward = v.reward(code)
    assert reward == 1.0


def test_dockerfile_invalid_stage_ref():
    code = """
FROM alpine:3.19
COPY --from=nonexistent_builder /app /app
USER appuser
HEALTHCHECK --interval=30s CMD curl -f http://localhost/ || exit 1
CMD ["./run"]
"""
    v = DockerfileVerifier()
    reward = v.reward(code)
    assert reward < 1.0
    oracle = DockerfileOracle()
    diags = oracle.diagnostics(code)
    assert any(d.code == "DOCKER_INVALID_STAGE_REF" for d in diags)


def test_dockerfile_syntax_error_unknown_instruction():
    code = """
FROM alpine:3.19
GIBBERISH_INSTRUCTION /foo
USER appuser
HEALTHCHECK --interval=30s CMD true
CMD ["sh"]
"""
    v = DockerfileVerifier()
    reward = v.reward(code)
    assert reward < 1.0
    oracle = DockerfileOracle()
    diags = oracle.diagnostics(code)
    assert any(d.code == "DOCKER_SYNTAX_UNKNOWN_INSTRUCTION" for d in diags)


def test_dockerfile_anti_goodhart_latest_tag():
    code = """
FROM python:latest
WORKDIR /app
USER appuser
HEALTHCHECK --interval=30s CMD true
CMD ["python", "main.py"]
"""
    v = DockerfileVerifier()
    assert v.reward(code) == -1.0
    hatches = find_docker_escape_hatches(code)
    assert "latest_tag" in hatches


def test_dockerfile_anti_goodhart_running_as_root():
    # Explicit USER root
    code_root = """
FROM python:3.11-slim
USER root
HEALTHCHECK --interval=30s CMD true
CMD ["python", "main.py"]
"""
    v = DockerfileVerifier()
    assert v.reward(code_root) == -1.0

    # Default root (missing USER directive before CMD)
    code_no_user = """
FROM python:3.11-slim
WORKDIR /app
HEALTHCHECK --interval=30s CMD true
CMD ["python", "main.py"]
"""
    assert v.reward(code_no_user) == -1.0


def test_dockerfile_anti_goodhart_healthcheck_none():
    code = """
FROM alpine:3.19
USER appuser
HEALTHCHECK NONE
CMD ["sleep", "infinity"]
"""
    v = DockerfileVerifier()
    assert v.reward(code) == -1.0
    hatches = find_docker_escape_hatches(code)
    assert "healthcheck_none" in hatches


def test_dockerfile_directive_suppression():
    code = """
# hadolint ignore=DL3006
FROM alpine:3.19
USER appuser
HEALTHCHECK --interval=30s CMD true
CMD ["sh"]
"""
    v = DockerfileVerifier()
    assert v.reward(code) == -1.0


# --------------------------------------------------------------------------- #
# 4. Orchestration (Kubernetes YAML & Helm) Tests
# --------------------------------------------------------------------------- #

def test_kubernetes_clean_deployment():
    code = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
  labels:
    app: api
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: api
        image: ghcr.io/org/api:v1.2.3
        resources:
          requests:
            cpu: "100m"
            memory: "128Mi"
          limits:
            cpu: "500m"
            memory: "256Mi"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8080
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
"""
    v = KubernetesVerifier()
    assert v.reward(code) == 1.0


def test_kubernetes_clean_service():
    code = """
apiVersion: v1
kind: Service
metadata:
  name: api-service
spec:
  ports:
  - port: 80
    targetPort: 8080
  selector:
    app: api
"""
    v = KubernetesVerifier()
    assert v.reward(code) == 1.0


def test_kubernetes_syntax_error():
    code = """
apiVersion: v1
kind: Service
metadata:
  name: [unclosed list
"""
    v = KubernetesVerifier()
    reward = v.reward(code)
    assert reward < 1.0
    oracle = KubernetesOracle()
    diags = oracle.diagnostics(code)
    assert any(d.code == "K8S_SYNTAX_ERROR" for d in diags)


def test_kubernetes_anti_goodhart_unbounded_memory():
    code = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: leaky-app
spec:
  template:
    spec:
      containers:
      - name: web
        image: nginx:1.25
        resources:
          requests:
            cpu: "100m"
            memory: "64Mi"
          limits:
            cpu: "200m"
        livenessProbe:
          httpGet:
            path: /
            port: 80
        readinessProbe:
          httpGet:
            path: /
            port: 80
"""
    v = KubernetesVerifier()
    assert v.reward(code) == -1.0
    hatches = find_kubernetes_escape_hatches(code)
    assert "unbounded_memory" in hatches


def test_kubernetes_anti_goodhart_missing_probes():
    code = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: probeless-app
spec:
  template:
    spec:
      containers:
      - name: web
        image: nginx:1.25
        resources:
          requests:
            cpu: "100m"
            memory: "64Mi"
          limits:
            cpu: "200m"
            memory: "128Mi"
"""
    v = KubernetesVerifier()
    assert v.reward(code) == -1.0
    hatches = find_kubernetes_escape_hatches(code)
    assert "missing_liveness_probe" in hatches or "missing_readiness_probe" in hatches


def test_kubernetes_helm_template_parsing():
    code = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Values.appName }}
spec:
  replicas: {{ .Values.replicaCount }}
  template:
    spec:
      containers:
      - name: app
        image: {{ .Values.image.repository }}:{{ .Values.image.tag }}
        resources:
          requests:
            cpu: "100m"
            memory: "128Mi"
          limits:
            cpu: "200m"
            memory: "256Mi"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8080
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
"""
    oracle = KubernetesOracle()
    diags = oracle.diagnostics(code)
    # Helm delimiters {{ }} must be cleanly handled without syntax crash
    assert not any(d.code == "K8S_SYNTAX_ERROR" for d in diags)


# --------------------------------------------------------------------------- #
# 5. Shell Scripting & POSIX Automation Tests
# --------------------------------------------------------------------------- #

def test_shell_clean_script():
    code = """#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="/tmp/output"
if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
fi

for item in 1 2 3; do
    echo "Item: $item"
done
"""
    v = ShellVerifier()
    assert v.reward(code) == 1.0


def test_shell_syntax_unterminated_quote():
    code = """#!/usr/bin/env bash
set -euo pipefail
echo "missing closing quote
"""
    v = ShellVerifier()
    reward = v.reward(code)
    assert reward < 1.0
    oracle = ShellOracle()
    diags = oracle.diagnostics(code)
    assert any(d.code == "SHELL_SYNTAX_UNTERMINATED_QUOTE" for d in diags)


def test_shell_syntax_unbalanced_if_fi():
    code = """#!/usr/bin/env bash
set -euo pipefail
if [ "$x" = "1" ]; then
    echo "one"
"""
    v = ShellVerifier()
    reward = v.reward(code)
    assert reward < 1.0
    oracle = ShellOracle()
    diags = oracle.diagnostics(code)
    assert any(d.code == "SHELL_SYNTAX_UNBALANCED_IF_FI" for d in diags)


def test_shell_missing_strict_mode():
    code = """#!/usr/bin/env bash
echo "Running without strict mode"
"""
    v = ShellVerifier()
    reward = v.reward(code)
    assert reward < 1.0
    oracle = ShellOracle()
    diags = oracle.diagnostics(code)
    assert any(d.code == "SHELL_MISSING_STRICT_MODE" for d in diags)


def test_shell_anti_goodhart_eval_injection():
    code = """#!/usr/bin/env bash
set -euo pipefail
eval "$USER_COMMAND"
"""
    v = ShellVerifier()
    assert v.reward(code) == -1.0
    hatches = find_shell_escape_hatches(code)
    assert "eval_injection" in hatches


def test_shell_anti_goodhart_unquoted_var_in_cmd():
    code = """#!/usr/bin/env bash
set -euo pipefail
rm -rf $DANGEROUS_DIR
"""
    v = ShellVerifier()
    assert v.reward(code) == -1.0
    hatches = find_shell_escape_hatches(code)
    assert "unquoted_var_in_cmd" in hatches


def test_shell_directive_suppression():
    code = """#!/usr/bin/env bash
# shellcheck disable=SC2086
set -euo pipefail
echo "test"
"""
    v = ShellVerifier()
    assert v.reward(code) == -1.0


# --------------------------------------------------------------------------- #
# 6. Web Core & UI Styling (HTML5, CSS3, Tailwind CSS) Tests
# --------------------------------------------------------------------------- #

def test_html_tailwind_clean_document():
    code = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Clean Dashboard</title>
</head>
<body class="bg-gray-50 text-gray-900">
  <header class="bg-white border-b border-gray-200 p-4">
    <h1 class="text-xl font-bold">Metrics</h1>
  </header>
  <main class="max-w-6xl mx-auto p-6 flex flex-col gap-6">
    <section class="grid grid-cols-3 gap-4">
      <div class="p-4 bg-white rounded shadow">
        <h2 class="text-sm font-medium text-gray-500">Total Users</h2>
        <p class="text-2xl font-bold">12,450</p>
      </div>
    </section>
    <img src="chart.png" alt="User Growth Chart" class="w-full h-auto rounded" />
    <button role="button" class="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded">
      Refresh Data
    </button>
  </main>
  <footer class="text-center text-xs text-gray-400 py-4">
    Dashboard Footer
  </footer>
</body>
</html>
"""
    v = HtmlTailwindVerifier()
    assert v.reward(code) == 1.0


def test_html_missing_img_alt():
    code = """
<main>
  <h1>Product</h1>
  <img src="product.jpg">
</main>
"""
    v = HtmlTailwindVerifier()
    reward = v.reward(code)
    assert reward < 1.0
    oracle = HtmlTailwindOracle()
    diags = oracle.diagnostics(code)
    assert any(d.code == "A11Y_IMG_MISSING_ALT" for d in diags)


def test_html_anti_goodhart_invalid_aria_role():
    code = """
<main>
  <div role="link-button">Click here</div>
</main>
"""
    v = HtmlTailwindVerifier()
    assert v.reward(code) == -1.0
    hatches = find_html_tailwind_escape_hatches(code)
    assert "invalid_aria_role" in hatches


def test_html_anti_goodhart_div_spam():
    # 5 nested divs without semantic containers (main, header, nav, section, article, footer)
    code = """
<div>
  <div>
    <div>
      <div>
        <div>
          <span>Non-semantic soup</span>
        </div>
      </div>
    </div>
  </div>
</div>
"""
    v = HtmlTailwindVerifier()
    assert v.reward(code) == -1.0
    hatches = find_html_tailwind_escape_hatches(code)
    assert "div_spam" in hatches


def test_tailwind_unknown_utility_class():
    code = """
<main class="max-w-4xl">
  <div class="col-super-stretch align-everything">
    <p>Testing unknown utility</p>
  </div>
</main>
"""
    v = HtmlTailwindVerifier()
    reward = v.reward(code)
    assert reward < 1.0
    oracle = HtmlTailwindOracle()
    diags = oracle.diagnostics(code)
    assert any(d.code == "TAILWIND_UNKNOWN_UTILITY" for d in diags)


def test_css_syntax_error_unbalanced_braces():
    code = """
<main>
  <style>
    .test { color: red;
  </style>
</main>
"""
    v = HtmlTailwindVerifier()
    reward = v.reward(code)
    assert reward < 1.0
    oracle = HtmlTailwindOracle()
    diags = oracle.diagnostics(code)
    assert any(d.code == "CSS_SYNTAX_UNBALANCED_BRACES" for d in diags)


# --------------------------------------------------------------------------- #
# 7. Unified CloudInfraVerifier & Auto-Routing Tests
# --------------------------------------------------------------------------- #

def test_unified_cloud_infra_routing():
    v = CloudInfraVerifier()

    # Route Terraform
    tf_code = 'resource "aws_s3_bucket" "b" { bucket = "my-bucket" }'
    assert v.reward(tf_code) == 1.0

    # Route Dockerfile
    docker_code = "FROM alpine:3.19\nUSER appuser\nHEALTHCHECK --interval=30s CMD true\nCMD [\"sh\"]"
    assert v.reward(docker_code) == 1.0

    # Route Kubernetes
    k8s_code = "apiVersion: v1\nkind: Service\nmetadata:\n  name: svc\nspec:\n  ports:\n  - port: 80"
    assert v.reward(k8s_code) == 1.0

    # Route Shell
    shell_code = "#!/bin/bash\nset -euo pipefail\necho 'hello'"
    assert v.reward(shell_code) == 1.0

    # Route HTML/Tailwind
    html_code = "<main class='p-4'><p>hello</p></main>"
    assert v.reward(html_code) == 1.0


# --------------------------------------------------------------------------- #
# 8. Telemetry Reporting (Tracks syntax vs linter vs escape-hatch penalties)
# --------------------------------------------------------------------------- #

def test_telemetry_penalty_tracking():
    v = TerraformVerifier()

    # 1. Clean completion
    clean = 'resource "aws_s3_bucket" "b" { bucket = "my-bucket" }'
    r1 = v.reward(clean)
    assert r1 == 1.0

    # 2. Syntax error completion
    syntax_err = 'resource "aws_s3_bucket" "b" {'
    r2 = v.reward(syntax_err)
    assert r2 < 1.0

    # 3. Linter/semantic error completion
    linter_err = 'resource "aws_instance" "web" { subnet_id = aws_subnet.nonexistent.id }'
    r3 = v.reward(linter_err)
    assert r3 < 1.0

    # 4. Escape hatch completion
    hacked = 'resource "aws_db_instance" "db" { password = "HardcodedSecretPassword123" }'
    r4 = v.reward(hacked)
    assert r4 == -1.0

    telem = v.telemetry()
    assert telem["n_samples"] == 4
    assert telem["n_clean"] == 1
    assert telem["n_hacked"] == 1
    assert telem["n_syntax_errors"] >= 1
    assert telem["n_linter_errors"] >= 1
    assert telem["n_escape_hatches"] >= 1
    assert telem["syntax_penalties"] > 0.0
    assert telem["linter_penalties"] > 0.0
    assert telem["escape_hatch_penalties"] > 0.0

    # Check unified verifier telemetry
    unified = CloudInfraVerifier()
    unified.reward(clean)
    u_telem = unified.telemetry()
    assert u_telem["verifier"] == "cloud_infra_unified"
    assert "sub_verifiers" in u_telem
    assert "terraform" in u_telem["sub_verifiers"]
    assert "docker" in u_telem["sub_verifiers"]


# --------------------------------------------------------------------------- #
# 9. Code Suite Benchmark Evaluator (evaluate_cloud_infra) Tests
# --------------------------------------------------------------------------- #

def test_evaluate_cloud_infra_suite():
    test_cases = [
        {
            "id": "tf-clean",
            "domain": "terraform",
            "code": 'resource "aws_s3_bucket" "b" { bucket = "prod-bucket" }',
            "expected_clean": True,
        },
        {
            "id": "tf-hacked",
            "domain": "terraform",
            "code": 'resource "aws_db" "d" { password = "plain_secret_pwd" }',
            "expected_clean": False,
        },
        {
            "id": "docker-clean",
            "domain": "docker",
            "code": "FROM alpine:3.19\nUSER appuser\nHEALTHCHECK --interval=30s CMD true\nCMD [\"sh\"]",
            "expected_clean": True,
        },
    ]

    res = evaluate_cloud_infra(test_cases)
    assert res["n_cases"] == 3
    assert res["n_passed"] == 3
    assert res["accuracy"] == 1.0
    assert "overall" in res
    assert "by_bucket" in res
    assert "terraform" in res["by_bucket"]
    assert "docker" in res["by_bucket"]


# --------------------------------------------------------------------------- #
# 10. Injectable Seam & Degenerate Output Guards
# --------------------------------------------------------------------------- #

def test_cloud_infra_fake_oracle_injection():
    fake = FakeOracle(raises=True)
    v = TerraformVerifier(oracle=fake, on_error="skip")
    assert v.reward('resource "aws_instance" "a" {}') is None
    assert fake.n_calls == 1


def test_cloud_infra_fail_fast_on_degenerate():
    fake = FakeOracle()
    v = TerraformVerifier(oracle=fake, fail_fast=True)
    # Empty completion short-circuits to -1.0 without calling oracle
    r = v.reward("")
    assert r == -1.0
    assert fake.n_calls == 0
