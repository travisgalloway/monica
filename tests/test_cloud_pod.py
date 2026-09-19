"""Unit tests for RunPod cloud pod management utility and lifecycle tracking (#354).

Covers:
  - Hardware template presets and registry integrity
  - Configuration loading, precedence hierarchy, and SSH key discovery
  - Lifecycle tracking persistence and activity heartbeat recording
  - Limit evaluation (hard budget cap, max runtime, and idle timeout)
  - Mocked RunPod GraphQL API requests and mutations
  - Mocked RunPod REST API requests and error handling
  - Automated lifecycle watcher loop and termination/stop enforcement
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

# Add repo root to path for test imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cloud_pod import (
    DEFAULT_TRACKER_FILE,
    POD_TEMPLATES,
    LifecycleTracker,
    RunPodClient,
    evaluate_limits,
    get_ssh_key,
    load_env,
    main,
    watch_pod,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pod Templates
# ─────────────────────────────────────────────────────────────────────────────


def test_pod_templates_registry() -> None:
    """Verify all formal pod templates exist and have valid hardware specs."""
    expected_templates = {"rtx4090", "a40", "a100-pcie", "a100-sxm4", "h100-sxm"}
    assert set(POD_TEMPLATES.keys()) == expected_templates

    for key, tmpl in POD_TEMPLATES.items():
        assert tmpl["name"] == key
        assert tmpl["gpu"]
        assert "runpod/pytorch:" in tmpl["image"]
        assert tmpl["cuda_version"] == "12.4.1"
        assert tmpl["disk_gb"] >= 50
        assert tmpl["volume_gb"] == 0  # Standalone pods default to volume 0
        assert tmpl["cloud_type"] in ("ALL", "COMMUNITY", "SECURE")
        assert tmpl["approx_hourly_cost"] > 0.0
        assert tmpl["description"]


def test_templates_cli_output(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify templates subcommand prints formatted table and valid JSON."""
    with mock.patch("sys.argv", ["cloud_pod.py", "templates"]):
        main()
    captured = capsys.readouterr()
    assert "Template" in captured.out
    assert "rtx4090" in captured.out
    assert "a100-sxm4" in captured.out

    with mock.patch("sys.argv", ["cloud_pod.py", "templates", "--json"]):
        main()
    captured_json = capsys.readouterr()
    data = json.loads(captured_json.out)
    assert "rtx4090" in data
    assert data["rtx4090"]["approx_hourly_cost"] == 0.44


# ─────────────────────────────────────────────────────────────────────────────
# 2. Configuration & Environment Precedence
# ─────────────────────────────────────────────────────────────────────────────


def test_load_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify load_env reads .env and respects existing os.environ."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "RUNPOD_API_KEY=test_key_env\nRUNPOD_MAX_BUDGET_USD=15.0\n# comment\n",
        encoding="utf-8",
    )

    with mock.patch("scripts.cloud_pod.REPO_ROOT", tmp_path):
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        monkeypatch.delenv("RUNPOD_MAX_BUDGET_USD", raising=False)
        env = load_env()
        assert env["RUNPOD_API_KEY"] == "test_key_env"
        assert env["RUNPOD_MAX_BUDGET_USD"] == "15.0"


def test_get_ssh_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_ssh_key honors RUNPOD_SSH_KEY override."""
    monkeypatch.setenv("RUNPOD_SSH_KEY", "ssh-ed25519 AAAAB3NzaC1yc2E test-key")
    assert get_ssh_key() == "ssh-ed25519 AAAAB3NzaC1yc2E test-key"


def test_get_ssh_key_from_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify get_ssh_key reads from ~/.ssh when env is unset."""
    monkeypatch.delenv("RUNPOD_SSH_KEY", raising=False)
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    key_file = ssh_dir / "id_ed25519.pub"
    key_file.write_text("ssh-ed25519 AAAAC3NzaC1yc2E file-key\n", encoding="utf-8")

    with mock.patch("pathlib.Path.home", return_value=tmp_path):
        assert get_ssh_key() == "ssh-ed25519 AAAAC3NzaC1yc2E file-key"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Lifecycle Tracker & Heartbeat
# ─────────────────────────────────────────────────────────────────────────────


def test_lifecycle_tracker_persistence(tmp_path: Path) -> None:
    """Verify tracker records, persists, and reloads pod metadata."""
    tracker_file = tmp_path / "tracker.json"
    tracker = LifecycleTracker(tracker_file)

    tracker.track_pod(
        pod_id="pod-xyz",
        name="test-poc",
        gpu="NVIDIA A40",
        cost_per_hr=0.40,
        max_runtime_seconds=7200.0,
        max_budget_usd=10.0,
        idle_timeout_seconds=1800.0,
        auto_action="terminate",
    )

    # Reload in a new instance
    reloaded = LifecycleTracker(tracker_file)
    rec = reloaded.get_pod("pod-xyz")
    assert rec is not None
    assert rec["name"] == "test-poc"
    assert rec["cost_per_hr"] == 0.40
    assert rec["max_budget_usd"] == 10.0
    assert rec["auto_action"] == "terminate"
    assert rec["status"] == "RUNNING"


def test_lifecycle_tracker_heartbeat(tmp_path: Path) -> None:
    """Verify recording heartbeat updates last_heartbeat timestamp."""
    tracker_file = tmp_path / "tracker.json"
    tracker = LifecycleTracker(tracker_file)

    tracker.track_pod(pod_id="pod-123")
    old_hb = tracker.get_pod("pod-123")["last_heartbeat"]

    new_time = old_hb + 500.0
    tracker.record_heartbeat("pod-123", timestamp=new_time)

    rec = tracker.get_pod("pod-123")
    assert rec["last_heartbeat"] == new_time


def test_heartbeat_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify heartbeat CLI command touches the tracker without requiring API credentials."""
    tracker_file = tmp_path / "tracker.json"
    with mock.patch("scripts.cloud_pod.DEFAULT_TRACKER_FILE", tracker_file), mock.patch(
        "sys.argv", ["cloud_pod.py", "heartbeat", "pod-abc", "--timestamp", "1700000000.0"]
    ):
        main()

    captured = capsys.readouterr()
    assert "Recorded activity heartbeat for pod pod-abc." in captured.out

    tracker = LifecycleTracker(tracker_file)
    rec = tracker.get_pod("pod-abc")
    assert rec is not None
    assert rec["last_heartbeat"] == 1700000000.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Limit Evaluation (Budget, Max Runtime, Idle Timeout)
# ─────────────────────────────────────────────────────────────────────────────


def test_evaluate_limits_healthy() -> None:
    """Verify healthy pod within limits triggers no breach."""
    record = {
        "pod_id": "pod-1",
        "started_at": 1000.0,
        "last_heartbeat": 1000.0,
        "cost_per_hr": 1.0,
        "max_budget_usd": 10.0,
        "max_runtime_seconds": 3600.0,
        "idle_timeout_seconds": 600.0,
    }
    # Uptime: 1800s (0.5h), spend: $0.50, idle: 300s
    breached, reason, details = evaluate_limits(
        record=record,
        uptime_seconds=1800.0,
        cost_per_hr=1.0,
        now=1300.0,
    )
    assert not breached
    assert reason is None
    assert details is None


def test_evaluate_limits_hard_budget_exceeded() -> None:
    """Verify hard budget stop triggers when estimated spend meets or exceeds budget cap."""
    record = {
        "pod_id": "pod-1",
        "started_at": 1000.0,
        "last_heartbeat": 1000.0,
        "cost_per_hr": 2.0,
        "max_budget_usd": 5.0,
        "max_runtime_seconds": 36000.0,
        "idle_timeout_seconds": 3600.0,
    }
    # Uptime: 10800s (3h) @ $2.00/hr = $6.00 spend (budget: $5.00)
    breached, reason, details = evaluate_limits(
        record=record,
        uptime_seconds=10800.0,
        cost_per_hr=2.0,
        now=1000.0,
    )
    assert breached
    assert reason == "BUDGET_EXCEEDED"
    assert "Estimated spend $6.00 reached or exceeded budget limit of $5.00" in (details or "")


def test_evaluate_limits_max_runtime_exceeded() -> None:
    """Verify max runtime limit triggers when uptime meets or exceeds limit."""
    record = {
        "pod_id": "pod-1",
        "started_at": 1000.0,
        "last_heartbeat": 1000.0,
        "cost_per_hr": 0.10,
        "max_budget_usd": 100.0,
        "max_runtime_seconds": 3600.0,  # 1 hour
        "idle_timeout_seconds": 1800.0,
    }
    # Uptime: 3601s, spend: $0.10, now: 1100s (idle: 100s)
    breached, reason, details = evaluate_limits(
        record=record,
        uptime_seconds=3601.0,
        cost_per_hr=0.10,
        now=1100.0,
    )
    assert breached
    assert reason == "MAX_RUNTIME_EXCEEDED"
    assert "Total runtime 3601.0s reached or exceeded max runtime limit of 3600.0s" in (details or "")


def test_evaluate_limits_idle_timeout_exceeded() -> None:
    """Verify idle timeout triggers when elapsed time since last heartbeat exceeds threshold."""
    record = {
        "pod_id": "pod-1",
        "started_at": 1000.0,
        "last_heartbeat": 1000.0,
        "cost_per_hr": 1.0,
        "max_budget_usd": 100.0,
        "max_runtime_seconds": 36000.0,
        "idle_timeout_seconds": 600.0,  # 10 minutes
    }
    # Now: 1700s (idle for 700s > 600s threshold), uptime: 700s, spend: $0.19
    breached, reason, details = evaluate_limits(
        record=record,
        uptime_seconds=700.0,
        cost_per_hr=1.0,
        now=1700.0,
    )
    assert breached
    assert reason == "IDLE_TIMEOUT"
    assert "No activity heartbeat received for 700.0s (idle timeout threshold: 600.0s)" in (details or "")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Mocked RunPod GraphQL API
# ─────────────────────────────────────────────────────────────────────────────


def test_graphql_get_pod() -> None:
    """Verify get_pod parses RunPod GraphQL response correctly."""
    mock_pod_data = {
        "id": "pod-12345",
        "name": "monica-cuda-poc",
        "desiredStatus": "RUNNING",
        "costPerHr": 0.44,
        "uptimeSeconds": 3600,
        "machine": {"gpuDisplayName": "NVIDIA GeForce RTX 4090", "costPerHr": 0.44},
        "runtime": {
            "uptimeInSeconds": 3600,
            "ports": [{"ip": "198.51.100.1", "privatePort": 22, "publicPort": 12345}],
        },
    }

    def mock_transport(mode: str, query: str = "") -> dict:
        assert mode == "graphql"
        assert 'pod(input: {podId: "pod-12345"})' in query
        return {"data": {"pod": mock_pod_data}}

    client = RunPodClient(api_key="mock_key", api_mode="graphql", transport=mock_transport)
    pod = client.get_pod("pod-12345")
    assert pod is not None
    assert pod["id"] == "pod-12345"
    assert pod["desiredStatus"] == "RUNNING"
    assert pod["machine"]["gpuDisplayName"] == "NVIDIA GeForce RTX 4090"


def test_graphql_get_pods() -> None:
    """Verify get_pods parses list of pods from RunPod GraphQL myself query."""
    mock_pods_list = [
        {"id": "pod-1", "name": "pod-one", "desiredStatus": "RUNNING"},
        {"id": "pod-2", "name": "pod-two", "desiredStatus": "STOPPED"},
    ]

    def mock_transport(mode: str, query: str = "") -> dict:
        assert "myself" in query
        return {"data": {"myself": {"pods": mock_pods_list}}}

    client = RunPodClient(api_key="mock_key", api_mode="graphql", transport=mock_transport)
    pods = client.get_pods()
    assert len(pods) == 2
    assert pods[0]["id"] == "pod-1"


def test_graphql_create_pod() -> None:
    """Verify create_pod issues deployment mutation with sanitized parameters."""
    def mock_transport(mode: str, query: str = "") -> dict:
        assert "podFindAndDeployOnDemand" in query
        assert "NVIDIA GeForce RTX 4090" in query
        return {"data": {"podFindAndDeployOnDemand": {"id": "pod-new", "name": "poc-run", "costPerHr": 0.44}}}

    client = RunPodClient(api_key="mock_key", api_mode="graphql", transport=mock_transport)
    res = client.create_pod(
        name="poc-run",
        image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        gpu_type_id="NVIDIA GeForce RTX 4090",
        volume_in_gb=0,
        container_disk_in_gb=50,
    )
    assert res["id"] == "pod-new"
    assert res["costPerHr"] == 0.44


def test_graphql_stop_and_terminate() -> None:
    """Verify stop_pod and terminate_pod issue respective GraphQL mutations."""
    stopped = False
    terminated = False

    def mock_transport(mode: str, query: str = "") -> dict:
        nonlocal stopped, terminated
        if "podStop" in query:
            stopped = True
            return {"data": {"podStop": {"id": "pod-1", "desiredStatus": "STOPPED"}}}
        if "podTerminate" in query:
            terminated = True
            return {"data": {"podTerminate": None}}
        raise ValueError(f"Unexpected query: {query}")

    client = RunPodClient(api_key="mock_key", api_mode="graphql", transport=mock_transport)
    stop_res = client.stop_pod("pod-1")
    assert stopped
    assert stop_res["desiredStatus"] == "STOPPED"

    term_res = client.terminate_pod("pod-1")
    assert terminated
    assert term_res["status"] == "TERMINATED"


def test_graphql_error_handling() -> None:
    """Verify GraphQL error payload raises RuntimeError with clear message."""
    def mock_transport(mode: str, query: str = "") -> dict:
        return {"errors": [{"message": "Pod not found or access denied."}]}

    client = RunPodClient(api_key="mock_key", api_mode="graphql", transport=mock_transport)
    with pytest.raises(RuntimeError, match="RunPod GraphQL error: Pod not found or access denied."):
        client.get_pod("nonexistent-pod")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Mocked RunPod REST API
# ─────────────────────────────────────────────────────────────────────────────


def test_rest_api_lifecycle() -> None:
    """Verify REST API calls get, stop, and terminate endpoints properly."""
    endpoints_called: list[tuple[str, str]] = []

    def mock_transport(mode: str, path: str = "", method: str = "GET", body: dict | None = None) -> dict:
        assert mode == "rest"
        endpoints_called.append((method, path))
        if method == "GET" and path == "v1/pods/pod-rest-1":
            return {"id": "pod-rest-1", "desiredStatus": "RUNNING", "costPerHr": 0.40}
        if method == "POST" and path == "v1/pods/pod-rest-1/stop":
            return {"id": "pod-rest-1", "status": "STOPPED"}
        if method == "DELETE" and path == "v1/pods/pod-rest-1":
            return {"id": "pod-rest-1", "status": "TERMINATED"}
        return {}

    client = RunPodClient(api_key="mock_key", api_mode="rest", transport=mock_transport)
    pod = client.get_pod("pod-rest-1")
    assert pod["id"] == "pod-rest-1"

    stop_res = client.stop_pod("pod-rest-1")
    assert stop_res["status"] == "STOPPED"

    term_res = client.terminate_pod("pod-rest-1")
    assert term_res["status"] == "TERMINATED"

    assert ("GET", "v1/pods/pod-rest-1") in endpoints_called
    assert ("POST", "v1/pods/pod-rest-1/stop") in endpoints_called
    assert ("DELETE", "v1/pods/pod-rest-1") in endpoints_called


# ─────────────────────────────────────────────────────────────────────────────
# 7. Automated Lifecycle Watcher & Limit Enforcement
# ─────────────────────────────────────────────────────────────────────────────


def test_watch_pod_healthy_once(tmp_path: Path) -> None:
    """Verify watch_pod returns 0 when pod is within all limits."""
    tracker_file = tmp_path / "tracker.json"
    tracker = LifecycleTracker(tracker_file)
    tracker.track_pod(
        pod_id="pod-healthy",
        cost_per_hr=0.40,
        max_budget_usd=10.0,
        max_runtime_seconds=7200.0,
        idle_timeout_seconds=1800.0,
    )

    def mock_transport(mode: str, query: str = "") -> dict:
        return {
            "data": {
                "pod": {
                    "id": "pod-healthy",
                    "desiredStatus": "RUNNING",
                    "costPerHr": 0.40,
                    "runtime": {"uptimeInSeconds": 600},
                }
            }
        }

    client = RunPodClient(api_key="mock_key", api_mode="graphql", transport=mock_transport)
    code = watch_pod(
        client=client,
        tracker=tracker,
        pod_id="pod-healthy",
        once=True,
    )
    assert code == 0


def test_watch_pod_terminates_on_budget_exceeded(tmp_path: Path) -> None:
    """Verify watch_pod executes pod termination when hard budget limit is exceeded."""
    tracker_file = tmp_path / "tracker.json"
    tracker = LifecycleTracker(tracker_file)
    tracker.track_pod(
        pod_id="pod-budget-breach",
        cost_per_hr=2.00,
        max_budget_usd=5.00,
        auto_action="terminate",
    )

    terminated = False

    def mock_transport(mode: str, query: str = "") -> dict:
        nonlocal terminated
        if "pod(" in query:
            return {
                "data": {
                    "pod": {
                        "id": "pod-budget-breach",
                        "desiredStatus": "RUNNING",
                        "costPerHr": 2.00,
                        "runtime": {"uptimeInSeconds": 10800},  # 3 hrs @ $2 = $6 > $5
                    }
                }
            }
        if "podTerminate" in query:
            terminated = True
            return {"data": {"podTerminate": None}}
        return {}

    client = RunPodClient(api_key="mock_key", api_mode="graphql", transport=mock_transport)
    code = watch_pod(
        client=client,
        tracker=tracker,
        pod_id="pod-budget-breach",
        once=True,
    )
    assert code == 2
    assert terminated

    rec = tracker.get_pod("pod-budget-breach")
    assert rec["status"] == "BUDGET_EXCEEDED"
    assert rec["action_taken"] == "terminate"


def test_watch_pod_stops_on_idle_timeout(tmp_path: Path) -> None:
    """Verify watch_pod stops pod when activity heartbeat times out."""
    tracker_file = tmp_path / "tracker.json"
    tracker = LifecycleTracker(tracker_file)
    # Set last heartbeat in the past
    past_time = time.time() - 3600.0  # 1 hr ago
    tracker.track_pod(
        pod_id="pod-idle-breach",
        cost_per_hr=0.40,
        max_budget_usd=50.00,
        idle_timeout_seconds=600.0,  # 10 min threshold
        auto_action="stop",
    )
    tracker.records["pod-idle-breach"]["last_heartbeat"] = past_time
    tracker.save()

    stopped = False

    def mock_transport(mode: str, query: str = "") -> dict:
        nonlocal stopped
        if "pod(" in query:
            return {
                "data": {
                    "pod": {
                        "id": "pod-idle-breach",
                        "desiredStatus": "RUNNING",
                        "costPerHr": 0.40,
                        "runtime": {"uptimeInSeconds": 3600},
                    }
                }
            }
        if "podStop" in query:
            stopped = True
            return {"data": {"podStop": {"id": "pod-idle-breach", "desiredStatus": "STOPPED"}}}
        return {}

    client = RunPodClient(api_key="mock_key", api_mode="graphql", transport=mock_transport)
    code = watch_pod(
        client=client,
        tracker=tracker,
        pod_id="pod-idle-breach",
        once=True,
    )
    assert code == 2
    assert stopped

    rec = tracker.get_pod("pod-idle-breach")
    assert rec["status"] == "IDLE_TIMEOUT"
    assert rec["action_taken"] == "stop"


def test_watch_pod_dry_run(tmp_path: Path) -> None:
    """Verify dry_run records breach without calling stop or terminate."""
    tracker_file = tmp_path / "tracker.json"
    tracker = LifecycleTracker(tracker_file)
    tracker.track_pod(
        pod_id="pod-dry",
        cost_per_hr=5.00,
        max_budget_usd=1.00,
        auto_action="terminate",
    )

    terminated = False

    def mock_transport(mode: str, query: str = "") -> dict:
        nonlocal terminated
        if "pod(" in query:
            return {
                "data": {
                    "pod": {
                        "id": "pod-dry",
                        "desiredStatus": "RUNNING",
                        "costPerHr": 5.00,
                        "runtime": {"uptimeInSeconds": 3600},
                    }
                }
            }
        if "podTerminate" in query:
            terminated = True
            return {"data": {"podTerminate": None}}
        return {}

    client = RunPodClient(api_key="mock_key", api_mode="graphql", transport=mock_transport)
    code = watch_pod(
        client=client,
        tracker=tracker,
        pod_id="pod-dry",
        once=True,
        dry_run=True,
    )
    assert code == 2
    assert not terminated  # dry_run prevented actual termination
    rec = tracker.get_pod("pod-dry")
    assert rec["status"] == "DRY_RUN_BUDGET_EXCEEDED"


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI Subcommand Invocations via main()
# ─────────────────────────────────────────────────────────────────────────────


def test_main_missing_api_key(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify commands requiring API key exit gracefully when RUNPOD_API_KEY is missing."""
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with mock.patch("scripts.cloud_pod.load_env", return_value={}):
        with pytest.raises(SystemExit) as exc_info:
            with mock.patch("sys.argv", ["cloud_pod.py", "list"]):
                main()
        assert "RUNPOD_API_KEY not found" in str(exc_info.value)


def test_main_list_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify list subcommand invokes client and formats active pod list."""
    mock_pods = [
        {
            "id": "pod-1",
            "name": "monica-run",
            "desiredStatus": "RUNNING",
            "machine": {"gpuDisplayName": "NVIDIA A40"},
        }
    ]
    with mock.patch("scripts.cloud_pod.load_env", return_value={"RUNPOD_API_KEY": "dummy_key"}), mock.patch.object(
        RunPodClient, "get_pods", return_value=mock_pods
    ):
        with mock.patch("sys.argv", ["cloud_pod.py", "list"]):
            main()
        captured = capsys.readouterr()
        assert "Active pods (1):" in captured.out
        assert "pod-1" in captured.out
        assert "monica-run" in captured.out


def test_main_status_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify status subcommand prints pod status and tracker metrics."""
    mock_pod = {
        "id": "pod-123",
        "name": "monica-test",
        "desiredStatus": "RUNNING",
        "costPerHr": 0.44,
        "runtime": {"uptimeInSeconds": 3600},
        "machine": {"gpuDisplayName": "NVIDIA GeForce RTX 4090"},
    }
    with mock.patch("scripts.cloud_pod.load_env", return_value={"RUNPOD_API_KEY": "dummy_key"}), mock.patch.object(
        RunPodClient, "get_pod", return_value=mock_pod
    ):
        with mock.patch("sys.argv", ["cloud_pod.py", "status", "pod-123"]):
            main()
        captured = capsys.readouterr()
        assert "=== RunPod Instance Status ===" in captured.out
        assert "pod-123" in captured.out
        assert "NVIDIA GeForce RTX 4090" in captured.out
        assert "$0.44" in captured.out


def test_main_stop_and_terminate_subcommands(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Verify stop and terminate CLI subcommands call client and update tracker."""
    tracker_file = tmp_path / "tracker.json"
    with mock.patch("scripts.cloud_pod.DEFAULT_TRACKER_FILE", tracker_file), mock.patch(
        "scripts.cloud_pod.load_env", return_value={"RUNPOD_API_KEY": "dummy_key"}
    ), mock.patch.object(RunPodClient, "stop_pod", return_value={"status": "STOPPED"}), mock.patch.object(
        RunPodClient, "terminate_pod", return_value={"status": "TERMINATED"}
    ):
        with mock.patch("sys.argv", ["cloud_pod.py", "stop", "pod-target"]):
            main()
        out_stop = capsys.readouterr().out
        assert "Stopping pod pod-target..." in out_stop

        tracker = LifecycleTracker(tracker_file)
        assert tracker.get_pod("pod-target")["status"] == "STOPPED"

        with mock.patch("sys.argv", ["cloud_pod.py", "terminate", "pod-target"]):
            main()
        out_term = capsys.readouterr().out
        assert "Terminating pod pod-target..." in out_term
        assert tracker.get_pod("pod-target")["status"] == "TERMINATED"


def test_main_launch_subcommand(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Verify launch command resolves template, calls client, and registers tracker."""
    tracker_file = tmp_path / "tracker.json"
    mock_new_pod = {
        "id": "pod-new-777",
        "name": "custom-name",
        "desiredStatus": "RUNNING",
        "costPerHr": 0.40,
    }
    with mock.patch("scripts.cloud_pod.DEFAULT_TRACKER_FILE", tracker_file), mock.patch(
        "scripts.cloud_pod.load_env", return_value={"RUNPOD_API_KEY": "dummy_key"}
    ), mock.patch.object(RunPodClient, "create_pod", return_value=mock_new_pod):
        with mock.patch(
            "sys.argv",
            [
                "cloud_pod.py",
                "launch",
                "--template",
                "a40",
                "--name",
                "custom-name",
                "--max-budget",
                "12.50",
                "--no-wait",
            ],
        ):
            main()

        captured = capsys.readouterr()
        assert "Pod created! ID: pod-new-777" in captured.out
        assert "Hard budget cap: $12.50" in captured.out

        tracker = LifecycleTracker(tracker_file)
        rec = tracker.get_pod("pod-new-777")
        assert rec is not None
        assert rec["name"] == "custom-name"
        assert rec["gpu"] == "NVIDIA A40"
        assert rec["max_budget_usd"] == 12.50
