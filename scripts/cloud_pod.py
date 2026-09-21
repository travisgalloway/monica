#!/usr/bin/env python3
"""CLI utility to manage on-demand RunPod CUDA instances for Monica (#198, #354).

Supports:
    python scripts/cloud_pod.py launch --template rtx4090 --name monica-cuda-poc --max-budget 10.0 --idle-timeout 30
    python scripts/cloud_pod.py list
    python scripts/cloud_pod.py status <pod_id>
    python scripts/cloud_pod.py heartbeat <pod_id>
    python scripts/cloud_pod.py watch <pod_id> [--max-budget 10.0] [--idle-timeout 30] [--auto-action terminate]
    python scripts/cloud_pod.py stop <pod_id>
    python scripts/cloud_pod.py terminate <pod_id>
    python scripts/cloud_pod.py templates
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKER_FILE = REPO_ROOT / "runs" / "cloud_pod_tracker.json"

POD_TEMPLATES: dict[str, dict[str, Any]] = {
    "rtx4090": {
        "name": "rtx4090",
        "gpu": "NVIDIA GeForce RTX 4090",
        "image": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "cuda_version": "12.4.1",
        "disk_gb": 50,
        "volume_gb": 0,
        "cloud_type": "ALL",
        "approx_hourly_cost": 0.44,
        "description": "Single-GPU consumer tier for small POC training and fast iteration (24GB VRAM)",
    },
    "a40": {
        "name": "a40",
        "gpu": "NVIDIA A40",
        "image": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "cuda_version": "12.4.1",
        "disk_gb": 50,
        "volume_gb": 0,
        "cloud_type": "COMMUNITY",
        "approx_hourly_cost": 0.40,
        "description": "Single-GPU datacenter tier for Tier 1 POC training with 48GB VRAM headroom",
    },
    "a100-pcie": {
        "name": "a100-pcie",
        "gpu": "NVIDIA A100-PCIE-80GB",
        "image": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "cuda_version": "12.4.1",
        "disk_gb": 100,
        "volume_gb": 0,
        "cloud_type": "SECURE",
        "approx_hourly_cost": 1.89,
        "description": "High-memory single-GPU datacenter tier for 1B dense training (80GB VRAM)",
    },
    "a100-sxm4": {
        "name": "a100-sxm4",
        "gpu": "NVIDIA A100-SXM4-80GB",
        "image": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "cuda_version": "12.4.1",
        "disk_gb": 100,
        "volume_gb": 0,
        "cloud_type": "SECURE",
        "approx_hourly_cost": 2.49,
        "description": "SXM4 NVLink multi-GPU candidate for FSDP2 + Expert Parallelism",
    },
    "h100-sxm": {
        "name": "h100-sxm",
        "gpu": "NVIDIA H100 80GB HBM3",
        "image": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "cuda_version": "12.4.1",
        "disk_gb": 100,
        "volume_gb": 0,
        "cloud_type": "SECURE",
        "approx_hourly_cost": 3.89,
        "description": "Hopper architecture tier for FP8 expert GEMM verification (#240)",
    },
}


def load_env() -> dict[str, str]:
    """Load configuration from .env file and environment variables."""
    env: dict[str, str] = {}
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                clean_k = k.strip()
                clean_v = v.strip("'\"")
                env[clean_k] = clean_v
                if clean_k not in os.environ:
                    os.environ[clean_k] = clean_v
    for k, v in os.environ.items():
        env[k] = v
    return env


def get_ssh_key() -> str:
    """Retrieve public SSH key from environment or ~/.ssh standard paths."""
    env_key = os.environ.get("RUNPOD_SSH_KEY")
    if env_key:
        return env_key.strip()
    key_path = Path.home() / ".ssh" / "id_ed25519.pub"
    if key_path.exists():
        return key_path.read_text(encoding="utf-8").strip()
    rsa_path = Path.home() / ".ssh" / "id_rsa.pub"
    if rsa_path.exists():
        return rsa_path.read_text(encoding="utf-8").strip()
    return ""


class LifecycleTracker:
    """Tracks pod lifecycle, heartbeat activity, and cumulative expenditure."""

    def __init__(self, tracker_file: Path | str | None = None) -> None:
        if tracker_file is None:
            env_path = os.environ.get("RUNPOD_TRACKER_FILE")
            self.tracker_file = Path(env_path) if env_path else DEFAULT_TRACKER_FILE
        else:
            self.tracker_file = Path(tracker_file)
        self.records: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        if self.tracker_file.exists():
            try:
                data = json.loads(self.tracker_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                return {}
        return {}

    def save(self) -> None:
        self.tracker_file.parent.mkdir(parents=True, exist_ok=True)
        self.tracker_file.write_text(json.dumps(self.records, indent=2), encoding="utf-8")

    def track_pod(
        self,
        pod_id: str,
        name: str = "",
        gpu: str = "",
        cost_per_hr: float = 0.0,
        max_runtime_seconds: float = 0.0,
        max_budget_usd: float = 0.0,
        idle_timeout_seconds: float = 0.0,
        auto_action: str = "terminate",
    ) -> dict[str, Any]:
        now = time.time()
        record = self.records.get(pod_id, {})
        record.update(
            {
                "pod_id": pod_id,
                "name": name or record.get("name", ""),
                "gpu": gpu or record.get("gpu", ""),
                "cost_per_hr": cost_per_hr if cost_per_hr > 0 else record.get("cost_per_hr", 0.0),
                "created_at": record.get("created_at", now),
                "started_at": record.get("started_at", now),
                "last_heartbeat": record.get("last_heartbeat", now),
                "max_runtime_seconds": (
                    max_runtime_seconds if max_runtime_seconds > 0 else record.get("max_runtime_seconds", 0.0)
                ),
                "max_budget_usd": max_budget_usd if max_budget_usd > 0 else record.get("max_budget_usd", 0.0),
                "idle_timeout_seconds": (
                    idle_timeout_seconds if idle_timeout_seconds > 0 else record.get("idle_timeout_seconds", 0.0)
                ),
                "auto_action": auto_action or record.get("auto_action", "terminate"),
                "status": record.get("status", "RUNNING"),
                "action_taken": record.get("action_taken", None),
                "action_timestamp": record.get("action_timestamp", None),
                "total_spend_usd": record.get("total_spend_usd", 0.0),
                "total_runtime_seconds": record.get("total_runtime_seconds", 0.0),
            }
        )
        self.records[pod_id] = record
        self.save()
        return record

    def record_heartbeat(self, pod_id: str, timestamp: float | None = None) -> bool:
        now = timestamp if timestamp is not None else time.time()
        if pod_id not in self.records:
            self.track_pod(pod_id)
        self.records[pod_id]["last_heartbeat"] = now
        self.save()
        return True

    def get_pod(self, pod_id: str) -> dict[str, Any] | None:
        self.records = self._load()
        return self.records.get(pod_id)

    def update_status(
        self,
        pod_id: str,
        status: str,
        action_taken: str | None = None,
        spend: float | None = None,
        runtime: float | None = None,
    ) -> None:
        if pod_id not in self.records:
            self.track_pod(pod_id)
        rec = self.records[pod_id]
        rec["status"] = status
        if action_taken is not None:
            rec["action_taken"] = action_taken
            rec["action_timestamp"] = time.time()
        if spend is not None:
            rec["total_spend_usd"] = spend
        if runtime is not None:
            rec["total_runtime_seconds"] = runtime
        self.save()


def evaluate_limits(
    record: dict[str, Any],
    uptime_seconds: float,
    cost_per_hr: float,
    now: float | None = None,
) -> tuple[bool, str | None, str | None]:
    """Evaluate whether runtime, budget, or idle thresholds have been breached.

    Returns:
        (is_breached, breach_reason, breach_details)
        Breach reasons: 'BUDGET_EXCEEDED', 'MAX_RUNTIME_EXCEEDED', 'IDLE_TIMEOUT'.
    """
    now = now if now is not None else time.time()
    max_budget = float(record.get("max_budget_usd") or 0.0)
    max_runtime = float(record.get("max_runtime_seconds") or 0.0)
    idle_timeout = float(record.get("idle_timeout_seconds") or 0.0)

    # 1. Hard Budget Limit
    effective_rate = cost_per_hr if cost_per_hr > 0 else float(record.get("cost_per_hr") or 0.0)
    estimated_spend = (max(0.0, uptime_seconds) / 3600.0) * effective_rate
    if max_budget > 0.0 and estimated_spend >= max_budget:
        return (
            True,
            "BUDGET_EXCEEDED",
            f"Estimated spend ${estimated_spend:.2f} reached or exceeded budget limit of ${max_budget:.2f} "
            f"(rate: ${effective_rate:.2f}/hr, uptime: {uptime_seconds:.0f}s)",
        )

    # 2. Max Runtime Limit
    if max_runtime > 0.0 and uptime_seconds >= max_runtime:
        return (
            True,
            "MAX_RUNTIME_EXCEEDED",
            f"Total runtime {uptime_seconds:.1f}s reached or exceeded max runtime limit of {max_runtime:.1f}s "
            f"({max_runtime / 3600.0:.2f}h)",
        )

    # 3. Idle Timeout Limit
    last_heartbeat = float(record.get("last_heartbeat") or record.get("started_at") or now)
    idle_duration = max(0.0, now - last_heartbeat)
    if idle_timeout > 0.0 and idle_duration >= idle_timeout:
        return (
            True,
            "IDLE_TIMEOUT",
            f"No activity heartbeat received for {idle_duration:.1f}s (idle timeout threshold: {idle_timeout:.1f}s)",
        )

    return False, None, None


class RunPodClient:
    """Client for RunPod GraphQL and REST APIs."""

    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.runpod.io",
        api_mode: str = "graphql",
        transport: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.api_mode = api_mode  # "graphql", "rest", or "sdk"
        self.transport = transport

    def execute_graphql(self, query: str) -> dict[str, Any]:
        if self.transport is not None:
            res = self.transport("graphql", query=query)
        else:
            url = f"{self.api_url}/graphql"
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "monica-cloud-pod/1.0",
                "Authorization": f"Bearer {self.api_key}",
            }
            data = json.dumps({"query": query}).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"GraphQL request failed with HTTP {exc.code}: {body}") from exc
            except Exception as exc:
                raise RuntimeError(f"GraphQL request failed: {exc}") from exc

        if "errors" in res and res["errors"]:
            msg = res["errors"][0].get("message", "Unknown GraphQL error")
            raise RuntimeError(f"RunPod GraphQL error: {msg}")
        return res

    def execute_rest(
        self,
        path: str,
        method: str = "GET",
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.transport is not None:
            return self.transport("rest", path=path, method=method, body=body)
        url = f"{self.api_url}/{path.lstrip('/')}"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "monica-cloud-pod/1.0",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=payload, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"REST request {method} {url} failed with HTTP {exc.code}: {err_body}") from exc
        except Exception as exc:
            raise RuntimeError(f"REST request {method} {url} failed: {exc}") from exc

    def get_pod(self, pod_id: str) -> dict[str, Any] | None:
        if self.api_mode == "sdk":
            try:
                import runpod

                runpod.api_key = self.api_key
                return runpod.get_pod(pod_id)
            except ImportError:
                pass

        if self.api_mode == "rest":
            try:
                res = self.execute_rest(f"v1/pods/{pod_id}", method="GET")
                return res.get("pod", res)
            except Exception as e:
                if "404" in str(e):
                    return None
                raise

        query = f"""
        query pod {{
            pod(input: {{podId: "{pod_id}"}}) {{
                id
                name
                desiredStatus
                costPerHr
                uptimeSeconds
                containerDiskInGb
                volumeInGb
                imageName
                machine {{
                    gpuDisplayName
                    costPerHr
                }}
                runtime {{
                    uptimeInSeconds
                    ports {{
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                        type
                    }}
                }}
            }}
        }}
        """
        res = self.execute_graphql(query)
        return res.get("data", {}).get("pod")

    def get_pods(self) -> list[dict[str, Any]]:
        if self.api_mode == "sdk":
            try:
                import runpod

                runpod.api_key = self.api_key
                return runpod.get_pods()
            except ImportError:
                pass

        if self.api_mode == "rest":
            res = self.execute_rest("v1/pods", method="GET")
            if isinstance(res, list):
                return res
            return res.get("pods", [])

        query = """
        query {
            myself {
                pods {
                    id
                    name
                    desiredStatus
                    costPerHr
                    uptimeSeconds
                    machine {
                        gpuDisplayName
                        costPerHr
                    }
                    runtime {
                        uptimeInSeconds
                        ports {
                            ip
                            isIpPublic
                            privatePort
                            publicPort
                            type
                        }
                    }
                }
            }
        }
        """
        res = self.execute_graphql(query)
        myself = res.get("data", {}).get("myself") or {}
        return myself.get("pods", [])

    def stop_pod(self, pod_id: str) -> dict[str, Any]:
        if self.api_mode == "sdk":
            try:
                import runpod

                runpod.api_key = self.api_key
                return runpod.stop_pod(pod_id)
            except ImportError:
                pass

        if self.api_mode == "rest":
            return self.execute_rest(f"v1/pods/{pod_id}/stop", method="POST")

        mutation = f"""
        mutation {{
            podStop(input: {{ podId: "{pod_id}" }}) {{
                id
                desiredStatus
            }}
        }}
        """
        res = self.execute_graphql(mutation)
        return res.get("data", {}).get("podStop", {})

    def terminate_pod(self, pod_id: str) -> dict[str, Any]:
        if self.api_mode == "sdk":
            try:
                import runpod

                runpod.api_key = self.api_key
                res = runpod.terminate_pod(pod_id)
                return res if isinstance(res, dict) else {"id": pod_id, "status": "TERMINATED"}
            except ImportError:
                pass

        if self.api_mode == "rest":
            return self.execute_rest(f"v1/pods/{pod_id}", method="DELETE")

        mutation = f"""
        mutation {{
            podTerminate(input: {{ podId: "{pod_id}" }})
        }}
        """
        res = self.execute_graphql(mutation)
        return res.get("data", {}).get("podTerminate") or {"id": pod_id, "status": "TERMINATED"}

    def create_pod(
        self,
        name: str,
        image_name: str,
        gpu_type_id: str,
        cloud_type: str = "ALL",
        volume_in_gb: int = 0,
        container_disk_in_gb: int = 50,
        ports: str = "22/tcp",
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if self.api_mode == "sdk":
            try:
                import runpod

                runpod.api_key = self.api_key
                return runpod.create_pod(
                    name=name,
                    image_name=image_name,
                    gpu_type_id=gpu_type_id,
                    cloud_type=cloud_type,
                    volume_in_gb=volume_in_gb,
                    container_disk_in_gb=container_disk_in_gb,
                    start_ssh=True,
                    ports=ports,
                    env=env or {},
                )
            except ImportError:
                pass

        if self.api_mode == "rest":
            body = {
                "name": name,
                "imageName": image_name,
                "gpuTypeId": gpu_type_id,
                "cloudType": cloud_type,
                "volumeInGb": volume_in_gb,
                "containerDiskInGb": container_disk_in_gb,
                "ports": ports,
                "env": env or {},
            }
            return self.execute_rest("v1/pods", method="POST", body=body)

        env_items = []
        if isinstance(env, dict):
            for k, v in env.items():
                env_items.append(f"{{ key: {json.dumps(str(k))}, value: {json.dumps(str(v))} }}")
        elif isinstance(env, list):
            for item in env:
                if isinstance(item, dict):
                    k = item.get("key", "")
                    v = item.get("value", "")
                    env_items.append(f"{{ key: {json.dumps(str(k))}, value: {json.dumps(str(v))} }}")
        env_graphql = "[" + ", ".join(env_items) + "]"

        mutation = f"""
        mutation {{
            podFindAndDeployOnDemand(input: {{
                name: "{name}",
                imageName: "{image_name}",
                gpuTypeId: "{gpu_type_id}",
                gpuCount: 1,
                cloudType: {cloud_type},
                volumeInGb: {volume_in_gb},
                containerDiskInGb: {container_disk_in_gb},
                startSsh: true,
                ports: "{ports}",
                env: {env_graphql}
            }}) {{
                id
                name
                desiredStatus
                costPerHr
            }}
        }}
        """
        res = self.execute_graphql(mutation)
        return res.get("data", {}).get("podFindAndDeployOnDemand", {}) or {}


def watch_pod(
    client: RunPodClient,
    tracker: LifecycleTracker,
    pod_id: str,
    max_runtime_hours: float | None = None,
    max_budget: float | None = None,
    idle_timeout_minutes: float | None = None,
    auto_action: str | None = None,
    poll_interval: int = 30,
    once: bool = False,
    dry_run: bool = False,
) -> int:
    """Watch a pod and enforce lifecycle limits (max runtime, budget cap, idle timeout).

    Returns:
        0 if watching completed or limits healthy
        1 if pod not found or API error
        2 if limit breached and action executed
    """
    record = tracker.get_pod(pod_id)
    if record is None:
        record = tracker.track_pod(pod_id)

    if max_runtime_hours is not None:
        record["max_runtime_seconds"] = max_runtime_hours * 3600.0
    if max_budget is not None:
        record["max_budget_usd"] = max_budget
    if idle_timeout_minutes is not None:
        record["idle_timeout_seconds"] = idle_timeout_minutes * 60.0
    if auto_action is not None:
        record["auto_action"] = auto_action
    tracker.save()

    while True:
        try:
            info = client.get_pod(pod_id)
        except Exception as e:
            print(f"Error fetching status for pod {pod_id}: {e}", file=sys.stderr)
            return 1

        if not info:
            print(f"Pod {pod_id} not found on RunPod.")
            tracker.update_status(pod_id, "NOT_FOUND")
            return 1

        desired_status = str(info.get("desiredStatus", "")).upper()
        if desired_status in ("TERMINATED", "EXITED"):
            print(f"Pod {pod_id} has terminated ({desired_status}).")
            tracker.update_status(pod_id, desired_status)
            return 0
        if desired_status == "STOPPED" and record.get("auto_action") == "stop":
            print(f"Pod {pod_id} is stopped.")
            tracker.update_status(pod_id, "STOPPED")
            return 0

        runtime_dict = info.get("runtime") or {}
        uptime = runtime_dict.get("uptimeInSeconds")
        if uptime is None:
            uptime = info.get("uptimeSeconds")
        if uptime is None:
            uptime = max(0.0, time.time() - float(record.get("started_at", time.time())))
        else:
            uptime = float(uptime)

        cost_per_hr = info.get("costPerHr")
        if cost_per_hr is None:
            cost_per_hr = info.get("machine", {}).get("costPerHr")
        if cost_per_hr is None:
            cost_per_hr = record.get("cost_per_hr", 0.44)
        cost_per_hr = float(cost_per_hr)

        now = time.time()
        is_breached, reason, details = evaluate_limits(record, uptime, cost_per_hr, now=now)

        if is_breached and reason:
            action = record.get("auto_action", "terminate")
            spend = (uptime / 3600.0) * cost_per_hr
            print(f"\n[ALERT] Limit breached for pod {pod_id}!")
            print(f"  Reason: {reason}")
            print(f"  Details: {details}")
            print(f"  Enforcing action: {action} (dry_run={dry_run})")
            if not dry_run:
                if action == "stop":
                    client.stop_pod(pod_id)
                else:
                    client.terminate_pod(pod_id)
                tracker.update_status(pod_id, reason, action_taken=action, spend=spend, runtime=uptime)
            else:
                tracker.update_status(
                    pod_id,
                    f"DRY_RUN_{reason}",
                    action_taken=f"DRY_RUN_{action}",
                    spend=spend,
                    runtime=uptime,
                )
            return 2

        spend = (uptime / 3600.0) * cost_per_hr
        last_hb = float(record.get("last_heartbeat") or record.get("started_at") or now)
        idle_s = max(0.0, now - last_hb)
        budget_str = (
            f"${spend:.2f}/${record.get('max_budget_usd', 0.0):.2f}"
            if record.get("max_budget_usd")
            else f"${spend:.2f}"
        )
        idle_str = (
            f"{idle_s:.0f}s/{record.get('idle_timeout_seconds', 0.0):.0f}s"
            if record.get("idle_timeout_seconds")
            else f"{idle_s:.0f}s"
        )
        print(
            f"Pod {pod_id} healthy: status={desired_status} | uptime={uptime:.0f}s | "
            f"spend={budget_str} | idle={idle_str}"
        )

        if once:
            return 0
        time.sleep(poll_interval)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    # Launch
    p_launch = sub.add_parser("launch", help="Launch a RunPod instance")
    p_launch.add_argument("--template", choices=list(POD_TEMPLATES.keys()), help="Hardware template preset")
    p_launch.add_argument("--name", help="Pod name (default from RUNPOD_POD_NAME or 'monica-cuda-poc')")
    p_launch.add_argument("--gpu", help="GPU type ID (default from RUNPOD_GPU_TYPE or template)")
    p_launch.add_argument("--image", help="Docker image (default from RUNPOD_IMAGE_NAME or template)")
    p_launch.add_argument("--volume-gb", type=int, help="Volume size in GB (0 = standalone, default 0)")
    p_launch.add_argument("--disk-gb", type=int, help="Container disk in GB (default from template or 50)")
    p_launch.add_argument("--cloud-type", choices=("ALL", "COMMUNITY", "SECURE"), help="Cloud type")
    p_launch.add_argument("--max-runtime-hours", type=float, help="Max runtime in hours before auto stop/terminate")
    p_launch.add_argument("--max-budget", type=float, help="Max budget in USD before auto stop/terminate")
    p_launch.add_argument("--idle-timeout", type=float, help="Idle timeout in minutes before auto stop/terminate")
    p_launch.add_argument(
        "--auto-action",
        choices=("stop", "terminate"),
        help="Action on limit breach (default from RUNPOD_AUTO_ACTION or 'terminate')",
    )
    p_launch.add_argument(
        "--wait",
        action="store_true",
        default=True,
        help="Wait until pod is RUNNING and print SSH command (default: True)",
    )
    p_launch.add_argument(
        "--no-wait",
        action="store_false",
        dest="wait",
        help="Do not wait for pod to become RUNNING",
    )
    p_launch.add_argument(
        "--watch",
        action="store_true",
        default=False,
        help="Enter lifecycle watch loop immediately after launch",
    )

    # List
    p_list = sub.add_parser("list", help="List active pods")
    p_list.add_argument("--json", action="store_true", help="Output raw JSON")

    # Status
    p_status = sub.add_parser("status", help="Get pod status and lifecycle limits")
    p_status.add_argument("pod_id", help="Pod ID")
    p_status.add_argument("--json", action="store_true", help="Output raw JSON")

    # Heartbeat
    p_hb = sub.add_parser("heartbeat", help="Record activity heartbeat for pod")
    p_hb.add_argument("pod_id", help="Pod ID")
    p_hb.add_argument("--timestamp", type=float, default=None, help="Explicit epoch timestamp")

    # Watch
    p_watch = sub.add_parser("watch", help="Watch pod and enforce automated lifecycle limits")
    p_watch.add_argument("pod_id", help="Pod ID")
    p_watch.add_argument("--max-runtime-hours", type=float, help="Max runtime in hours")
    p_watch.add_argument("--max-budget", type=float, help="Max budget in USD")
    p_watch.add_argument("--idle-timeout", type=float, help="Idle timeout in minutes")
    p_watch.add_argument("--auto-action", choices=("stop", "terminate"), help="Action on limit breach")
    p_watch.add_argument("--poll-interval", type=int, default=30, help="Polling interval in seconds (default: 30)")
    p_watch.add_argument(
        "--once",
        action="store_true",
        default=False,
        help="Evaluate limits once and exit (exit code 2 if breached, 0 if healthy)",
    )
    p_watch.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Report breaches without executing stop or terminate",
    )

    # Stop
    p_stop = sub.add_parser("stop", help="Stop pod (preserves disk, pauses compute billing)")
    p_stop.add_argument("pod_id", help="Pod ID")

    # Terminate
    p_term = sub.add_parser("terminate", help="Terminate pod (ceases all billing)")
    p_term.add_argument("pod_id", help="Pod ID")

    # Templates
    p_tmpl = sub.add_parser("templates", help="List formalized hardware templates")
    p_tmpl.add_argument("--json", action="store_true", help="Output JSON")

    args = ap.parse_args()

    # Templates command requires no API credentials
    if args.command == "templates":
        if args.json:
            print(json.dumps(POD_TEMPLATES, indent=2))
            return
        print(f"{'Template':<12} {'GPU':<28} {'Disk/Vol':<10} {'CUDA':<8} {'Cost/hr':<9} {'Description'}")
        print("-" * 105)
        for key, tmpl in POD_TEMPLATES.items():
            print(
                f"{key:<12} {tmpl['gpu']:<28} "
                f"{f'{tmpl["disk_gb"]}/{tmpl["volume_gb"]}GB':<10} "
                f"{tmpl['cuda_version']:<8} "
                f"{f'${tmpl["approx_hourly_cost"]:.2f}':<9} "
                f"{tmpl['description']}"
            )
        return

    tracker = LifecycleTracker()

    # Heartbeat command requires no API credentials
    if args.command == "heartbeat":
        tracker.record_heartbeat(args.pod_id, timestamp=args.timestamp)
        print(f"Recorded activity heartbeat for pod {args.pod_id}.")
        return

    env = load_env()
    api_key = env.get("RUNPOD_API_KEY")
    if not api_key:
        sys.exit("RUNPOD_API_KEY not found in .env or environment.")

    api_mode = env.get("RUNPOD_API_MODE", "graphql")
    api_url = env.get("RUNPOD_API_BASE_URL", "https://api.runpod.io")
    client = RunPodClient(api_key=api_key, api_url=api_url, api_mode=api_mode)

    if args.command == "list":
        pods = client.get_pods()
        if args.json:
            print(json.dumps(pods, indent=2))
            return
        print(f"Active pods ({len(pods)}):")
        for p in pods:
            pid = p.get("id")
            gpu_name = p.get("machine", {}).get("gpuDisplayName", "Unknown")
            rec = tracker.get_pod(pid) if pid else None
            extra = ""
            if rec:
                spend = rec.get("total_spend_usd", 0.0)
                limit = rec.get("max_budget_usd", 0.0)
                extra = f" | Tracked Spend: ${spend:.2f} (Cap: ${limit:.2f})"
            print(
                f"  - ID: {pid} | Name: {p.get('name')} | "
                f"Status: {p.get('desiredStatus')} | GPU: {gpu_name}{extra}"
            )
        return

    if args.command == "status":
        pod = client.get_pod(args.pod_id)
        if not pod:
            sys.exit(f"Pod {args.pod_id} not found.")
        if args.json:
            print(json.dumps(pod, indent=2))
            return
        rec = tracker.get_pod(args.pod_id) or {}
        runtime_dict = pod.get("runtime") or {}
        uptime = runtime_dict.get("uptimeInSeconds") or pod.get("uptimeSeconds") or 0
        rate = pod.get("costPerHr") or pod.get("machine", {}).get("costPerHr") or rec.get("cost_per_hr", 0.44)
        spend = (float(uptime) / 3600.0) * float(rate)
        now = time.time()
        last_hb = float(rec.get("last_heartbeat") or rec.get("started_at") or now)
        idle_s = max(0.0, now - last_hb)

        print("=== RunPod Instance Status ===")
        print(f"Pod ID:             {pod.get('id')}")
        print(f"Name:               {pod.get('name')}")
        print(f"Status:             {pod.get('desiredStatus')}")
        print(f"GPU:                {pod.get('machine', {}).get('gpuDisplayName', 'Unknown')}")
        print(f"Hourly Rate:        ${float(rate):.2f}/hr")
        print(f"Uptime:             {uptime}s ({uptime / 3600.0:.2f}h)")
        print(f"Estimated Spend:    ${spend:.2f}")
        print(f"Last Heartbeat:     {idle_s:.0f}s ago")
        if rec:
            print("--- Lifecycle Guard Limits ---")
            print(f"Max Budget:         ${float(rec.get('max_budget_usd', 0.0)):.2f}")
            print(f"Max Runtime:        {float(rec.get('max_runtime_seconds', 0.0)) / 3600.0:.2f}h")
            print(f"Idle Timeout:       {float(rec.get('idle_timeout_seconds', 0.0)) / 60.0:.1f}m")
            print(f"Auto Action:        {rec.get('auto_action')}")
            print(f"Tracker Status:     {rec.get('status')}")
            if rec.get("action_taken"):
                print(f"Action Executed:    {rec.get('action_taken')}")
        return

    if args.command == "stop":
        print(f"Stopping pod {args.pod_id}...")
        res = client.stop_pod(args.pod_id)
        tracker.update_status(args.pod_id, "STOPPED", action_taken="stop")
        print("Result:", res)
        return

    if args.command == "terminate":
        print(f"Terminating pod {args.pod_id}...")
        res = client.terminate_pod(args.pod_id)
        tracker.update_status(args.pod_id, "TERMINATED", action_taken="terminate")
        print("Result:", res)
        return

    if args.command == "watch":
        code = watch_pod(
            client=client,
            tracker=tracker,
            pod_id=args.pod_id,
            max_runtime_hours=args.max_runtime_hours,
            max_budget=args.max_budget,
            idle_timeout_minutes=args.idle_timeout,
            auto_action=args.auto_action,
            poll_interval=args.poll_interval,
            once=args.once,
            dry_run=args.dry_run,
        )
        sys.exit(code)

    if args.command == "launch":
        # Resolve defaults from template, environment, and CLI flags
        tmpl_key = args.template or env.get("RUNPOD_TEMPLATE")
        tmpl = POD_TEMPLATES.get(tmpl_key, {}) if tmpl_key else {}

        name = args.name or env.get("RUNPOD_POD_NAME") or env.get("RUNPOD_DEFAULT_NAME") or "monica-cuda-poc"
        gpu = args.gpu or env.get("RUNPOD_GPU_TYPE") or tmpl.get("gpu") or "NVIDIA GeForce RTX 4090"
        image = (
            args.image
            or env.get("RUNPOD_IMAGE_NAME")
            or tmpl.get("image")
            or "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
        )
        volume_gb = (
            args.volume_gb
            if args.volume_gb is not None
            else int(env.get("RUNPOD_VOLUME_GB") or tmpl.get("volume_gb", 0))
        )
        disk_gb = (
            args.disk_gb
            if args.disk_gb is not None
            else int(env.get("RUNPOD_CONTAINER_DISK_GB") or env.get("RUNPOD_DISK_GB") or tmpl.get("disk_gb", 50))
        )
        cloud_type = args.cloud_type or env.get("RUNPOD_CLOUD_TYPE") or tmpl.get("cloud_type", "ALL")

        max_runtime_hours = (
            args.max_runtime_hours
            if args.max_runtime_hours is not None
            else float(env.get("RUNPOD_MAX_RUNTIME_HOURS") or 0.0)
        )
        max_budget = (
            args.max_budget if args.max_budget is not None else float(env.get("RUNPOD_MAX_BUDGET_USD") or 0.0)
        )
        idle_timeout_min = (
            args.idle_timeout
            if args.idle_timeout is not None
            else float(env.get("RUNPOD_IDLE_TIMEOUT_MINUTES") or 0.0)
        )
        auto_action = args.auto_action or env.get("RUNPOD_AUTO_ACTION") or "terminate"

        pub_key = get_ssh_key()
        pod_env = {
            "PUBLIC_KEY": pub_key,
            "AWS_ENDPOINT_URL_S3": env.get("AWS_ENDPOINT_URL_S3", ""),
            "AWS_ACCESS_KEY_ID": env.get("AWS_ACCESS_KEY_ID", ""),
            "AWS_SECRET_ACCESS_KEY": env.get("AWS_SECRET_ACCESS_KEY", ""),
            "AWS_DEFAULT_REGION": "auto",
            "R2_BUCKET": env.get("R2_BUCKET", "monica-training"),
        }

        print(f"Launching pod '{name}' with GPU '{gpu}' (template: {tmpl_key or 'custom'})...")
        pod = client.create_pod(
            name=name,
            image_name=image,
            gpu_type_id=gpu,
            cloud_type=cloud_type,
            volume_in_gb=volume_gb,
            container_disk_in_gb=disk_gb,
            ports="22/tcp",
            env=pod_env,
        )
        pod_id = pod.get("id")
        if not pod_id:
            sys.exit(f"Failed to create pod: {pod}")

        rate = float(pod.get("costPerHr") or tmpl.get("approx_hourly_cost", 0.44))
        tracker.track_pod(
            pod_id=pod_id,
            name=name,
            gpu=gpu,
            cost_per_hr=rate,
            max_runtime_seconds=max_runtime_hours * 3600.0,
            max_budget_usd=max_budget,
            idle_timeout_seconds=idle_timeout_min * 60.0,
            auto_action=auto_action,
        )
        print(f"Pod created! ID: {pod_id} (estimated rate: ${rate:.2f}/hr)")
        if max_budget > 0:
            print(f"  Hard budget cap: ${max_budget:.2f} ({auto_action})")
        if max_runtime_hours > 0:
            print(f"  Max runtime: {max_runtime_hours:.1f}h ({auto_action})")
        if idle_timeout_min > 0:
            print(f"  Idle timeout: {idle_timeout_min:.1f}m ({auto_action})")

        if args.wait or args.watch:
            print("Waiting for pod to be RUNNING...")
            for _ in range(60):
                time.sleep(5)
                info = client.get_pod(pod_id)
                if not info:
                    continue
                status = info.get("desiredStatus")
                runtime = info.get("runtime") or {}
                ports = runtime.get("ports", [])
                if runtime and ports:
                    ssh_port = None
                    ip = None
                    for p in ports:
                        if p.get("privatePort") == 22:
                            ssh_port = p.get("publicPort")
                            ip = p.get("ip")
                            break
                    if ssh_port and ip:
                        print("\n=== POD IS READY ===")
                        print(f"Pod ID:      {pod_id}")
                        print(f"IP:          {ip}")
                        print(f"SSH Port:    {ssh_port}")
                        print("SSH Command:")
                        print(f"  ssh root@{ip} -p {ssh_port} -i ~/.ssh/id_ed25519")
                        break
                print(f"  Current status: {status} ...")
            else:
                print("Timed out waiting for SSH port. Check status via: python scripts/cloud_pod.py status", pod_id)

        if args.watch:
            print(f"\nEntering lifecycle watch loop for pod {pod_id}...")
            sys.exit(
                watch_pod(
                    client=client,
                    tracker=tracker,
                    pod_id=pod_id,
                    max_runtime_hours=max_runtime_hours,
                    max_budget=max_budget,
                    idle_timeout_minutes=idle_timeout_min,
                    auto_action=auto_action,
                )
            )


if __name__ == "__main__":
    main()
