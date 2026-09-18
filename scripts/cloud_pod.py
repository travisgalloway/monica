#!/usr/bin/env python3
"""CLI utility to manage on-demand RunPod CUDA instances for Monica (#198).

Supports:
    python scripts/cloud_pod.py launch --gpu "NVIDIA GeForce RTX 4090" --name monica-cuda-poc
    python scripts/cloud_pod.py list
    python scripts/cloud_pod.py status <pod_id>
    python scripts/cloud_pod.py terminate <pod_id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_env() -> dict[str, str]:
    env = {}
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip("\"'")
                os.environ[k.strip()] = v.strip("\"'")
    return env


def _get_ssh_key() -> str:
    key_path = Path.home() / ".ssh" / "id_ed25519.pub"
    if key_path.exists():
        return key_path.read_text().strip()
    rsa_path = Path.home() / ".ssh" / "id_rsa.pub"
    if rsa_path.exists():
        return rsa_path.read_text().strip()
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    # Launch
    p_launch = sub.add_parser("launch", help="Launch a RunPod instance")
    p_launch.add_argument("--name", default="monica-cuda-poc", help="pod name")
    p_launch.add_argument("--gpu", default="NVIDIA GeForce RTX 4090", help="GPU type ID")
    p_launch.add_argument("--image", default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                          help="Docker image")
    p_launch.add_argument("--volume-gb", type=int, default=50, help="volume size in GB")
    p_launch.add_argument("--disk-gb", type=int, default=50, help="container disk in GB")
    p_launch.add_argument("--cloud-type", choices=("ALL", "COMMUNITY", "SECURE"), default="ALL")
    p_launch.add_argument("--wait", action="store_true", default=True,
                          help="wait until pod is RUNNING and print SSH command")

    # List
    sub.add_parser("list", help="List active pods")

    # Status
    p_status = sub.add_parser("status", help="Get pod status")
    p_status.add_argument("pod_id", help="pod ID")

    # Terminate
    p_term = sub.add_parser("terminate", help="Terminate pod")
    p_term.add_argument("pod_id", help="pod ID")

    args = ap.parse_args()
    env = _load_env()

    api_key = env.get("RUNPOD_API_KEY")
    if not api_key:
        sys.exit("RUNPOD_API_KEY not found in .env or environment.")

    import runpod
    runpod.api_key = api_key

    if args.command == "list":
        pods = runpod.get_pods()
        print(f"Active pods ({len(pods)}):")
        for p in pods:
            gpu_name = p.get("machine", {}).get("gpuDisplayName", "Unknown")
            print(f"  - ID: {p.get('id')} | Name: {p.get('name')} | Status: {p.get('desiredStatus')} | GPU: {gpu_name}")
        return

    if args.command == "status":
        pod = runpod.get_pod(args.pod_id)
        if not pod:
            sys.exit(f"Pod {args.pod_id} not found.")
        print(json.dumps(pod, indent=2))
        return

    if args.command == "terminate":
        print(f"Terminating pod {args.pod_id}...")
        res = runpod.terminate_pod(args.pod_id)
        print("Result:", res)
        return

    if args.command == "launch":
        pub_key = _get_ssh_key()
        pod_env = {
            "PUBLIC_KEY": pub_key,
            "AWS_ENDPOINT_URL_S3": env.get("AWS_ENDPOINT_URL_S3", ""),
            "AWS_ACCESS_KEY_ID": env.get("AWS_ACCESS_KEY_ID", ""),
            "AWS_SECRET_ACCESS_KEY": env.get("AWS_SECRET_ACCESS_KEY", ""),
            "AWS_DEFAULT_REGION": "auto",
            "R2_BUCKET": env.get("R2_BUCKET", "monica-training"),
        }
        print(f"Launching pod '{args.name}' with GPU '{args.gpu}'...")
        pod = runpod.create_pod(
            name=args.name,
            image_name=args.image,
            gpu_type_id=args.gpu,
            cloud_type=args.cloud_type,
            volume_in_gb=args.volume_gb,
            container_disk_in_gb=args.disk_gb,
            start_ssh=True,
            env=pod_env,
        )
        pod_id = pod.get("id")
        print(f"Pod created! ID: {pod_id}")

        if args.wait:
            print("Waiting for pod to be RUNNING...")
            for _ in range(60):
                time.sleep(5)
                info = runpod.get_pod(pod_id)
                status = info.get("desiredStatus")
                runtime = info.get("runtime", {})
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
                        print(f"Pod ID: {pod_id}")
                        print(f"IP: {ip}")
                        print(f"SSH Port: {ssh_port}")
                        print(f"SSH Command:")
                        print(f"  ssh root@{ip} -p {ssh_port} -i ~/.ssh/id_ed25519")
                        return
                print(f"  Current status: {status} ...")
            print("Timed out waiting for SSH port. Check status via: python scripts/cloud_pod.py status", pod_id)


if __name__ == "__main__":
    main()
