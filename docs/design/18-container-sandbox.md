# Containerized Execution Sandbox for Agent Tools and RLVR Verifiers (#353)

This document describes the design and implementation of the containerized execution sandbox backend (`src/runtime/sandbox.py`, #353) for autonomous coding agent tools and Reinforcement Learning from Verifiable Rewards (RLVR) reward verifiers.

---

## 1. Motivation and Environmental Isolation

Current evaluation, RLVR reward grading (`src/train/verifiers.py`), and autonomous agent tool execution (`src/agent/runtime.py`) historically operated directly on the host machine via local subprocesses. As established in container orchestration literature (e.g. arXiv:2609.20804, Harbor) and rigorous AI safety standards:
- Executing untrusted candidate model solutions or bash scripts directly on the host system presents escape, denial-of-service, and system state pollution risks.
- Reward verifiers in RLVR sweeps execute thousands of model-generated rollouts; a single fork bomb or unbounded memory allocation can crash training nodes.
- Environmental reproducibility requires strict control over installed packages, network access, and execution limits.

To provide safe, reproducible execution without introducing heavy third-party framework dependencies, Monica implements a standard-library-first containerized execution backend supporting Docker and Podman CLI backends.

---

## 2. Architecture Overview (`src/runtime/sandbox.py`)

The sandbox architecture separates execution provider interfaces from concrete environments above the hardware seam:

```
                            +--------------------+
                            |  SandboxProvider   |
                            |       (ABC)        |
                            +---------+----------+
                                      |
                 +--------------------+--------------------+
                 |                                         |
      +----------v-----------+                  +----------v-----------+
      | LocalSandboxProvider |                  |   ContainerSandbox   |
      |   (Host Subprocess)  |                  |   (Docker / Podman)  |
      +----------------------+                  +----------------------+
                 |                                         |
         Trusted Smoke Runs                      Untrusted Agent Tools &
          (Fast CI tests)                             RLVR Verifiers
```

### Components

1. **`ExecutionResult`**:
   Standardized result container tracking `exit_code`, captured `stdout`, `stderr`, elapsed `duration_s`, `timed_out` flag, and optional `error` details. Evaluates `is_error` as `True` when `exit_code != 0` or `timed_out` is `True`.

2. **`SandboxConfig`**:
   Declarative configuration defining:
   - `image`: Target container image (default `python:3.12-slim`).
   - `engine`: Container runtime CLI (`docker` or `podman`).
   - `workspace_dir`: Host repository or scratch path to mount.
   - `container_workdir`: In-container mount destination (default `/workspace`).
   - `cpu_limit`: Maximum CPU count/quota (e.g., `2.0`).
   - `memory_limit`: Memory constraint string (e.g., `"2g"` or `"512m"`).
   - `pids_limit`: Maximum process IDs to prevent fork bombs (default `256`).
   - `network`: Network isolation mode (`"none"` for isolated RLVR/evals, `"bridge"`, `"host"`).
   - `read_only`: Boolean flag to mount rootfs/workspace read-only.
   - `cap_drop`: Linux capabilities dropped (default `["ALL"]`).
   - `env`: Environment variable mappings injected into containers.
   - `persistent`: Boolean flag indicating persistent container session lifecycle.

3. **`SandboxProvider` (ABC)**:
   Base execution protocol defining `run_command`, with convenience methods `run_python` (running `["python3", "-c", code]`) and `run_bash` (running `["/bin/bash", "-c", command]`), plus context manager hooks (`__enter__`, `__exit__`, `close`).

---

## 3. Ephemeral & Persistent Container Lifecycles

`ContainerSandbox` supports two complementary execution lifecycles:

### 1. Ephemeral Per-Command Mode (Default)
Each command executes in a newly launched, self-contained container:
```bash
docker run --rm \
  --name monica_sb_<uuid> \
  --cpus 2.0 \
  --memory 2g \
  --pids-limit 256 \
  --network none \
  --cap-drop ALL \
  -v /path/to/host/repo:/workspace:rw \
  -w /workspace \
  -e KEY=VAL \
  python:3.12-slim /bin/bash -c "<command>"
```
- **Guaranteed Cleanup**: Ephemeral containers are tagged with unique names and the `--rm` flag. If the command exceeds `timeout_s`, Python's `subprocess.TimeoutExpired` initiates an immediate asynchronous cleanup: `docker rm -f <container_name>`.
- **Stateless Isolation**: Guarantees zero residual state between successive verifier runs or rollout candidates.

### 2. Persistent Container Session Mode
For high-turn agent sessions or large test suites where per-command container startup overhead is undesirable:
- `start()` launches a long-running container: `docker run -d --name <name> ... sleep infinity`.
- Successive commands execute via `docker exec -w <workdir> <container_id> <cmd>`.
- `stop()` / `__exit__()` forcefully terminates and prunes the session container (`docker rm -f <container_id>`).

---

## 4. Integration with Verifiers & Agent Tools

### `CodeVerifier` Integration (`src/train/verifiers.py`)
`CodeVerifier` computes RLVR reward signals as the fraction of unit test assertions passing against candidate code:
- **Local Guard**: Local subprocess execution runs untrusted code directly on the host and requires an explicit `enabled=True` opt-in.
- **Container Sandbox**: Passing a container sandbox (`sandbox=ContainerSandbox(...)` or `sandbox="docker"`) routes execution into the isolated container. Because containerization guarantees environmental safety, evaluation runs without requiring the dangerous host `enabled=True` flag:

```python
from src.train.verifiers import CodeVerifier
from src.runtime.sandbox import ContainerSandbox

# Isolated RLVR grading inside an ephemeral container
sandbox = ContainerSandbox(image="python:3.12-slim", network="none")
verifier = CodeVerifier(sandbox=sandbox, timeout=5.0)

reward = verifier.reward("def add(a, b): return a + b", ["assert add(1, 2) == 3"])
# reward == 1.0
```

### `WorkspaceToolExecutor` & `AgentRuntime` Integration (`src/agent/runtime.py`)
`WorkspaceToolExecutor` manages repository operations for autonomous coding agents:
- `execute_bash` routes commands to `self.sandbox.run_bash(command, cwd=..., timeout=...)` when a sandbox is configured.
- Host filesystem operations (`view_file`, `edit_file`, `write_file`) operate directly on the mounted workspace directory, ensuring instant static analysis, linting, and Git tracking without synchronization overhead.
- `AgentRuntime` and `run_agent_loop` accept an optional `sandbox` parameter, propagating it to the underlying tool executor.

---

## 5. Seamless Toggling

The factory `create_sandbox` and environment variable `MONICA_SANDBOX_BACKEND` allow seamless switching between environments:

| Mode / Backend | Engine | Use Case |
|---|---|---|
| `"local"` / `"subprocess"` | Host Python | Fast unit tests and trusted smoke gates |
| `"docker"` | Docker CLI | Ephemeral containerized agent execution & RLVR sweeps |
| `"podman"` | Podman CLI | Rootless daemonless containerized execution |
| `"auto"` | Detected / Env | Automatic detection: inspects `MONICA_SANDBOX_BACKEND` or `docker`/`podman` in PATH |
