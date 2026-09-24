"""Build the scale corpus with datatrove (#80, #252) — the pod/cluster driver.

Runs the staged datatrove pipeline (`src/data/datatrove_pipeline.py`): ingest a source, apply the
project filters (reusing `src/data/filters.py` semantics), write cleaned text shards, then
optionally run cross-source MinHash dedup. The cleaned shards then feed the native Swift
`monica-tokenize pack` step (or `src/data/shard.py`) to emit packed uint16 shards
(`.bin` + `.bounds` + `manifest.json`) readable directly by `src/data/loader.py`.

MUST run in the py3.11 datatrove venv (`.venv-dt`), not the main py3.14 env:

    # Local sample shard generation & verification (#252 Phase 1):
    .venv-dt/bin/python scripts/build_corpus.py --source sample \
        --out data/mhm_sample_clean --pack --shards-out data/mhm_sample_shards \
        --quality --license-filter --scrub --decontam --val-tokens 1024

    # Pod/cluster production run (#252 Phase 2):
    .venv-dt/bin/python scripts/build_corpus.py --source fineweb-edu \
        --out s3://monica-training/reserve-pretrain --executor slurm --tasks 200 \
        --quality --license-filter --scrub --decontam --dedup
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def find_monica_tokenize(explicit: str | Path | None = None) -> Path | None:
    """Locate the monica-tokenize binary across known build directories."""
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    env = os.environ.get("MONICA_TOKENIZE")
    if env and Path(env).exists():
        return Path(env)
    for root in (REPO_ROOT, Path("/Users/travisgalloway/github/monica")):
        swift_dir = root / "swift"
        for mode in ("release", "debug"):
            for cand in (
                swift_dir / ".build" / mode / "monica-tokenize",
                swift_dir / ".build" / "arm64-apple-macosx" / mode / "monica-tokenize",
            ):
                if cand.exists():
                    return cand
    return None


def find_default_tokenizer(explicit: str | Path | None = None) -> Path | None:
    """Locate default MHM tokenizer vocabulary (preferring 32k/49k MHM vocabs)."""
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    candidates = [
        REPO_ROOT / "data" / "tokenizer" / "vocab-32768.json",
        REPO_ROOT / "configs" / "tokenizer" / "vocab.json",
        Path.home() / "monica-data" / "vocab-sweep-251-half" / "tok" / "vocab-32768.json",
        Path.home() / "monica-data" / "vocab-sweep-251" / "tok" / "vocab-32768.json",
        Path.home() / "monica-data" / "vocab-sweep-251-half" / "tok" / "vocab-49152.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    tok_bin = find_monica_tokenize()
    if tok_bin is not None:
        fixture = REPO_ROOT / "swift" / "Fixtures" / "parity-corpus.jsonl"
        if fixture.exists():
            out_tok = REPO_ROOT / "data" / "tokenizer" / "vocab-32768.json"
            out_tok.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([str(tok_bin), "train", "--in", str(fixture), "--out", str(out_tok), "--vocab-size", "32768"], check=True)
            if out_tok.exists():
                return out_tok
    return None


def ensure_sample_corpus(out_file: Path) -> Path:
    """Generate a rich sample source repository mixture exercising #357, #358, and #359."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists() and out_file.stat().st_size > 5000:
        return out_file

    parity_corpus = REPO_ROOT / "swift" / "Fixtures" / "parity-corpus.jsonl"
    humaneval = REPO_ROOT / "eval_sets" / "humaneval_ts" / "humaneval_ts.jsonl"

    contam_line = ""
    if humaneval.exists():
        with open(humaneval, encoding="utf-8") as f:
            first = f.readline()
            if first.strip():
                contam_line = json.loads(first)["prompt"]

    records = []

    # Project 1: auth-gateway (TypeScript multi-file repository)
    repo1 = "auth-gateway"
    p1_files = [
        ("src/types/token.ts", """export interface TokenPayload {
    sub: string;
    username: string;
    role: "admin" | "member" | "guest";
    iat: number;
    exp: number;
}

export interface SessionContext {
    token: string;
    payload: TokenPayload;
    active: boolean;
}
"""),
        ("src/utils/crypto.ts", """import { TokenPayload } from "../types/token";

export function generateNonce(length: number = 16): string {
    const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let nonce = "";
    for (let i = 0; i < length; i += 1) {
        const idx = Math.floor(Math.random() * chars.length);
        nonce += chars.charAt(idx);
    }
    return nonce;
}

export function validateExpiry(payload: TokenPayload): boolean {
    const now = Math.floor(Date.now() / 1000);
    if (payload.exp <= now) {
        return false;
    }
    return true;
}
"""),
        ("src/services/session.ts", """import { SessionContext, TokenPayload } from "../types/token";
import { generateNonce, validateExpiry } from "../utils/crypto";

export class SessionService {
    private activeSessions = new Map<string, SessionContext>();

    public createSession(payload: TokenPayload): SessionContext {
        const nonce = generateNonce(32);
        const context: SessionContext = {
            token: nonce,
            payload,
            active: validateExpiry(payload),
        };
        this.activeSessions.set(nonce, context);
        return context;
    }

    public getSession(token: string): SessionContext | undefined {
        const session = this.activeSessions.get(token);
        if (session && !validateExpiry(session.payload)) {
            session.active = false;
        }
        return session;
    }
}
"""),
        ("src/middleware/guard.ts", """import { SessionService } from "../services/session";
import { SessionContext } from "../types/token";

export function authorizeRole(
    sessionService: SessionService,
    token: string,
    requiredRole: string
): boolean {
    const session = sessionService.getSession(token);
    if (!session || !session.active) {
        return false;
    }
    if (requiredRole === "admin" && session.payload.role !== "admin") {
        return false;
    }
    return true;
}
"""),
        ("src/index.ts", """import { SessionService } from "./services/session";
import { authorizeRole } from "./middleware/guard";
import { TokenPayload } from "./types/token";

export function bootstrapAuthApp(): void {
    const svc = new SessionService();
    const adminUser: TokenPayload = {
        sub: "user-101",
        username: "system_admin",
        role: "admin",
        iat: Math.floor(Date.now() / 1000),
        exp: Math.floor(Date.now() / 1000) + 3600,
    };
    const session = svc.createSession(adminUser);
    const authorized = authorizeRole(svc, session.token, "admin");
    console.log(`Auth initialization complete. Authorized: ${authorized}`);
}
"""),
    ]
    for p, code in p1_files:
        # Replicate code blocks to provide adequate sequence tokens
        body = code + "\n" + code
        records.append({"text": body, "path": p, "repo": repo1, "metadata": {"lang": "typescript", "license": "mit", "repo": repo1, "is_code": True}})

    # Project 2: matrix-compute (Python multi-file repository)
    repo2 = "matrix-compute"
    p2_files = [
        ("math_engine/config.py", """from dataclasses import dataclass

@dataclass
class EngineConfig:
    dimension: int = 128
    precision: str = "float32"
    device: str = "cpu"
    normalize_inputs: bool = True
"""),
        ("math_engine/linear.py", """from math_engine.config import EngineConfig

class Matrix2D:
    def __init__(self, rows: int, cols: int, config: EngineConfig | None = None):
        self.rows = rows
        self.cols = cols
        self.config = config or EngineConfig()
        self.data: list[list[float]] = [[0.0 for _ in range(cols)] for _ in range(rows)]

    def set_identity(self) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                self.data[r][c] = 1.0 if r == c else 0.0

    def trace(self) -> float:
        total = 0.0
        limit = min(self.rows, self.cols)
        for i in range(limit):
            total += self.data[i][i]
        return total
"""),
        ("math_engine/statistics.py", """from math_engine.linear import Matrix2D

def compute_row_means(mat: Matrix2D) -> list[float]:
    means = []
    for row in mat.data:
        if not row:
            means.append(0.0)
            continue
        total = sum(row)
        means.append(total / float(len(row)))
    return means
"""),
        ("math_engine/solver.py", """from math_engine.config import EngineConfig
from math_engine.linear import Matrix2D
from math_engine.statistics import compute_row_means

class LinearSolver:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.matrix = Matrix2D(config.dimension, config.dimension, config)
        self.matrix.set_identity()

    def solve_trace_and_means(self) -> tuple[float, list[float]]:
        t = self.matrix.trace()
        m = compute_row_means(self.matrix)
        return t, m
"""),
        ("main.py", """from math_engine.config import EngineConfig
from math_engine.solver import LinearSolver

def run_computation():
    cfg = EngineConfig(dimension=64)
    solver = LinearSolver(cfg)
    t, means = solver.solve_trace_and_means()
    print(f"Computed matrix trace: {t} across {len(means)} dimensions")

if __name__ == "__main__":
    run_computation()
"""),
    ]
    for p, code in p2_files:
        body = code + "\n" + code
        records.append({"text": body, "path": p, "repo": repo2, "metadata": {"lang": "python", "license": "apache-2.0", "repo": repo2, "is_code": True}})

    # Project 3: event-dispatcher (TypeScript multi-file repository)
    repo3 = "event-dispatcher"
    p3_files = [
        ("src/schemas/events.ts", """export interface BaseEvent {
    eventId: string;
    timestamp: number;
    sourceService: string;
}

export interface UserRegistrationEvent extends BaseEvent {
    userId: string;
    email: string;
    tier: "free" | "pro";
}
"""),
        ("src/pipes/validator.ts", """import { BaseEvent, UserRegistrationEvent } from "../schemas/events";

export function validateEventStructure(evt: BaseEvent): boolean {
    if (!evt.eventId || evt.eventId.length === 0) {
        return false;
    }
    if (evt.timestamp <= 0) {
        return false;
    }
    return true;
}

export function isUserRegistration(evt: BaseEvent): evt is UserRegistrationEvent {
    return (evt as UserRegistrationEvent).userId !== undefined;
}
"""),
        ("src/dispatcher.ts", """import { BaseEvent } from "./schemas/events";
import { validateEventStructure, isUserRegistration } from "./pipes/validator";

export class EventDispatcher {
    private handlers = new Map<string, Array<(e: BaseEvent) => void>>();

    public registerHandler(eventType: string, handler: (e: BaseEvent) => void): void {
        const list = this.handlers.get(eventType) || [];
        list.push(handler);
        this.handlers.set(eventType, list);
    }

    public dispatch(evt: BaseEvent): boolean {
        if (!validateEventStructure(evt)) {
            return false;
        }
        const listeners = this.handlers.get(evt.sourceService) || [];
        for (const fn of listeners) {
            fn(evt);
        }
        return true;
    }
}
"""),
    ]
    for p, code in p3_files:
        body = code + "\n" + code
        records.append({"text": body, "path": p, "repo": repo3, "metadata": {"lang": "typescript", "license": "mit", "repo": repo3, "is_code": True}})

    # Additional text and prose records from parity corpus
    if parity_corpus.exists():
        with open(parity_corpus, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    records.append({
                        "text": item.get("text", ""),
                        "metadata": {"lang": "en", "license": "odc-by", "is_code": False},
                    })

    # Add secret line to verify scrubber
    records.append({
        "text": """// Sensitive configuration file\nconst AWS_SECRET_KEY = "AKIA1234567890EXAMPLE";\nexport function getKey(): string { return AWS_SECRET_KEY; }\n""",
        "path": "src/secret.ts",
        "metadata": {"lang": "typescript", "license": "mit", "is_code": True},
    })

    # Add contam line to verify decontaminator
    if contam_line:
        records.append({
            "text": contam_line,
            "path": "eval_contam.ts",
            "metadata": {"lang": "typescript", "license": "mit", "is_code": True},
        })

    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return out_file

def find_default_sample(explicit: str | Path | None = None) -> Path | None:
    """Locate sample corpus file, creating rich sample repo mixture if none exists."""
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    candidates = [
        REPO_ROOT / "data" / "sample" / "sample.jsonl",
        Path.home() / "monica-data" / "vocab-sweep-251-half" / "sample" / "sample.jsonl",
        Path.home() / "monica-data" / "vocab-sweep-251" / "sample" / "sample.jsonl",
        REPO_ROOT / "data" / "sample_slice" / "raw.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return c
    return ensure_sample_corpus(REPO_ROOT / "data" / "sample" / "sample.jsonl")


def iter_cleaned_docs(cleaned_dir: Path | str) -> Iterator[dict]:
    """Iterate over cleaned JSONL/GZ files emitted by Datatrove."""
    cleaned_dir = Path(cleaned_dir)
    files = sorted(glob.glob(f"{cleaned_dir}/**/*.jsonl*", recursive=True))
    for fp in files:
        if fp.endswith(".gz"):
            with gzip.open(fp, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        elif fp.endswith(".jsonl"):
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)


def consolidate_cleaned(cleaned_dir: Path | str, out_file: Path | str) -> Tuple[int, Dict[str, Any], int]:
    """Consolidate cleaned records to a single cleaned.jsonl and compute per-language mix."""
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    n_docs = 0
    total_bytes = 0
    lang_counts: Dict[str, int] = {}
    lang_bytes: Dict[str, int] = {}

    with open(out_file, "w", encoding="utf-8") as f:
        for doc in iter_cleaned_docs(cleaned_dir):
            text = doc.get("text", "")
            if not text:
                continue
            f.write(json.dumps({"text": text}) + "\n")
            n_docs += 1
            n_b = len(text.encode("utf-8"))
            total_bytes += n_b
            md = doc.get("metadata") or {}
            lang = doc.get("lang") or md.get("lang") or "unknown"
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
            lang_bytes[lang] = lang_bytes.get(lang, 0) + n_b

    lang_mix = {}
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        b = lang_bytes[lang]
        lang_mix[lang] = {
            "docs": count,
            "bytes": b,
            "pct_docs": round(100.0 * count / n_docs, 2) if n_docs else 0.0,
            "pct_bytes": round(100.0 * b / total_bytes, 2) if total_bytes else 0.0,
        }
    return n_docs, lang_mix, total_bytes


def extract_and_sort_repos(
    docs_iterable: Iterable[dict],
    dag_sort: bool = True,
) -> Tuple[List[dict], List[dict]]:
    """Group repository file entries, apply topological sort via import DAG (#359),
    and construct repo projects manifest.

    Returns:
        (sorted_docs, repo_projects)
    """
    from src.data.repo_graph import build_repo_graph

    repos: Dict[str, Dict[str, str]] = {}
    standalone_docs: List[dict] = []

    for doc in docs_iterable:
        text = doc.get("text", "")
        if not text:
            continue
        md = doc.get("metadata") or {}
        repo_name = doc.get("repo") or md.get("repo")
        path = doc.get("path") or md.get("path")
        if repo_name and path:
            if repo_name not in repos:
                repos[repo_name] = {}
            repos[repo_name][path] = text
        else:
            standalone_docs.append(doc)

    all_sorted_docs: List[dict] = []
    repo_projects: List[dict] = []

    for repo_name, file_map in repos.items():
        if dag_sort and len(file_map) > 1:
            try:
                graph = build_repo_graph(file_map)
                sorted_paths = graph.topological_sort()
            except Exception:
                sorted_paths = sorted(file_map.keys())
        else:
            sorted_paths = sorted(file_map.keys())

        file_entries = []
        for p in sorted_paths:
            content = file_map[p]
            file_entries.append({"path": p, "content": content})
            lang = "typescript" if p.endswith((".ts", ".tsx")) else "python" if p.endswith(".py") else "text"
            all_sorted_docs.append({
                "text": content,
                "path": p,
                "repo": repo_name,
                "lang": lang,
                "metadata": {"lang": lang, "license": "mit", "repo": repo_name, "is_code": True},
            })
        repo_projects.append({"repo": repo_name, "files": file_entries})

    all_sorted_docs.extend(standalone_docs)
    return all_sorted_docs, repo_projects


def pack_cleaned_shards(cleaned_jsonl: Path, shards_out: Path, tokenizer_path: Path,
                        seq_len: int = 8192, shard_size_mb: int = 512,
                        tokenize_bin: Path | None = None,
                        chunk_align: int | None = None,
                        fim_rate: float | None = None,
                        fim_seed: int | None = None,
                        fim_mode: str | None = None,
                        repo_manifest: Path | None = None) -> dict:
    """Pack cleaned.jsonl or repo_manifest into uint16 .bin + .bounds + manifest.json shards."""
    shards_out = Path(shards_out)
    shards_out.mkdir(parents=True, exist_ok=True)
    if tokenize_bin is not None:
        cmd = [
            str(tokenize_bin), "pack",
            "--tokenizer", str(tokenizer_path),
            "--out", str(shards_out),
            "--seq-len", str(seq_len),
            "--shard-size-mb", str(shard_size_mb),
        ]
        if repo_manifest is not None and Path(repo_manifest).exists():
            cmd.extend(["--repo-manifest", str(repo_manifest)])
        else:
            cmd.extend(["--in", str(cleaned_jsonl)])
        if chunk_align:
            cmd.extend(["--chunk-align", str(chunk_align)])
        if fim_rate is not None:
            cmd.extend(["--fim-rate", str(fim_rate)])
        if fim_seed is not None:
            cmd.extend(["--fim-seed", str(fim_seed)])
        if fim_mode is not None:
            cmd.extend(["--fim-mode", str(fim_mode)])
        subprocess.run(cmd, check=True)
    else:
        # Fallback to Python pack_sequences
        from src.data.shard import pack_sequences
        from src.data.tokenize import ByteTokenizer
        tok = ByteTokenizer()
        docs = []
        with open(cleaned_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line).get("text", ""))
        tokenized = [tok.encode(d) for d in docs if d]
        pack_sequences(tokenized, shards_out, seq_len=seq_len,
                       shard_size_mb=shard_size_mb, tokenizer="code")

    manifest_path = shards_out / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"pack step failed to emit {manifest_path}")
    return json.loads(manifest_path.read_text())


def run_corpus_pipeline(
    source: str = "sample",
    out_dir: Path | str = "data/mhm_sample_clean",
    *,
    from_jsonl: Path | str | None = None,
    from_repo: Path | str | None = None,
    repo_name: str | None = None,
    dag_sort: bool = True,
    repo_manifest: Path | str | None = None,
    limit: int = -1,
    split: str = "train",
    executor_kind: str = "local",
    tasks: int = 1,
    workers: int = 1,
    logging_dir: Path | str | None = None,
    quality: bool = True,
    license_filter: bool = True,
    drop_minified: bool = False,
    drop_autogen: bool = False,
    scrub: bool = True,
    decontam: bool = True,
    decontam_file: Path | str | None = None,
    dedup: bool = False,
    pack: bool = False,
    tokenizer: Path | str | None = None,
    shards_out: Path | str | None = None,
    seq_len: int = 8192,
    shard_size_mb: int = 512,
    val_tokens: int | None = None,
    tokenize_bin: Path | str | None = None,
    fim_mode: str = "joint",
    fim_rate: float | None = 0.5,
    fim_seed: int | None = 42,
    r2_sync_uri: str | None = None,
) -> dict:
    """End-to-end driver: clean -> optional dedup -> consolidate -> pack -> split."""
    from src.data import datatrove_pipeline as dt

    out_dir = Path(out_dir)
    logging_dir = Path(logging_dir or (out_dir / "logs"))

    # 1. Resolve source reader and repository manifests
    sample_path = None
    repo_projects_list = []
    active_repo_manifest = Path(repo_manifest) if repo_manifest else None

    if source == "repo" or from_repo is not None:
        target_repo = Path(from_repo or from_jsonl or "eval_sets/code_recall/fixture_repo.jsonl")
        if not target_repo.exists():
            raise FileNotFoundError(f"Repository path not found: {target_repo}")
        raw_items = []
        if target_repo.is_dir():
            from src.data.repo_graph import build_repo_graph
            g = build_repo_graph(target_repo)
            for f in g.files:
                fp = target_repo / f
                if fp.exists():
                    raw_items.append({"path": f, "text": fp.read_text(encoding="utf-8"), "repo": repo_name or target_repo.stem})
        else:
            with open(target_repo, encoding="utf-8") as rf:
                for line in rf:
                    if line.strip():
                        obj = json.loads(line)
                        if "files" in obj and isinstance(obj["files"], list):
                            rname = obj.get("repo") or obj.get("repo_name") or repo_name or target_repo.stem
                            for f in obj["files"]:
                                raw_items.append({
                                    "path": f.get("path"),
                                    "text": f.get("content") or f.get("text") or "",
                                    "repo": rname,
                                    "metadata": {"lang": "typescript" if str(f.get("path")).endswith((".ts", ".tsx")) else "python", "license": "mit", "repo": rname, "is_code": True},
                                })
                        elif "path" in obj:
                            raw_items.append({
                                "path": obj.get("path"),
                                "text": obj.get("text") or obj.get("content") or "",
                                "repo": repo_name or target_repo.stem,
                                "metadata": obj.get("metadata") or {"lang": "typescript" if str(obj.get("path")).endswith((".ts", ".tsx")) else "python", "license": "mit", "repo": repo_name or target_repo.stem, "is_code": True},
                            })
        sorted_docs, repo_projects_list = extract_and_sort_repos(raw_items, dag_sort=dag_sort)
        temp_input = out_dir / "raw_repo_input.jsonl"
        temp_input.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_input, "w", encoding="utf-8") as wf:
            for doc in sorted_docs:
                wf.write(json.dumps(doc) + "\n")
        reader = dt.jsonl_reader(str(temp_input), limit=limit)
    elif source == "fineweb-edu":
        reader = dt.fineweb_edu_reader(limit=limit, split=split)
    elif source == "stack-v2":
        reader = dt.stack_v2_reader(limit=limit)
    elif source == "essential-web":
        reader = dt.essential_web_reader(limit=limit)
    elif source == "pretrain-extract":
        reader = dt.composite_reader([dt.stack_v2_reader(limit=limit), dt.essential_web_reader(limit=limit)])
    elif source in ("sample", "jsonl"):
        sample_path = find_default_sample(from_jsonl)
        if sample_path is None or not Path(sample_path).exists():
            raise FileNotFoundError(
                f"Sample corpus not found (given: {from_jsonl}). Provide --from-jsonl <path> "
                "or run scripts/vocab_sweep.py --sample-only to populate monica-data/."
            )
        # Check if sample source contains repository file entries to sort by DAG (#359)
        if dag_sort:
            sample_docs = []
            with open(sample_path, encoding="utf-8") as sf:
                for line in sf:
                    if line.strip():
                        sample_docs.append(json.loads(line))
            sorted_docs, repo_projects_list = extract_and_sort_repos(sample_docs, dag_sort=dag_sort)
            if repo_projects_list:
                temp_input = out_dir / "sorted_sample_input.jsonl"
                temp_input.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_input, "w", encoding="utf-8") as wf:
                    for doc in sorted_docs:
                        wf.write(json.dumps(doc) + "\n")
                reader = dt.jsonl_reader(str(temp_input), limit=limit)
            else:
                reader = dt.jsonl_reader(str(sample_path), limit=limit)
        else:
            reader = dt.jsonl_reader(str(sample_path), limit=limit)
    else:
        raise ValueError(f"Unknown source {source!r}")

    # 2. Decontamination setup
    decon = None
    decontam_path = None
    if decontam or decontam_file:
        decontam_path = Path(decontam_file) if decontam_file else (REPO_ROOT / "eval_sets/decontam/blocklist.txt")
        if decontam_path.exists():
            from src.data.dedup import Decontaminator
            with open(decontam_path, encoding="utf-8") as f:
                decon = Decontaminator.from_texts(f)

    # 3. Clean pipeline
    pipeline = dt.clean_pipeline(
        reader, str(out_dir), quality=quality, license_filter=license_filter,
        drop_minified=drop_minified, drop_autogen=drop_autogen, scrub=scrub,
        decontaminator=decon)
    executor = dt.make_executor(pipeline, str(logging_dir / "clean"), kind=executor_kind,
                                tasks=tasks, workers=workers)
    executor.run()
    cleaned_dir = out_dir / "cleaned"
    print(f"clean pass complete -> {cleaned_dir}")

    # 4. Optional MinHash dedup
    text_dir = cleaned_dir
    if dedup:
        dedup_dir = out_dir / "dedup"
        dt.run_minhash_dedup(str(cleaned_dir), str(dedup_dir),
                             kind=executor_kind, tasks=tasks, workers=workers,
                             logging_dir=str(logging_dir / "dedup"))
        text_dir = dedup_dir / "deduplicated"
        print(f"dedup complete -> {text_dir}")

    # 5. Consolidate into cleaned.jsonl and calculate stats
    cleaned_jsonl = out_dir / "cleaned.jsonl"
    n_cleaned, lang_mix, total_bytes = consolidate_cleaned(text_dir, cleaned_jsonl)

    # Extract source count from stats if possible
    n_source = n_cleaned
    stats_file = logging_dir / "clean" / "stats.json"
    if stats_file.exists():
        try:
            stats_data = json.loads(stats_file.read_text())
            for step in stats_data:
                if "READER" in step.get("name", ""):
                    doc_stats = step.get("stats", {}).get("documents", {})
                    if isinstance(doc_stats, dict) and "total" in doc_stats:
                        n_source = int(doc_stats["total"])
                    elif "documents" in step.get("stats", {}):
                        n_source = int(step["stats"]["documents"])
                    break
        except Exception:
            pass
    if limit > 0 and n_source > limit:
        n_source = limit

    n_dropped = max(0, n_source - n_cleaned)
    filter_rate = {
        "n_source": n_source,
        "n_cleaned": n_cleaned,
        "n_dropped": n_dropped,
        "drop_rate": round(n_dropped / n_source, 4) if n_source > 0 else 0.0,
    }

    decontam_info = {
        "applied": decon is not None,
        "blocklist": str(decontam_path.relative_to(REPO_ROOT)) if decontam_path and decontam_path.is_relative_to(REPO_ROOT) else str(decontam_path) if decontam_path else None,
        "eval_sets": ["eval_sets/humaneval_ts", "eval_sets/ts_error_injection"] if decon is not None else [],
    }

    result = {
        "source": source,
        "cleaned_jsonl": str(cleaned_jsonl),
        "n_cleaned": n_cleaned,
        "total_bytes": total_bytes,
        "language_mix": lang_mix,
        "filter_rate": filter_rate,
        "decontamination": decontam_info,
    }

    # 6. Packing
    if pack:
        shards_dir = Path(shards_out or (out_dir / "shards"))
        tok_bin = find_monica_tokenize(tokenize_bin)
        tok_file = find_default_tokenizer(tokenizer)
        if tok_bin is not None and tok_file is None:
            raise FileNotFoundError(
                f"Tokenizer JSON file not found (given: {tokenizer}). Provide --tokenizer <path>."
            )

        # If repository projects were extracted, write repo manifest JSONL
        if repo_projects_list and active_repo_manifest is None:
            # Map cleaned texts into repo projects
            cleaned_map = {}
            for doc in iter_cleaned_docs(text_dir):
                md = doc.get("metadata") or {}
                p = doc.get("path") or md.get("path")
                t = doc.get("text")
                if p and t:
                    cleaned_map[p] = t
            cleaned_repos = []
            for rp in repo_projects_list:
                rf = [f for f in rp["files"] if f["path"] in cleaned_map]
                if rf:
                    cleaned_repos.append({
                        "repo": rp["repo"],
                        "files": [{"path": f["path"], "content": cleaned_map[f["path"]]} for f in rf],
                    })
            if cleaned_repos:
                active_repo_manifest = out_dir / "repo_manifest.jsonl"
                with open(active_repo_manifest, "w", encoding="utf-8") as rmf:
                    for cr in cleaned_repos:
                        rmf.write(json.dumps(cr) + "\n")

        manifest = pack_cleaned_shards(
            cleaned_jsonl, shards_dir, tok_file if tok_file else Path("dummy"),
            seq_len=seq_len, shard_size_mb=shard_size_mb,
            tokenize_bin=tok_bin,
            fim_rate=fim_rate,
            fim_seed=fim_seed,
            fim_mode=fim_mode,
            repo_manifest=active_repo_manifest,
        )

        # Enrich manifest with language mix, filter rate, and decontamination provenance
        manifest["language_mix"] = lang_mix
        manifest["filter_rate"] = filter_rate
        manifest["decontamination"] = decontam_info
        manifest["total_raw_bytes"] = total_bytes
        manifest["fim_mode"] = fim_mode
        manifest["fim_rate"] = fim_rate
        manifest["repo_dag_sorted"] = dag_sort
        if repo_projects_list:
            manifest["repos"] = [r["repo"] for r in repo_projects_list]
        if tok_file:
            manifest["tokenizer_file"] = tok_file.name

        (shards_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        result["shards_dir"] = str(shards_dir)
        result["manifest"] = manifest

        # 7. Validation split
        if val_tokens:
            from src.data.split import split_shards
            split_dir = shards_dir / "split"
            tr_path, va_path = split_shards(shards_dir, split_dir, val_tokens=val_tokens)
            result["split"] = {
                "train_bin": str(tr_path),
                "val_bin": str(va_path),
                "val_tokens": val_tokens,
            }
            print(f"validation split ({val_tokens} tokens) -> {split_dir}")

        # 8. Optional Cloudflare R2 mirror (#370 Stage 2)
        if r2_sync_uri:
            from src.data.r2_sync import sync_up
            sync_up(shards_dir, r2_sync_uri)
            result["r2_sync_uri"] = r2_sync_uri
            print(f"synced packed shards to R2 -> {r2_sync_uri}")

        print(f"packed {manifest.get('n_sequences', 0)} seq x {seq_len} "
              f"({manifest.get('n_tokens', 0)} tokens) -> {shards_dir}")

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("fineweb-edu", "jsonl", "sample", "repo", "stack-v2", "essential-web", "pretrain-extract"), default="sample",
                    help="corpus source (fineweb-edu, local jsonl, sample mixture, repo, stack-v2, essential-web, or pretrain-extract; #70/#252/#370/#418)")
    ap.add_argument("--extract-raw", action="store_true",
                    help="extract raw uncompressed documents to <out> with manifest (#418)")
    ap.add_argument("--stack-v2-limit", type=int, default=1000,
                    help="max Stack v2 documents to extract (default 1000; -1 for no cap)")
    ap.add_argument("--essential-web-limit", type=int, default=1000,
                    help="max Essential-Web documents to extract (default 1000; -1 for no cap)")
    ap.add_argument("--from-jsonl", default=None,
                    help="path to input JSONL file or directory (for --source jsonl or sample)")
    ap.add_argument("--from-repo", default=None,
                    help="path to repository directory or jsonl for --source repo")
    ap.add_argument("--repo-name", default=None,
                    help="repository name for --source repo")
    ap.add_argument("--dag-sort", action=argparse.BooleanOptionalAction, default=True,
                    help="topologically sort multi-file repository dependencies (#359)")
    ap.add_argument("--repo-manifest", default=None,
                    help="path to custom repository manifest JSONL for packing")
    ap.add_argument("--fim-mode", choices=("psm", "spm", "joint"), default="joint",
                    help="FIM sentinel ordering mode: psm, spm, or joint (#358)")
    ap.add_argument("--fim-rate", type=float, default=0.5,
                    help="probability of FIM transformation on code documents/files (default 0.5; #215/#358)")
    ap.add_argument("--fim-seed", type=int, default=42,
                    help="deterministic seed for FIM transformations (default 42)")
    ap.add_argument("--r2-sync", default=None,
                    help="destination Cloudflare R2 / S3 URI to mirror packed shards (docs/infrastructure.md)")
    ap.add_argument("--out", required=True,
                    help="output directory prefix (writes <out>/cleaned, <out>/cleaned.jsonl)")
    ap.add_argument("--limit", type=int, default=-1, help="max docs to read (-1 = no cap)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--executor", choices=("local", "slurm"), default="local")
    ap.add_argument("--tasks", type=int, default=1, help="parallel tasks (shards)")
    ap.add_argument("--workers", type=int, default=1, help="concurrent local workers")
    ap.add_argument("--logging-dir", default=None, help="datatrove logs/stats (default <out>/logs)")
    # Stage-3 filters
    ap.add_argument("--quality", action="store_true", help="text-quality heuristics")
    ap.add_argument("--license-filter", action="store_true", help="permissive-only gate on code")
    ap.add_argument("--drop-minified", action="store_true")
    ap.add_argument("--drop-autogen", action="store_true")
    ap.add_argument("--scrub", action="store_true", help="redact secrets/PII")
    ap.add_argument("--decontam", action="store_true",
                    help="filter docs overlapping eval benchmarks (default blocklist: eval_sets/decontam/blocklist.txt)")
    ap.add_argument("--decontam-file", default=None,
                    help="custom text file of eval-benchmark lines to strip (13/7-gram overlap)")
    # Stage-4 dedup
    ap.add_argument("--dedup", action="store_true", help="run cross-source MinHash dedup after clean")
    # Stage-5 tokenization & packing
    ap.add_argument("--pack", action="store_true",
                    help="tokenize and pack cleaned docs into .bin + .bounds + manifest.json shards")
    ap.add_argument("--tokenizer", default=None,
                    help="path to tokenizer.json / vocab-32768.json for monica-tokenize pack")
    ap.add_argument("--shards-out", default=None,
                    help="output directory for packed shards (default: <out>/shards)")
    ap.add_argument("--seq-len", type=int, default=8192,
                    help="packed sequence length (default: 8192)")
    ap.add_argument("--shard-size-mb", type=int, default=512,
                    help="packed shard size budget in MB (default: 512)")
    ap.add_argument("--val-tokens", type=int, default=None,
                    help="carve out a validation split with split_shards")
    ap.add_argument("--tokenize-bin", default=None,
                    help="path to monica-tokenize binary")
    args = ap.parse_args()

    if args.extract_raw or args.source == "pretrain-extract":
        from src.data import datatrove_pipeline as dt
        manifest = dt.run_raw_extraction(
            out_uri=args.out,
            stack_v2_limit=args.stack_v2_limit if args.source != "essential-web" else 0,
            essential_web_limit=args.essential_web_limit if args.source != "stack-v2" else 0,
            executor_kind=args.executor,
            tasks=args.tasks,
            workers=args.workers,
            logging_dir=args.logging_dir,
        )
        print(f"Raw extraction completed -> {args.out}")
        print(f"Total documents: {manifest.get('document_count', 0)}, Total volume: {manifest.get('total_volume_bytes', 0)} bytes")
        return

    run_corpus_pipeline(
        source=args.source,
        out_dir=args.out,
        from_jsonl=args.from_jsonl,
        limit=args.limit,
        split=args.split,
        executor_kind=args.executor,
        tasks=args.tasks,
        workers=args.workers,
        logging_dir=args.logging_dir,
        quality=args.quality,
        license_filter=args.license_filter,
        drop_minified=args.drop_minified,
        drop_autogen=args.drop_autogen,
        scrub=args.scrub,
        decontam=args.decontam or (args.decontam_file is not None),
        decontam_file=args.decontam_file,
        dedup=args.dedup,
        pack=args.pack,
        tokenizer=args.tokenizer,
        shards_out=args.shards_out,
        seq_len=args.seq_len,
        shard_size_mb=args.shard_size_mb,
        val_tokens=args.val_tokens,
        tokenize_bin=args.tokenize_bin,
        from_repo=args.from_repo,
        repo_name=args.repo_name,
        dag_sort=args.dag_sort,
        repo_manifest=args.repo_manifest,
        fim_mode=args.fim_mode,
        fim_rate=args.fim_rate,
        fim_seed=args.fim_seed,
        r2_sync_uri=args.r2_sync,
    )

    # A streaming HF reader truncated by --limit leaves a non-daemon prefetch thread alive, which
    # hangs the interpreter at teardown after all shards/markers are already flushed. Force a clean
    # exit ONLY in that case.
    if args.source == "fineweb-edu" and args.limit and args.limit > 0:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
