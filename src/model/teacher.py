"""The conversion-teacher seam: a frozen, forward-only teacher protocol.

THIS MODULE MUST NOT IMPORT ANY BACKEND (no `mlx`, no `torch`/CUDA). Like
`interface.ModelInterface`, it is the portable contract that the distillation
stages depend on; each backend provides a concrete `ConversionTeacher` (the MLX
one is `mlx_teacher.MLXConversionTeacher`).

The conversion teacher is the transformer the student is *built from*
(`Qwen/Qwen3-4B-Thinking-2507` by default — a thinking-mode reasoning model on the
token-aligned Qwen3 vocab; see `docs/design/10-distillation.md`).
It is run **forward only** — never trained — so it follows the same frozen-reference
pattern DPO already uses (`mlx_train_step.make_dpo_train_step` holds a distinct
`ref_model` the optimizer never touches). Two distillation issues consume it:

  * student init (#99) maps the teacher's attention Q/K/V/O onto the student SSM's
    C/B/input/output projections — hence `attention_projection`.
  * the distill loss (#100) matches the student against the teacher's top-k logits
    and optional hidden states — hence `forward(return_hidden=...)` / `topk_logits`.

Per the seam, portable code sees only opaque arrays plus a `to_numpy` converter and
these accessors — never the backend array type. The teacher reports **no** trainable
parameters (`trainable_parameters() == {}`), so it is structurally excluded from any
optimizer / resume bundle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Opaque, backend-defined arrays (an MLX array, a torch tensor, ...). Code above the
# seam treats these as blobs it converts with `to_numpy`.
Array = Any


@dataclass
class TeacherConfig:
    """Architecture of a Qwen2/Qwen3-family conversion teacher.

    The default teacher is `Qwen/Qwen3-4B-Thinking-2507` (a Qwen3 decoder — see
    `qwen3_4b_thinking`). The older Qwen2 fixtures `openr1_distill_7b` /
    `qwen_1_5b` are retained for back-compat. The fields are exactly what the
    forward pass and the #99 projection mapping need; nothing here imports a
    backend, so the config is shared across backends like `MambaConfig`.

    Qwen3 differs from Qwen2 in two ways the backend forward honors: per-head
    RMSNorm on Q and K before RoPE, and **no** QKV bias (so a Qwen3 teacher's
    `AttnProjections` carry `None` biases). This config does not encode those
    flags — the backend keys off the absence of bias/norm weights in its
    checkpoint mapping — but `tie_embeddings` (true for Qwen3-4B) is a field.
    """

    vocab_size: int              # MODEL embedding width (may be padded above the tokenizer vocab)
    d_model: int                 # hidden_size
    n_layers: int                # num_hidden_layers
    n_heads: int                 # num_attention_heads (query heads)
    n_kv_heads: int              # num_key_value_heads (GQA groups; == n_heads if MHA)
    head_dim: int                # per-head width (Qwen2-1.5B: 128, not d_model//n_heads)
    intermediate_size: int       # SwiGLU MLP inner width
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    model_id: Optional[str] = None   # HF repo id, for from_pretrained / provenance
    # The TOKENIZER vocab, when the model embedding is padded above it (Qwen2.5: model 151936,
    # tokenizer 151646). None => no padding (use `vocab_size`). `forward`/`topk_logits` expose
    # logits/indices over `effective_vocab_size`, so a student with the tokenizer vocab can
    # consume them — the padded rows are never emitted as teacher targets.
    tokenizer_vocab_size: Optional[int] = None

    @classmethod
    def qwen3_4b_thinking(cls) -> "TeacherConfig":
        """`Qwen/Qwen3-4B-Thinking-2507` — the default conversion teacher.

        Qwen's 4B thinking-mode reasoning model (Apache-2.0). It carries
        thinking-mode CoT (the main above-weight-class lever for the student) and
        sits on the unified Qwen3 BPE vocab, which is token-aligned with Qwen2.5 —
        so logit KD stays aligned (Qwen3.5-4B was rejected: it moved to a new 250k
        vocab that breaks aligned KD and bloats the student embedding).

        Architecture (Qwen3): vocab 151936 is the padded *model* embedding; the
        Qwen3 tokenizer vocab is 151669 (Qwen2.5's 151646 plus a few added control
        tokens incl. the `<think>`/`</think>` pair). `head_dim` 128 is NOT
        `d_model // n_heads` (2560/32 = 80), so the explicit field matters
        (`q_dim` 4096, `kv_dim` 1024). Qwen3 has per-head Q/K RMSNorm before RoPE
        and no QKV bias; `from_hf_dict` reads the authoritative values from the
        checkpoint's `config.json` at load — these are the offline fixture.
        """
        return cls(
            vocab_size=151936, d_model=2560, n_layers=36, n_heads=32, n_kv_heads=8,
            head_dim=128, intermediate_size=9728, rms_norm_eps=1e-6, rope_theta=5000000.0,
            tie_embeddings=True, model_id="Qwen/Qwen3-4B-Thinking-2507",
            tokenizer_vocab_size=151669,   # Qwen3 tokenizer vocab; padded rows 151669..151936 unused
        )

    @classmethod
    def openr1_distill_7b(cls) -> "TeacherConfig":
        """`open-r1/OpenR1-Distill-7B` — the default, fully-open conversion teacher.

        HuggingFace's Open-R1 reproduction of R1 distillation: SFT of
        Qwen2.5-Math-7B on the openly released Mixture-of-Thoughts / OpenR1-Math-220k
        / CodeForces-CoTs reasoning traces (open data + recipe, Apache-2.0). It keeps
        the Qwen2 architecture and Qwen2.5 tokenizer, so it is a drop-in for the MLX
        Qwen2 forward; Open-R1 extends RoPE theta to 300k for a 32k context.

        Dims are the Qwen2.5-Math-7B architecture (vocab 152064 is the padded *model*
        embedding; the Qwen2.5 tokenizer vocab is 151646). `from_hf_dict` reads the
        authoritative values from the checkpoint's `config.json` at load time; these
        match that config and serve as the offline fixture.

        NOTE: HF `config.json` carries no `tokenizer_vocab_size`, so a teacher built
        via `from_pretrained(..., config=None)` gets `from_hf_dict`'s default
        (`tokenizer_vocab_size=None` -> `effective_vocab_size == vocab_size`, 152064)
        and would emit logits/top-k over the padded rows the student vocab (151646)
        cannot represent. When loading real weights, pass this config (or otherwise set
        `tokenizer_vocab_size`) so outputs are sliced to the tokenizer vocab.
        """
        return cls(
            vocab_size=152064, d_model=3584, n_layers=28, n_heads=28, n_kv_heads=4,
            head_dim=128, intermediate_size=18944, rms_norm_eps=1e-6, rope_theta=300000.0,
            tie_embeddings=False, model_id="open-r1/OpenR1-Distill-7B",
            tokenizer_vocab_size=151646,   # student/tokenizer vocab; padded rows 151646..152064 unused
        )

    @classmethod
    def qwen_1_5b(cls) -> "TeacherConfig":
        """DeepSeek-R1-Distill-Qwen-1.5B (== Qwen2.5-Math-1.5B architecture).

        Note vocab 151936 is the *model* embedding width (padded); the Qwen2.5
        tokenizer vocab is 151646 (`docs/design/10-distillation.md`). The student
        config uses 151646; the distill loss (#100) matches on top-k indices, which
        fall inside the shared lower range.
        """
        return cls(
            vocab_size=151936, d_model=1536, n_layers=28, n_heads=12, n_kv_heads=2,
            head_dim=128, intermediate_size=8960, rms_norm_eps=1e-6, rope_theta=10000.0,
            tie_embeddings=True, model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            tokenizer_vocab_size=151646,   # student/tokenizer vocab; padded rows 151646..151936 unused
        )

    @classmethod
    def tiny(cls) -> "TeacherConfig":
        """A toy Qwen2 teacher for offline tests / small local checks (byte vocab)."""
        return cls(
            vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
            head_dim=16, intermediate_size=128, rms_norm_eps=1e-6, rope_theta=10000.0,
            tie_embeddings=True, model_id=None,
        )

    @classmethod
    def from_hf_dict(cls, hf: Dict[str, Any]) -> "TeacherConfig":
        """Build from a HuggingFace Qwen2/Qwen3 `config.json` dict (same field schema)."""
        n_heads = int(hf["num_attention_heads"])
        d_model = int(hf["hidden_size"])
        head_dim = int(hf.get("head_dim", d_model // n_heads))
        return cls(
            vocab_size=int(hf["vocab_size"]),
            d_model=d_model,
            n_layers=int(hf["num_hidden_layers"]),
            n_heads=n_heads,
            n_kv_heads=int(hf.get("num_key_value_heads", n_heads)),
            head_dim=head_dim,
            intermediate_size=int(hf["intermediate_size"]),
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-6)),
            rope_theta=float(hf.get("rope_theta", 10000.0)),
            tie_embeddings=bool(hf.get("tie_word_embeddings", True)),
            model_id=hf.get("_name_or_path"),
        )

    @property
    def effective_vocab_size(self) -> int:
        """The vocab the teacher emits logits/top-k over: the tokenizer vocab when the model
        embedding is padded above it, otherwise the full `vocab_size`."""
        return self.tokenizer_vocab_size or self.vocab_size

    @property
    def q_dim(self) -> int:
        """Width of the concatenated query projection (n_heads * head_dim)."""
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        """Width of each of the key/value projections (n_kv_heads * head_dim)."""
        return self.n_kv_heads * self.head_dim

    def validate(self) -> None:
        if self.n_kv_heads <= 0 or self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_kv_heads={self.n_kv_heads} must divide n_heads={self.n_heads} (GQA)."
            )
        if self.head_dim <= 0 or self.head_dim % 2 != 0:
            raise ValueError(f"head_dim={self.head_dim} must be positive and even (RoPE).")
        for name in ("vocab_size", "d_model", "n_layers", "intermediate_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.tokenizer_vocab_size is not None and not (
                0 < self.tokenizer_vocab_size <= self.vocab_size):
            raise ValueError(
                f"tokenizer_vocab_size={self.tokenizer_vocab_size} must be in "
                f"(0, vocab_size={self.vocab_size}].")


@dataclass
class AttnProjections:
    """One layer's attention projection weights, for the #99 init mapping.

    Weights are opaque backend arrays in the HF row-major convention (`y = x @ W.T`):
    `q` is (q_dim, d_model), `k`/`v` are (kv_dim, d_model), `o` is (d_model, q_dim).
    Qwen2 has biases on q/k/v and none on o; **Qwen3 has no QKV bias**, so a Qwen3
    teacher leaves the `*_bias` fields `None`. Qwen3's per-head Q/K RMSNorm is part
    of the frozen attention block and is NOT carried here — the #99 init maps only
    the Q/K/V/O projections onto the student SSM's C/B/input/output (so a kept-and-
    frozen attention layer drops the teacher's QK-norm; acceptable for the POC, the
    init is judged by the distillation curve, not exactness).
    """

    q: Array
    k: Array
    v: Array
    o: Array
    q_bias: Optional[Array] = None
    k_bias: Optional[Array] = None
    v_bias: Optional[Array] = None


@dataclass
class TeacherForward:
    """Result of a teacher forward pass. `logits` is (batch, seq, vocab).

    `hidden_states`, when requested, is a tuple of length `n_layers + 1`: the embedding
    output followed by each decoder layer's output, each (batch, seq, d_model) — the HF
    `output_hidden_states` convention MOHAWK hidden-alignment (#100) consumes.
    """

    logits: Array
    hidden_states: Optional[Tuple[Array, ...]] = None


class ConversionTeacher(ABC):
    """Frozen forward-only teacher contract. Concrete impls live below the seam."""

    #: Architecture, the single source of truth for the teacher's shapes.
    config: TeacherConfig

    @property
    def n_layers(self) -> int:
        return self.config.n_layers

    @abstractmethod
    def forward(self, token_batch: Array, *, return_hidden: bool = False) -> TeacherForward:
        """Forward over `token_batch` (batch, seq) ids. Returns logits (+ hidden states).

        No gradient flows into the teacher's weights: the implementation wraps its
        outputs in the backend's stop-gradient so the teacher stays frozen even when
        its logits/hidden states are composed into a student loss.
        """

    @abstractmethod
    def topk_logits(self, token_batch: Array, k: int) -> Tuple[Array, Array]:
        """Top-`k` logits over the vocab. Returns (values, indices), each (batch, seq, k),
        values descending. This is the per-token teacher signal cached by the precompute
        (#94) and matched with KL by the distill loss (#100)."""

    @abstractmethod
    def attention_projection(self, layer: int) -> AttnProjections:
        """The Q/K/V/O projection weights of decoder `layer` (0-indexed), for #99."""

    @abstractmethod
    def embedding_matrix(self) -> Array:
        """The token-embedding matrix `(vocab_size, d_model)`, for the #99 init.

        Shape is `(vocab_size, d_model)` — the *full padded* model vocab, NOT pre-clipped
        to `effective_vocab_size` (padded rows are at the end, above the tokenizer vocab).
        The student init crops it to the student shape with `_fit`, so the row-crop drops
        exactly the padding and the column-crop selects the student's residual subspace;
        an impl that pre-clipped would make `_fit` crop live rows. Opaque,
        stop-gradient-wrapped (frozen teacher)."""

    @abstractmethod
    def lm_head_matrix(self) -> Array:
        """The output (unembedding) matrix `(vocab_size, d_model)`, for the #99 init.

        Equals `embedding_matrix()` when the teacher ties embeddings. Same convention and
        cropping contract as `embedding_matrix`."""

    @abstractmethod
    def to_numpy(self, array: Array) -> Any:
        """Convert an opaque teacher array to a numpy array (the seam converter)."""

    def trainable_parameters(self) -> Dict[str, Array]:
        """The frozen contract: a conversion teacher has NO trainable parameters, so it
        is excluded from the optimizer and the resume bundle. Overridable, but the empty
        default is what makes the freeze structural rather than merely conventional."""
        return {}
