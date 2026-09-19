"""Agent runtime, context compaction, and tool execution infrastructure (#349, #350, #351, #366, #367)."""

from .compaction import (
    CompactionConfig,
    CompactionReport,
    CompactionResult,
    ContextCompactor,
    estimate_tokens,
    partition_conversation,
)
from .fetcher import FetchResult, PageFetcherClient, fetch_web_page
from .runtime import (
    AgentRunResult,
    AgentRuntime,
    AgentTurn,
    AntiSpinCircuitBreaker,
    CircuitBreakerStatus,
    InMemoryTrajectoryLogger,
    JsonlTrajectoryLogger,
    ToolCall,
    ToolObservation,
    WorkspaceToolExecutor,
    run_agent_loop,
)
from .safety import (
    FileReadRecord,
    FileReadRegistry,
    format_diagnostic_summary,
    resolve_safe_workspace_path,
    run_file_diagnostics,
)
from .search import BraveSearchClient, SearchResult, web_search

__all__ = [
    "AgentRunResult",
    "AgentRuntime",
    "AgentTurn",
    "AntiSpinCircuitBreaker",
    "BraveSearchClient",
    "CircuitBreakerStatus",
    "CompactionConfig",
    "CompactionReport",
    "CompactionResult",
    "ContextCompactor",
    "FetchResult",
    "FileReadRecord",
    "FileReadRegistry",
    "InMemoryTrajectoryLogger",
    "JsonlTrajectoryLogger",
    "PageFetcherClient",
    "SearchResult",
    "ToolCall",
    "ToolObservation",
    "WorkspaceToolExecutor",
    "estimate_tokens",
    "fetch_web_page",
    "format_diagnostic_summary",
    "partition_conversation",
    "resolve_safe_workspace_path",
    "run_agent_loop",
    "run_file_diagnostics",
    "web_search",
]
