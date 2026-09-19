"""Agent runtime, context compaction, and tool execution infrastructure (#349, #350, #351, #352, #366, #367)."""

from .compaction import (
    CompactionConfig,
    CompactionReport,
    CompactionResult,
    ContextCompactor,
    estimate_tokens,
    partition_conversation,
)
from .fetcher import FetchResult, PageFetcherClient, fetch_web_page
from .planning import (
    PlanItem,
    PlanManager,
    PlanningPolicy,
    parse_plan_markdown,
    resolve_planning_policy,
)
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
    TrajectoryTelemetry,
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
    "PlanItem",
    "PlanManager",
    "PlanningPolicy",
    "SearchResult",
    "ToolCall",
    "ToolObservation",
    "TrajectoryTelemetry",
    "WorkspaceToolExecutor",
    "estimate_tokens",
    "fetch_web_page",
    "format_diagnostic_summary",
    "parse_plan_markdown",
    "partition_conversation",
    "resolve_planning_policy",
    "resolve_safe_workspace_path",
    "run_agent_loop",
    "run_file_diagnostics",
    "web_search",
]
