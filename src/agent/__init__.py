"""Agent runtime and tool execution infrastructure (#349, #366, #367)."""

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
from .search import BraveSearchClient, SearchResult, web_search

__all__ = [
    "AgentRunResult",
    "AgentRuntime",
    "AgentTurn",
    "AntiSpinCircuitBreaker",
    "BraveSearchClient",
    "CircuitBreakerStatus",
    "FetchResult",
    "InMemoryTrajectoryLogger",
    "JsonlTrajectoryLogger",
    "PageFetcherClient",
    "SearchResult",
    "ToolCall",
    "ToolObservation",
    "WorkspaceToolExecutor",
    "fetch_web_page",
    "run_agent_loop",
    "web_search",
]
