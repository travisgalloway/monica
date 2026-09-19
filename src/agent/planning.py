"""Capability-adaptive planning scaffolding with out-of-history plan injection (#352).

Empirical findings in arXiv:2609.20804 uncover an inverse relationship between model
capability and planning utility:
- For sub-frontier models (<100B), planning is an indispensable accuracy scaffold
  (+11.6% SWE-Bench), preventing early task abandonment during initial code localization.
- For frontier models, planning functions as an exit-condition / checklist gate,
  cutting 30%–40% in API costs by suppressing wandering post-edit verification loops.
- To prevent context contamination, plans are maintained in harness state and injected
  as clean prompt blocks rather than accumulated as noisy message history.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class PlanningPolicy(str, Enum):
    """Capability-adaptive planning scaffolding policies (#352).

    - SUB_FRONTIER: For models <100B params. Enforces mandatory plan generation on turn 1
      and tracks progress to prevent premature task aborts.
    - FRONTIER: For frontier models. Configures the plan as a completion exit gate
      terminating cleanly upon checklist completion to suppress wandering verification loops.
    - ADAPTIVE: Automatically selects SUB_FRONTIER or FRONTIER based on model size / name.
    - NONE: Planning scaffolding disabled.
    """

    SUB_FRONTIER = "sub_frontier"
    FRONTIER = "frontier"
    ADAPTIVE = "adaptive"
    NONE = "none"


def resolve_planning_policy(
    policy: str | PlanningPolicy,
    model_name: str | None = None,
    model_size_b: float | None = None,
) -> PlanningPolicy:
    """Resolve an adaptive or string planning policy into a concrete policy enum."""
    if isinstance(policy, str):
        try:
            resolved = PlanningPolicy(policy.lower())
        except ValueError:
            resolved = PlanningPolicy.ADAPTIVE
    else:
        resolved = policy

    if resolved != PlanningPolicy.ADAPTIVE:
        return resolved

    if model_size_b is not None:
        if model_size_b < 100.0:
            return PlanningPolicy.SUB_FRONTIER
        return PlanningPolicy.FRONTIER

    if model_name is not None:
        sub_pattern = r"(?i)\b(7b|8b|14b|32b|70b|small|mini|flash|haiku|sub[-_]?frontier)\b"
        frontier_pattern = r"(?i)(frontier|gpt[-_]?4|claude[-_]?3|claude[-_]?sonnet|claude[-_]?opus|o1|o3|gemini[-_]?1\.5[-_]?pro|gemini[-_]?pro|405b)"
        if re.search(frontier_pattern, model_name):
            return PlanningPolicy.FRONTIER
        if re.search(sub_pattern, model_name):
            return PlanningPolicy.SUB_FRONTIER

    # Default for adaptive when unspecified is SUB_FRONTIER (<100B scaffold)
    return PlanningPolicy.SUB_FRONTIER


@dataclass
class PlanItem:
    """Individual checklist step in an agent execution plan."""

    index: int
    description: str
    completed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "description": self.description,
            "completed": self.completed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanItem:
        return cls(
            index=int(data["index"]),
            description=str(data["description"]),
            completed=bool(data.get("completed", False)),
        )

    def render(self) -> str:
        mark = "[x]" if self.completed else "[ ]"
        return f"- {mark} {self.index}. {self.description}"


def parse_plan_markdown(text: str) -> list[PlanItem]:
    """Parse a markdown checklist or numbered list into structured PlanItem objects."""
    items: list[PlanItem] = []
    lines = text.strip().split("\n")
    item_idx = 1

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Skip headers like "# Plan", "## Checklist", "**Plan:**"
        if stripped.startswith("#") or (stripped.startswith("**") and stripped.endswith("**")):
            continue

        # Match checkbox pattern: - [ ] 1. desc or - [x] desc or 1. [ ] desc
        cb_match = re.match(
            r"^[-*•]?\s*(?:(\d+)[\.\)]\s*)?\[([ xX])\]\s*(?:(\d+)[\.\)]\s*)?(.*)$",
            stripped,
        )
        if cb_match:
            is_checked = cb_match.group(2).lower() == "x"
            desc = cb_match.group(4).strip()
            # Clean any leftover leading numbers or symbols
            desc = re.sub(r"^\d+[\.\)]\s*", "", desc)
            if desc:
                items.append(PlanItem(index=item_idx, description=desc, completed=is_checked))
                item_idx += 1
            continue

        # Match numbered or bulleted line: 1. desc or - desc
        list_match = re.match(r"^[-*•]?\s*(?:(\d+)[\.\)]\s*)?(.*)$", stripped)
        if list_match:
            desc = list_match.group(2).strip()
            if desc and not desc.startswith("["):
                items.append(PlanItem(index=item_idx, description=desc, completed=False))
                item_idx += 1

    return items


class PlanManager:
    """External harness plan state manager with out-of-history prompt conditioning (#352).

    Maintains execution plans in harness state and renders clean {PLAN} blocks into agent
    prompt conditioning without accumulating noisy conversational planning turns in history.
    Enforces capability-adaptive policies:
    - Sub-frontier: Turn 1 mandatory plan generation and premature task abort prevention.
    - Frontier: Completion exit gate that terminates execution cleanly when all items are verified.
    """

    def __init__(
        self,
        policy: str | PlanningPolicy = PlanningPolicy.ADAPTIVE,
        *,
        model_name: str | None = None,
        model_size_b: float | None = None,
        initial_plan: str | list[str] | list[PlanItem] | None = None,
        exit_gate_enabled: bool = True,
        prevent_premature_abort: bool = True,
        enforce_turn1_plan: bool = True,
        injection_target: str = "system",
        max_abort_retries: int = 3,
    ) -> None:
        self.policy = policy
        self.model_name = model_name
        self.model_size_b = model_size_b
        self.exit_gate_enabled = bool(exit_gate_enabled)
        self.prevent_premature_abort = bool(prevent_premature_abort)
        self.enforce_turn1_plan = bool(enforce_turn1_plan)
        self.injection_target = injection_target
        self.max_abort_retries = int(max_abort_retries)

        self.initial_plan = initial_plan
        self.items: list[PlanItem] = []
        self.history: list[dict[str, Any]] = []
        self.abort_retry_count: int = 0

        if initial_plan is not None:
            self.set_plan(initial_plan)

    @property
    def effective_policy(self) -> PlanningPolicy:
        """Resolve current active policy accounting for adaptive model capability."""
        return resolve_planning_policy(
            self.policy,
            model_name=self.model_name,
            model_size_b=self.model_size_b,
        )

    @property
    def has_plan(self) -> bool:
        """True if one or more plan checklist items have been established."""
        return len(self.items) > 0

    @property
    def is_complete(self) -> bool:
        """True if a plan exists and every checklist item is marked completed ([x])."""
        return self.has_plan and all(item.completed for item in self.items)

    @property
    def completed_count(self) -> int:
        return sum(1 for item in self.items if item.completed)

    @property
    def total_count(self) -> int:
        return len(self.items)

    @property
    def remaining_count(self) -> int:
        return sum(1 for item in self.items if not item.completed)

    @property
    def remaining_items(self) -> list[PlanItem]:
        return [item for item in self.items if not item.completed]

    def reset(self) -> None:
        """Reset plan state and retry counters for a new execution session."""
        self.abort_retry_count = 0
        self.history.clear()
        if self.initial_plan is not None:
            self.set_plan(self.initial_plan)
        else:
            self.items.clear()

    def set_plan(self, plan: str | list[str] | list[PlanItem] | list[dict[str, Any]]) -> list[PlanItem]:
        """Initialize or replace the plan items."""
        new_items: list[PlanItem] = []

        if isinstance(plan, str):
            new_items = parse_plan_markdown(plan)
        elif isinstance(plan, list):
            for i, raw in enumerate(plan):
                if isinstance(raw, PlanItem):
                    new_items.append(PlanItem(index=i + 1, description=raw.description, completed=raw.completed))
                elif isinstance(raw, dict):
                    desc = raw.get("description", raw.get("desc", f"Step {i + 1}"))
                    done = bool(raw.get("completed", raw.get("done", False)))
                    new_items.append(PlanItem(index=i + 1, description=str(desc), completed=done))
                elif isinstance(raw, str):
                    if "- [" in raw or raw.strip().startswith("["):
                        parsed = parse_plan_markdown(raw)
                        new_items.extend(parsed)
                    else:
                        new_items.append(PlanItem(index=i + 1, description=raw.strip(), completed=False))

        # Ensure 1-based sequential indices
        for idx, it in enumerate(new_items):
            it.index = idx + 1

        self.items = new_items
        self.history.append({
            "action": "set_plan",
            "total_items": len(self.items),
            "completed_items": self.completed_count,
        })
        return self.items

    def update_step(self, step: int, completed: bool = True) -> PlanItem:
        """Update the checklist state of a single step by 1-based index ([ ] -> [x])."""
        for item in self.items:
            if item.index == step:
                old_state = item.completed
                item.completed = bool(completed)
                self.history.append({
                    "action": "update_step",
                    "step": step,
                    "old_completed": old_state,
                    "new_completed": item.completed,
                })
                return item

        raise ValueError(
            f"Step {step} not found in plan. Plan contains {len(self.items)} items (1..{len(self.items)})."
        )

    def execute_update_plan(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle invocations of the update_plan tool from the model."""
        try:
            # 1. Initialize or replace plan if 'plan' or 'steps' argument is provided
            if arguments.get("plan"):
                self.set_plan(arguments["plan"])
            elif "steps" in arguments and isinstance(arguments["steps"], list) and arguments["steps"]:
                self.set_plan(arguments["steps"])

            # 2. Apply batch updates if 'updates' argument is provided
            updates = arguments.get("updates")
            if isinstance(updates, list):
                for upd in updates:
                    if isinstance(upd, dict) and "step" in upd:
                        step_num = int(upd["step"])
                        is_comp = bool(upd.get("completed", True))
                        self.update_step(step_num, is_comp)

            # 3. Apply single step update if 'step' or 'step_index' is provided
            step_arg = arguments.get("step", arguments.get("step_index"))
            if step_arg is not None:
                step_num = int(step_arg)
                # Parse completed status
                if "completed" in arguments:
                    is_comp = bool(arguments["completed"])
                elif "status" in arguments:
                    status_str = str(arguments["status"]).lower()
                    is_comp = status_str in ("completed", "done", "finished", "pass", "ok")
                else:
                    is_comp = True
                self.update_step(step_num, is_comp)

            return {
                "status": "ok",
                "message": f"Plan updated: {self.completed_count}/{self.total_count} steps completed.",
                "plan": self.render_plan(),
                "completed_items": self.completed_count,
                "total_items": self.total_count,
                "all_completed": self.is_complete,
                "exit_gate_ready": self.is_complete if self.effective_policy == PlanningPolicy.FRONTIER else False,
            }
        except Exception as e:  # noqa: BLE001
            return {
                "status": "error",
                "error": f"Failed to update plan: {e}",
                "is_error": True,
            }

    def render_plan(self) -> str:
        """Render active plan checklist items as markdown."""
        if not self.items:
            return "[No active plan]"
        return "\n".join(item.render() for item in self.items)

    def render_plan_block(self) -> str:
        """Render the clean {PLAN} block injected into prompt conditioning."""
        lines = ["{PLAN}"]
        if not self.items:
            lines.append("[No active plan]")
            if self.effective_policy == PlanningPolicy.SUB_FRONTIER:
                lines.append("[MANDATORY PLAN GENERATION: Establish your initial checklist on turn 1 using update_plan.]")
            elif self.effective_policy == PlanningPolicy.FRONTIER:
                lines.append("[EXIT GATE ACTIVE: Establish verification checklist; terminate immediately upon completion.]")
        else:
            for item in self.items:
                lines.append(item.render())

            if self.effective_policy == PlanningPolicy.FRONTIER:
                if self.is_complete:
                    lines.append("\n[EXIT GATE: All verification items checked off. Terminate immediately with your final answer.]")
                else:
                    lines.append("\n[EXIT GATE ACTIVE: Terminate immediately once all verification items are checked off.]")
            elif self.effective_policy == PlanningPolicy.SUB_FRONTIER:
                if self.is_complete:
                    lines.append("\n[ALL PLAN ITEMS COMPLETED: Proceed to final answer.]")
                else:
                    lines.append(f"\n[SCAFFOLDING ACTIVE: {self.remaining_count} item(s) pending. Complete all items before finishing.]")

        return "\n".join(lines)

    def inject_plan(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Inject plan state out-of-history into prompt conditioning without mutating conversational turns."""
        plan_block = self.render_plan_block()
        conditioned = copy.deepcopy(messages)

        # 1. If {PLAN} placeholder is present in any message, substitute it
        replaced = False
        for msg in conditioned:
            content = msg.get("content", "")
            if "{PLAN}" in content:
                msg["content"] = content.replace("{PLAN}", plan_block)
                replaced = True

        # 2. If no placeholder was found, inject plan block into system message
        if not replaced and conditioned:
            if conditioned[0].get("role") == "system":
                sys_content = conditioned[0].get("content", "")
                conditioned[0]["content"] = f"{sys_content}\n\n{plan_block}"
            else:
                conditioned.insert(0, {"role": "system", "content": plan_block})

            # Optional injection into turn prompt (last user message)
            if self.injection_target in ("turn_prompt", "both") and len(conditioned) > 1:
                last_msg = conditioned[-1]
                if last_msg.get("role") == "user":
                    last_msg["content"] = f"{last_msg.get('content', '')}\n\n{plan_block}"

        return conditioned

    def should_exit_gate_terminate(self) -> bool:
        """Check if exit-gate policy triggers termination upon checklist completion (#352)."""
        if self.effective_policy == PlanningPolicy.FRONTIER:
            return self.exit_gate_enabled and self.is_complete
        return False

    def check_premature_abort(self, turn_idx: int) -> tuple[bool, str | None]:
        """Detect and prevent premature task aborts under sub-frontier scaffolding policy.

        Returns (should_prevent_abort, redirection_message).
        """
        if self.effective_policy != PlanningPolicy.SUB_FRONTIER or not self.prevent_premature_abort:
            return False, None

        # Turn 1: Enforce mandatory plan generation
        if turn_idx == 1 and not self.has_plan and self.enforce_turn1_plan:
            if self.abort_retry_count >= self.max_abort_retries:
                return False, None
            self.abort_retry_count += 1
            return True, (
                "[PLAN ENFORCEMENT] Premature task abort prevented. Mandatory plan generation "
                "is required on turn 1 before completing tasks. Establish a plan checklist "
                "using the update_plan tool."
            )

        # Subsequent turns: Prevent premature completion while checklist items remain incomplete
        if self.has_plan and not self.is_complete:
            if self.abort_retry_count >= self.max_abort_retries:
                return False, None
            self.abort_retry_count += 1
            remaining_str = "\n".join(item.render() for item in self.remaining_items)
            return True, (
                f"[PLAN ENFORCEMENT] Premature task abort prevented. {self.remaining_count} checklist "
                f"item(s) are still pending completion:\n{remaining_str}\n"
                "Complete all checklist items and update the plan states before finishing."
            )

        return False, None

    def get_turn1_reminder(self) -> str | None:
        """Reminder injected when turn 1 executed tools without establishing a mandatory plan."""
        if self.effective_policy == PlanningPolicy.SUB_FRONTIER and self.enforce_turn1_plan and not self.has_plan:
            return (
                "[PLAN REMINDER] Plan generation is mandatory on turn 1. "
                "Please call update_plan with your initial checklist."
            )
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize plan manager state for telemetry and evaluation."""
        return {
            "policy": self.policy.value if isinstance(self.policy, PlanningPolicy) else str(self.policy),
            "effective_policy": self.effective_policy.value,
            "has_plan": self.has_plan,
            "is_complete": self.is_complete,
            "completed_count": self.completed_count,
            "total_count": self.total_count,
            "items": [item.to_dict() for item in self.items],
            "history": list(self.history),
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
