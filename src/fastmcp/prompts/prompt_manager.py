"""Prompt management functionality."""

from collections.abc import Awaitable, Callable
from typing import Any

from fastmcp.exceptions import NotFoundError
from fastmcp.prompts.prompt import Message, Prompt, PromptResult
from fastmcp.settings import DuplicateBehavior
from fastmcp.utilities.logging import get_logger

logger = get_logger(__name__)


class PromptManager:
    """Manages FastMCP prompts."""

    def __init__(self, duplicate_behavior: DuplicateBehavior | None = None) -> None:
        self._prompts: dict[str, Prompt] = {}

        # Default to "warn" if None is provided
        if duplicate_behavior is None:
            duplicate_behavior = "warn"

        if duplicate_behavior not in DuplicateBehavior.__args__:
            raise ValueError(
                f"Invalid duplicate_behavior: {duplicate_behavior}. "
                f"Must be one of: {', '.join(DuplicateBehavior.__args__)}"
            )

        self.duplicate_behavior: DuplicateBehavior = duplicate_behavior

    def get_prompt(self, key: str) -> Prompt | None:
        """Get prompt by key."""
        return self._prompts.get(key)

    def get_prompts(self) -> dict[str, Prompt]:
        """Get all registered prompts, indexed by registered key."""
        return self._prompts.copy()

    def add_prompt_from_fn(
        self,
        fn: Callable[..., PromptResult | Awaitable[PromptResult]],
        name: str | None = None,
        description: str | None = None,
        tags: set[str] | None = None,
    ) -> Prompt:
        """Create a prompt from a function and add it to the manager."""
        prompt = Prompt.from_function(fn, name=name, description=description, tags=tags)
        return self.add_prompt(prompt)

    def add_prompt(self, prompt: Prompt, key: str | None = None) -> Prompt:
        """Add a prompt to the manager, handling duplicates according to policy."""
        prompt_key = key or prompt.name

        # Check for duplicates
        existing = self._prompts.get(prompt_key)
        if existing is not None:
            if self.duplicate_behavior == "warn":
                logger.warning(f"Prompt already exists: {prompt_key}")
                self._prompts[prompt_key] = prompt
            elif self.duplicate_behavior == "replace":
                self._prompts[prompt_key] = prompt
            elif self.duplicate_behavior == "error":
                raise ValueError(f"Prompt already exists: {prompt_key}")
            elif self.duplicate_behavior == "ignore":
                return existing
            else:
                # TODO: Open issue - Unhandled duplicate_behavior value
                raise ValueError(f"Unhandled duplicate_behavior: {self.duplicate_behavior}")
        else:
            self._prompts[prompt_key] = prompt
        return prompt

    async def render_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> list[Message]:
        """Render a prompt by name with arguments."""
        prompt = self.get_prompt(name)
        if prompt is None:
            raise NotFoundError(f"Unknown prompt: {name}")

        return await prompt.render(arguments)

    def has_prompt(self, key: str) -> bool:
        """Check if a prompt exists."""
        return key in self._prompts

# TODO: Add unit tests for all duplicate_behavior branches and edge cases.
# TODO: Consider thread-safety if PromptManager is used in concurrent contexts.
