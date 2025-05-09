from typing import Any, cast
from urllib.parse import quote

import mcp.types
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.types import (
    BlobResourceContents,
    EmbeddedResource,
    GetPromptResult,
    ImageContent,
    TextContent,
    TextResourceContents,
)
from pydantic.networks import AnyUrl

from fastmcp.client import Client
from fastmcp.exceptions import NotFoundError
from fastmcp.prompts import Message, Prompt
from fastmcp.resources import Resource, ResourceTemplate
from fastmcp.server.context import Context
from fastmcp.server.server import FastMCP
from fastmcp.tools.tool import Tool
from fastmcp.utilities.func_metadata import func_metadata
from fastmcp.utilities.logging import get_logger

logger = get_logger(__name__)


def _proxy_passthrough(*args, **kwargs):
    # This is a placeholder passthrough function for proxy objects.
    pass


class ProxyTool(Tool):
    def __init__(self, client: "Client", **kwargs):
        super().__init__(**kwargs)
        self._client = client

    @classmethod
    async def from_client(cls, client: "Client", tool: mcp.types.Tool) -> "ProxyTool":
        return cls(
            client=client,
            name=tool.name,
            description=tool.description,
            parameters=tool.inputSchema,
            fn=_proxy_passthrough,
            fn_metadata=func_metadata(_proxy_passthrough),
            is_async=True,
        )

    async def run(
        self, arguments: dict[str, Any], context: Context | None = None
    ) -> Any:
        # The client context manager will swallow any exceptions inside a TaskGroup,
        # so we return the raw result and raise an exception ourselves if needed.
        async with self._client:
            result = await self._client.call_tool(
                self.name, arguments, _return_raw_result=True
            )
        if result.isError:
            raise ValueError(cast(mcp.types.TextContent, result.content[0]).text)
        return result.content


class ProxyResource(Resource):
    def __init__(
        self, client: "Client", *, _value: str | bytes | None = None, **kwargs
    ):
        super().__init__(**kwargs)
        self._client = client
        self._value = _value

    @classmethod
    async def from_client(
        cls, client: "Client", resource: mcp.types.Resource
    ) -> "ProxyResource":
        return cls(
            client=client,
            uri=resource.uri,
            name=resource.name,
            description=resource.description,
            mime_type=resource.mimeType,
        )

    async def read(self) -> str | bytes:
        if self._value is not None:
            return self._value

        async with self._client:
            result = await self._client.read_resource(self.uri)
        if isinstance(result[0], TextResourceContents):
            return result[0].text
        elif isinstance(result[0], BlobResourceContents):
            return result[0].blob
        else:
            raise ValueError(f"Unsupported content type: {type(result[0])}")


class ProxyTemplate(ResourceTemplate):
    def __init__(self, client: "Client", **kwargs):
        super().__init__(**kwargs)
        self._client = client

    @classmethod
    async def from_client(
        cls, client: "Client", template: mcp.types.ResourceTemplate
    ) -> "ProxyTemplate":
        return cls(
            client=client,
            uri_template=template.uriTemplate,
            name=template.name,
            description=template.description,
            fn=_proxy_passthrough,
            parameters={},
        )

    async def create_resource(self, uri: str, params: dict[str, Any]) -> ProxyResource:
        # Don't use the provided uri, because it may not be the same as the
        # uri_template on the remote server.
        # Quote params to ensure they are valid for the uri_template.
        parameterized_uri = self.uri_template.format(
            **{k: quote(str(v), safe="") for k, v in params.items()}
        )
        async with self._client:
            result = await self._client.read_resource(parameterized_uri)

        if isinstance(result[0], TextResourceContents):
            value = result[0].text
        elif isinstance(result[0], BlobResourceContents):
            value = result[0].blob
        else:
            raise ValueError(f"Unsupported content type: {type(result[0])}")

        return ProxyResource(
            client=self._client,
            uri=parameterized_uri,
            name=self.name,
            description=self.description,
            mime_type=result[0].mimeType,
            contents=result,
            _value=value,
        )


class ProxyPrompt(Prompt):
    def __init__(self, client: "Client", **kwargs):
        super().__init__(**kwargs)
        self._client = client

    @classmethod
    async def from_client(
        cls, client: "Client", prompt: mcp.types.Prompt
    ) -> "ProxyPrompt":
        return cls(
            client=client,
            name=prompt.name,
            description=prompt.description,
            arguments=[a.model_dump() for a in (prompt.arguments or [])],
            fn=_proxy_passthrough,
        )

    async def render(self, arguments: dict[str, Any]) -> list[Message]:
        async with self._client:
            result = await self._client.get_prompt(self.name, arguments)
        return [Message(role=m.role, content=m.content) for m in result]


class FastMCPProxy(FastMCP):
    def __init__(self, client: "Client", **kwargs):
        super().__init__(**kwargs)
        self.client = client

    async def get_tools(self) -> dict[str, Tool]:
        tools = await super().get_tools()

        async with self.client:
            for tool in await self.client.list_tools():
                tool_proxy = await ProxyTool.from_client(self.client, tool)
                tools[tool_proxy.name] = tool_proxy

        return tools

    async def get_resources(self) -> dict[str, Resource]:
        resources = await super().get_resources()

        async with self.client:
            for resource in await self.client.list_resources():
                resource_proxy = await ProxyResource.from_client(self.client, resource)
                resources[str(resource_proxy.uri)] = resource_proxy

        return resources

    async def get_resource_templates(self) -> dict[str, ResourceTemplate]:
        templates = await super().get_resource_templates()

        async with self.client:
            for template in await self.client.list_resource_templates():
                template_proxy = await ProxyTemplate.from_client(self.client, template)
                templates[template_proxy.uri_template] = template_proxy

        return templates

    async def get_prompts(self) -> dict[str, Prompt]:
        prompts = await super().get_prompts()

        async with self.client:
            for prompt in await self.client.list_prompts():
                prompt_proxy = await ProxyPrompt.from_client(self.client, prompt)
                prompts[prompt_proxy.name] = prompt_proxy
        return prompts

    async def _mcp_call_tool(
        self, key: str, arguments: dict[str, Any]
    ) -> list[TextContent | ImageContent | EmbeddedResource]:
        try:
            result = await super()._mcp_call_tool(key, arguments)
            return result
        except NotFoundError:
            async with self.client:
                result = await self.client.call_tool(key, arguments)
            return result

    async def _mcp_read_resource(self, uri: AnyUrl | str) -> list[ReadResourceContents]:
        try:
            result = await super()._mcp_read_resource(uri)
            return result
        except NotFoundError:
            async with self.client:
                resource = await self.client.read_resource(uri)
                if isinstance(resource[0], TextResourceContents):
                    content = resource[0].text
                elif isinstance(resource[0], BlobResourceContents):
                    content = resource[0].blob
                else:
                    raise ValueError(f"Unsupported content type: {type(resource[0])}")

            return [
                ReadResourceContents(content=content, mime_type=resource[0].mimeType)
            ]

    async def _mcp_get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> GetPromptResult:
        try:
            result = await super()._mcp_get_prompt(name, arguments)
            return result
        except NotFoundError:
            async with self.client:
                result = await self.client.get_prompt(name, arguments)
            return GetPromptResult(messages=result)
