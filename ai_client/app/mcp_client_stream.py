"""
MCP Client: Connect to MCP Server, convert tools to OpenAI function calling format, and handle streaming chat with tool calls.
"""
import asyncio
import json
from dataclasses import dataclass
from typing import AsyncGenerator

from fastmcp import Client
from openai import AsyncOpenAI
from opentelemetry import context as otel_context
from opentelemetry import trace

_tracer = trace.get_tracer("pankb.agent")


@dataclass
class AgentEvent:
    """Event yielded during chat processing for frontend rendering"""
    type: str  # "tool_start", "tool_result", "text", "usage"
    content: str | None = None  # json formatted tool result (chart, table, rag) or plain text for tool call errors
    tool_name: str | None = None
    tool_args: dict | None = None
    result_type: str | None = None  # "chart", "table", "string", or None for plain text
    parsed_data: dict | None = None
    # Token usage (only for type="usage")
    tokens_in: int | None = None
    tokens_out: int | None = None


class MCPClient:
    def __init__(
        self,
        mcp_server_url: str,
        openai_client: AsyncOpenAI,
        model: str = "gpt-4o-mini",
        auth=None,
        system_prompt: str | None = None,
        prompt_version: str = "unknown",
    ):
        if not mcp_server_url:
            raise ValueError("mcp_server_url is required")
        if not openai_client:
            raise ValueError("openai_client is required")
        if not auth:
            raise ValueError("auth is required (use BearerAuth)")

        self.mcp_server_url = mcp_server_url
        self.model = model
        self.auth = auth
        self.openai = openai_client
        self.tools_cache: list[dict] = []
        self.messages: list[dict] = []
        self.system_prompt = system_prompt
        self.prompt_version = prompt_version

    def _create_mcp_client(self) -> Client:
        return Client(self.mcp_server_url, auth=self.auth)

    def clear_history(self):
        """Clear conversation history"""
        self.messages = []

    async def connect(self):
        """Connect to MCP Server and fetch tools"""
        async with self._create_mcp_client() as client:
            mcp_tools = await client.list_tools()
            self.tools_cache = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema or {"type": "object", "properties": {}},
                    "strict": False,
                }
                for t in mcp_tools
            ]

    def _parse_result(self, result_str: str) -> tuple[str | None, dict | None]:
        """Parse tool result for chart/table/url data"""
        try:
            data = json.loads(result_str)
            if isinstance(data, dict) and data.get("type") in ("chart", "table", "string", "url"):
                return data["type"], data
        except (json.JSONDecodeError, TypeError):
            pass
        return None, None

    async def _call_tool(self, client: Client, name: str, args: dict) -> tuple[str, str | None, dict | None]:
        """
        Call MCP tool on an already-connected client and return
        (result_str, result_type, parsed_data).
        """
        with _tracer.start_as_current_span(f"tool.{name}") as span:
            # OpenInference semantic conventions: render as a TOOL span in Phoenix
            span.set_attribute("openinference.span.kind", "TOOL")
            span.set_attribute("tool.name", name)
            try:
                span.set_attribute("tool.parameters", json.dumps(args))
            except (TypeError, ValueError):
                pass

            try:
                result = await client.call_tool(name, args)
                result_str = ""
                result_type = None
                parsed_data = None
                for content in result.content:
                    if hasattr(content, "text"):
                        result_str = content.text
                        result_type, parsed_data = self._parse_result(result_str)
                span.set_attribute("tool.result_type", result_type or "text")
                span.set_attribute("tool.result_chars", len(result_str))

                # Per-result-type metadata so Phoenix shows what actually came back,
                # not just "12453 chars of something". Keep payloads small.
                if result_type == "chart" and parsed_data:
                    span.set_attribute("tool.chart.title", parsed_data.get("title", ""))
                    series = (parsed_data.get("data") or {}).get("series") or []
                    span.set_attribute("tool.chart.series_count", len(series))
                    span.set_attribute(
                        "tool.chart.series_names",
                        json.dumps([s.get("name", "?") for s in series][:20]),
                    )
                elif result_type == "table" and parsed_data:
                    span.set_attribute("tool.table.title", parsed_data.get("title", ""))
                    span.set_attribute(
                        "tool.table.row_count",
                        parsed_data.get("row_count", len(parsed_data.get("rows", []) or [])),
                    )
                elif result_type == "url" and parsed_data:
                    span.set_attribute("tool.url.title", parsed_data.get("title", ""))
                    span.set_attribute("tool.url.value", parsed_data.get("url", ""))
                elif name == "search_pangenome_literature" and result_str:
                    # RAG returns markdown with "### Document N:" per chunk. Count +
                    # preview is enough to debug "did we retrieve anything reasonable?"
                    # without dumping the full payload.
                    span.set_attribute("rag.chunk_count", result_str.count("### Document "))
                    span.set_attribute("rag.documents_markdown", result_str)

                return result_str, result_type, parsed_data
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                return f"Error: {e}", None, None

    async def chat(
        self,
        user_message: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        Process user message and yield events (streaming).
        Using OpenAI Responses API.

        Yields:
            AgentEvent(type="tool_start") - tool is being called
            AgentEvent(type="tool_result") - tool finished with result
            AgentEvent(type="text") - streaming text chunk (one per chunk)
        """
        # Add user message to input
        self.messages.append({"role": "user", "content": user_message})

        # Root span for the whole user turn. Manual start + context.attach so
        # the span both encloses the async generator across yields AND becomes
        # the parent of any child spans (iteration, tool, auto-instrumented LLM).
        agent_span = _tracer.start_span("agent.chat")
        agent_span.set_attribute("openinference.span.kind", "AGENT")
        agent_span.set_attribute("agent.model", self.model)
        agent_span.set_attribute("agent.prompt_version", self.prompt_version)
        agent_span.set_attribute("input.value", user_message[:2000])
        if session_id:
            agent_span.set_attribute("session.id", session_id)
        if user_id:
            agent_span.set_attribute("user.id", user_id)
        agent_token = otel_context.attach(trace.set_span_in_context(agent_span))
        total_tool_calls = 0
        iterations_used = 0
        total_tokens_in = 0
        total_tokens_out = 0
        final_output = ""

        try:
            max_iterations = 5
            for iteration in range(max_iterations):
                iterations_used = iteration + 1
                iter_span = _tracer.start_span(f"agent.iteration.{iteration}")
                iter_span.set_attribute("openinference.span.kind", "CHAIN")
                iter_span.set_attribute("agent.iteration", iteration)
                iter_token = otel_context.attach(trace.set_span_in_context(iter_span))
                try:
                    # Filter messages to only include valid API items
                    api_input = self._filter_messages_for_api()

                    # Streaming call using Responses API
                    # (auto-instrumented by openinference-instrumentation-openai)
                    stream = await self.openai.responses.create(
                        model=self.model,
                        input=api_input,
                        instructions=self.system_prompt,
                        tools=self.tools_cache or None,
                        store=False,
                        stream=True,
                        temperature=0,  # Deterministic output for strict instruction following
                    )

                    # Collect streamed content and final response
                    full_content = ""
                    function_calls: list[dict] = []
                    usage_info: dict | None = None

                    async for event in stream:
                        # Handle text output deltas (for streaming display)
                        if event.type == "response.output_text.delta":
                            full_content += event.delta
                            yield AgentEvent(type="text", content=event.delta)

                        # Handle tool call output - extract complete function calls
                        elif event.type == "response.output_item.done":
                            if hasattr(event.item, "type") and event.item.type == "function_call":
                                function_calls.append({
                                    "call_id": event.item.call_id,
                                    "name": event.item.name,
                                    "arguments": event.item.arguments
                                })

                        # Capture usage from response.completed event
                        elif event.type == "response.completed":
                            if hasattr(event, "response") and hasattr(event.response, "usage"):
                                usage = event.response.usage
                                usage_info = {
                                    "input_tokens": usage.input_tokens,
                                    "output_tokens": usage.output_tokens
                                }

                    if usage_info:
                        total_tokens_in += usage_info["input_tokens"]
                        total_tokens_out += usage_info["output_tokens"]
                        iter_span.set_attribute("llm.token_count.prompt", usage_info["input_tokens"])
                        iter_span.set_attribute("llm.token_count.completion", usage_info["output_tokens"])

                    iter_span.set_attribute("agent.tool_calls_planned", len(function_calls))

                    # Handle function calls if any
                    if function_calls:
                        total_tool_calls += len(function_calls)
                        # Add function_call items to messages for next API call
                        for fc in function_calls:
                            self.messages.append({
                                "type": "function_call",
                                "call_id": fc["call_id"],
                                "name": fc["name"],
                                "arguments": fc["arguments"]
                            })

                        # Open MCP session once and run all tool calls in parallel.
                        parsed_args = [json.loads(fc["arguments"]) if fc["arguments"] else {} for fc in function_calls]
                        for fc, args in zip(function_calls, parsed_args):
                            yield AgentEvent(type="tool_start", tool_name=fc["name"], tool_args=args)
                        async with self._create_mcp_client() as mcp:
                            tool_results = await asyncio.gather(*[
                                self._call_tool(mcp, fc["name"], args)
                                for fc, args in zip(function_calls, parsed_args)
                            ])

                        # Process each result in original call order
                        for fc, args, (result_str, result_type, parsed_data) in zip(function_calls, parsed_args, tool_results):
                            name = fc["name"]
                            yield AgentEvent(
                                type="tool_result",
                                content=result_str,
                                tool_name=name,
                                tool_args=args,
                                result_type=result_type,
                                parsed_data=parsed_data
                            )

                            # Prepare content for LLM
                            if result_type == "chart" and parsed_data:
                                llm_content = (
                                    f"[Chart rendered successfully: {parsed_data.get('title', 'Untitled')}]\n"
                                    "The interactive chart is already visible to the user above. "
                                )
                            elif result_type == "table" and parsed_data:
                                row_count = parsed_data.get("row_count", len(parsed_data.get("rows", [])))
                                llm_content = f"[Table displayed: {parsed_data.get('title', 'Data')} - {row_count} rows]"
                            elif result_type == "url" and parsed_data:
                                llm_content = f"[URL provided: {parsed_data.get('title', 'Link')} - {parsed_data.get('url', '')}]"
                            else:
                                llm_content = result_str

                            # Add function_call_output to messages (Responses API format)
                            self.messages.append({
                                "type": "function_call_output",
                                "call_id": fc["call_id"],
                                "output": llm_content
                            })

                            # Store extra data for UI rendering (we'll need this for history)
                            # Add a marker message for UI (will be filtered when sending to API)
                            self.messages.append({
                                "role": "tool",  # Marker for UI rendering
                                "tool_call_id": fc["call_id"],
                                "content": llm_content,
                                "tool_name": name,
                                "tool_args": args,
                                "result_type": result_type,
                                "parsed_data": parsed_data
                            })

                            # Add instruction for RAG tool results
                            if name == "search_pangenome_literature":
                                self.messages.append({
                                    "role": "system",
                                    "content": (
                                        "INSTRUCTION: Answer the user's question based ONLY on the documents above. "
                                        "Do NOT use your own knowledge. Summarize key points and cite sources with titles/URLs. "
                                        "If the documents don't contain relevant information, say 'I don't have information about this in my knowledge base.'"
                                    )
                                })

                        # Loop back so the LLM can see tool outputs and decide:
                        # call more tools, or emit a final assistant message.
                        otel_context.detach(iter_token)
                        iter_span.end()
                        continue

                    # No function calls — AI is done (either final text or empty response)
                    if full_content:
                        self.messages.append({"role": "assistant", "content": full_content})
                        final_output = full_content

                    # Yield usage event at the end of the response
                    # We only recorded the final round of token usage here, but token captured by phoenix is correct
                    if usage_info:
                        yield AgentEvent(
                            type="usage",
                            tokens_in=usage_info["input_tokens"],
                            tokens_out=usage_info["output_tokens"]
                        )
                    otel_context.detach(iter_token)
                    iter_span.end()
                    break
                except Exception:
                    otel_context.detach(iter_token)
                    iter_span.end()
                    raise
        finally:
            agent_span.set_attribute("agent.iterations_used", iterations_used)
            agent_span.set_attribute("agent.total_tool_calls", total_tool_calls)
            agent_span.set_attribute("agent.hit_iteration_cap", iterations_used >= 5)
            agent_span.set_attribute("llm.token_count.prompt", total_tokens_in)
            agent_span.set_attribute("llm.token_count.completion", total_tokens_out)
            if final_output:
                agent_span.set_attribute("output.value", final_output[:8000])
            otel_context.detach(agent_token)
            agent_span.end()

    def _filter_messages_for_api(self) -> list[dict]:
        """
        Filter messages to only include items valid for Responses API input.
        Removes UI-only marker messages (role="tool").
        """
        valid_items = []
        for msg in self.messages:
            # Skip UI-only tool marker messages
            if msg.get("role") == "tool":
                continue
            # Responses API items: function_call, function_call_output
            if msg.get("type") in ("function_call", "function_call_output"):
                valid_items.append(msg)
            # Standard message format: user, assistant, system
            elif msg.get("role") in ("user", "assistant", "system"):
                valid_items.append(msg)
        return valid_items
