"""Shared metadata-only recording for explicit real-API benchmarks."""

import time


class RecordingLLM:
    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    @property
    def model_name(self):
        return self.inner.model_name

    async def _run(self, request, handler=None):
        start = time.perf_counter()
        call = {
            "provider_message_chars": sum(len(m.content) for m in request.messages),
            "task": request.task,
            "max_output_tokens": request.max_tokens,
            "target_count": len(request.review_context.get("target_ids", [])),
        }
        try:
            result = (
                await self.inner.complete_json_with_tools(request, handler)
                if handler is not None
                else await self.inner.complete_json(request)
            )
            call.update(
                usage=result.usage,
                trace=result.agent_trace.model_dump() if result.agent_trace else None,
            )
            return result
        except Exception as error:
            call["error_type"] = type(error).__name__
            call["error"] = str(error)[:200]
            raise
        finally:
            call["seconds"] = round(time.perf_counter() - start, 4)
            self.calls.append(call)

    async def complete_json(self, request):
        return await self._run(request)

    async def complete_json_with_tools(self, request, handler):
        return await self._run(request, handler)
