from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import uvicorn
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from minisgl.utils import init_logger

from .config import MultiModelConfig, ModelInstanceConfig

logger = init_logger(__name__, "MultiModelGateway")


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAICompletionRequest(BaseModel):
    model: str
    prompt: str | None = None
    messages: List[Message] | None = None
    max_tokens: int = 16
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: List[str] = []
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    ignore_eos: bool = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mini-sglang"
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


@dataclass
class InstanceStatus:
    model_name: str
    base_url: str
    pid: int | None = None
    healthy: bool = False
    last_error: str | None = None
    last_checked_at: float | None = None
    started_at: float = field(default_factory=time.time)


class MultiModelRuntime:
    def __init__(self, config: MultiModelConfig, statuses: Dict[str, List[InstanceStatus]]):
        self.config = config
        self.statuses = statuses
        self._rr_index: Dict[str, int] = {name: 0 for name in statuses}
        self._lock = threading.Lock()

    def list_models(self) -> List[ModelCard]:
        result = []
        for model_name, instances in self.statuses.items():
            root = instances[0].base_url
            result.append(ModelCard(id=model_name, root=root))
        return result

    def get_instance(self, model_name: str) -> InstanceStatus:
        instances = self.statuses.get(model_name)
        if not instances:
            raise HTTPException(status_code=404, detail=f"Unknown model '{model_name}'")

        with self._lock:
            start = self._rr_index[model_name]
            for offset in range(len(instances)):
                index = (start + offset) % len(instances)
                if instances[index].healthy:
                    self._rr_index[model_name] = (index + 1) % len(instances)
                    return instances[index]

            raise HTTPException(
                status_code=503,
                detail=f"No healthy instance available for model '{model_name}'",
            )

    def dump_status(self) -> Dict[str, List[Dict[str, object]]]:
        return {
            model_name: [
                {
                    "base_url": status.base_url,
                    "pid": status.pid,
                    "healthy": status.healthy,
                    "last_error": status.last_error,
                    "last_checked_at": status.last_checked_at,
                    "started_at": status.started_at,
                }
                for status in statuses
            ]
            for model_name, statuses in self.statuses.items()
        }


def _proxy_stream(url: str, body: bytes) -> Iterator[bytes]:
    req = Request(url=url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=600) as resp:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                yield chunk
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=exc.code, detail=detail) from exc
    except URLError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream unavailable: {exc}") from exc


def _health_check_once(status: InstanceStatus) -> None:
    status.last_checked_at = time.time()
    try:
        with urlopen(f"{status.base_url}/v1", timeout=2) as resp:
            status.healthy = 200 <= resp.status < 500
            status.last_error = None
    except Exception as exc:
        status.healthy = False
        status.last_error = str(exc)


def create_app(runtime: MultiModelRuntime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        stop_event = threading.Event()

        def _watch_health() -> None:
            while not stop_event.is_set():
                for statuses in runtime.statuses.values():
                    for status in statuses:
                        _health_check_once(status)
                stop_event.wait(3)

        thread = threading.Thread(target=_watch_health, name="multi-model-health", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop_event.set()
            thread.join(timeout=1)

    app = FastAPI(title="MiniSGL Multi-Model Gateway", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        return ModelList(data=runtime.list_models())

    @app.get("/admin/instances")
    async def list_instances():
        return runtime.dump_status()

    @app.post("/v1/chat/completions")
    async def chat_completions(req: OpenAICompletionRequest, _: FastAPIRequest):
        target = runtime.get_instance(req.model)
        body = json.dumps(req.model_dump()).encode("utf-8")
        stream = req.stream
        generator = _proxy_stream(f"{target.base_url}/v1/chat/completions", body)
        media_type = "text/event-stream" if stream else "application/json"
        return StreamingResponse(generator, media_type=media_type)

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(_, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return app


def run_gateway(config: MultiModelConfig, statuses: Dict[str, List[InstanceStatus]]) -> None:
    runtime = MultiModelRuntime(config, statuses)
    app = create_app(runtime)
    uvicorn.run(app, host=config.gateway.host, port=config.gateway.port)
