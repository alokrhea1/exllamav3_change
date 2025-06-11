#!/usr/bin/env python3
"""
openai_server.py – expose exllamav3 as an OpenAI-compatible REST API
--------------------------------------------------------------------
• /v1/models            – minimal model list (2024-12-01 schema)
• /v1/chat/completions  – chat endpoint (stream / non-stream)
• /v1/completions       – plain completion endpoint (stream / non-stream)

Drop-in for Mikupad (or any client that speaks the official OpenAI spec).
"""

from __future__ import annotations

import os, sys, json, time, argparse
from typing import List, Optional, Generator as TypingGen

import torch  # noqa – required by exllamav3
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware  # ← NEW
from pydantic import BaseModel, Field
import uvicorn

# --------------------------------------------------------------------- exllamav3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Generator as ExGen, Job, model_init  # noqa
from exllamav3.generator.sampler import ComboSampler  # noqa
from chat_templates import prompt_formats  # noqa

# ------------------------------------------------------------------ CLI args

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run exllamav3 as an OpenAI-style server")
    model_init.add_args(p, cache=True)  # brings in --model_dir, dtype flags, etc.

    # minimal inference controls
    p.add_argument("--mode", required=True,
                   help="Key in chat_templates.prompt_formats (e.g. 'raw', 'mistral')")
    p.add_argument("--system_prompt")
    p.add_argument("--max_response_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--min_p", type=float, default=0.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--presence_penalty", type=float, default=0.0)
    p.add_argument("--frequency_penalty", type=float, default=0.0)
    p.add_argument("--penalty_range", type=int, default=1024)

    # server
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5001)
    return p


# -------------------------------------------------------------- Pydantic models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = Field(False, description="Return SSE stream if true")

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False


# ------------------------------------------------------------- Core wrapper
class ExLlamaEngine:
    """Singleton wrapper so the model stays resident in memory."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model, self.config, self.cache, self.tokenizer = model_init.init(args)
        self.context_len = self.cache.max_num_tokens

        self.model_id = f"exllama-{args.mode}"
        self.prompt_tmpl = prompt_formats[args.mode]("User", "Assistant")
        self.system_prompt = (
            args.system_prompt or self.prompt_tmpl.default_system_prompt()
        )

    # ------------------------------------------------- Sampling factory
    def sampler(self, temp: Optional[float], top_p: Optional[float]) -> ComboSampler:
        return ComboSampler(
            rep_p=self.args.repetition_penalty,
            pres_p=self.args.presence_penalty,
            freq_p=self.args.frequency_penalty,
            rep_sustain_range=self.args.penalty_range,
            rep_decay_range=self.args.penalty_range,
            temperature=temp if temp is not None else self.args.temperature,
            min_p=self.args.min_p,
            top_k=self.args.top_k,
            top_p=top_p if top_p is not None else self.args.top_p,
            temp_last=True,
        )

    # ------------------------------------------------ Prompt building
    def build_prompt(self, messages: List[Message]) -> str:
        ctx: list[tuple[Optional[str], Optional[str]]] = []
        for m in messages:
            if m.role == "system":
                self.system_prompt = m.content
            elif m.role == "user":
                ctx.append((m.content, None))
            elif m.role == "assistant":
                ctx.append((None, m.content))
            else:
                raise HTTPException(400, f"unknown role: {m.role}")
        return self.prompt_tmpl.format(self.system_prompt, ctx)

    # ------------------------------------------------ Inference generator
    def generate(
        self,
        prompt: str,
        sampler: ComboSampler,
        max_new_tokens: int,
        *,
        stream: bool = False,
    ) -> TypingGen[str, None, str]:
        ids = self.tokenizer.encode(
            prompt,
            add_bos=self.prompt_tmpl.add_bos(),
            encode_special_tokens=True,
        )
        # truncate from the head if context is too long
        while ids.shape[-1] + max_new_tokens + 1 > self.context_len:
            ids = ids[:, ids.shape[-1] // 2 :]

        stop_conditions = self.prompt_tmpl.stop_conditions(self.tokenizer)
        tt = self.prompt_tmpl.thinktag()

        job = Job(
            input_ids=ids,
            max_new_tokens=max_new_tokens,
            stop_conditions=stop_conditions,
            sampler=sampler,
            banned_strings=[tt[0], tt[1]],
        )

        gen = ExGen(model=self.model, cache=self.cache, tokenizer=self.tokenizer)
        gen.enqueue(job)

        if stream:
            for r in gen.iterate():
                txt = r.get("text", "")
                if txt:
                    yield txt
        else:
            chunks: list[str] = []
            for r in gen.iterate():
                if r.get("text"):
                    chunks.append(r["text"])
            yield "".join(chunks)


# ---------------------------------------------------------------- FastAPI app

def create_app(engine: ExLlamaEngine) -> FastAPI:
    app = FastAPI()

    # --------------- NEW: Enable CORS for any origin / method / header
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----------------------------------------------------------- /v1/models
    @app.get("/v1/models")
    async def list_models():
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model_id,
                    "object": "model",
                    "created": now,
                    "owned_by": "you",
                }
            ],
        }

    # --------------------------------------------------- /v1/chat/completions
    @app.post("/v1/chat/completions")
    async def chat_completion(req: ChatRequest):
        max_tok = req.max_tokens or engine.args.max_response_tokens
        sampler = engine.sampler(req.temperature, req.top_p)
        prompt = engine.build_prompt(req.messages)

        # token count for usage
        prompt_ids = engine.tokenizer.encode(prompt, add_bos=engine.prompt_tmpl.add_bos(), encode_special_tokens=True)
        prompt_tokens = prompt_ids.shape[-1]

        # STREAMING ----------------------------------------------------
        if req.stream:
            start_ts = int(time.time())
            first_chunk_flag = [True]  # mutable closure

            def event_stream():
                for chunk in engine.generate(prompt, sampler, max_tok, stream=True):
                    data_obj = {
                        "id": f"chatcmpl-{start_ts}",
                        "object": "chat.completion.chunk",
                        "created": start_ts,
                        "model": engine.model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": (
                                    {"role": "assistant"} if first_chunk_flag[0] else {}
                                ) | {"content": chunk},
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(data_obj, ensure_ascii=False)}\n\n"
                    first_chunk_flag[0] = False
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        # NON-STREAM ---------------------------------------------------
        text = "".join(engine.generate(prompt, sampler, max_tok, stream=False))
        now = int(time.time())
        completion_tokens = engine.tokenizer.encode(text).shape[-1]
        return JSONResponse(
            {
                "id": f"chatcmpl-{now}",
                "object": "chat.completion",
                "created": now,
                "model": engine.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )

    # ------------------------------------------------------ /v1/completions
    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        max_tok = req.max_tokens or engine.args.max_response_tokens
        sampler = engine.sampler(req.temperature, req.top_p)
        prompt = req.prompt

        prompt_ids = engine.tokenizer.encode(prompt, add_bos=engine.prompt_tmpl.add_bos(), encode_special_tokens=True)
        prompt_tokens = prompt_ids.shape[-1]

        # STREAM --------------------------------------------------------
        if req.stream:
            start_ts = int(time.time())
            def event_stream():
                for chunk in engine.generate(prompt, sampler, max_tok, stream=True):
                    data_obj = {
                        "id": f"cmpl-{start_ts}",
                        "object": "text_completion_chunk",
                        "created": start_ts,
                        "model": engine.model_id,
                        "choices": [
                            {
                                "index": 0,
                                "text": chunk,
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(data_obj, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(event_stream(), media_type="text/event-stream")

        # NON-STREAM ---------------------------------------------------
        text = "".join(engine.generate(prompt, sampler, max_tok, stream=False))
        now = int(time.time())
        completion_tokens = engine.tokenizer.encode(text).shape[-1]
        return JSONResponse(
            {
                "id": f"cmpl-{now}",
                "object": "text_completion",
                "created": now,
                "model": engine.model_id,
                "choices": [
                    {
                        "index": 0,
                        "text": text,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )

    return app


# ------------------------------------------------------------------- main
if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    engine = ExLlamaEngine(args)
    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
