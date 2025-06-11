#!/usr/bin/env python3
"""
openai_server.py – expose exllamav3 as an OpenAI-style REST API
---------------------------------------------------------------
• POST /v1/chat/completions   – chat endpoint (stream / no-stream)
• POST /v1/completions        – plain completion endpoint
• GET  /v1/models             – minimal model listing
"""

import os, sys, json, time, argparse, itertools
from typing import List, Dict, Any, Optional, Generator as TypingGen

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# --- bring exllamav3 into path ------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Generator as ExGen, Job, model_init
from exllamav3.generator.sampler import ComboSampler
from chat_templates import prompt_formats           # your existing templates
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------- Build args
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    model_init.add_args(p, cache=True)  # <-- all the usual exllama flags
    p.add_argument("--mode",  required=True,
                   help="key in chat_templates.prompt_formats (e.g. 'mistral')")
    p.add_argument("--system_prompt", default=None)
    p.add_argument("--max_response_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int,   default=0)
    p.add_argument("--min_p", type=float, default=0.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--presence_penalty",  type=float, default=0.0)
    p.add_argument("--frequency_penalty", type=float, default=0.0)
    p.add_argument("--penalty_range",     type=int,   default=1024)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default=5001, type=int)
    return p


# -------------------------------------------------------------- OpenAI schema
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


# -------------------------------------------------------------- Helper logic
class ExLlamaEngine:
    """
    Wrap your previous CLI script into a state-ful singleton.
    Avoids reloading the model for every request.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model, self.config, self.cache, self.tokenizer = model_init.init(args)
        self.context_len = self.cache.max_num_tokens

        self.prompt_tmpl = prompt_formats[args.mode]("User", "Assistant")
        self.system_prompt = (args.system_prompt or
                              self.prompt_tmpl.default_system_prompt())

    def sampler(self, req_temp, req_top_p) -> ComboSampler:
        """Return a ComboSampler object filled from request-level or default args."""
        return ComboSampler(
            rep_p=self.args.repetition_penalty,
            pres_p=self.args.presence_penalty,
            freq_p=self.args.frequency_penalty,
            rep_sustain_range=self.args.penalty_range,
            rep_decay_range=self.args.penalty_range,
            temperature=req_temp if req_temp is not None else self.args.temperature,
            min_p=self.args.min_p,
            top_k=self.args.top_k,
            top_p=req_top_p if req_top_p is not None else self.args.top_p,
            temp_last=True,   # mimic your original CLI default
        )

    def build_prompt(self, messages: List[Message]) -> str:
        """
        Convert OP's OpenAI-style message list into the template string
        understood by exllamav3.
        The first assistant/system message is mapped into the 'system prompt'.
        """
        context_pairs = []
        for msg in messages:
            if msg.role == "system":
                # Only the *last* system message wins
                self.system_prompt = msg.content
            elif msg.role == "user":
                context_pairs.append((msg.content, None))
            elif msg.role == "assistant":
                context_pairs.append((None, msg.content))
            else:
                raise HTTPException(400, f"unknown role: {msg.role}")

        return self.prompt_tmpl.format(self.system_prompt, context_pairs)

    def generate(
        self,
        prompt: str,
        sampler: ComboSampler,
        max_new_tokens: int,
        stream: bool = False,
    ) -> TypingGen[str, None, str]:
        """
        Yield chunks (for stream=True) or the full text at the end.
        """
        # Tokenize
        ids = self.tokenizer.encode(
            prompt,
            add_bos=self.prompt_tmpl.add_bos(),
            encode_special_tokens=True,
        )
        # Trim head if needed
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
        gen = ExGen(
            model=self.model,
            cache=self.cache,
            tokenizer=self.tokenizer,
        )
        gen.enqueue(job)

        full_text = []
        if stream:
            for r in gen.iterate():
                txt = r.get("text", "")
                if txt:
                    full_text.append(txt)
                    yield txt
        else:
            for r in gen.iterate():
                if r.get("text"):
                    full_text.append(r["text"])
            yield "".join(full_text)

# ------------------------------------------------------------------- FastAPI
def create_app(engine: ExLlamaEngine) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/models")
    async def list_models():
        return {"data": [{"id": "exllama-"+engine.args.mode,
                          "object": "model",
                          "owned_by": "you"}]}

    @app.post("/v1/chat/completions")
    async def chat_completion(req: ChatRequest):
        max_tok = req.max_tokens or engine.args.max_response_tokens
        sampler = engine.sampler(req.temperature, req.top_p)
        prompt = engine.build_prompt(req.messages)

        def mk_stream():
            idx = 0
            start = time.time()
            for chunk in engine.generate(prompt, sampler, max_tok, stream=True):
                data = {
                    "id": f"chatcmpl-{int(start*1000)}",
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                idx += 1
            yield "data: [DONE]\n\n"

        if req.stream:
            return StreamingResponse(mk_stream(),
                                     media_type="text/event-stream")
        else:
            text = "".join(engine.generate(prompt, sampler, max_tok, stream=False))
            return JSONResponse({
                "id": f"chatcmpl-{int(time.time()*1000)}",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                },
            })

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        max_tok = req.max_tokens or engine.args.max_response_tokens
        sampler = engine.sampler(req.temperature, req.top_p)

        prompt = req.prompt

        def mk_stream():
            start = time.time()
            for chunk in engine.generate(prompt, sampler, max_tok, stream=True):
                data = {
                    "id": f"cmpl-{int(start*1000)}",
                    "object": "text_completion_chunk",
                    "choices": [{
                        "index": 0,
                        "text": chunk,
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        if req.stream:
            return StreamingResponse(mk_stream(),
                                     media_type="text/event-stream")
        else:
            text = "".join(engine.generate(prompt, sampler, max_tok, stream=False))
            return JSONResponse({
                "id": f"cmpl-{int(time.time()*1000)}",
                "object": "text_completion",
                "choices": [{
                    "index": 0,
                    "text": text,
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                },
            })

    return app


# -------------------------------------------------------------- entry point
if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    engine = ExLlamaEngine(args)
    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
