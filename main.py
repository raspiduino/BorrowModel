import asyncio
import json
import os
import re
import sqlite3
import time
import uuid
import traceback
from enum import Enum
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
import httpx

# ==========================================
# Configuration & Environment Fallbacks
# ==========================================
HOST = os.getenv("PROXY_HOST", "127.0.0.1")
PORT = int(os.getenv("PROXY_PORT", "9908"))

LOCAL_API_BASE = os.getenv("LOCAL_API_BASE", "http://127.0.0.1:11434/v1")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "none")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "Qwen3.8-27B")

# Provider 1 (Main Frontier Heavy Solver)
PROVIDER1_API_BASE = os.getenv("PROVIDER1_API_BASE", "https://api.openai.com/v1")
PROVIDER1_API_KEY = os.getenv("PROVIDER1_API_KEY", "sk-...")
PROVIDER1_MODEL_NAME = os.getenv("PROVIDER1_MODEL_NAME", "gpt-4o")

# Provider 2 (Independent Secondary Flash Verifier)
PROVIDER2_ENABLED = os.getenv("PROVIDER2_ENABLED", "false").lower() == "true"
PROVIDER2_API_BASE = os.getenv("PROVIDER2_API_BASE", "https://api.openai.com/v1")
PROVIDER2_API_KEY = os.getenv("PROVIDER2_API_KEY", "sk-...")
PROVIDER2_MODEL_NAME = os.getenv("PROVIDER2_MODEL_NAME", "gpt-4o-mini")

DB_FILE = os.getenv("LOG_DB_PATH", "audit_logs.db")
MAX_CORRECTION_ROUNDS = 3

app = FastAPI(title="BorrowModel")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Database Concurrency Engine (WAL Mode)
# ==========================================
db_write_lock = asyncio.Lock()
state_lock = asyncio.Lock()

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_turns (
            id TEXT PRIMARY KEY,
            created_at REAL,
            raw_query TEXT,
            routing_mode TEXT,
            status TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS turn_subrequests (
            id TEXT PRIMARY KEY,
            turn_id TEXT,
            step_order INTEGER,
            created_at REAL,
            latency_ms REAL,
            phase_type TEXT,
            is_external INTEGER,
            target_engine TEXT,
            prompt TEXT,
            response TEXT,
            certainty TEXT,
            audit_notes TEXT,
            status TEXT,
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            FOREIGN KEY (turn_id) REFERENCES user_turns(id)
        )
    """)
    for col in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        try:
            c.execute(f"ALTER TABLE turn_subrequests ADD COLUMN {col} INTEGER DEFAULT 0;")
        except sqlite3.OperationalError:
            pass
    c.execute("CREATE INDEX IF NOT EXISTS idx_subreq_turn ON turn_subrequests(turn_id);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_turns_created ON user_turns(created_at DESC);")
    conn.commit()
    conn.close()

init_db()

async def safe_db_write(query: str, params: tuple = ()):
    async with db_write_lock:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(query, params)
        conn.commit()
        conn.close()

async def db_start_turn(turn_id: str, raw_query: str):
    await safe_db_write("INSERT OR REPLACE INTO user_turns (id, created_at, raw_query, routing_mode, status) VALUES (?, ?, ?, ?, ?)",
                        (turn_id, time.time(), raw_query, "EVALUATING", "IN_PROGRESS"))

async def db_update_turn_mode(turn_id: str, routing_mode: str):
    await safe_db_write("UPDATE user_turns SET routing_mode = ? WHERE id = ?", (routing_mode, turn_id))

async def db_finish_turn(turn_id: str, status: str = "COMPLETED"):
    await safe_db_write("UPDATE user_turns SET status = ? WHERE id = ?", (status, turn_id))

async def db_record_subrequest(subreq_id: str, turn_id: str, step_order: int, latency_ms: float,
                              phase_type: str, is_external: bool, target: str, prompt: str,
                              response: str = "", certainty: str = "N/A", audit_notes: str = "", status: str = "COMPLETED",
                              prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0):
    await safe_db_write("""
        INSERT OR REPLACE INTO turn_subrequests 
        (id, turn_id, step_order, created_at, latency_ms, phase_type, is_external, target_engine, prompt, response, certainty, audit_notes, status, prompt_tokens, completion_tokens, total_tokens)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (subreq_id, turn_id, step_order, time.time(), latency_ms, phase_type, 1 if is_external else 0,
          target, prompt, response, certainty, audit_notes, status, prompt_tokens, completion_tokens, total_tokens))

# ==========================================
# In-Memory State
# ==========================================
class GatePolicy(str, Enum):
    AUTO_APPROVE = "auto"
    MANUAL_APPROVE_ALL = "manual"
    CONFIDENCE_BASED = "confidence"

class AuditCertainty(str, Enum):
    HIGH_CERTAINTY_CLEAN = "high_certainty_clean"
    UNCERTAIN = "uncertain"
    FLAGGED_LEAK = "flagged_leak"

class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

class QueueRecord(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    id: str
    turn_id: str
    round_idx: int = 1
    created_at: float
    target_provider: str
    spec_type: str
    prompt_text: str
    certainty: AuditCertainty
    audit_notes: str
    evaluator_raw_response: Optional[str] = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    event: asyncio.Event = Field(default_factory=asyncio.Event, exclude=True)

class ProxyState:
    def __init__(self):
        self.policy: GatePolicy = GatePolicy.MANUAL_APPROVE_ALL
        self.queue: Dict[str, QueueRecord] = {}

state = ProxyState()

# ==========================================
# Transports & Helpers
# ==========================================
def build_payload(system_prompt: str, history: List[Dict[str, str]], user_prompt: str) -> List[Dict[str, str]]:
    safe_history = [m for m in history if m.get("role") != "system"]
    return [{"role": "system", "content": system_prompt}] + safe_history + [{"role": "user", "content": user_prompt}]

def format_sse(text: str) -> str:
    return f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n"

async def call_llm(base_url: str, api_key: str, model: str, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: Optional[int] = None):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens: payload["max_tokens"] = max_tokens
        
    start_t = time.perf_counter()
    async with httpx.AsyncClient(timeout=600.0) as client:
        res = await client.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload)
        latency = (time.perf_counter() - start_t) * 1000.0
        if res.status_code != 200:
            raise HTTPException(status_code=502, detail=f"LLM upstream error ({base_url}): {res.text}")
        data = res.json()
        usage = data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        return data["choices"][0]["message"].get("content", ""), latency, usage

async def stream_llm(base_url: str, api_key: str, model: str, messages: List[Dict[str, str]], temperature: float = 0.0):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "stream": True, "stream_options": {"include_usage": True}}
    start_t = time.perf_counter()
    async with httpx.AsyncClient(timeout=600.0) as client:
        async with client.stream("POST", f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload) as res:
            if res.status_code != 200:
                err = await res.aread()
                raise HTTPException(status_code=502, detail=f"LLM stream error: {err.decode()}")
            
            final_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            async for line in res.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        data = json.loads(line[6:])
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            chunk_content = delta.get("content", "") or ""
                            chunk_reasoning = delta.get("reasoning_content", "") or ""
                            if chunk_content or chunk_reasoning:
                                yield chunk_content, chunk_reasoning, None
                        if "usage" in data and data["usage"]:
                            final_usage = data["usage"]
                    except json.JSONDecodeError: pass
            latency = (time.perf_counter() - start_t) * 1000.0
            yield "", "", (latency, final_usage)

async def evaluate_audit_certainty(abstract_prompt: str, turn_id: str, step_order: int) -> tuple[AuditCertainty, str, str]:
    sys_prompt = (
        "You are an adversarial industrial espionage auditor. Judge if this text leaks proprietary architecture or intent.\n"
        "Respond STRICTLY in JSON format: {\"certainty\": \"high_certainty_clean\"|\"uncertain\"|\"flagged_leak\", \"notes\": \"...\"}"
    )
    
    op_id = str(uuid.uuid4())
    try:
        raw_response, latency, usage = await call_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, 
                                                      [{"role": "system", "content": sys_prompt}, {"role": "user", "content": abstract_prompt}])
        clean_res = re.sub(r"<think>.*?(</think>|$)", "", raw_response, flags=re.DOTALL | re.IGNORECASE).strip()
        match = re.search(r"\{.*\}", clean_res, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else json.loads(clean_res)
        certainty = AuditCertainty(parsed.get("certainty", "uncertain"))
        notes = parsed.get("notes", "No detailed reasoning provided.")
    except Exception as e:
        certainty = AuditCertainty.UNCERTAIN
        notes = f"Audit skipped due to local model parsing/timeout error: {str(e)}"
        raw_response = raw_response if 'raw_response' in locals() else ""
        latency = latency if 'latency' in locals() else 0.0
        usage = usage if 'usage' in locals() else {}
        
    await db_record_subrequest(op_id, turn_id, step_order, latency, "INTERNAL_AUDIT", False,
                               f"Local ({LOCAL_MODEL_NAME})", abstract_prompt, raw_response, certainty.value, notes, "COMPLETED",
                               usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), usage.get("total_tokens", 0))
    return certainty, notes, raw_response

async def queue_and_await_approval(record: QueueRecord, step_order: int) -> bool:
    await db_record_subrequest(record.id, record.turn_id, step_order, 0.0, f"GATE_{record.spec_type.upper()}",
                               True, record.target_provider, record.prompt_text, "", record.certainty.value, record.audit_notes, record.status.value)
    
    if state.policy == GatePolicy.AUTO_APPROVE or (state.policy == GatePolicy.CONFIDENCE_BASED and record.certainty == AuditCertainty.HIGH_CERTAINTY_CLEAN):
        record.status = ApprovalStatus.APPROVED
        await db_record_subrequest(record.id, record.turn_id, step_order, 0.0, f"GATE_{record.spec_type.upper()}",
                                   True, record.target_provider, record.prompt_text, "", record.certainty.value, record.audit_notes, "APPROVED_AUTO")
        return True

    async with state_lock: state.queue[record.id] = record
    try:
        await asyncio.wait_for(record.event.wait(), timeout=900.0)
    except asyncio.TimeoutError:
        async with state_lock: state.queue.pop(record.id, None)
        await db_record_subrequest(record.id, record.turn_id, step_order, 0.0, f"GATE_{record.spec_type.upper()}",
                                   True, record.target_provider, record.prompt_text, "", record.certainty.value, record.audit_notes, "REJECTED_TIMEOUT")
        return False
    except asyncio.CancelledError:
        async with state_lock: state.queue.pop(record.id, None)
        raise

    async with state_lock: state.queue.pop(record.id, None)
    await db_record_subrequest(record.id, record.turn_id, step_order, 0.0, f"GATE_{record.spec_type.upper()}",
                               True, record.target_provider, record.prompt_text, "", record.certainty.value, record.audit_notes, f"MANUAL_{record.status.value.upper()}")
    return record.status == ApprovalStatus.APPROVED

def extract_specs(raw_text: str) -> tuple[Optional[str], Optional[str]]:
    s_match = re.search(r"===\s*SOLVER_SPEC\s*===(.*?)(?:===\s*FALSIFICATION_SPEC\s*===|$)", raw_text, re.DOTALL | re.IGNORECASE)
    f_match = re.search(r"===\s*FALSIFICATION_SPEC\s*===(.*?)$", raw_text, re.DOTALL | re.IGNORECASE)
    return s_match.group(1).strip() if s_match else None, f_match.group(1).strip() if f_match else None

# ==========================================
# Core Adaptive Pipeline (Sequential Logic Fix)
# ==========================================
async def run_adaptive_pipeline(turn_id: str, user_query: str, conversation_history: List[Dict[str, str]]):
    await db_start_turn(turn_id, user_query)
    step = 1

    thinking_open = True
    yield format_sse("<think>\n=== 1. LOCAL TRIAGE ===\n")

    try:
        triage_sys = (
            "You are an air-gapped scientific lead architect. Analyze the user's query in context.\n"
            "Determine if this request requires delegating a dense mathematical/algorithmic bottleneck "
            "to external frontier oracle solvers, or if it can be resolved locally.\n\n"
            "Format your decision STRICTLY as:\nDECISION: [LOCAL or DELEGATE]\nREASON: [Brief justification]\n\n"
            "If DELEGATE, append:\n===SOLVER_SPEC===\n<stateless abstract prompt>\n===FALSIFICATION_SPEC===\n<stateless declarative proposition>"
        )

        triage_payload = build_payload(triage_sys, conversation_history, f"EVALUATE AND ROUTE THIS REQUEST:\n{user_query}")
        
        triage_raw = ""
        async for c_content, c_reasoning, finish_data in stream_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, triage_payload):
            if c_reasoning: yield format_sse(c_reasoning)
            if c_content:
                triage_raw += c_content
                yield format_sse(c_content)
            if finish_data:
                t_lat, t_usage = finish_data
                await db_record_subrequest(str(uuid.uuid4()), turn_id, step, t_lat, "LOCAL_TRIAGE", False, f"Local ({LOCAL_MODEL_NAME})", user_query, triage_raw, "N/A", "Assessed routing", "COMPLETED", t_usage.get("prompt_tokens", 0), t_usage.get("completion_tokens", 0), t_usage.get("total_tokens", 0))
        step += 1

        solver_spec, falsification_spec = extract_specs(triage_raw)
        needs_delegation = bool(re.search(r"DECISION:\s*DELEGATE", triage_raw, re.IGNORECASE)) and bool(solver_spec)

        # ====================================================
        # BRANCH A: PURE LOCAL RESOLUTION
        # ====================================================
        if not needs_delegation:
            await db_update_turn_mode(turn_id, "LOCAL_ONLY")
            local_sys = "You are an expert scientific assistant. Provide a comprehensive answer directly."
            
            yield format_sse("\n\n=== 2. LOCAL EXECUTION ===\n")
            
            direct_ans = ""
            chat_payload = build_payload(local_sys, conversation_history, user_query)
            async for c_content, c_reasoning, finish_data in stream_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, chat_payload):
                if c_reasoning: yield format_sse(c_reasoning)
                if c_content:
                    if thinking_open:
                        yield format_sse("\n</think>\n\n")
                        thinking_open = False
                    direct_ans += c_content
                    yield format_sse(c_content)
                if finish_data:
                    a_lat, a_usage = finish_data
                    await db_record_subrequest(str(uuid.uuid4()), turn_id, step, a_lat, "LOCAL_EXECUTION", False, f"Local ({LOCAL_MODEL_NAME})", user_query, direct_ans, "N/A", "Initial local resolution", "COMPLETED", a_usage.get("prompt_tokens", 0), a_usage.get("completion_tokens", 0), a_usage.get("total_tokens", 0))
            step += 1

            correction_round = 0
            while correction_round < MAX_CORRECTION_ROUNDS:
                correction_round += 1
                
                if not thinking_open:
                    yield format_sse(f"\n\n<think>\n=== AUDIT VERIFICATION (Round {correction_round}) ===\n")
                    thinking_open = True
                else:
                    yield format_sse(f"\n\n=== AUDIT VERIFICATION (Round {correction_round}) ===\n")

                falsify_prompt = f"Analyze this explanation:\n\n{direct_ans[:2500]}\n\nIf an admissible flaw exists, output: 'VERDICT: FALSIFIED' and explain.\nIf unconditionally sound, output: 'VERDICT: SOUND'"
                falsify_res = ""
                async for c_content, c_reasoning, finish_data in stream_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, [{"role": "system", "content": falsify_prompt}, {"role": "user", "content": direct_ans[:2000]}]):
                    if c_reasoning: yield format_sse(c_reasoning)
                    if c_content:
                        falsify_res += c_content
                        yield format_sse(c_content)
                    if finish_data:
                        f_lat, f_usage = finish_data
                        await db_record_subrequest(str(uuid.uuid4()), turn_id, step, f_lat, f"LOCAL_FALSIFICATION_CHECK_R{correction_round}", False, f"Local ({LOCAL_MODEL_NAME})", direct_ans[:500] + "...", falsify_res, "N/A", f"Falsification round {correction_round}", "COMPLETED", f_usage.get("prompt_tokens", 0), f_usage.get("completion_tokens", 0), f_usage.get("total_tokens", 0))
                step += 1

                if "VERDICT: FALSIFIED" in falsify_res:
                    if correction_round < MAX_CORRECTION_ROUNDS:
                        counterexample = falsify_res.split("VERDICT: FALSIFIED")[-1].strip()
                        yield format_sse("\n\n=== RE-GENERATING SOLUTION ===\n")
                        re_prompt = f"ORIGINAL GOAL: {user_query}\n\nYOUR PREVIOUS ANSWER:\n{direct_ans}\n\nREFEREE'S REFUTATION:\n{counterexample}\n\nRe-answer comprehensively, explicitly resolving this counterexample."
                        direct_ans = ""
                        async for c_content, c_reasoning, finish_data in stream_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, build_payload(local_sys, conversation_history, re_prompt)):
                            if c_reasoning: yield format_sse(c_reasoning)
                            if c_content:
                                if thinking_open:
                                    yield format_sse("\n</think>\n\n")
                                    thinking_open = False
                                direct_ans += c_content
                                yield format_sse(c_content)
                            if finish_data:
                                r_lat, r_usage = finish_data
                                await db_record_subrequest(str(uuid.uuid4()), turn_id, step, r_lat, f"LOCAL_CORRECTION_REANSWER_R{correction_round}", False, f"Local ({LOCAL_MODEL_NAME})", counterexample, direct_ans, "N/A", "Corrected answer re-synthesized", "COMPLETED", r_usage.get("prompt_tokens", 0), r_usage.get("completion_tokens", 0), r_usage.get("total_tokens", 0))
                        step += 1
                    else:
                        yield format_sse(f"\n\n⚠️ Boundary Warning: Verification unresolved after {MAX_CORRECTION_ROUNDS} rounds.\n")
                        break
                else:
                    break

            if thinking_open:
                yield format_sse("\n</think>\n\n")
                thinking_open = False

            await db_finish_turn(turn_id, "COMPLETED")
            return

        # ====================================================
        # BRANCH B: HYBRID EGRESS DELEGATION
        # ====================================================
        await db_update_turn_mode(turn_id, "HYBRID_EGRESS")
        yield format_sse("\n\n=== 2. ROUTING: HYBRID EGRESS DELEGATION ===\n")

        correction_round = 0
        all_clear = False
        final_solver_res, final_verifier_res = "", ""

        while correction_round < MAX_CORRECTION_ROUNDS and not all_clear:
            correction_round += 1
            yield format_sse(f"\n=== HYBRID EXECUTION (Round {correction_round}) ===\n")

            if not falsification_spec:
                yield format_sse("> Synthesizing missing declarative proposition...\n")
                falsification_spec, f_lat, f_usg = await call_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, [{"role": "system", "content": "Extract the core mathematical invariant proposition."}, {"role": "user", "content": solver_spec}])
                await db_record_subrequest(str(uuid.uuid4()), turn_id, step, f_lat, f"PROPOSITION_SYNTHESIS_R{correction_round}", False, f"Local ({LOCAL_MODEL_NAME})", solver_spec, falsification_spec, "N/A", "Synthesized missing invariant", "COMPLETED", f_usg.get("prompt_tokens", 0), f_usg.get("completion_tokens", 0), f_usg.get("total_tokens", 0))
                step += 1

            # We audit both specifications concurrently to check for leaks
            yield format_sse("> Auditing Prompts for Leakage...\n")
            s_cert, s_notes, s_raw = await evaluate_audit_certainty(solver_spec, turn_id, step)
            v_cert, v_notes, v_raw = await evaluate_audit_certainty(falsification_spec, turn_id, step + 1)
            step += 2

            rec_solver = QueueRecord(id=str(uuid.uuid4()), turn_id=turn_id, round_idx=correction_round, created_at=time.time(), target_provider=f"Provider 1 ({PROVIDER1_MODEL_NAME})", spec_type=f"solver_r{correction_round}", prompt_text=solver_spec, certainty=s_cert, audit_notes=s_notes, evaluator_raw_response=s_raw)
            rec_verifier = QueueRecord(id=str(uuid.uuid4()), turn_id=turn_id, round_idx=correction_round, created_at=time.time(), target_provider=f"Provider 2 ({PROVIDER2_MODEL_NAME})" if PROVIDER2_ENABLED else f"Local Keystone ({LOCAL_MODEL_NAME})", spec_type=f"verifier_r{correction_round}", prompt_text=falsification_spec, certainty=v_cert, audit_notes=v_notes, evaluator_raw_response=v_raw)

            # Concurrent Gating: User approves both rules at the exact same time
            yield format_sse("> Awaiting Gateway Approvals...\n")
            if PROVIDER2_ENABLED:
                approved_s, approved_v = await asyncio.gather(queue_and_await_approval(rec_solver, step), queue_and_await_approval(rec_verifier, step + 1))
                step += 2
                if not (approved_s and approved_v): raise HTTPException(status_code=403, detail="Egress specs rejected.")
            else:
                approved_s = await queue_and_await_approval(rec_solver, step)
                step += 1
                if not approved_s: raise HTTPException(status_code=403, detail="Solver spec rejected.")

            # SEQUENTIAL EXECUTION: The Solver runs first.
            yield format_sse(f"\n--- SOLVER DERIVATION (Provider 1) ---\n")
            final_solver_res = ""
            solver_step_id = str(uuid.uuid4())
            await db_record_subrequest(solver_step_id, turn_id, step, 0.0, f"EXTERNAL_SOLVER_R{correction_round}", True, f"Provider 1 ({PROVIDER1_MODEL_NAME})", solver_spec, "[Streaming in progress...]", s_cert.value, "Frontier derivation started", "IN_PROGRESS")
            
            async for c_content, c_reasoning, finish_data in stream_llm(PROVIDER1_API_BASE, PROVIDER1_API_KEY, PROVIDER1_MODEL_NAME, [{"role": "user", "content": solver_spec}]):
                if c_reasoning: yield format_sse(c_reasoning)
                if c_content:
                    final_solver_res += c_content
                    yield format_sse(c_content)
                if finish_data:
                    lat, usg = finish_data
                    await db_record_subrequest(solver_step_id, turn_id, step, lat, f"EXTERNAL_SOLVER_R{correction_round}", True, f"Provider 1 ({PROVIDER1_MODEL_NAME})", solver_spec, final_solver_res, s_cert.value, "Frontier derivation completed", "COMPLETED", usg.get("prompt_tokens", 0), usg.get("completion_tokens", 0), usg.get("total_tokens", 0))
            step += 1

            # SEQUENTIAL EXECUTION: The Verifier evaluates the actual derived solution
            yield format_sse(f"\n\n--- REFEREE VERDICT ---\n")
            final_verifier_res = ""
            verifier_step_id = str(uuid.uuid4())
            
            actual_referee_prompt = (
                f"Evaluate the following derived solution against the stated constraints:\n\n"
                f"CONSTRAINTS / PROPOSITION:\n{falsification_spec}\n\n"
                f"DERIVED SOLUTION:\n{final_solver_res}\n\n"
                f"If the solution violates the constraints, misses requirements, or contains an admissible flaw, output 'VERDICT: FALSIFIED' and explain the flaw.\n"
                f"If the solution strictly satisfies all constraints and is unconditionally sound, output 'VERDICT: SOUND'."
            )

            verifier_api_base = PROVIDER2_API_BASE if PROVIDER2_ENABLED else LOCAL_API_BASE
            verifier_api_key = PROVIDER2_API_KEY if PROVIDER2_ENABLED else LOCAL_API_KEY
            verifier_model = PROVIDER2_MODEL_NAME if PROVIDER2_ENABLED else LOCAL_MODEL_NAME
            v_target_name = f"Provider 2 ({PROVIDER2_MODEL_NAME})" if PROVIDER2_ENABLED else f"Local Keystone ({LOCAL_MODEL_NAME})"
            v_is_ext = True if PROVIDER2_ENABLED else False

            await db_record_subrequest(verifier_step_id, turn_id, step, 0.0, f"VERIFIER_R{correction_round}", v_is_ext, v_target_name, actual_referee_prompt, "[Streaming in progress...]", v_cert.value, "Referee check started", "IN_PROGRESS")

            async for c_content, c_reasoning, finish_data in stream_llm(verifier_api_base, verifier_api_key, verifier_model, [{"role": "user", "content": actual_referee_prompt}]):
                if c_reasoning: yield format_sse(c_reasoning)
                if c_content:
                    final_verifier_res += c_content
                    yield format_sse(c_content)
                if finish_data:
                    lat, usg = finish_data
                    await db_record_subrequest(verifier_step_id, turn_id, step, lat, f"VERIFIER_R{correction_round}", v_is_ext, v_target_name, actual_referee_prompt, final_verifier_res, v_cert.value, "Impartial referee check completed", "COMPLETED", usg.get("prompt_tokens", 0), usg.get("completion_tokens", 0), usg.get("total_tokens", 0))
            step += 1

            if "VERDICT: FALSIFIED" in final_verifier_res:
                if correction_round < MAX_CORRECTION_ROUNDS:
                    counterexample = final_verifier_res.split("VERDICT: FALSIFIED")[-1].strip()
                    yield format_sse("\n=== RE-SPECIFICATION ===\n> Triggering Counter-Model Refinement...\n")
                    re_spec_prompt = (
                        f"ORIGINAL HIGH-LEVEL OBJECTIVE: {user_query}\n\nYOUR PREVIOUS SOLVER_SPEC:\n{solver_spec}\n\nYOUR PREVIOUS FALSIFICATION_SPEC:\n{falsification_spec}\n\nREFEREE'S REFUTATION:\n{counterexample}\n\n"
                        "Formulate revised formal specifications that solve the COMPLETE original goal while explicitly eliminating this counterexample.\nFormat output EXACTLY as:\n===SOLVER_SPEC===\n<revised comprehensive solver prompt>\n===FALSIFICATION_SPEC===\n<revised formal proposition and constraints>"
                    )
                    revised_raw = ""
                    async for c_content, c_reasoning, finish_data in stream_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, build_payload(triage_sys, conversation_history, re_spec_prompt)):
                        if c_reasoning: yield format_sse(c_reasoning)
                        if c_content:
                            revised_raw += c_content
                            yield format_sse(c_content)
                        if finish_data:
                            r_lat, r_usg = finish_data
                            await db_record_subrequest(str(uuid.uuid4()), turn_id, step, r_lat, f"RE_SPEC_GENERATION_R{correction_round}", False, f"Local ({LOCAL_MODEL_NAME})", counterexample, revised_raw, "N/A", "Re-synthesized formal specifications following refutation", "COMPLETED", r_usg.get("prompt_tokens", 0), r_usg.get("completion_tokens", 0), r_usg.get("total_tokens", 0))
                    step += 1

                    rev_solver, rev_falsify = extract_specs(revised_raw)
                    if rev_solver and rev_falsify:
                        solver_spec, falsification_spec = rev_solver, rev_falsify
                    else:
                        raise HTTPException(status_code=422, detail="Local enclave failed to emit valid delimiters on re-specification.")
                else:
                    yield format_sse(f"\n\n⚠️ Boundary Warning: Verification unresolved after {MAX_CORRECTION_ROUNDS} rounds.\n")
                    all_clear = False
            else:
                all_clear = True

        yield format_sse("\n=== 3. FINAL INTEGRATION ===\nSynthesizing verified solution...\n")

        assemble_sys = "You are the local integration architect. Synthesize the user's request, external derivation, and referee report into a cohesive, verified final answer. Acknowledge boundary limitations if refuted."
        assemble_user = f"ORIGINAL QUERY:\n{user_query}\n\nEXTERNAL DERIVATION:\n{final_solver_res}\n\nREFEREE REPORT:\n{final_verifier_res}"
        
        final_res = ""
        async for c_content, c_reasoning, finish_data in stream_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, build_payload(assemble_sys, conversation_history, assemble_user)):
            if c_reasoning: yield format_sse(c_reasoning)
            if c_content:
                if thinking_open:
                    yield format_sse("\n</think>\n\n")
                    thinking_open = False
                final_res += c_content
                yield format_sse(c_content)
            if finish_data:
                a_lat, a_usage = finish_data
                await db_record_subrequest(str(uuid.uuid4()), turn_id, step, a_lat, "LOCAL_ASSEMBLY", False, f"Local ({LOCAL_MODEL_NAME})", assemble_user, final_res, "N/A", "Context reintegration completed", "COMPLETED", a_usage.get("prompt_tokens", 0), a_usage.get("completion_tokens", 0), a_usage.get("total_tokens", 0))

        if thinking_open:
            yield format_sse("\n</think>\n\n")
            thinking_open = False

        await db_finish_turn(turn_id, "COMPLETED")
        return

    except asyncio.CancelledError:
        await db_finish_turn(turn_id, "CANCELLED_BY_USER")
        raise
    except BaseException as e:
        err_trace = traceback.format_exc()
        print(f"CRITICAL ERROR in turn {turn_id}:\n{err_trace}")
        await db_finish_turn(turn_id, f"FAILED: {str(e)}")
        if thinking_open:
            yield format_sse(f"\n\n=== PIPELINE ERROR ===\n{err_trace}\n</think>\n\n")
        else:
            yield format_sse(f"\n\n❌ **Pipeline Error:**\n```\n{err_trace}\n```\n")
        raise e

# ==========================================
# FastAPI Endpoints
# ==========================================
def clean_incoming_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cleaned = []
    for m in messages:
        c = m.get("content", "")
        c = re.sub(r"<think>.*?(</think>|$)", "", c, flags=re.DOTALL | re.IGNORECASE).strip()
        if c: cleaned.append({**m, "content": c})
    return cleaned

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "BorrowModel", "object": "model", "created": int(time.time()), "owned_by": "local-enclave"}]}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    raw_messages = body.get("messages", [])
    
    messages = clean_incoming_messages(raw_messages)
    user_query = next((m["content"] for m in reversed(messages) if m["role"] == "user"), None)
    if not user_query:
        raise HTTPException(status_code=400, detail="No user message provided.")

    turn_id = str(uuid.uuid4())
    history_context = messages[:-1]

    q = asyncio.Queue()

    async def producer():
        try:
            async for chunk in run_adaptive_pipeline(turn_id, user_query, history_context):
                await q.put(chunk)
            await q.put(None)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            err_trace = traceback.format_exc()
            print(f"PIPELINE EXCEPTION:\n{err_trace}")
            await q.put(format_sse(f"\n\n<think>\n=== PIPELINE ERROR ===\n{err_trace}\n</think>\n\n"))
            await q.put(None)

    async def consumer():
        prod_task = asyncio.create_task(producer())
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(q.get(), timeout=3.0)
                    if chunk is None:
                        break
                    yield chunk
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            if not prod_task.done():
                prod_task.cancel()
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT SUM(prompt_tokens), SUM(completion_tokens), SUM(total_tokens) FROM turn_subrequests WHERE turn_id = ?", (turn_id,))
        row = c.fetchone()
        conn.close()
        if row and row[2]:
            usage_chunk = {
                "id": f"chatcmpl-{turn_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "BorrowModel",
                "choices": [],
                "usage": {"prompt_tokens": row[0] or 0, "completion_tokens": row[1] or 0, "total_tokens": row[2] or 0}
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n"
        
        yield "data: [DONE]\n\n"

    return StreamingResponse(consumer(), media_type="text/event-stream")

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    with open("dashboard.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/state")
async def get_api_state():
    async with state_lock:
        return {"policy": state.policy.value, "queue": [r.model_dump() for r in state.queue.values()]}

@app.get("/api/logs")
async def get_paginated_logs(page: int = 1, page_size: int = 10):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM user_turns")
    total_turns = c.fetchone()[0]
    offset = (page - 1) * page_size
    c.execute("SELECT id, created_at, raw_query, routing_mode, status FROM user_turns ORDER BY created_at DESC LIMIT ? OFFSET ?", (page_size, offset))
    turns = c.fetchall()

    grouped_data = []
    for t in turns:
        c.execute("SELECT id, step_order, created_at, latency_ms, phase_type, is_external, target_engine, prompt, response, certainty, audit_notes, status, prompt_tokens, completion_tokens, total_tokens FROM turn_subrequests WHERE turn_id = ? ORDER BY step_order ASC", (t["id"],))
        subreqs = [dict(row) for row in c.fetchall()]
        ext_count = sum(1 for s in subreqs if s["is_external"] == 1)
        local_count = sum(1 for s in subreqs if s["is_external"] == 0)
        total_lat = sum(s["latency_ms"] for s in subreqs)
        total_toks = sum(s["total_tokens"] for s in subreqs)

        grouped_data.append({
            "turn_id": t["id"], "created_at": t["created_at"], "raw_query": t["raw_query"], "routing_mode": t["routing_mode"], "status": t["status"],
            "metrics": {"total_subrequests": len(subreqs), "external_count": ext_count, "local_count": local_count, "total_latency_ms": round(total_lat, 1), "total_tokens": total_toks},
            "subrequests": subreqs
        })
    conn.close()
    return {"page": page, "page_size": page_size, "total_turns": total_turns, "total_pages": (total_turns + page_size - 1) // page_size if total_turns > 0 else 1, "data": grouped_data}

@app.get("/api/tokens")
async def get_token_stats(page: int = 1, page_size: int = 25):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("SELECT SUM(prompt_tokens), SUM(completion_tokens), SUM(total_tokens) FROM turn_subrequests")
    totals = c.fetchone()
    
    c.execute("SELECT COUNT(DISTINCT turn_id) FROM turn_subrequests")
    total_turns = c.fetchone()[0]
    
    offset = (page - 1) * page_size
    c.execute("""
        SELECT turn_id, MAX(created_at) as started, 
               SUM(prompt_tokens) as pt, SUM(completion_tokens) as ct, SUM(total_tokens) as tt
        FROM turn_subrequests 
        GROUP BY turn_id 
        ORDER BY started DESC 
        LIMIT ? OFFSET ?
    """, (page_size, offset))
    turns = c.fetchall()
    
    data = []
    for t in turns:
        c.execute("SELECT phase_type, target_engine, prompt_tokens, completion_tokens, total_tokens FROM turn_subrequests WHERE turn_id = ? ORDER BY step_order ASC", (t["turn_id"],))
        subs = [dict(row) for row in c.fetchall()]
        data.append({
            "turn_id": t["turn_id"], "started": t["started"], "prompt_tokens": t["pt"], "completion_tokens": t["ct"], "total_tokens": t["tt"], "subrequests": subs
        })
        
    conn.close()
    return {
        "totals": {"prompt": totals[0] or 0, "completion": totals[1] or 0, "total": totals[2] or 0},
        "page": page, "page_size": page_size, "total_turns": total_turns,
        "total_pages": (total_turns + page_size - 1) // page_size if total_turns > 0 else 1,
        "data": data
    }

@app.post("/api/policy")
async def update_policy(payload: Dict[str, str]):
    new_policy = payload.get("policy")
    if new_policy in [e.value for e in GatePolicy]:
        async with state_lock: state.policy = GatePolicy(new_policy)
        return {"status": "ok", "policy": state.policy}
    raise HTTPException(status_code=400, detail="Invalid policy.")

@app.post("/api/resolve/{req_id}")
async def resolve_request(req_id: str, payload: Dict[str, str]):
    action = payload.get("action")
    async with state_lock:
        record = state.queue.get(req_id)
        if not record: raise HTTPException(status_code=404, detail="Request ID not found.")
        
        if action == "reject":
            turn_id = record.turn_id
            for sibling in list(state.queue.values()):
                if sibling.turn_id == turn_id:
                    sibling.status = ApprovalStatus.REJECTED
                    sibling.event.set()
        else:
            record.status = ApprovalStatus.APPROVED
            record.event.set()
    return {"status": "resolved", "id": req_id, "action": action}

if __name__ == "__main__":
    import uvicorn
    print(f"[*] Starting BorrowModel Proxy securely on http://{HOST}:{PORT}")
    print(f"[*] Live Dashboard: http://{HOST}:{PORT}/dashboard")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)