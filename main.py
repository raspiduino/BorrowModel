import asyncio
import json
import os
import re
import sqlite3
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
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

# Provider 2 (Independent Secondary Flash Verifier - Optional)
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
            status TEXT
        )
    """)
    # Migration: Add token tracking columns if they don't exist
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
    await safe_db_write(
        "INSERT OR REPLACE INTO user_turns (id, created_at, raw_query, routing_mode, status) VALUES (?, ?, ?, ?, ?)",
        (turn_id, time.time(), raw_query, "EVALUATING", "IN_PROGRESS")
    )

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
# Memory Sync & Approval Management
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
# LLM Transport Client & Payload Builders
# ==========================================
def build_payload(system_prompt: str, history: List[Dict[str, str]], user_prompt: str) -> List[Dict[str, str]]:
    """Guarantees strict [System, History..., User] sequence."""
    safe_history = [m for m in history if m.get("role") != "system"]
    return [{"role": "system", "content": system_prompt}] + safe_history + [{"role": "user", "content": user_prompt}]

def format_sse(text: str) -> str:
    return f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n"

async def call_llm(base_url: str, api_key: str, model: str, messages: List[Dict[str, str]], 
                   temperature: float = 0.0, max_tokens: Optional[int] = None) -> tuple[str, float, dict]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens:
        payload["max_tokens"] = max_tokens
        
    start_t = time.perf_counter()
    async with httpx.AsyncClient(timeout=180.0) as client:
        res = await client.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload)
        latency = (time.perf_counter() - start_t) * 1000.0
        if res.status_code != 200:
            raise HTTPException(status_code=502, detail=f"LLM upstream error ({base_url}): {res.text}")
        data = res.json()
        usage = data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        return data["choices"][0]["message"]["content"], latency, usage

async def stream_llm(base_url: str, api_key: str, model: str, messages: List[Dict[str, str]], temperature: float = 0.0):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model, "messages": messages, "temperature": temperature, 
        "stream": True, "stream_options": {"include_usage": True}
    }
    start_t = time.perf_counter()
    async with httpx.AsyncClient(timeout=180.0) as client:
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
                            if "content" in delta and delta["content"]:
                                yield delta["content"], None
                        if "usage" in data and data["usage"]:
                            final_usage = data["usage"]
                    except json.JSONDecodeError:
                        pass
            latency = (time.perf_counter() - start_t) * 1000.0
            yield "", (latency, final_usage)

async def evaluate_audit_certainty(abstract_prompt: str, turn_id: str, step_order: int) -> tuple[AuditCertainty, str, str]:
    system_prompt = (
        "You are an adversarial industrial espionage auditor inspecting an outgoing prompt.\n"
        "Your task is to judge whether this text leaks proprietary architecture, secret intent, or a trackable project fingerprint.\n"
        "1. Reverse Engineering Test: Can an external vendor infer the user's specific application or private domain?\n"
        "2. Contextual Constant Test: Inspect numerical constants.\n"
        "3. Multi-turn Correlation: Is the prompt strictly stateless?\n\n"
        "Respond STRICTLY in JSON format with two keys:\n"
        "- 'certainty': Exactly one of ['high_certainty_clean', 'uncertain', 'flagged_leak']\n"
        "- 'notes': Clear, point-by-point explanation of reasoning."
    )
    raw_response, latency, usage = await call_llm(
        LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME,
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": abstract_prompt}],
        temperature=0.0
    )
    
    op_id = str(uuid.uuid4())
    try:
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else json.loads(raw_response)
        certainty = AuditCertainty(parsed.get("certainty", "uncertain"))
        notes = parsed.get("notes", "No detailed reasoning provided.")
    except Exception as e:
        certainty = AuditCertainty.UNCERTAIN
        notes = f"Audit JSON parse error: {str(e)}"

    await db_record_subrequest(op_id, turn_id, step_order, latency, "INTERNAL_AUDIT", False,
                               f"Local ({LOCAL_MODEL_NAME})", abstract_prompt, raw_response,
                               certainty.value, notes, "COMPLETED", usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), usage.get("total_tokens", 0))
    return certainty, notes, raw_response

async def queue_and_await_approval(record: QueueRecord, step_order: int) -> bool:
    await db_record_subrequest(record.id, record.turn_id, step_order, 0.0, f"GATE_{record.spec_type.upper()}",
                               True, record.target_provider, record.prompt_text, "",
                               record.certainty.value, record.audit_notes, record.status.value)

    if state.policy == GatePolicy.AUTO_APPROVE:
        record.status = ApprovalStatus.APPROVED
        await db_record_subrequest(record.id, record.turn_id, step_order, 0.0, f"GATE_{record.spec_type.upper()}",
                                   True, record.target_provider, record.prompt_text, "",
                                   record.certainty.value, record.audit_notes, "APPROVED_AUTO")
        return True

    if state.policy == GatePolicy.CONFIDENCE_BASED and record.certainty == AuditCertainty.HIGH_CERTAINTY_CLEAN:
        record.status = ApprovalStatus.APPROVED
        await db_record_subrequest(record.id, record.turn_id, step_order, 0.0, f"GATE_{record.spec_type.upper()}",
                                   True, record.target_provider, record.prompt_text, "",
                                   record.certainty.value, record.audit_notes, "APPROVED_CONFIDENCE")
        return True

    async with state_lock:
        state.queue[record.id] = record

    try:
        await asyncio.wait_for(record.event.wait(), timeout=900.0)
    except asyncio.TimeoutError:
        async with state_lock:
            record.status = ApprovalStatus.REJECTED
            state.queue.pop(record.id, None)
        await db_record_subrequest(record.id, record.turn_id, step_order, 0.0, f"GATE_{record.spec_type.upper()}",
                                   True, record.target_provider, record.prompt_text, "",
                                   record.certainty.value, record.audit_notes, "REJECTED_TIMEOUT")
        return False
    except asyncio.CancelledError:
        async with state_lock:
            state.queue.pop(record.id, None)
        raise

    async with state_lock:
        state.queue.pop(record.id, None)

    await db_record_subrequest(record.id, record.turn_id, step_order, 0.0, f"GATE_{record.spec_type.upper()}",
                               True, record.target_provider, record.prompt_text, "",
                               record.certainty.value, record.audit_notes, f"MANUAL_{record.status.value.upper()}")
    return record.status == ApprovalStatus.APPROVED

def extract_specs(raw_text: str) -> tuple[Optional[str], Optional[str]]:
    solver_match = re.search(r"===\s*SOLVER_SPEC\s*===(.*?)(?:===\s*FALSIFICATION_SPEC\s*===|$)", raw_text, re.DOTALL | re.IGNORECASE)
    falsify_match = re.search(r"===\s*FALSIFICATION_SPEC\s*===(.*?)$", raw_text, re.DOTALL | re.IGNORECASE)
    
    solver_spec = solver_match.group(1).strip() if solver_match else None
    falsify_spec = falsify_match.group(1).strip() if falsify_match else None
    return solver_spec, falsify_spec

# ==========================================
# Core Adaptive Pipeline (Streaming Async Gen)
# ==========================================
async def run_adaptive_pipeline(turn_id: str, user_query: str, conversation_history: List[Dict[str, str]]):
    await db_start_turn(turn_id, user_query)
    step = 1

    try:
        triage_sys = (
            "You are an air-gapped scientific lead architect. Analyze the user's query in context.\n"
            "Determine if this request requires delegating a dense formal/mathematical bottleneck "
            "to external frontier oracle solvers, or if it can be resolved locally.\n\n"
            "Format your decision STRICTLY as:\n"
            "DECISION: [LOCAL or DELEGATE]\n"
            "REASON: [Brief justification]\n\n"
            "If DELEGATE, append:\n"
            "===SOLVER_SPEC===\n"
            "<stateless abstract implementation prompt>\n"
            "===FALSIFICATION_SPEC===\n"
            "<stateless formal proposition for a referee>"
        )

        triage_payload = build_payload(triage_sys, conversation_history, f"EVALUATE AND ROUTE THIS REQUEST:\n{user_query}")
        triage_raw, t_lat, t_usage = await call_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, triage_payload)
        
        solver_spec, falsification_spec = extract_specs(triage_raw)
        needs_delegation = bool(re.search(r"DECISION:\s*DELEGATE", triage_raw, re.IGNORECASE)) and bool(solver_spec)
        
        await db_record_subrequest(str(uuid.uuid4()), turn_id, step, t_lat, "LOCAL_TRIAGE", False,
                                   f"Local ({LOCAL_MODEL_NAME})", user_query, triage_raw, "N/A", 
                                   "Assessed routing", "COMPLETED", t_usage.get("prompt_tokens", 0), t_usage.get("completion_tokens", 0), t_usage.get("total_tokens", 0))
        step += 1

        # ====================================================
        # BRANCH A: PURE LOCAL RESOLUTION
        # ====================================================
        if not needs_delegation:
            await db_update_turn_mode(turn_id, "LOCAL_ONLY")
            
            local_sys = (
                "You are an expert scientific assistant operating in a secure local enclave. "
                "Provide a comprehensive, rigorous answer directly from your knowledge base."
            )
            chat_payload = build_payload(local_sys, conversation_history, user_query)
            
            # Stream Initial Local Generation
            yield format_sse("<think>\n=== LOCAL GENERATION ===\n</think>\n")
            direct_ans = ""
            async for chunk, finish_data in stream_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, chat_payload):
                if chunk:
                    direct_ans += chunk
                    yield format_sse(chunk)
                if finish_data:
                    a_lat, a_usage = finish_data
                    await db_record_subrequest(str(uuid.uuid4()), turn_id, step, a_lat, "LOCAL_EXECUTION", False,
                                               f"Local ({LOCAL_MODEL_NAME})", user_query, direct_ans, "N/A",
                                               "Initial local resolution", "COMPLETED", 
                                               a_usage.get("prompt_tokens", 0), a_usage.get("completion_tokens", 0), a_usage.get("total_tokens", 0))
            step += 1

            correction_round = 0
            while correction_round < MAX_CORRECTION_ROUNDS:
                correction_round += 1
                falsify_prompt = (
                    "You are an impartial formal referee. Analyze this explanation:\n\n"
                    f"{direct_ans[:2500]}\n\n"
                    "If an admissible mathematical/logical flaw exists, output: 'VERDICT: FALSIFIED' and explain the counterexample.\n"
                    "If universally sound, output: 'VERDICT: SOUND'"
                )
                falsify_res, f_lat, f_usage = await call_llm(
                    LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME,
                    [{"role": "system", "content": falsify_prompt}, {"role": "user", "content": direct_ans[:2000]}]
                )
                await db_record_subrequest(str(uuid.uuid4()), turn_id, step, f_lat, f"LOCAL_FALSIFICATION_CHECK_R{correction_round}", False,
                                           f"Local ({LOCAL_MODEL_NAME})", direct_ans[:500] + "...", falsify_res, "N/A",
                                           f"Falsification round {correction_round}", "COMPLETED",
                                           f_usage.get("prompt_tokens", 0), f_usage.get("completion_tokens", 0), f_usage.get("total_tokens", 0))
                step += 1

                if "VERDICT: FALSIFIED" in falsify_res:
                    if correction_round < MAX_CORRECTION_ROUNDS:
                        counterexample = falsify_res.split("VERDICT: FALSIFIED")[-1].strip()
                        yield format_sse(f"\n\n<think>\n=== REFEREE REFUTATION ===\n{falsify_res}\n\n=== RE-GENERATING ===\n</think>\n")
                        
                        re_prompt = (
                            f"ORIGINAL GOAL: {user_query}\n\n"
                            f"YOUR PREVIOUS FLAWED ANSWER:\n{direct_ans}\n\n"
                            f"REFEREE'S REFUTATION & COUNTEREXAMPLE:\n{counterexample}\n\n"
                            "Re-answer the original question comprehensively, resolving this counterexample."
                        )
                        re_payload = build_payload(local_sys, conversation_history, re_prompt)
                        
                        direct_ans = ""
                        async for chunk, finish_data in stream_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, re_payload):
                            if chunk:
                                direct_ans += chunk
                                yield format_sse(chunk)
                            if finish_data:
                                r_lat, r_usage = finish_data
                                await db_record_subrequest(str(uuid.uuid4()), turn_id, step, r_lat, f"LOCAL_CORRECTION_REANSWER_R{correction_round}", False,
                                                           f"Local ({LOCAL_MODEL_NAME})", counterexample, direct_ans, "N/A",
                                                           f"Corrected answer re-synthesized", "COMPLETED",
                                                           r_usage.get("prompt_tokens", 0), r_usage.get("completion_tokens", 0), r_usage.get("total_tokens", 0))
                        step += 1
                    else:
                        yield format_sse(f"\n\n<think>\n⚠️️ Boundary Warning: Verification unresolved after {MAX_CORRECTION_ROUNDS} rounds:\n{falsify_res}\n</think>\n")
                        break
                else:
                    break

            await db_finish_turn(turn_id, "COMPLETED")
            return

        # ====================================================
        # BRANCH B: HYBRID EGRESS DELEGATION (Frontier Solver + Keystone Verifier)
        # ====================================================
        await db_update_turn_mode(turn_id, "HYBRID_EGRESS")

        correction_round = 0
        all_clear = False
        final_solver_res = ""
        final_verifier_res = ""

        while correction_round < MAX_CORRECTION_ROUNDS and not all_clear:
            correction_round += 1

            if not falsification_spec:
                prop_sys = "Extract the mathematical invariant asserted by this problem into a declarative statement."
                falsification_spec, f_lat, f_usage = await call_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, 
                                                           [{"role": "system", "content": prop_sys}, {"role": "user", "content": solver_spec}])
                await db_record_subrequest(str(uuid.uuid4()), turn_id, step, f_lat, f"PROPOSITION_SYNTHESIS_R{correction_round}", False,
                                           f"Local ({LOCAL_MODEL_NAME})", solver_spec, falsification_spec, "N/A",
                                           "Synthesized missing declarative invariant", "COMPLETED",
                                           f_usage.get("prompt_tokens", 0), f_usage.get("completion_tokens", 0), f_usage.get("total_tokens", 0))
                step += 1

            referee_prompt = (
                "You are an impartial formal referee. Evaluate this proposition:\n\n"
                f"{falsification_spec}\n\n"
                "If an admissible counterexample exists, state: 'VERDICT: FALSIFIED' and explain.\n"
                "If unconditionally true, state: 'VERDICT: SOUND'"
            )

            s_cert, s_notes, s_raw = await evaluate_audit_certainty(solver_spec, turn_id, step)
            v_cert, v_notes, v_raw = await evaluate_audit_certainty(referee_prompt, turn_id, step + 1)
            step += 2

            rec_solver = QueueRecord(
                id=str(uuid.uuid4()), turn_id=turn_id, round_idx=correction_round, created_at=time.time(),
                target_provider=f"Provider 1 ({PROVIDER1_MODEL_NAME})", spec_type=f"solver_r{correction_round}",
                prompt_text=solver_spec, certainty=s_cert, audit_notes=s_notes, evaluator_raw_response=s_raw
            )
            rec_verifier = QueueRecord(
                id=str(uuid.uuid4()), turn_id=turn_id, round_idx=correction_round, created_at=time.time(),
                target_provider=f"Provider 2 Flash ({PROVIDER2_MODEL_NAME})" if PROVIDER2_ENABLED else f"Local Keystone ({LOCAL_MODEL_NAME})",
                spec_type=f"verifier_r{correction_round}", prompt_text=referee_prompt, certainty=v_cert, audit_notes=v_notes, evaluator_raw_response=v_raw
            )

            if PROVIDER2_ENABLED:
                approved_s, approved_v = await asyncio.gather(
                    queue_and_await_approval(rec_solver, step),
                    queue_and_await_approval(rec_verifier, step + 1)
                )
                step += 2
                if not (approved_s and approved_v):
                    raise HTTPException(status_code=403, detail="Egress specs rejected or timed out.")
            else:
                approved_s = await queue_and_await_approval(rec_solver, step)
                step += 1
                if not approved_s:
                    raise HTTPException(status_code=403, detail="Solver spec rejected or timed out.")

            async def run_p1(step_idx: int):
                ans, lat, usage = await call_llm(PROVIDER1_API_BASE, PROVIDER1_API_KEY, PROVIDER1_MODEL_NAME, [{"role": "user", "content": solver_spec}])
                await db_record_subrequest(str(uuid.uuid4()), turn_id, step_idx, lat, f"EXTERNAL_SOLVER_R{correction_round}", True,
                                           f"Provider 1 ({PROVIDER1_MODEL_NAME})", solver_spec, ans, s_cert.value,
                                           "Frontier derivation completed", "COMPLETED", usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), usage.get("total_tokens", 0))
                return ans

            async def run_verifier(step_idx: int):
                if PROVIDER2_ENABLED:
                    ans, lat, usage = await call_llm(PROVIDER2_API_BASE, PROVIDER2_API_KEY, PROVIDER2_MODEL_NAME, [{"role": "user", "content": referee_prompt}], max_tokens=512)
                    await db_record_subrequest(str(uuid.uuid4()), turn_id, step_idx, lat, f"BLIND_FALSIFICATION_VERIFIER_R{correction_round}", True,
                                               f"Provider 2 ({PROVIDER2_MODEL_NAME})", referee_prompt, ans, v_cert.value,
                                               "Impartial referee check completed", "COMPLETED", usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), usage.get("total_tokens", 0))
                    return ans
                else:
                    ans, lat, usage = await call_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, [{"role": "user", "content": referee_prompt}], max_tokens=512)
                    await db_record_subrequest(str(uuid.uuid4()), turn_id, step_idx, lat, f"LOCAL_KEYSTONE_VERIFIER_R{correction_round}", False,
                                               f"Local Keystone ({LOCAL_MODEL_NAME})", referee_prompt, ans, "HIGH_CERTAINTY_CLEAN",
                                               "Impartial local referee check completed", "COMPLETED", usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), usage.get("total_tokens", 0))
                    return ans

            solver_res, verifier_res = await asyncio.gather(run_p1(step), run_verifier(step + 1))
            step += 2

            final_solver_res = solver_res
            final_verifier_res = verifier_res

            # Yield the reasoning traces for the user's UI
            yield format_sse(f"<think>\n=== ROUND {correction_round} SOLVER ===\n{final_solver_res}\n\n=== ROUND {correction_round} VERIFIER ===\n{final_verifier_res}\n</think>\n\n")

            if "VERDICT: FALSIFIED" in verifier_res:
                if correction_round < MAX_CORRECTION_ROUNDS:
                    counterexample = verifier_res.split("VERDICT: FALSIFIED")[-1].strip()
                    re_spec_prompt = (
                        f"ORIGINAL HIGH-LEVEL OBJECTIVE: {user_query}\n\n"
                        f"YOUR PREVIOUS SOLVER_SPEC:\n{solver_spec}\n\n"
                        f"YOUR PREVIOUS FALSIFICATION_SPEC:\n{falsification_spec}\n\n"
                        f"REFEREE'S REFUTATION:\n{counterexample}\n\n"
                        "Formulate revised formal specifications that solve the COMPLETE original goal while explicitly eliminating this counterexample.\n"
                        "Format output EXACTLY as:\n"
                        "===SOLVER_SPEC===\n<revised comprehensive solver prompt>\n"
                        "===FALSIFICATION_SPEC===\n<revised formal proposition and constraints>"
                    )
                    
                    re_payload = build_payload(triage_sys, conversation_history, re_spec_prompt)
                    revised_raw, r_lat, r_usage = await call_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, re_payload)
                    await db_record_subrequest(str(uuid.uuid4()), turn_id, step, r_lat, f"RE_SPEC_GENERATION_R{correction_round}", False,
                                               f"Local ({LOCAL_MODEL_NAME})", counterexample, revised_raw, "N/A",
                                               "Re-synthesized formal specifications following refutation", "COMPLETED",
                                               r_usage.get("prompt_tokens", 0), r_usage.get("completion_tokens", 0), r_usage.get("total_tokens", 0))
                    step += 1
                    
                    yield format_sse(f"<think>\n=== RE-SPECIFICATION ===\n{revised_raw}\n</think>\n\n")

                    rev_solver, rev_falsify = extract_specs(revised_raw)
                    if rev_solver and rev_falsify:
                        solver_spec = rev_solver
                        falsification_spec = rev_falsify
                    else:
                        raise HTTPException(status_code=422, detail="Local enclave failed to emit valid delimiters on re-specification.")
                else:
                    yield format_sse(f"<think>\n⚠️ Boundary Warning: Verification unresolved after {MAX_CORRECTION_ROUNDS} rounds:\n{final_verifier_res}\n</think>\n")
                    all_clear = False
            else:
                all_clear = True

        assemble_sys = (
            "You are the local integration architect. You have:\n"
            "1. The user's original request.\n"
            "2. An externally derived analytical solution.\n"
            "3. An independent referee verification report.\n\n"
            "Synthesize these into a cohesive, verified, domain-contextualized final answer."
        )
        assemble_user = f"ORIGINAL QUERY:\n{user_query}\n\nEXTERNAL DERIVATION:\n{final_solver_res}\n\nREFEREE REPORT:\n{final_verifier_res}"
        assemble_payload = build_payload(assemble_sys, conversation_history, assemble_user)
        
        # Stream Final Assembly
        yield format_sse("<think>\n=== FINAL ASSEMBLY (Local) ===\n</think>\n")
        final_res = ""
        async for chunk, finish_data in stream_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, assemble_payload):
            if chunk:
                final_res += chunk
                yield format_sse(chunk)
            if finish_data:
                a_lat, a_usage = finish_data
                await db_record_subrequest(str(uuid.uuid4()), turn_id, step, a_lat, "LOCAL_ASSEMBLY", False,
                                           f"Local ({LOCAL_MODEL_NAME})", assemble_user, final_res, "N/A",
                                           "Context reintegration completed", "COMPLETED",
                                           a_usage.get("prompt_tokens", 0), a_usage.get("completion_tokens", 0), a_usage.get("total_tokens", 0))

        await db_finish_turn(turn_id, "COMPLETED")
        return

    except asyncio.CancelledError:
        await db_finish_turn(turn_id, "CANCELLED_BY_USER")
        raise
    except BaseException as e:
        await db_finish_turn(turn_id, f"FAILED: {str(e)}")
        raise e

# ==========================================
# FastAPI Endpoints & Context Cleaner
# ==========================================
def clean_incoming_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cleaned = []
    for m in messages:
        c = m.get("content", "")
        # Remove past reasoning thoughts to drastically save context space
        c = re.sub(r"<think>.*?</think>", "", c, flags=re.DOTALL | re.IGNORECASE).strip()
        if c:
            cleaned.append({**m, "content": c})
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

    # Use a Queue to separate the background processing loop from the keep-alive loop
    q = asyncio.Queue()

    async def producer():
        try:
            async for chunk in run_adaptive_pipeline(turn_id, user_query, history_context):
                await q.put(chunk)
            await q.put(None) # Signal EOF
        except asyncio.CancelledError:
            pass
        except Exception as e:
            await q.put(format_sse(f"\n\n<think>Pipeline Error: {str(e)}</think>\n\n"))
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
        yield "data: [DONE]\n\n"

    return StreamingResponse(consumer(), media_type="text/event-stream")

# ==========================================
# Dashboard & Non-Blocking Read Endpoints
# ==========================================
@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    with open("dashboard.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/state")
async def get_api_state():
    async with state_lock:
        return {
            "policy": state.policy.value,
            "queue": [r.model_dump() for r in state.queue.values()]
        }

@app.get("/api/logs")
async def get_paginated_logs(page: int = 1, page_size: int = 10):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM user_turns")
    total_turns = c.fetchone()[0]

    offset = (page - 1) * page_size
    c.execute("""
        SELECT id, created_at, raw_query, routing_mode, status 
        FROM user_turns 
        ORDER BY created_at DESC 
        LIMIT ? OFFSET ?
    """, (page_size, offset))
    turns = c.fetchall()

    grouped_data = []
    for t in turns:
        t_id = t["id"]
        c.execute("""
            SELECT id, step_order, created_at, latency_ms, phase_type, is_external, 
                   target_engine, prompt, response, certainty, audit_notes, status, prompt_tokens, completion_tokens, total_tokens
            FROM turn_subrequests 
            WHERE turn_id = ? 
            ORDER BY step_order ASC, created_at ASC
        """, (t_id,))
        subreqs = [dict(row) for row in c.fetchall()]
        
        ext_count = sum(1 for s in subreqs if s["is_external"] == 1)
        local_count = sum(1 for s in subreqs if s["is_external"] == 0)
        total_lat = sum(s["latency_ms"] for s in subreqs)

        grouped_data.append({
            "turn_id": t_id,
            "created_at": t["created_at"],
            "raw_query": t["raw_query"],
            "routing_mode": t["routing_mode"],
            "status": t["status"],
            "metrics": {
                "total_subrequests": len(subreqs),
                "external_count": ext_count,
                "local_count": local_count,
                "total_latency_ms": round(total_lat, 1)
            },
            "subrequests": subreqs
        })

    conn.close()
    return {
        "page": page,
        "page_size": page_size,
        "total_turns": total_turns,
        "total_pages": (total_turns + page_size - 1) // page_size if total_turns > 0 else 1,
        "data": grouped_data
    }

@app.post("/api/policy")
async def update_policy(payload: Dict[str, str]):
    new_policy = payload.get("policy")
    if new_policy in [e.value for e in GatePolicy]:
        async with state_lock:
            state.policy = GatePolicy(new_policy)
        return {"status": "ok", "policy": state.policy}
    raise HTTPException(status_code=400, detail="Invalid policy.")

@app.post("/api/resolve/{req_id}")
async def resolve_request(req_id: str, payload: Dict[str, str]):
    action = payload.get("action")
    async with state_lock:
        record = state.queue.get(req_id)
        if not record:
            raise HTTPException(status_code=404, detail="Request ID not found in queue.")
        
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
    print(f"[*] SQLite Disk Logging Enabled: {DB_FILE} (WAL mode)")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)