import asyncio
import json
import os
import random
import re
import sqlite3
import time
import uuid
import traceback
from enum import Enum
from typing import Any, Dict, List, Optional
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
import httpx

# ==========================================
# Configuration & Environment
# ==========================================
HOST = os.getenv("PROXY_HOST", "127.0.0.1")
PORT = int(os.getenv("PROXY_PORT", "9908"))

LOCAL_API_BASE = os.getenv("LOCAL_API_BASE", "http://127.0.0.1:11434/v1")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "none")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "Qwen3.8-27B")

PROVIDER1_API_BASE = os.getenv("PROVIDER1_API_BASE", "https://api.openai.com/v1")
PROVIDER1_API_KEY = os.getenv("PROVIDER1_API_KEY", "sk-...")
PROVIDER1_MODEL_NAME = os.getenv("PROVIDER1_MODEL_NAME", "gpt-4o")

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
        try: c.execute(f"ALTER TABLE turn_subrequests ADD COLUMN {col} INTEGER DEFAULT 0;")
        except sqlite3.OperationalError: pass
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
    await safe_db_write("INSERT OR REPLACE INTO user_turns (id, created_at, raw_query, routing_mode, status) VALUES (?, ?, ?, ?, ?)", (turn_id, time.time(), raw_query, "EVALUATING", "IN_PROGRESS"))
async def db_update_turn_mode(turn_id: str, routing_mode: str):
    await safe_db_write("UPDATE user_turns SET routing_mode = ? WHERE id = ?", (routing_mode, turn_id))
async def db_finish_turn(turn_id: str, status: str = "COMPLETED"):
    await safe_db_write("UPDATE user_turns SET status = ? WHERE id = ?", (status, turn_id))
async def db_record_subrequest(subreq_id: str, turn_id: str, step_order: int, latency_ms: float, phase_type: str, is_external: bool, target: str, prompt: str, response: str = "", certainty: str = "N/A", audit_notes: str = "", status: str = "COMPLETED", prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0):
    await safe_db_write("INSERT OR REPLACE INTO turn_subrequests (id, turn_id, step_order, created_at, latency_ms, phase_type, is_external, target_engine, prompt, response, certainty, audit_notes, status, prompt_tokens, completion_tokens, total_tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (subreq_id, turn_id, step_order, time.time(), latency_ms, phase_type, 1 if is_external else 0, target, prompt, response, certainty, audit_notes, status, prompt_tokens, completion_tokens, total_tokens))

# ==========================================
# In-Memory State & Task Registry
# ==========================================
class GatePolicy(str, Enum):
    AUTO_APPROVE = "auto"
    MANUAL_APPROVE_ALL = "manual"
    CONFIDENCE_BASED = "confidence"

class AuditCertainty(str, Enum):
    HIGH_CERTAINTY_CLEAN = "high_certainty_clean"
    UNCERTAIN = "uncertain"
    FLAGGED_LEAK = "flagged_leak"
    DECOY = "decoy"

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
    resolution_action: str = ""
    feedback_text: str = ""
    event: asyncio.Event = Field(default_factory=asyncio.Event, exclude=True)

class ProxyState:
    def __init__(self):
        self.policy: GatePolicy = GatePolicy.MANUAL_APPROVE_ALL
        self.assembly_policy: GatePolicy = GatePolicy.AUTO_APPROVE
        self.force_delegate: bool = False
        self.auto_redact: bool = True
        self.enable_verifier: bool = True
        self.enable_decoys: bool = True
        self.queue: Dict[str, QueueRecord] = {}

state = ProxyState()
active_turn_tasks: Dict[str, asyncio.Task] = {}  
turn_background_tasks: Dict[str, set[asyncio.Task]] = defaultdict(set)
global_external_cache: Dict[str, Dict] = {} 

# ==========================================
# Transports & Helpers
# ==========================================
def build_payload(system_prompt: str, history: List[Dict[str, str]], user_prompt: str) -> List[Dict[str, str]]:
    safe_history = [m for m in history if m.get("role") != "system"]
    return [{"role": "system", "content": system_prompt}] + safe_history + [{"role": "user", "content": user_prompt}]

def format_sse(text: str) -> str: return f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n"

def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

def encapsulate_external_think(text: str) -> str:
    return re.sub(r"<think>(.*?)</think>", r"<external_reasoning>\1</external_reasoning>", text, flags=re.DOTALL | re.IGNORECASE)

async def call_llm(base_url: str, api_key: str, model: str, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: Optional[int] = None):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "stream": False}
    if max_tokens: payload["max_tokens"] = max_tokens
    
    start_t = time.perf_counter()
    async with httpx.AsyncClient(timeout=600.0) as client:
        res = await client.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload)
        lat = (time.perf_counter() - start_t) * 1000.0
        
        if res.status_code != 200: 
            raise HTTPException(status_code=502, detail=f"LLM upstream error ({base_url}): {res.text}")
        
        text_res = res.text
        if "text/event-stream" in res.headers.get("Content-Type", "") or text_res.strip().startswith("data:"):
            full_content = ""
            for line in text_res.splitlines():
                line = line.strip()
                if line.startswith("data:") and "[DONE]" not in line:
                    try:
                        d = json.loads(line[5:].strip())
                        if "choices" in d and d["choices"]:
                            delta = d["choices"][0].get("delta", {})
                            full_content += delta.get("content", "")
                    except: pass
            return full_content, lat, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        try:
            data = res.json()
            return data["choices"][0]["message"].get("content", ""), lat, data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON response from upstream provider: {text_res[:200]}")

async def stream_llm(base_url: str, api_key: str, model: str, messages: List[Dict[str, str]], temperature: float = 0.0):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "stream": True, "stream_options": {"include_usage": True}}
    start_t = time.perf_counter()
    async with httpx.AsyncClient(timeout=600.0) as client:
        async with client.stream("POST", f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload) as res:
            if res.status_code != 200: raise HTTPException(status_code=502, detail=f"LLM stream error: {await res.aread()}")
            final_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            async for line in res.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        data = json.loads(line[6:])
                        if "choices" in data and data["choices"]:
                            d = data["choices"][0].get("delta", {})
                            c, r = d.get("content", ""), d.get("reasoning_content", "")
                            if c or r: yield c, r, None
                        if "usage" in data and data["usage"]: final_usage = data["usage"]
                    except: pass
            yield "", "", ((time.perf_counter() - start_t) * 1000.0, final_usage)

async def evaluate_audit_certainty(abstract_prompt: str, turn_id: str, step_order: int) -> tuple[AuditCertainty, str, str]:
    sys_prompt = (
        "You are an adversarial industrial espionage auditor. Your ONLY duty is security auditing.\n\n"
        "WARNING: YOU ARE STRICTLY FORBIDDEN FROM SOLVING OR ANSWERING THE USER's PROBLEM.\n"
        "DO NOT write code. DO NOT provide mathematical derivations. DO NOT answer the user's question.\n\n"
        "Judge if the text leaks proprietary identities, confidential organization assets, or private project data.\n"
        "CRITICAL DISTINCTION: Detailed formal descriptions, mathematical formulations, theoretical equations, and standard discipline-wide scientific/engineering terminology are completely SAFE and are NOT leaks. It is expected that the text describes novel, advanced, or experimental methodologies.\n\n"
        "You must ONLY flag explicit proprietary identifiers: specific corporate/institution names, internal project codenames, private repository paths, confidential asset handles, or explicitly unnormalized production-scale metric constants.\n\n"
        "Respond STRICTLY and ONLY in JSON format: {\"certainty\": \"high_certainty_clean\"|\"uncertain\"|\"flagged_leak\", \"notes\": \"...\"}"
    )
    op_id = str(uuid.uuid4())
    try:
        raw_response, latency, usage = await call_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, [{"role": "system", "content": sys_prompt}, {"role": "user", "content": abstract_prompt}])
        clean_res = strip_think(raw_response)
        match = re.search(r"\{.*\}", clean_res, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else json.loads(clean_res)
        certainty = AuditCertainty(parsed.get("certainty", "uncertain"))
        notes = parsed.get("notes", "No detailed reasoning provided.")
    except Exception as e:
        certainty, notes, raw_response, latency, usage = AuditCertainty.UNCERTAIN, f"Audit skipped due to error: {str(e)}", "", 0.0, {}
    await db_record_subrequest(op_id, turn_id, step_order, latency, "INTERNAL_AUDIT", False, f"Local ({LOCAL_MODEL_NAME})", abstract_prompt, raw_response, certainty.value, notes, "COMPLETED", usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), usage.get("total_tokens", 0))
    return certainty, notes, raw_response

async def queue_and_await_approval(record: QueueRecord, step_order: int) -> tuple[str, str, str]:
    await db_record_subrequest(record.id, record.turn_id, step_order, 0.0, f"GATE_{record.spec_type.upper()}", True, record.target_provider, record.prompt_text, "", record.certainty.value, record.audit_notes, record.status.value)
    
    active_policy = state.assembly_policy if record.spec_type == "assembly" else state.policy

    if active_policy == GatePolicy.AUTO_APPROVE or (active_policy == GatePolicy.CONFIDENCE_BASED and record.certainty == AuditCertainty.HIGH_CERTAINTY_CLEAN):
        record.status = ApprovalStatus.APPROVED
        await db_record_subrequest(record.id, record.turn_id, step_order, 0.0, f"GATE_{record.spec_type.upper()}", True, record.target_provider, record.prompt_text, "", record.certainty.value, record.audit_notes, "APPROVED_AUTO")
        return "approve", record.prompt_text, ""

    async with state_lock: state.queue[record.id] = record
    try: await asyncio.wait_for(record.event.wait(), timeout=900.0)
    except asyncio.TimeoutError:
        async with state_lock: state.queue.pop(record.id, None)
        await db_record_subrequest(record.id, record.turn_id, step_order, 0.0, f"GATE_{record.spec_type.upper()}", True, record.target_provider, record.prompt_text, "", record.certainty.value, record.audit_notes, "REJECTED_TIMEOUT")
        return "reject", record.prompt_text, ""
    except asyncio.CancelledError:
        async with state_lock: state.queue.pop(record.id, None)
        raise

    async with state_lock: state.queue.pop(record.id, None)
    await db_record_subrequest(record.id, record.turn_id, step_order, 0.0, f"GATE_{record.spec_type.upper()}", True, record.target_provider, record.prompt_text, "", record.certainty.value, record.audit_notes, f"MANUAL_{record.resolution_action.upper()}")
    return record.resolution_action, record.prompt_text, record.feedback_text

def extract_specs(raw_text: str) -> tuple[Optional[str], Optional[str]]:
    s_matches = re.findall(r"===\s*SOLVER_SPEC\s*===(.*?)(?:===\s*FALSIFICATION_SPEC\s*===|$)", raw_text, re.DOTALL | re.IGNORECASE)
    f_matches = re.findall(r"===\s*FALSIFICATION_SPEC\s*===(.*?)$", raw_text, re.DOTALL | re.IGNORECASE)
    
    s_text = strip_think(s_matches[-1]).strip() if s_matches else None
    f_text = strip_think(f_matches[-1]).strip() if f_matches else None
    return s_text, f_text

# ==========================================
# Core Adaptive Pipeline
# ==========================================
async def run_adaptive_pipeline(turn_id: str, user_query: str, conversation_history: List[Dict[str, str]]):
    await db_start_turn(turn_id, user_query)
    step = 1
    thinking_open = True
    local_subtasks: set[asyncio.Task] = set()

    yield format_sse("<think>\n=== 1. LOCAL TRIAGE ===\n")

    try:
        # --- FAST-FORWARD CACHE CHECK ---
        cache_key = f"{len(conversation_history)}_{user_query}"
        fast_forward = False
        if cache_key in global_external_cache and not state.force_delegate:
            fast_forward = True
            yield format_sse("=== ⚡ CACHE HIT: FAST-FORWARDING TO ASSEMBLY ===\n")
            cached = global_external_cache[cache_key]
            clean_solver_res = cached["solver"]
            clean_verifier_res = cached["verifier"]
            falsification_spec = cached.get("falsification", "N/A")
            await db_record_subrequest(str(uuid.uuid4()), turn_id, step, 0.0, "CACHE_HIT", False, "System", user_query, "Bypassed egress loops.", "N/A", "Loaded from cache", "COMPLETED")
            step += 1

        if not fast_forward:
            falsify_instruction = "\n===FALSIFICATION_SPEC===\n[Emit ONLY the stateless declarative proposition to falsify.]" if state.enable_verifier else ""
            
            if state.force_delegate:
                triage_sys = (
                    "You are an air-gapped data-sanitization router. Your EXCLUSIVE purpose is to anonymize text. The system policy strictly FORCES this request to be delegated to external solvers.\n"
                    "WARNING: YOU ARE STRICTLY FORBIDDEN FROM SOLVING THE PROBLEM YOURSELF. DO NOT write code to solve it. DO NOT provide mathematical proofs. DO NOT answer the user's query.\n\n"
                    "CRITICAL ANONYMIZATION RULES:\n"
                    "1. STRIP ALL specific entity names, organization handles, proprietary codenames, and unnormalized production constants.\n"
                    "2. PRESERVE the core scientific, mathematical, physical, or algorithmic mechanics. Do not over-sanitize into vacuous generalities if the problem relies on concrete domain laws or formulations.\n"
                    "3. RETAIN canonical discipline-wide terminology and formal mathematical representations while using generic variable symbols and normalized dummy dimensions.\n\n"
                    "Format your output STRICTLY AND ONLY as:\n"
                    "===SOLVER_SPEC===\n[Emit ONLY the anonymized abstract problem statement here. DO NOT SOLVE IT.]"
                    f"{falsify_instruction}"
                )
                yield format_sse("> Forced Egress Routing Engaged. Bypassing local assessment...\n")
            else:
                triage_sys = (
                    "You are an air-gapped data-sanitization router. Analyze the user's query to determine routing.\n"
                    "WARNING: YOU ARE STRICTLY FORBIDDEN FROM SOLVING THE PROBLEM YOURSELF DURING TRIAGE. DO NOT write code, proofs, or solutions. Your ONLY job is to evaluate routing and anonymize the text.\n\n"
                    "Determine if this request requires delegating a dense mathematical, structural, or algorithmic bottleneck to external solvers, or if it can be resolved locally.\n\n"
                    "CRITICAL ANONYMIZATION RULES FOR DELEGATION:\n"
                    "1. STRIP ALL specific entity names, organization handles, proprietary codenames, and unnormalized production constants.\n"
                    "2. PRESERVE the core scientific, mathematical, physical, or algorithmic mechanics. Do not over-sanitize into vacuous generalities.\n"
                    "3. RETAIN canonical discipline-wide terminology and formal mathematical representations while using generic variable symbols and normalized dummy dimensions.\n\n"
                    "Format your decision STRICTLY AND ONLY as:\nDECISION: [LOCAL or DELEGATE]\nREASON: [Brief justification for routing only]\n\n"
                    "If DELEGATE, append:\n===SOLVER_SPEC===\n[Emit ONLY the anonymized abstract problem statement here. DO NOT SOLVE IT.]"
                    f"{falsify_instruction}"
                )

            triage_raw = ""
            async for c_content, c_reasoning, finish_data in stream_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, build_payload(triage_sys, conversation_history, f"EVALUATE AND ROUTE THIS REQUEST:\n{user_query}")):
                if c_reasoning: yield format_sse(c_reasoning)
                if c_content: triage_raw += c_content; yield format_sse(c_content)
                if finish_data:
                    await db_record_subrequest(str(uuid.uuid4()), turn_id, step, finish_data[0], "LOCAL_TRIAGE", False, f"Local ({LOCAL_MODEL_NAME})", user_query, triage_raw, "N/A", "Assessed routing", "COMPLETED", finish_data[1].get("prompt_tokens", 0), finish_data[1].get("completion_tokens", 0), finish_data[1].get("total_tokens", 0))
            step += 1

            solver_spec, falsification_spec = extract_specs(triage_raw)
            
            if state.force_delegate:
                needs_delegation = True
                if not solver_spec: solver_spec = strip_think(triage_raw)
            else:
                needs_delegation = bool(re.search(r"DECISION:\s*DELEGATE", triage_raw, re.IGNORECASE)) and bool(solver_spec)

            # ====================================================
            # BRANCH A: PURE LOCAL RESOLUTION
            # ====================================================
            if not needs_delegation:
                await db_update_turn_mode(turn_id, "LOCAL_ONLY")
                yield format_sse("\n\n=== 2. LOCAL EXECUTION ===\n")
                direct_ans = ""
                async for c_content, c_reasoning, finish_data in stream_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, build_payload("You are an expert scientific assistant. Provide a comprehensive answer directly.", conversation_history, user_query)):
                    if c_reasoning: yield format_sse(c_reasoning)
                    if c_content:
                        if thinking_open: yield format_sse("\n</think>\n\n"); thinking_open = False
                        direct_ans += c_content; yield format_sse(c_content)
                    if finish_data: await db_record_subrequest(str(uuid.uuid4()), turn_id, step, finish_data[0], "LOCAL_EXECUTION", False, f"Local ({LOCAL_MODEL_NAME})", user_query, direct_ans, "N/A", "Initial local resolution", "COMPLETED", finish_data[1].get("prompt_tokens",0), finish_data[1].get("completion_tokens",0), finish_data[1].get("total_tokens",0))
                step += 1

                if state.enable_verifier:
                    correction_round = 0
                    while correction_round < MAX_CORRECTION_ROUNDS:
                        correction_round += 1
                        if not thinking_open: yield format_sse(f"\n\n<think>\n=== AUDIT VERIFICATION (Round {correction_round}) ===\n"); thinking_open = True
                        else: yield format_sse(f"\n\n=== AUDIT VERIFICATION (Round {correction_round}) ===\n")

                        falsify_res = ""
                        referee_sys = (
                            "Analyze this explanation:\n\n"
                            f"{direct_ans[:2500]}\n\n"
                            "DO NOT attempt to solve the provided problem yourself. Your role is purely to referee and verify.\n"
                            "If an admissible flaw exists, output: 'VERDICT: FALSIFIED' and explain.\n"
                            "If unconditionally sound, output: 'VERDICT: SOUND'"
                        )
                        async for c_content, c_reasoning, finish_data in stream_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, [{"role": "system", "content": referee_sys}, {"role": "user", "content": direct_ans[:2000]}]):
                            if c_reasoning: yield format_sse(c_reasoning)
                            if c_content: falsify_res += c_content; yield format_sse(c_content)
                            if finish_data: await db_record_subrequest(str(uuid.uuid4()), turn_id, step, finish_data[0], f"LOCAL_FALSIFICATION_CHECK_R{correction_round}", False, f"Local ({LOCAL_MODEL_NAME})", direct_ans[:500]+"...", falsify_res, "N/A", "Falsification round", "COMPLETED", finish_data[1].get("prompt_tokens",0), finish_data[1].get("completion_tokens",0), finish_data[1].get("total_tokens",0))
                        step += 1

                        if "VERDICT: FALSIFIED" in falsify_res:
                            if correction_round < MAX_CORRECTION_ROUNDS:
                                counterexample = falsify_res.split("VERDICT: FALSIFIED")[-1].strip()
                                yield format_sse("\n\n=== RE-GENERATING SOLUTION ===\n")
                                direct_ans = ""
                                async for c_content, c_reasoning, finish_data in stream_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, build_payload("You are an expert scientific assistant.", conversation_history, f"ORIGINAL GOAL: {user_query}\n\nYOUR PREVIOUS ANSWER:\n{direct_ans}\n\nREFEREE'S REFUTATION:\n{counterexample}\n\nRe-answer comprehensively, explicitly resolving this counterexample.")):
                                    if c_reasoning: yield format_sse(c_reasoning)
                                    if c_content:
                                        if thinking_open: yield format_sse("\n</think>\n\n"); thinking_open = False
                                        direct_ans += c_content; yield format_sse(c_content)
                                    if finish_data: await db_record_subrequest(str(uuid.uuid4()), turn_id, step, finish_data[0], f"LOCAL_CORRECTION_R{correction_round}", False, f"Local ({LOCAL_MODEL_NAME})", counterexample, direct_ans, "N/A", "Corrected", "COMPLETED", finish_data[1].get("prompt_tokens",0), finish_data[1].get("completion_tokens",0), finish_data[1].get("total_tokens",0))
                                step += 1
                            else:
                                yield format_sse(f"\n\n⚠️ Boundary Warning: Unresolved after {MAX_CORRECTION_ROUNDS} rounds.\n")
                                break
                        else: break
                else:
                    yield format_sse("\n<think> Referee verification disabled. Proceeding with unverified local answer...</think>\n")

                if thinking_open: yield format_sse("\n</think>\n\n"); thinking_open = False
                await db_finish_turn(turn_id, "COMPLETED")
                return

            # ====================================================
            # BRANCH B: HYBRID EGRESS DELEGATION
            # ====================================================
            await db_update_turn_mode(turn_id, "HYBRID_EGRESS")
            yield format_sse("\n\n=== 2. ROUTING: HYBRID EGRESS DELEGATION ===\n")

            # --- BACKGROUND DECOY SYNTHESIS & DISPATCH ---
            decoy_base_step = step
            
            if state.enable_decoys:
                yield format_sse("> 🎭 Background synthesis of orthogonal decoy started...\n")
                synth_step_id = str(uuid.uuid4())
                disp_step_id = str(uuid.uuid4())
                
                async def run_decoy_flow(step_idx, s_id, d_id):
                    try:
                        await db_record_subrequest(s_id, turn_id, step_idx, 0.0, "DECOY_SYNTHESIS", False, f"Local ({LOCAL_MODEL_NAME})", f"Avoid domain: {user_query[:50]}...", "[Synthesizing...]", "decoy", "Synthesis started", "IN_PROGRESS")
                        
                        chaff_sys = (
                            "You are an automated technical decoy generator. Generate ONE highly technical, self-contained mathematical, algorithmic, or physics query.\n"
                            "WARNING: YOU ARE STRICTLY FORBIDDEN FROM SOLVING OR ANSWERING THE PROBLEM. DO NOT write proofs or implementations.\n"
                            "CRITICAL RULE: It MUST be completely ORTHOGONAL (unrelated) to the user's actual domain provided below.\n"
                            "Make it graduate-level rigor. Output ONLY the query text. No conversational introductions or markdown formatting."
                        )
                        raw, lat_s, usg_s = await call_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, [{"role": "system", "content": chaff_sys}, {"role": "user", "content": f"USER REAL DOMAIN TO AVOID: {user_query}"}])
                        decoy_prompt = strip_think(raw)
                        
                        await db_record_subrequest(s_id, turn_id, step_idx, lat_s, "DECOY_SYNTHESIS", False, f"Local ({LOCAL_MODEL_NAME})", f"Avoid domain: {user_query[:50]}...", decoy_prompt, "decoy", "Synthesized dynamic decoy", "COMPLETED", usg_s.get("prompt_tokens",0), usg_s.get("completion_tokens",0), usg_s.get("total_tokens",0))
                        
                        if not decoy_prompt: raise ValueError("Empty decoy prompt synthesized.")
                        
                        rec_decoy = QueueRecord(
                            id=str(uuid.uuid4()), turn_id=turn_id, round_idx=1,
                            created_at=time.time(), target_provider=f"Provider 1 ({PROVIDER1_MODEL_NAME})",
                            spec_type="decoy", prompt_text=decoy_prompt, certainty=AuditCertainty.DECOY,
                            audit_notes="Orthogonal decoy query synthesized. Safe for dispatch."
                        )
                        action, text_d, _ = await queue_and_await_approval(rec_decoy, step_idx + 1)
                        
                        if action == "reject":
                            await db_record_subrequest(d_id, turn_id, step_idx + 2, 0.0, "DECOY_DISPATCH", True, "System", text_d, "[Rejected by User Gate]", "decoy", "Cancelled", "CANCELLED")
                            return
                            
                        await db_record_subrequest(d_id, turn_id, step_idx + 2, 0.0, "DECOY_DISPATCH", True, f"Provider 1 ({PROVIDER1_MODEL_NAME})", text_d, "[Executing...]", "decoy", "Decoy dispatched", "IN_PROGRESS")
                        
                        ans, lat_d, usg_d = await call_llm(PROVIDER1_API_BASE, PROVIDER1_API_KEY, PROVIDER1_MODEL_NAME, [{"role": "user", "content": text_d}], max_tokens=350)
                        ans_clean = strip_think(ans)
                        
                        await db_record_subrequest(d_id, turn_id, step_idx + 2, lat_d, "DECOY_DISPATCH", True, f"Provider 1 ({PROVIDER1_MODEL_NAME})", text_d, ans_clean[:400] + "...", "decoy", "Decoy completed", "COMPLETED", usg_d.get("prompt_tokens",0), usg_d.get("completion_tokens",0), usg_d.get("total_tokens",0))
                    
                    except asyncio.CancelledError:
                        await db_record_subrequest(s_id, turn_id, step_idx, 0.0, "DECOY_SYNTHESIS", False, "System", "Decoy synthesis", "[Aborted]", "decoy", "Cancelled", "CANCELLED")
                        await db_record_subrequest(d_id, turn_id, step_idx + 2, 0.0, "DECOY_DISPATCH", True, "System", "Decoy dispatch", "[Aborted]", "decoy", "Cancelled", "CANCELLED")
                        raise
                    except Exception as e:
                        prompt_log = decoy_prompt if 'decoy_prompt' in locals() else "[Synthesis Failed]"
                        await db_record_subrequest(d_id, turn_id, step_idx + 2, 0.0, "DECOY_DISPATCH", True, f"Provider 1 ({PROVIDER1_MODEL_NAME})", prompt_log, f"[Error: {str(e)}]", "decoy", "Error", "FAILED")

                decoy_bg_task = asyncio.create_task(run_decoy_flow(decoy_base_step, synth_step_id, disp_step_id))
                turn_background_tasks[turn_id].add(decoy_bg_task)
                decoy_bg_task.add_done_callback(lambda t: turn_background_tasks[turn_id].discard(t))
                step += 3 

            correction_round = 0
            spec_approved = False

            # --- AUDITING & REDACTION LOOP ---
            while not spec_approved and correction_round < MAX_CORRECTION_ROUNDS:
                correction_round += 1
                yield format_sse(f"\n=== EGRESS AUDIT (Round {correction_round}) ===\n")

                if state.enable_verifier and not falsification_spec:
                    yield format_sse("> Synthesizing missing declarative proposition...\n")
                    falsification_sys = (
                        "You are an abstraction engine. Your ONLY job is to extract the core formal, mathematical, or physical invariant proposition that must be satisfied by the text provided.\n\n"
                        "WARNING: YOU ARE STRICTLY FORBIDDEN FROM SOLVING THE PROBLEM. DO NOT provide the implementation or the mathematical proof. DO NOT write code.\n\n"
                        "State the invariant purely as a formal declarative condition without meta-labels, conversational filler, or solutions."
                    )
                    falsification_spec, f_lat, f_usg = await call_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, [{"role": "system", "content": falsification_sys}, {"role": "user", "content": solver_spec}])
                    await db_record_subrequest(str(uuid.uuid4()), turn_id, step, f_lat, f"PROPOSITION_SYNTHESIS_R{correction_round}", False, f"Local ({LOCAL_MODEL_NAME})", solver_spec, falsification_spec, "N/A", "Synthesized missing invariant", "COMPLETED", f_usg.get("prompt_tokens", 0), f_usg.get("completion_tokens", 0), f_usg.get("total_tokens", 0))
                    step += 1

                yield format_sse("> Auditing Prompts...\n")
                s_cert, s_notes, s_raw = await evaluate_audit_certainty(solver_spec, turn_id, step)
                step += 1
                
                if state.enable_verifier:
                    v_cert, v_notes, v_raw = await evaluate_audit_certainty(falsification_spec, turn_id, step)
                    step += 1
                else:
                    v_cert, v_notes, v_raw = AuditCertainty.HIGH_CERTAINTY_CLEAN, "Verifier Disabled", ""

                # AUTO ACTOR-CRITIC REDACTION
                auto_redact_round = 0
                needs_redact = (s_cert != AuditCertainty.HIGH_CERTAINTY_CLEAN) or (state.enable_verifier and v_cert != AuditCertainty.HIGH_CERTAINTY_CLEAN)
                while state.auto_redact and auto_redact_round < 2 and needs_redact:
                    auto_redact_round += 1
                    yield format_sse(f"\n> ⚠️ Internal Auditor flagged a potential leak (Auto-Redaction Phase {auto_redact_round}). Fixing...\n")
                    
                    redaction_prompt = (
                        "You are a strict data privacy redaction agent. Your ONLY job is to anonymize the text below to satisfy the auditor's feedback.\n"
                        "WARNING: YOU ARE STRICTLY FORBIDDEN FROM SOLVING THE PROBLEM. DO NOT write code or answer the question.\n\n"
                        f"PREVIOUS SOLVER_SPEC:\n{solver_spec}\n"
                        f"AUDITOR FEEDBACK (Solver): {s_notes}\n"
                    )
                    if state.enable_verifier:
                        redaction_prompt += f"PREVIOUS FALSIFICATION_SPEC:\n{falsification_spec}\nAUDITOR FEEDBACK (Falsify): {v_notes}\n"
                        redaction_prompt += "\nINSTRUCTION: Rewrite BOTH specifications to eliminate leaks. Format STRICTLY AND ONLY as:\n===SOLVER_SPEC===\n<rewritten prompt>\n===FALSIFICATION_SPEC===\n<rewritten proposition>"
                    else:
                        redaction_prompt += "\nINSTRUCTION: Rewrite the specification to eliminate leaks. Format STRICTLY AND ONLY as:\n===SOLVER_SPEC===\n<rewritten prompt>"
                    
                    redacted_raw = ""
                    async for c_content, c_reasoning, finish_data in stream_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, build_payload(triage_sys, conversation_history, redaction_prompt)):
                        if c_reasoning: yield format_sse(c_reasoning)
                        if c_content: redacted_raw += c_content; yield format_sse(c_content)
                    step += 1

                    rev_solver, rev_falsify = extract_specs(redacted_raw)
                    if rev_solver: 
                        solver_spec = rev_solver
                        if state.enable_verifier and rev_falsify: falsification_spec = rev_falsify
                    else: break

                    yield format_sse("\n> Re-Auditing Redacted Prompts...\n")
                    s_cert, s_notes, s_raw = await evaluate_audit_certainty(solver_spec, turn_id, step)
                    step += 1
                    if state.enable_verifier:
                        v_cert, v_notes, v_raw = await evaluate_audit_certainty(falsification_spec, turn_id, step)
                        step += 1
                    needs_redact = (s_cert != AuditCertainty.HIGH_CERTAINTY_CLEAN) or (state.enable_verifier and v_cert != AuditCertainty.HIGH_CERTAINTY_CLEAN)

                # DASHBOARD GATEWAY
                rec_solver = QueueRecord(id=str(uuid.uuid4()), turn_id=turn_id, round_idx=correction_round, created_at=time.time(), target_provider=f"Provider 1 ({PROVIDER1_MODEL_NAME})", spec_type=f"solver_r{correction_round}", prompt_text=solver_spec, certainty=s_cert, audit_notes=s_notes, evaluator_raw_response=s_raw)
                
                yield format_sse("> Awaiting Dashboard Gateway Approval...\n")
                if state.enable_verifier:
                    rec_verifier = QueueRecord(id=str(uuid.uuid4()), turn_id=turn_id, round_idx=correction_round, created_at=time.time(), target_provider=f"Provider 2 ({PROVIDER2_MODEL_NAME})" if PROVIDER2_ENABLED else f"Local Keystone ({LOCAL_MODEL_NAME})", spec_type=f"verifier_r{correction_round}", prompt_text=falsification_spec, certainty=v_cert, audit_notes=v_notes, evaluator_raw_response=v_raw)
                    if PROVIDER2_ENABLED:
                        res_s, res_v = await asyncio.gather(queue_and_await_approval(rec_solver, step), queue_and_await_approval(rec_verifier, step + 1))
                        action_s, text_s, feed_s = res_s
                        action_v, text_v, feed_v = res_v
                        step += 2
                        if action_s == "reject" or action_v == "reject": raise HTTPException(status_code=403, detail="Egress specs rejected.")
                    else:
                        action_s, text_s, feed_s = await queue_and_await_approval(rec_solver, step)
                        action_v, text_v, feed_v = "approve", falsification_spec, ""
                        step += 1
                        if action_s == "reject": raise HTTPException(status_code=403, detail="Solver spec rejected.")
                else:
                    action_s, text_s, feed_s = await queue_and_await_approval(rec_solver, step)
                    action_v, text_v, feed_v = "approve", "N/A", ""
                    step += 1
                    if action_s == "reject": raise HTTPException(status_code=403, detail="Solver spec rejected.")

                # MANUAL FEEDBACK LOOP
                if action_s == "feedback" or action_v == "feedback":
                    yield format_sse("\n> 🔄 User submitted manual rewrite feedback. Regenerating specs...\n")
                    active_feed = feed_s if action_s == "feedback" else feed_v
                    
                    feedback_step_id = str(uuid.uuid4())
                    await db_record_subrequest(feedback_step_id, turn_id, step, 0.0, f"MANUAL_REWRITE_R{correction_round}", False, f"Local ({LOCAL_MODEL_NAME})", active_feed, "[Rewriting...]", "N/A", "Applying user feedback", "IN_PROGRESS")
                    
                    manual_prompt = (
                        "You are a strict data privacy redaction agent. The user requested manual revisions to your specifications.\n\n"
                        "WARNING: YOU ARE STRICTLY FORBIDDEN FROM SOLVING THE PROBLEM OR WRITING CODE.\n\n"
                        f"PREVIOUS SOLVER_SPEC:\n{solver_spec}\n"
                    )
                    if state.enable_verifier: manual_prompt += f"PREVIOUS FALSIFICATION_SPEC:\n{falsification_spec}\n"
                    manual_prompt += f"USER FEEDBACK:\n{active_feed}\n\nINSTRUCTION: Rewrite perfectly adhering to feedback without solving it. Format STRICTLY AND ONLY as:\n===SOLVER_SPEC===\n<rewritten prompt>\n"
                    if state.enable_verifier: manual_prompt += "===FALSIFICATION_SPEC===\n<rewritten proposition>"

                    redacted_raw = ""
                    async for c_content, c_reasoning, finish_data in stream_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, build_payload(triage_sys, conversation_history, manual_prompt)):
                        if c_reasoning: yield format_sse(c_reasoning)
                        if c_content: redacted_raw += c_content; yield format_sse(c_content)
                        if finish_data:
                            await db_record_subrequest(feedback_step_id, turn_id, step, finish_data[0], f"MANUAL_REWRITE_R{correction_round}", False, f"Local ({LOCAL_MODEL_NAME})", active_feed, redacted_raw, "N/A", "Feedback applied", "COMPLETED", finish_data[1].get("prompt_tokens",0), finish_data[1].get("completion_tokens",0), finish_data[1].get("total_tokens",0))
                    step += 1
                    
                    rev_solver, rev_falsify = extract_specs(redacted_raw)
                    if rev_solver: 
                        solver_spec = rev_solver
                        if state.enable_verifier and rev_falsify: falsification_spec = rev_falsify
                    else: solver_spec = strip_think(redacted_raw)
                    continue

                solver_spec = text_s
                if state.enable_verifier: falsification_spec = text_v
                spec_approved = True

            # --- EXECUTION STAGE ---
            yield format_sse("> Dispatched to External Solvers...\n")
            yield format_sse(f"\n--- SOLVER DERIVATION (Provider 1) ---\n")
            
            final_solver_res = ""
            solver_step_id = str(uuid.uuid4())
            await db_record_subrequest(solver_step_id, turn_id, step, 0.0, f"EXTERNAL_SOLVER_R{correction_round}", True, f"Provider 1 ({PROVIDER1_MODEL_NAME})", solver_spec, "[Streaming...]", s_cert.value, "Derivation started", "IN_PROGRESS")
            
            solver_gen = stream_llm(PROVIDER1_API_BASE, PROVIDER1_API_KEY, PROVIDER1_MODEL_NAME, [{"role": "user", "content": solver_spec}])
            async for c_content, c_reasoning, finish_data in solver_gen:
                if c_reasoning: yield format_sse(c_reasoning)
                if c_content: final_solver_res += c_content; yield format_sse(c_content)
                if finish_data: await db_record_subrequest(solver_step_id, turn_id, step, finish_data[0], f"EXTERNAL_SOLVER_R{correction_round}", True, f"Provider 1 ({PROVIDER1_MODEL_NAME})", solver_spec, final_solver_res, s_cert.value, "Derivation completed", "COMPLETED", finish_data[1].get("prompt_tokens",0), finish_data[1].get("completion_tokens",0), finish_data[1].get("total_tokens",0))
            step += 1

            if state.enable_verifier:
                yield format_sse(f"\n\n> Waiting for Referee Verdict...\n")
                
                async def run_verifier(step_idx: int):
                    actual_referee_prompt = f"Evaluate the derived solution against the constraints:\n\nCONSTRAINTS:\n{falsification_spec}\n\nDERIVED SOLUTION:\n{final_solver_res}\n\nIf the solution contains an admissible flaw, output 'VERDICT: FALSIFIED' and explain.\nIf strictly sound, output 'VERDICT: SOUND'."
                    v_api, v_key, v_mod = (PROVIDER2_API_BASE, PROVIDER2_API_KEY, PROVIDER2_MODEL_NAME) if PROVIDER2_ENABLED else (LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME)
                    await db_record_subrequest(str(uuid.uuid4()), turn_id, step_idx, 0.0, f"VERIFIER_R{correction_round}", PROVIDER2_ENABLED, v_mod, actual_referee_prompt, "[Streaming...]", v_cert.value, "Referee check started", "IN_PROGRESS")
                    ans, lat, usg = await call_llm(v_api, v_key, v_mod, [{"role": "user", "content": actual_referee_prompt}], max_tokens=None)
                    await db_record_subrequest(str(uuid.uuid4()), turn_id, step_idx, lat, f"VERIFIER_R{correction_round}", PROVIDER2_ENABLED, v_mod, actual_referee_prompt, ans, v_cert.value, "Referee check completed", "COMPLETED", usg.get("prompt_tokens",0), usg.get("completion_tokens",0), usg.get("total_tokens",0))
                    return ans

                verifier_task = asyncio.create_task(run_verifier(step))
                local_subtasks.add(verifier_task)
                final_verifier_res = await verifier_task
                local_subtasks.discard(verifier_task)
                step += 1

                yield format_sse(f"\n--- REFEREE VERDICT ---\n{final_verifier_res}\n")

                if "VERDICT: FALSIFIED" in final_verifier_res:
                    yield format_sse("\n⚠️ Boundary Warning: Derivation Falsified by Referee. External loop ends.\n")
            else:
                final_verifier_res = "Referee verification is disabled by user policy."
                yield format_sse(f"\n<think> Referee verification disabled. Proceeding with unverified local answer...</think>\n")

            clean_solver_res = strip_think(final_solver_res)
            clean_verifier_res = strip_think(final_verifier_res)
            
            global_external_cache[cache_key] = {"solver": clean_solver_res, "verifier": clean_verifier_res, "falsification": falsification_spec}

        # --- ASSEMBLY GATING & INTEGRATION ---
        yield format_sse("\n=== 3. FINAL INTEGRATION ===\nSynthesizing verified solution...\n")
        
        assemble_sys = (
            "You are the Lead Integration Architect operating in an air-gapped secure enclave. "
            "Your objective is to solve the user's original domain-specific problem by directly adapting a provided external mathematical/algorithmic derivation.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. DOMAIN MAPPING: The EXTERNAL DERIVATION is structurally correct but uses generic/dummy variables. You MUST map these abstract variables back to the specific proprietary names, tensors, and concepts used in the ORIGINAL QUERY.\n"
            "2. REFEREE COMPLIANCE: If the REFEREE REPORT indicates 'FALSIFIED', explicitly acknowledge the flaw and adjust the integrated solution to resolve it.\n"
            "3. DIRECT ACTION: Do not loop your reasoning or repeat instructions. Do not meta-narrate your thought process outside of your <think> tags. Provide the final, cohesive, production-ready solution directly."
        )
        
        assemble_payload = (
            f"=== ORIGINAL USER QUERY ===\n{user_query}\n\n"
            f"=== EXTERNAL DERIVATION (VERIFIED) ===\n{clean_solver_res}\n\n"
            f"=== REFEREE REPORT ===\n{clean_verifier_res}"
        )

        rec_assembly = QueueRecord(
            id=str(uuid.uuid4()), turn_id=turn_id, round_idx=1,
            created_at=time.time(), target_provider=f"Local ({LOCAL_MODEL_NAME})", spec_type="assembly",
            prompt_text=assemble_payload, certainty=AuditCertainty.HIGH_CERTAINTY_CLEAN,
            audit_notes="Review or modify the synthesis payload before the local model assembles the final answer."
        )

        if state.assembly_policy == GatePolicy.MANUAL_APPROVE_ALL:
            yield format_sse("\n> ⏳ Awaiting Dashboard Gateway Approval for Final Assembly...\n")
            action, approved_payload, _ = await queue_and_await_approval(rec_assembly, step)
            step += 1
            if action == "reject":
                yield format_sse("\n❌ Assembly rejected by user via Dashboard.\n")
                await db_finish_turn(turn_id, "CANCELLED_BY_USER")
                return
            assemble_payload = approved_payload

        final_res = ""
        async for c_content, c_reasoning, finish_data in stream_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, build_payload(assemble_sys, conversation_history, assemble_payload)):
            if c_reasoning: yield format_sse(c_reasoning)
            if c_content:
                if thinking_open: yield format_sse("\n</think>\n\n"); thinking_open = False
                final_res += c_content; yield format_sse(c_content)
            if finish_data: await db_record_subrequest(str(uuid.uuid4()), turn_id, step, finish_data[0], "LOCAL_ASSEMBLY", False, f"Local ({LOCAL_MODEL_NAME})", assemble_payload, final_res, "N/A", "Context reintegration", "COMPLETED", finish_data[1].get("prompt_tokens",0), finish_data[1].get("completion_tokens",0), finish_data[1].get("total_tokens",0))

        if thinking_open: yield format_sse("\n</think>\n\n"); thinking_open = False
        await db_finish_turn(turn_id, "COMPLETED")
        return

    except asyncio.CancelledError:
        for t in list(local_subtasks):
            if not t.done(): t.cancel()
        async with state_lock:
            for q_id, q_rec in list(state.queue.items()):
                if q_rec.turn_id == turn_id:
                    state.queue.pop(q_id, None)
        await db_finish_turn(turn_id, "CANCELLED_BY_USER")
        raise
    except BaseException as e:
        for t in list(local_subtasks):
            if not t.done(): t.cancel()
        err_trace = traceback.format_exc()
        await db_finish_turn(turn_id, f"FAILED: {str(e)}")
        if thinking_open: yield format_sse(f"\n\n=== PIPELINE ERROR ===\n{err_trace}\n</think>\n\n")
        else: yield format_sse(f"\n\n❌ **Pipeline Error:**\n```\n{err_trace}\n```\n")
        raise e

# ==========================================
# FastAPI Endpoints
# ==========================================
def clean_incoming_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cleaned = []
    for m in messages:
        c = m.get("content", "")
        c = strip_think(c)
        if c: cleaned.append({**m, "content": c})
    return cleaned

@app.get("/v1/models")
async def list_models(): return {"object": "list", "data": [{"id": "BorrowModel", "object": "model", "created": int(time.time()), "owned_by": "local-enclave"}]}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = clean_incoming_messages(body.get("messages", []))
    user_query = next((m["content"] for m in reversed(messages) if m["role"] == "user"), None)
    if not user_query: raise HTTPException(status_code=400, detail="No user message provided.")

    turn_id = str(uuid.uuid4())
    q = asyncio.Queue(maxsize=20)

    async def producer():
        try:
            async for chunk in run_adaptive_pipeline(turn_id, user_query, messages[:-1]): await q.put(chunk)
            await q.put(None)
        except asyncio.CancelledError:
            await q.put(None)
            raise
        except Exception as e:
            await q.put(format_sse(f"\n\n<think>\n=== PIPELINE ERROR ===\n{traceback.format_exc()}\n</think>\n\n"))
            await q.put(None)

    prod_task = asyncio.create_task(producer())
    active_turn_tasks[turn_id] = prod_task

    async def monitor_disconnect():
        try:
            while not prod_task.done():
                if await request.is_disconnected():
                    prod_task.cancel()
                    break
                await asyncio.sleep(0.2)
        except asyncio.CancelledError: pass

    monitor_task = asyncio.create_task(monitor_disconnect())

    async def consumer():
        is_cancelled = False
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(q.get(), timeout=1.0)
                    if chunk is None: break
                    yield chunk
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        prod_task.cancel()
                        break
                    yield ": keepalive\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            is_cancelled = True
            prod_task.cancel()
            raise
        finally:
            active_turn_tasks.pop(turn_id, None)
            monitor_task.cancel()
            if not prod_task.done():
                prod_task.cancel()
                try: await prod_task
                except asyncio.CancelledError: pass
            
            if not is_cancelled:
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute("SELECT SUM(prompt_tokens), SUM(completion_tokens), SUM(total_tokens) FROM turn_subrequests WHERE turn_id = ?", (turn_id,))
                row = c.fetchone()
                conn.close()
                if row and row[2]:
                    yield f"data: {json.dumps({'id': f'chatcmpl-{turn_id}', 'object': 'chat.completion.chunk', 'model': 'BorrowModel', 'choices': [], 'usage': {'prompt_tokens': row[0] or 0, 'completion_tokens': row[1] or 0, 'total_tokens': row[2] or 0}})}\n\n"
                yield "data: [DONE]\n\n"

    return StreamingResponse(consumer(), media_type="text/event-stream")

class RegenAssemblyPayload(BaseModel):
    prompt: str

@app.post("/api/assembly/regenerate/{turn_id}")
async def regenerate_assembly(turn_id: str, req: RegenAssemblyPayload):
    prompt_text = req.prompt
    op_id = str(uuid.uuid4())
    await db_record_subrequest(op_id, turn_id, 999, 0.0, "LOCAL_ASSEMBLY_REGEN", False, f"Local ({LOCAL_MODEL_NAME})", prompt_text, "[Executing...]", "N/A", "Regenerating assembly locally", "IN_PROGRESS")
    try:
        ans, lat, usg = await call_llm(LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL_NAME, [{"role": "user", "content": prompt_text}], max_tokens=None)
        await db_record_subrequest(op_id, turn_id, 999, lat, "LOCAL_ASSEMBLY_REGEN", False, f"Local ({LOCAL_MODEL_NAME})", prompt_text, ans, "N/A", "Regenerated successfully", "COMPLETED", usg.get("prompt_tokens",0), usg.get("completion_tokens",0), usg.get("total_tokens",0))
        return {"status": "ok", "response": ans}
    except Exception as e:
        await db_record_subrequest(op_id, turn_id, 999, 0.0, "LOCAL_ASSEMBLY_REGEN", False, "System", prompt_text, f"Failed: {str(e)}", "N/A", "Error", "FAILED")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cancel/{turn_id}")
async def api_cancel_turn(turn_id: str):
    task = active_turn_tasks.get(turn_id)
    if task and not task.done():
        task.cancel()
        
    if turn_id in turn_background_tasks:
        for t in list(turn_background_tasks[turn_id]):
            if not t.done(): t.cancel()
    
    async with state_lock:
        for q_id, q_rec in list(state.queue.items()):
            if q_rec.turn_id == turn_id:
                q_rec.status = ApprovalStatus.REJECTED
                q_rec.event.set()
                
    await db_finish_turn(turn_id, "CANCELLED_BY_USER")
    return {"status": "cancelled", "turn_id": turn_id}

@app.post("/api/cache/clear")
async def clear_cache():
    global_external_cache.clear()
    return {"status": "ok"}

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    with open("dashboard.html", "r", encoding="utf-8") as f: return HTMLResponse(content=f.read())

@app.get("/api/state")
async def get_api_state():
    async with state_lock: 
        return {
            "policy": state.policy.value,
            "assembly_policy": state.assembly_policy.value,
            "force_delegate": state.force_delegate, 
            "auto_redact": state.auto_redact, 
            "enable_verifier": state.enable_verifier, 
            "enable_decoys": state.enable_decoys,
            "queue": [r.model_dump() for r in state.queue.values()]
        }

@app.get("/api/logs")
async def get_paginated_logs(page: int = 1, page_size: int = 10):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    total_turns = c.execute("SELECT COUNT(*) FROM user_turns").fetchone()[0]
    turns = c.execute("SELECT id, created_at, raw_query, routing_mode, status FROM user_turns ORDER BY created_at DESC LIMIT ? OFFSET ?", (page_size, (page - 1) * page_size)).fetchall()
    grouped_data = []
    for t in turns:
        subreqs = [dict(row) for row in c.execute("SELECT id, step_order, created_at, latency_ms, phase_type, is_external, target_engine, prompt, response, certainty, audit_notes, status, prompt_tokens, completion_tokens, total_tokens FROM turn_subrequests WHERE turn_id = ? ORDER BY step_order ASC", (t["id"],)).fetchall()]
        grouped_data.append({
            "turn_id": t["id"], "created_at": t["created_at"], "raw_query": t["raw_query"], "routing_mode": t["routing_mode"], "status": t["status"],
            "metrics": {"total_subrequests": len(subreqs), "external_count": sum(1 for s in subreqs if s["is_external"] == 1), "local_count": sum(1 for s in subreqs if s["is_external"] == 0), "total_latency_ms": round(sum(s["latency_ms"] for s in subreqs), 1), "total_tokens": sum(s["total_tokens"] for s in subreqs)},
            "subrequests": subreqs
        })
    conn.close()
    return {"page": page, "page_size": page_size, "total_turns": total_turns, "total_pages": (total_turns + page_size - 1) // page_size if total_turns > 0 else 1, "data": grouped_data}

@app.get("/api/tokens")
async def get_token_stats(page: int = 1, page_size: int = 25):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    totals = c.execute("SELECT SUM(prompt_tokens), SUM(completion_tokens), SUM(total_tokens) FROM turn_subrequests").fetchone()
    total_turns = c.execute("SELECT COUNT(DISTINCT turn_id) FROM turn_subrequests").fetchone()[0]
    turns = c.execute("SELECT turn_id, MAX(created_at) as started, SUM(prompt_tokens) as pt, SUM(completion_tokens) as ct, SUM(total_tokens) as tt FROM turn_subrequests GROUP BY turn_id ORDER BY started DESC LIMIT ? OFFSET ?", (page_size, (page - 1) * page_size)).fetchall()
    data = [{"turn_id": t["turn_id"], "started": t["started"], "prompt_tokens": t["pt"], "completion_tokens": t["ct"], "total_tokens": t["tt"], "subrequests": [dict(row) for row in c.execute("SELECT phase_type, target_engine, prompt_tokens, completion_tokens, total_tokens FROM turn_subrequests WHERE turn_id = ? ORDER BY step_order ASC", (t["turn_id"],)).fetchall()]} for t in turns]
    conn.close()
    return {"totals": {"prompt": totals[0] or 0, "completion": totals[1] or 0, "total": totals[2] or 0}, "page": page, "page_size": page_size, "total_turns": total_turns, "total_pages": (total_turns + page_size - 1) // page_size if total_turns > 0 else 1, "data": data}

@app.post("/api/policy")
async def update_policy(payload: Dict[str, str]):
    new_policy = payload.get("policy")
    target = payload.get("target", "egress")
    if new_policy in [e.value for e in GatePolicy]:
        async with state_lock: 
            if target == "assembly": state.assembly_policy = GatePolicy(new_policy)
            else: state.policy = GatePolicy(new_policy)
        return {"status": "ok"}
    raise HTTPException(status_code=400, detail="Invalid policy.")

@app.post("/api/routing")
async def update_routing(payload: Dict[str, Any]):
    async with state_lock:
        if "force_delegate" in payload: state.force_delegate = payload["force_delegate"]
        if "auto_redact" in payload: state.auto_redact = payload["auto_redact"]
        if "enable_verifier" in payload: state.enable_verifier = payload["enable_verifier"]
        if "enable_decoys" in payload: state.enable_decoys = payload["enable_decoys"]
    return {"status": "ok"}

@app.post("/api/edit/{req_id}")
async def edit_request(req_id: str, payload: Dict[str, str]):
    async with state_lock:
        record = state.queue.get(req_id)
        if not record: raise HTTPException(status_code=404, detail="Request not found.")
        if "prompt_text" in payload: record.prompt_text = payload["prompt_text"]
    return {"status": "ok"}

@app.post("/api/resolve/{req_id}")
async def resolve_request(req_id: str, payload: Dict[str, str]):
    action = payload.get("action")
    async with state_lock:
        record = state.queue.get(req_id)
        if not record: raise HTTPException(status_code=404, detail="Request not found.")
        
        if action in ["reject", "feedback"]:
            turn_id = record.turn_id
            for sibling in list(state.queue.values()):
                if sibling.turn_id == turn_id:
                    sibling.resolution_action = action
                    sibling.feedback_text = payload.get("feedback_text", "")
                    sibling.status = ApprovalStatus.REJECTED if action == "reject" else ApprovalStatus.PENDING
                    sibling.event.set()
        else:
            record.resolution_action = "approve"
            if "prompt_text" in payload: record.prompt_text = payload["prompt_text"]
            record.status = ApprovalStatus.APPROVED
            record.event.set()
    return {"status": "resolved"}

if __name__ == "__main__":
    import uvicorn
    print(f"[*] Starting BorrowModel Proxy securely on http://{HOST}:{PORT}")
    print(f"[*] Live Dashboard: http://{HOST}:{PORT}/dashboard")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)