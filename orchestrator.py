import asyncio
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

# ---------- Configuration ----------
app = FastAPI(title="Multi-Agent Developer Orchestrator with Memory & Files")

# DeepSeek / OpenAI‑compatible client
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
)
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Workspace directory for generated files
WORKSPACE = Path("/workspace")
WORKSPACE.mkdir(exist_ok=True)

# In‑memory session store: session_id -> list of messages (role, content)
session_memory: Dict[str, List[Dict[str, str]]] = {}

# ---------- Request / Response models ----------
class OrchestrateRequest(BaseModel):
    task: str = Field(..., description="The software development task")
    session_id: Optional[str] = Field(None, description="Session ID for memory")

class OrchestrateResponse(BaseModel):
    supervisor_plan: str
    code: str
    debugger_review: Optional[str]
    ux_review: Optional[str]
    file_path: Optional[str]
    file_url: Optional[str]

# ---------- Agent definitions ----------
AGENT_PROMPTS = {
    "supervisor": (
        "You are a Senior Software Architect. "
        "Analyse the user's request and create a structured plan. "
        "Output a JSON object with keys: 'plan', 'technical_requirements', 'edge_cases', 'ux_considerations'. "
        "Be concise, actionable, and remember any previous context from the conversation history."
    ),
    "coder": (
        "You are a Senior Software Engineer. "
        "Write production‑ready code to fulfil the plan. "
        "Use the provided context (including previous messages) to maintain consistency. "
        "Return ONLY the code in a markdown block (```python ... ```) or plain text. Include comments."
    ),
    "debugger": (
        "You are a QA specialist. Review the generated code. "
        "List any logic errors, security issues, or potential runtime exceptions. "
        "For each issue, provide a description and suggested fix. If none, say 'No issues found.'"
    ),
    "ux_reviewer": (
        "You are a UX/Developer Experience expert. "
        "Review the code from a usability and structure perspective. "
        "Suggest improvements for readability, error messages, configuration handling, etc."
    ),
}

# ---------- Core logic ----------
async def run_agent(agent_name: str, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
    """Execute a single agent call."""
    system_prompt = AGENT_PROMPTS[agent_name]
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=full_messages,
            temperature=temperature,
            max_tokens=3000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Agent {agent_name} failed: {e}")

def extract_code_block(text: str) -> str:
    """Extract code from the first markdown code block."""
    match = re.search(r"```[\w]*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # fallback: return the whole text if no code block
    return text.strip()

async def orchestrate_task(task: str, session_id: str) -> OrchestrateResponse:
    # Retrieve or initialise session memory
    if session_id not in session_memory:
        session_memory[session_id] = []
    memory = session_memory[session_id]

    # Build the user message for the supervisor
    user_msg = f"Task: {task}"
    memory.append({"role": "user", "content": user_msg})

    # --- 1. Supervisor ---
    supervisor_plan = await run_agent("supervisor", memory, temperature=0.5)
    memory.append({"role": "assistant", "content": supervisor_plan, "name": "supervisor"})
    # Keep the plan as a nicely formatted string
    try:
        plan_data = json.loads(supervisor_plan)
        plan_str = json.dumps(plan_data, indent=2)
    except json.JSONDecodeError:
        plan_str = supervisor_plan

    # --- 2. Coder (receives the full memory) ---
    coder_input = [{"role": "user", "content": "Produce the complete code based on the supervisor's plan and any previous context."}]
    raw_code = await run_agent("coder", memory + coder_input, temperature=0.4)
    final_code = extract_code_block(raw_code)
    memory.append({"role": "assistant", "content": final_code, "name": "coder"})

    # Save the code to a file in the workspace
    session_dir = WORKSPACE / session_id
    session_dir.mkdir(exist_ok=True)
    file_path = session_dir / "code.py"
    file_path.write_text(final_code, encoding="utf-8")
    file_url = f"/file/{session_id}"

    # --- 3. Debugger and UX in parallel ---
    review_context = [{"role": "user", "content": f"Review the following code:\n\n{final_code}"}]
    debug_task = run_agent("debugger", memory + review_context, temperature=0.5)
    ux_task = run_agent("ux_reviewer", memory + review_context, temperature=0.6)
    debugger_review, ux_review = await asyncio.gather(debug_task, ux_task)

    # Update memory with reviews
    memory.append({"role": "assistant", "content": debugger_review, "name": "debugger"})
    memory.append({"role": "assistant", "content": ux_review, "name": "ux_reviewer"})

    return OrchestrateResponse(
        supervisor_plan=plan_str,
        code=final_code,
        debugger_review=debugger_review,
        ux_review=ux_review,
        file_path=str(file_path),
        file_url=file_url,
    )

# ---------- API Endpoints ----------
@app.get("/")
async def root():
    return {
        "message": "Multi-Agent Developer Orchestrator with Memory and File Storage",
        "agents": ["supervisor", "coder", "debugger", "ux_reviewer"],
        "endpoints": {
            "POST /orchestrate": "Submit a task (optionally with session_id)",
            "GET /file/{session_id}": "Download the generated code file",
            "GET /sessions": "List active session IDs",
        },
    }

@app.post("/orchestrate", response_model=OrchestrateResponse)
async def orchestrate(request: OrchestrateRequest):
    session_id = request.session_id or "default"
    try:
        result = await orchestrate_task(request.task, session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/file/{session_id}")
async def get_file(session_id: str):
    """Download the generated code file for a session."""
    file_path = WORKSPACE / session_id / "code.py"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="No file found for this session")
    return FileResponse(path=file_path, filename=f"{session_id}.py", media_type="text/x-python")

@app.get("/sessions")
async def list_sessions():
    """Return the list of active session IDs."""
    return {"sessions": list(session_memory.keys())}
