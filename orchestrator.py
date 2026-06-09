import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process
from openai import OpenAI

app = FastAPI()

# DeepSeek configuration
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Ensure CrewAI's internal OpenAI calls point to DeepSeek
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = base_url
os.environ["OPENAI_MODEL_NAME"] = model

# Define your three custom agents
main_dev = Agent(
    role="Main Developer",
    goal="Provide the high-level design and initial code implementation for the user's request.",
    backstory="Senior software architect who understands requirements deeply.",
    allow_delegation=False,
    verbose=True
)

session_dev = Agent(
    role="Session Developer",
    goal="Refine the code, fix syntax, add details, and store the final version.",
    backstory="Expert programmer who turns sketches into production-ready code.",
    allow_delegation=False,
    verbose=True
)

auditor = Agent(
    role="Code Auditor",
    goal="Audit the final code for bugs, security issues, and style. Provide a detailed report.",
    backstory="Meticulous QA engineer who never misses a flaw.",
    allow_delegation=False,
    verbose=True
)

class CodeTaskRequest(BaseModel):
    task: str

@app.get("/")
async def root():
    return {"status": "ok", "agents": ["main_dev", "session_dev", "auditor"]}

@app.post("/orchestrate")
async def orchestrate(request: CodeTaskRequest):
    user_task = request.task

    # Sequential workflow: Main Dev → Session Dev → Auditor
    task1 = Task(
        description=f"Design a solution and write the initial code for: {user_task}. Output only the code.",
        agent=main_dev,
        expected_output="Well-commented initial code implementation."
    )
    task2 = Task(
        description="Take the initial code, refine it for correctness, add missing parts, and produce the final polished code.",
        agent=session_dev,
        expected_output="Final production-ready code."
    )
    task3 = Task(
        description="Review the final code. List any potential bugs, security vulnerabilities, or style issues, and suggest fixes.",
        agent=auditor,
        expected_output="Detailed audit report."
    )

    crew = Crew(
        agents=[main_dev, session_dev, auditor],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        verbose=True
    )

    try:
        result = crew.kickoff()
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
