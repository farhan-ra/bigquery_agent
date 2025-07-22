import asyncio
import logging
import os
import sys
import tempfile
import traceback
from typing import Any
from pydantic import BaseModel
from termcolor import colored
from typing import Optional, List, Any, AsyncGenerator
import uuid

from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager

from dotenv import load_dotenv

from beeai_framework.agents import AgentExecutionConfig
from beeai_framework.agents.react import ReActAgent
from beeai_framework.backend import ChatModel
from beeai_framework.emitter import EmitterOptions, EventMeta
from beeai_framework.errors import FrameworkError
from beeai_framework.logger import Logger
from beeai_framework.memory import TokenMemory
from tools.bigquery import BigQueryTool
from beeai_framework.emitter import EventMeta
from beeai_framework.errors import FrameworkError
from fastapi.responses import StreamingResponse
import json
from beeai_framework.backend.message import UserMessage

# Load environment variables
load_dotenv()

WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_API_URL = os.getenv("WATSONX_API_URL")


app = FastAPI()

# Agent initialization (global, or use dependency injection)
agent: ReActAgent | None = None

# FastAPI request body schema
class ChatRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None

class AgentEvent(BaseModel):
    type: str
    message: str
    key: Optional[str] = None  # optional, for update events

class ChatResponse(BaseModel):
    response: str
    session_id: str 
    events: List[AgentEvent]

# Configure logging - using DEBUG instead of trace
logger = Logger("app", level=logging.DEBUG)

async def token_streamer(
    agent: ReActAgent,
    prompt: str,
    memory: TokenMemory,
) -> AsyncGenerator[str, None]:
    agent.memory = memory
    response = await agent.run(
        prompt=prompt,
        execution=AgentExecutionConfig(
            max_retries_per_step=3,
            total_max_retries=10,
            max_iterations=20,
        ),
    )

    async for chunk in response.stream_text():
        yield chunk

class SessionMemoryManager:
    def __init__(self, llm):
        self.llm = llm
        self.sessions: dict[str, TokenMemory] = {}

    def get_or_create(self, session_id: str | None) -> tuple[str, TokenMemory]:
        if not session_id:
            session_id = str(uuid.uuid4())
        if session_id not in self.sessions:
            self.sessions[session_id] = TokenMemory(self.llm)
        return session_id, self.sessions[session_id]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize global agent
    llm = ChatModel.from_name(
        "watsonx:meta-llama/llama-3-3-70b-instruct",
        {
            "project_id": os.getenv("WATSONX_PROJECT_ID"),
            "api_key": os.getenv("WATSONX_API_KEY"),
            "api_base": os.getenv("WATSONX_API_URL"),
        },
    )

    tools = [BigQueryTool()]
    agent = ReActAgent(llm=llm, tools=tools, memory=TokenMemory(llm))

    # # Add code interpreter tool if URL is configured
    # # for running numpy/matplotlib/etc.
    # code_interpreter_url = os.getenv("CODE_INTERPRETER_URL")
    # if code_interpreter_url:
    #     tools.append(
    #         PythonTool(
    #             code_interpreter_url,
    #             LocalPythonStorage(
    #                 local_working_dir=tempfile.mkdtemp("code_interpreter_source"),
    #                 interpreter_working_dir=os.getenv("CODE_INTERPRETER_TMPDIR", "./tmp/code_interpreter_target"),
    #             ),
    #         )
    #     )

    app.state.agent = agent # ✅ The agent is created once at startup and stored in app.state.agent.
    app.state.memory_manager = SessionMemoryManager(llm)
    yield
    # You can clean up resources here if needed

app = FastAPI(lifespan=lifespan)

def create_event_collector(event_list: list[AgentEvent]):
    def capture_event(data: Any, event: EventMeta):
        if event.name == "update":
            event_list.append(AgentEvent(
                type="update",
                key=data.update.key,
                message=str(data.update.parsed_value)
            ))
        elif event.name == "error":
            event_list.append(AgentEvent(
                type="error",
                message=FrameworkError.ensure(data.error).explain()
            ))
        elif event.name == "retry":
            event_list.append(AgentEvent(
                type="retry",
                message="retrying the action..."
            ))
        elif event.name == "start":
            event_list.append(AgentEvent(
                type="start",
                message="starting new iteration"
            ))
        elif event.name == "success":
            event_list.append(AgentEvent(
                type="success",
                message="success"
            ))
    return capture_event


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, request: Request):

    # Get the single agent instance created at startup
    agent: ReActAgent = request.app.state.agent
    memory_manager = request.app.state.memory_manager

    # Get or create memory session
    session_id, memory = memory_manager.get_or_create(req.session_id)

    # Attach session-specific memory to the existing agent instance
    agent.memory = memory

    # Prepare event list and hook
    collected_events: list[AgentEvent] = []
    capture_event = create_event_collector(collected_events)

    try:
        response = await agent.run(
            prompt=req.prompt,
            execution=AgentExecutionConfig(
                max_retries_per_step=3,
                total_max_retries=10,
                max_iterations=20,
            ),
        ).on("*", capture_event, EmitterOptions(match_nested=False))

        return ChatResponse(
            response=response.result.text,
            session_id=session_id,
            events=collected_events
        )

    except FrameworkError as e:
        return ChatResponse(
            response=f"Agent error: {e.explain()}",
            session_id=session_id,
            events=collected_events
        )
    except Exception as ex:
        return ChatResponse(
            response=f"Agent error: {str(ex)}",
            session_id=session_id,
            events=collected_events
        )

@app.post("/chat/stream")
async def chat_stream_endpoint(req: ChatRequest, request: Request):
    agent: ReActAgent = request.app.state.agent
    memory_manager = request.app.state.memory_manager
    session_id, memory = memory_manager.get_or_create(req.session_id)

    return StreamingResponse(
        token_streamer(agent, req.prompt, memory),
        media_type="text/plain",
        headers={"X-Session-ID": session_id}
    )