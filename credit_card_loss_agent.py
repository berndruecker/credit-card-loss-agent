# credit_card_loss_agent.py
# Requirements:
#   pip install fastapi uvicorn langchain langchain-openai pydantic
#   pip install "a2a-sdk[http-server]" starlette sse-starlette
#
# Run:
#   uvicorn credit_card_loss_agent:app --reload --host 0.0.0.0 --port 8000

import json
import logging
import re
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# ---- LangChain imports ----
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage, BaseMessage

# ===========================
# Mock back-end "core" logic
# ===========================

# In-memory state for demo
CARD_STATE = {
    # last4 -> state
    "1234": {"frozen": False, "replacement_order": None},
    "9876": {"frozen": True, "replacement_order": {"status": "in_transit", "eta_days": 2}},
}

def _ensure_card(last4: str):
    if last4 not in CARD_STATE:
        CARD_STATE[last4] = {"frozen": False, "replacement_order": None}

def freeze_card_fn(card_last4: str) -> str:
    _ensure_card(card_last4)
    CARD_STATE[card_last4]["frozen"] = True
    return f"Card •••• {card_last4} frozen."

def unfreeze_card_fn(card_last4: str) -> str:
    _ensure_card(card_last4)
    CARD_STATE[card_last4]["frozen"] = False
    return f"Card •••• {card_last4} unfrozen."

def report_lost_fn(card_last4: str, date_lost: Optional[str] = None) -> str:
    _ensure_card(card_last4)
    when = date_lost or datetime.utcnow().date().isoformat()
    return f"Lost report filed for card •••• {card_last4} (date_lost={when})."

def order_replacement_fn(card_last4: str, delivery_option: str = "standard") -> str:
    _ensure_card(card_last4)
    CARD_STATE[card_last4]["replacement_order"] = {
        "status": "processing",
        "delivery_option": delivery_option,
        "eta_days": 5 if delivery_option == "standard" else 2,
    }
    return f"Replacement ordered for •••• {card_last4} via {delivery_option}."

def check_replacement_status_fn(card_last4: str) -> str:
    _ensure_card(card_last4)
    ro = CARD_STATE[card_last4]["replacement_order"]
    if not ro:
        return f"No replacement order found for •••• {card_last4}."
    return f"Replacement status for •••• {card_last4}: {ro['status']}, ETA {ro.get('eta_days','?')} days."

# ===========================
# Tool schemas
# ===========================

class CardLast4(BaseModel):
    card_last4: str = Field(..., description="Last 4 digits of the card")

class ReportLostSchema(CardLast4):
    date_lost: Optional[str] = Field(None, description="ISO date when card was lost")

class OrderReplacementSchema(CardLast4):
    delivery_option: str = Field(
        "standard", description="Delivery option: 'standard' or 'express'"
    )

freeze_tool = StructuredTool.from_function(
    name="freeze_card",
    description="Freeze a card immediately.",
    func=lambda card_last4: freeze_card_fn(card_last4),
    args_schema=CardLast4,
)

unfreeze_tool = StructuredTool.from_function(
    name="unfreeze_card",
    description="Unfreeze a card.",
    func=lambda card_last4: unfreeze_card_fn(card_last4),
    args_schema=CardLast4,
)

report_lost_tool = StructuredTool.from_function(
    name="report_lost",
    description="Report a card as lost.",
    func=lambda card_last4, date_lost=None: report_lost_fn(card_last4, date_lost),
    args_schema=ReportLostSchema,
)

order_replacement_tool = StructuredTool.from_function(
    name="order_replacement",
    description="Order a replacement card.",
    func=lambda card_last4, delivery_option="standard": order_replacement_fn(card_last4, delivery_option),
    args_schema=OrderReplacementSchema,
)

check_replacement_status_tool = StructuredTool.from_function(
    name="check_replacement_status",
    description="Check the status of a replacement card order.",
    func=lambda card_last4: check_replacement_status_fn(card_last4),
    args_schema=CardLast4,
)

tools = [
    freeze_tool,
    unfreeze_tool,
    report_lost_tool,
    order_replacement_tool,
    check_replacement_status_tool,
]

# ===========================
# Heuristics for intent hints
# ===========================

ACTION_ALIASES = {
    "freeze": ["freeze", "block", "lock", "disable"],
    "unfreeze": ["unfreeze", "unblock", "unlock", "enable"],
    "report_lost": ["lost", "stolen", "misplaced"],
    "order_replacement": ["replacement", "replace", "new card", "reissue"],
    "check_replacement_status": ["status", "delivery", "where", "eta", "track", "tracking"],
}
def extract_intent_and_last4(text: str):
    text_l = (text or "").lower()
    m = re.search(r"\b(\d{4})\b", text_l)
    last4 = m.group(1) if m else None
    for action, keywords in ACTION_ALIASES.items():
        if any(k in text_l for k in keywords):
            return action, last4
    return None, last4

# ===========================
# Agent policy prompt
# ===========================

SYSTEM_POLICY = (
    "You are the Card Management Agent.\n\n"
    "SCOPE:\n"
    "- Only handle card actions: freeze/unfreeze, report lost, order replacement, check replacement status.\n\n"
    "BEHAVIOR:\n"
    "- If the user reports a lost/stolen card, perform these steps in order: (1) Freeze; (2) File lost report; "
    "(3) Order replacement (express if urgency indicated, else standard).\n"
    "- If the user asks to block/freeze, freeze.\n"
    "- If the user asks about delivery/status, check replacement status.\n"
    "- If an action and card last-4 are present, call the corresponding tool(s) immediately without clarifying.\n"
    "- If last-4 is missing but the action is clear, ask once for last-4. If unavailable, do best effort with provided info.\n"
    "- If request is out of scope (e.g., loans), return a JSON object indicating handoff is required with a concise reason.\n\n"
    "PARSING HINTS:\n"
    "- Treat any standalone 4-digit sequence as the card_last4.\n"
    "- Synonyms:\n"
    "  * freeze: freeze, block, lock, disable\n"
    "  * unfreeze: unfreeze, unblock, unlock, enable\n"
    "  * lost: lost, stolen, misplaced\n"
    "  * replacement: replacement, replace, reissue, new card\n"
    "  * status: status, track, tracking, delivery, ETA, where\n\n"
    "OUTPUT:\n"
    "- Final answer MUST be a single valid JSON object ONLY (no explanations, no prose, no markdown, no code fences).\n"
    "- The JSON schema is:\n"
    "  - actions: array of objects with keys: action (string), status (string)\n"
    "  - next_steps: string (optional)\n"
)

# IMPORTANT: include agent_scratchpad for tool-using agents
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_POLICY),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),  # required for tool use
    ]
)

# ===========================
# Build agent
# ===========================

llm = ChatOpenAI(
    model="gpt-4o-mini",  # any tool-calling-capable model
    temperature=0,
)

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

# Return intermediate steps so we can inspect actions_taken in REST
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    return_intermediate_steps=True,
)

# Helper: convert lightweight dict chat history to LangChain messages
def to_lc_messages(history: List[Dict[str, str]]) -> List[BaseMessage]:
    lc_msgs: List[BaseMessage] = []
    for m in history or []:
        role = (m.get("role") or m.get("type") or "").lower()
        content = m.get("content") or ""
        if role in ("human", "user"):
            lc_msgs.append(HumanMessage(content=content))
        elif role in ("ai", "assistant"):
            lc_msgs.append(AIMessage(content=content))
        else:
            lc_msgs.append(HumanMessage(content=content))
    return lc_msgs

# ===========================
# FastAPI (REST) – simple wrapper
# ===========================

app = FastAPI(title="Card Management Agent", version="1.3.0")

class A2ARequest(BaseModel):
    intent: Optional[str] = Field(None, description="e.g., 'report_lost', 'freeze', 'check_status'")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    input_text: Optional[str] = Field(None, description="Natural-language request from orchestrator or user")
    chat_history: List[Dict[str, str]] = Field(default_factory=list)

class A2AResponse(BaseModel):
    status: str
    result: Dict[str, Any]
    actions_taken: List[str] = []
    handoff_required: bool = False
    handoff_reason: Optional[str] = None

@app.post("/agents/card/handle", response_model=A2AResponse)
def handle(req: A2ARequest):
    agent_input = req.input_text or f"Intent: {req.intent}; Parameters: {req.parameters}"
    history_msgs = to_lc_messages(req.chat_history)

    parsed_action, parsed_last4 = extract_intent_and_last4(agent_input)
    hint = f"Parsed(action={parsed_action}, last4={parsed_last4})"
    agent_input_with_hint = f"{agent_input}\n\n{hint}"

    try:
        result = executor.invoke({"input": agent_input_with_hint, "chat_history": history_msgs})
        final_text = result.get("output", "")
        actions = []
        for step in result.get("intermediate_steps", []):
            if isinstance(step, dict) and "tool" in step:
                actions.append(step["tool"])
            elif isinstance(step, (list, tuple)) and step:
                ti = step[0]
                if isinstance(ti, dict) and "tool" in ti:
                    actions.append(ti["tool"])
                else:
                    actions.append(str(ti))
        return A2AResponse(status="ok", result={"message": final_text}, actions_taken=actions)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=A2AResponse(
                status="error",
                result={"error": str(e)},
                handoff_required=True,
                handoff_reason="Unhandled exception while performing card action.",
            ).model_dump(),
        )

# ===========================
# A2A protocol – Starlette app
# ===========================

logger = logging.getLogger("a2a-mount")
A2A_MOUNTED = False
A2A_MOUNT_PATH = "/a2a"

try:
    # Core A2A server pieces
    from a2a.server.agent_execution.agent_executor import AgentExecutor as A2AAgentExecutor
    from a2a.server.agent_execution.context import RequestContext
    from a2a.server.events.event_queue import EventQueue
    from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler

    # Public card + types
    from a2a.types import (
        AgentCard, AgentSkill, AgentCapabilities,
        Message, Role, Part, DataPart,
    )

    logger.info("a2a-sdk import OK, preparing to mount...")

    CARD_SKILLS = [
        AgentSkill(id="freeze_card", name="Freeze a card",
                   description="Freeze by last 4 digits.", tags=["cards","security"], examples=["freeze 1234"]),
        AgentSkill(id="unfreeze_card", name="Unfreeze a card",
                   description="Unfreeze a card.", tags=["cards","security"], examples=["unfreeze 1234"]),
        AgentSkill(id="report_lost", name="Report lost or stolen",
                   description="File a loss report and start replacement.", tags=["cards","loss"], examples=["I lost 1234 yesterday"]),
        AgentSkill(id="order_replacement", name="Order a replacement",
                   description="Order standard or express replacement.", tags=["cards","replacement"], examples=["replace 1234 express"]),
        AgentSkill(id="check_replacement_status", name="Check replacement status",
                   description="Check replacement order status.", tags=["cards","tracking"], examples=["status 1234"]),
    ]

    A2A_PUBLIC_CARD = AgentCard(
        name="Card Management Agent",
        description="Freezes/unfreezes cards, handles loss reports, orders replacements, and checks status.",
        url="http://localhost:8000",  # adjust in prod
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=CARD_SKILLS,
        supports_authenticated_extended_card=False,
    )

    # ---------- A2A helpers ----------

    def _to_dict(obj):
        """Best-effort convert Pydantic models or dicts to plain dicts."""
        if isinstance(obj, dict):
            return obj
        md = getattr(obj, "model_dump", None)  # Pydantic v2
        if callable(md):
            return md()
        dd = getattr(obj, "dict", None)        # Pydantic v1
        if callable(dd):
            return dd()
        js = getattr(obj, "json", None)
        if callable(js):
            try:
                return json.loads(js())
            except Exception:
                pass
        return {}

    def _extract_text_from_message(context: RequestContext) -> str:
        """Robustly extract user text from A2A RequestContext.message."""
        msg = getattr(context, "message", None)
        if not msg:
            return ""
        m = _to_dict(msg)                     # {'messageId':..., 'role':..., 'parts':[...]}
        parts = m.get("parts") or []
        texts = []
        for p in parts:
            pd = _to_dict(p)                  # e.g. {'kind':'text','text':'freeze 1234'}
            if isinstance(pd.get("text"), str):
                texts.append(pd["text"])
                continue
            if pd.get("kind") == "text" and isinstance(pd.get("data"), str):
                texts.append(pd["data"])
                continue
            for k in ("content", "value"):
                if isinstance(pd.get(k), str):
                    texts.append(pd[k])
                    break
        if not texts:
            for k in ("text", "content", "value"):
                if isinstance(m.get(k), str) and m[k].strip():
                    texts.append(m[k].strip())
                    break
        return " ".join(t.strip() for t in texts if isinstance(t, str)).strip()

    _JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)

    def _coerce_json(text: str) -> Dict[str, Any]:
        """
        Try to parse LLM output as JSON, stripping ``` fences if present.
        If parsing fails, wrap it so we still return structured data.
        """
        if not isinstance(text, str):
            return {"message": str(text)}
        m = _JSON_BLOCK.search(text)
        s = m.group(1) if m else text
        try:
            return json.loads(s)
        except Exception:
            return {"message": s}

    async def _emit_data_message(event_queue: EventQueue, context: RequestContext, payload: Dict[str, Any]) -> None:
        """Emit a DataPart message (clean, unescaped JSON)."""
        msg = Message(
            role=Role.agent,
            parts=[Part(root=DataPart(data=payload))],
            message_id=str(uuid.uuid4()),
            task_id=getattr(context, "task_id", None),
            context_id=getattr(context, "context_id", None),
        )
        await event_queue.enqueue_event(msg)

    # ---------- A2A executor ----------

    class CardAgentExecutor(A2AAgentExecutor):
        async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
            user_text = _extract_text_from_message(context)
            if not user_text:
                user_text = "Help with card management."

            parsed_action, parsed_last4 = extract_intent_and_last4(user_text)
            hint = f"Parsed(action={parsed_action}, last4={parsed_last4})"

            try:
                lc_result = executor.invoke({"input": f"{user_text}\n\n{hint}", "chat_history": []})
                final_text = lc_result.get("output", "") or str(lc_result)
            except Exception as e:
                final_text = json.dumps({
                    "handoff_required": True,
                    "reason": f"Error handling request: {e}",
                })

            payload = _coerce_json(final_text)
            await _emit_data_message(event_queue, context, payload)

        async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
            raise Exception("cancel not supported")

    a2a_handler = DefaultRequestHandler(
        agent_executor=CardAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    a2a_app_builder = A2AStarletteApplication(
        agent_card=A2A_PUBLIC_CARD,
        http_handler=a2a_handler,   # IMPORTANT: this enables the JSON-RPC endpoint
    )

    app.mount(A2A_MOUNT_PATH, a2a_app_builder.build())
    A2A_MOUNTED = True
    logger.info(f"A2A app mounted at {A2A_MOUNT_PATH}")

except Exception as e:
    logger.exception(f"Failed to mount A2A app: {e}")

# ---- Optional: minimal fallback well-known routes (discovery even if mount failed) ----
MINIMAL_AGENT_CARD = {
    "name": "Card Management Agent",
    "description": "Freezes/unfreezes cards, handles loss reports, orders replacements, and checks status.",
    "url": "http://localhost:8000",
    "version": "1.0.0",
    "default_input_modes": ["text"],
    "default_output_modes": ["text"],
    "capabilities": {"streaming": False},
    "skills": [
        {"id": "freeze_card", "name": "Freeze a card", "description": "Freeze by last 4 digits."},
        {"id": "unfreeze_card", "name": "Unfreeze a card", "description": "Unfreeze a card."},
        {"id": "report_lost", "name": "Report lost or stolen", "description": "File a loss report and start replacement."},
        {"id": "order_replacement", "name": "Order a replacement", "description": "Order standard or express replacement."},
        {"id": "check_replacement_status", "name": "Check replacement status", "description": "Check replacement order status."},
    ],
    "supports_authenticated_extended_card": False,
}

@app.get("/a2a/.well-known/agent-card.json")
def _fallback_agent_card():
    return MINIMAL_AGENT_CARD

@app.get("/a2a/.well-known/agent.json")
def _fallback_agent_json():
    return MINIMAL_AGENT_CARD

# Optional: health to confirm mount status
@app.get("/a2a")
def _a2a_health():
    return {"mounted": A2A_MOUNTED, "mount_path": A2A_MOUNT_PATH}

# ===========================
# Optional: run via `python credit_card_loss_agent.py`
# ===========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("credit_card_loss_agent:app", host="0.0.0.0", port=8000, reload=True)
