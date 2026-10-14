import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.explanation_service import client, MODEL_NAME

router = APIRouter(prefix="/chat", tags=["Chatbot"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list, max_length=12)
    analysis_context: dict | None = None


@router.post("")
def chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    history = []
    for item in request.history[-10:]:
        role = item.get("role")
        content = item.get("content")

        if role in ("user", "assistant") and isinstance(content, str):
            history.append({
                "role": role,
                "content": content[:2000],
            })

    context = request.analysis_context or {}

    prompt = f"""
You are DeepShield's AI chatbot, an assistant for understanding
deepfakes, media authenticity, and AI detection.

Rules:
- Answer clearly and helpfully.
- Use the supplied analysis context only when relevant.
- Never invent detector findings or claim certainty from a model score.
- Explain that AI detection can make mistakes.
- If no analysis context is provided, answer general questions normally.
- Do not reveal API keys, environment variables, or internal secrets.
- Keep answers concise and useful.

CURRENT ANALYSIS CONTEXT:
{json.dumps(context, ensure_ascii=False)[:10000]}

RECENT CONVERSATION:
{json.dumps(history, ensure_ascii=False)}

USER'S NEW MESSAGE:
{message}

Respond with plain text, not JSON.
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        answer = (response.text or "").strip()

        if not answer:
            raise ValueError("Gemini returned an empty response.")

        return {"answer": answer}

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="The chatbot could not generate a response. Please try again.",
        )