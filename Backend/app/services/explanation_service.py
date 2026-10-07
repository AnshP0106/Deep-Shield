import json
import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not configured."
    )


client = genai.Client(
    api_key=GEMINI_API_KEY
)


MODEL_NAME = "gemini-3.5-flash-lite"


def generate_explanation(
    media_type: str,
    detection_result: dict
) -> dict:

    prompt = f"""
You are DeepShield, an AI-powered deepfake detection assistant.

Your job is to explain a detector's result to a normal user.

IMPORTANT RULES:
1. Use ONLY the information provided in the detection result.
2. Do NOT invent visual, audio, or forensic evidence.
3. Do NOT claim certainty when the detector is uncertain.
4. Explain technical information in simple language.
5. Keep the explanation brief but useful.
6. The explanation should help a normal user understand WHY the detector reached its result.
7. If the result is uncertain, clearly say that.
8. Do not provide instructions for creating or manipulating deepfakes.

Media type:
{media_type}

Detection result:
{json.dumps(detection_result, indent=2, default=str)}

Return ONLY valid JSON in exactly this structure:

{{
    "summary": "One short sentence describing the result.",
    "why": "Brief explanation of the evidence or detector signals.",
    "simple_explanation": "Explain the result in very simple language.",
    "confidence_explanation": "Explain what the confidence score means.",
    "user_advice": "One short practical sentence for the user."
}}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    text = response.text.strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:

        return {
            "summary": text,
            "why": "",
            "simple_explanation": "",
            "confidence_explanation": "",
            "user_advice": ""
        }