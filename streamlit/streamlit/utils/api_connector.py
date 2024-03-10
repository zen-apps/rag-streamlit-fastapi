import requests
import os


def get_ai_response(question: str, history: str) -> dict:
    """Get AI response to question."""
    BACKEND_HOST = os.getenv("BACKEND_HOST")
    api_path = "v1/genai/ai_chat_retrieval/"
    api_url = f"{BACKEND_HOST}{api_path}"

    json_serielized_history = str(history)
    query = {"question": str(question), "conversation_history": json_serielized_history}
    response = requests.post(
        api_url, json=query, headers={"Content-Type": "application/json"}
    )
    if response.status_code != 200:
        raise ValueError(f"Error: {response.status_code}")
    return response.json()
