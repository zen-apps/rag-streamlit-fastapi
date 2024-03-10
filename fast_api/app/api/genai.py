import os
import json
from fastapi import APIRouter, Response

from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

import app.helpers.llm_tools as llm_tools

genai = APIRouter()


@genai.post("/ai_chat_retrieval/")
async def ai_chat(query: dict) -> Response:
    """Chat with the AI."""
    OPENAPI_KEY = os.getenv("OPENAI_API_KEY")
    llm = OpenAI(api_key=OPENAPI_KEY, temperature=0.0)

    rqa = llm_tools.run_retrieval_qa(llm)
    question = query["question"]
    history = query.get("conversation_history", "")

    qa_prompt_template = llm_tools.ai_retrieval_template()

    qa_prompt = PromptTemplate(
        input_variables=[
            "question",
            "conversation_history",
        ],
        template=qa_prompt_template,
    )

    prompt_filled_in = qa_prompt.format(
        **{
            "question": question,
            "conversation_history": history,
        }
    )

    response = rqa.invoke(prompt_filled_in)
    source_documents = response["source_documents"]
    docs_list = [doc.to_json() for doc in source_documents]
    docs_metadata = []
    for doc in docs_list:
        if "kwargs" in doc:
            page_content = doc["kwargs"]["page_content"]
            metadata = doc["kwargs"]["metadata"]
            docs_metadata.append({"page_content": page_content, "metadata": metadata})
    output_dict = {
        "retrieval_response": response["result"],
        "source_documents": docs_metadata,
    }
    return Response(json.dumps(output_dict), media_type="application/json")
