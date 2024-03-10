import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from utils.api_connector import get_ai_response

st.title("Consumer Reports EV Chatbot")

if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.session_state.messages = []


def run_llm_api_chat():
    """Run the LLM API chat."""
    gpt_source_documents = None
    st.chat_message("assistant").write(
        "Hello, I am your EV assistant.  How can I help you?"
    )
    if prompt := st.chat_input():
        llm_response = get_ai_response(
            question=prompt, history=st.session_state.messages
        )
        st.session_state.messages.append(HumanMessage(prompt))

        llm_retrieval_response = llm_response["retrieval_response"]

        st.session_state.messages.append(AIMessage(llm_retrieval_response))

        try:
            gpt_source_documents = llm_response["source_documents"]
        except Exception as e:
            gpt_source_documents = []

    for msg in st.session_state.messages:
        if type(msg) == HumanMessage:
            st.chat_message("human").write(msg.content)
        else:
            st.chat_message("assistant").write(msg.content)

    if gpt_source_documents is not None and len(gpt_source_documents) > 0:
        with st.expander("show source documents"):
            st.write(gpt_source_documents)

    with st.sidebar.expander("show all messages"):
        st.write(st.session_state.messages)


def main():
    """Run the main function."""
    st.sidebar.title("EV Chatbot GPT PDF RAG")
    source_url = "https://data.consumerreports.org/wp-content/uploads/2022/01/Consumer-Reports-Insights-for-More-Reliable-Electric-Vehicles-Jan-2022.pdf"
    st.sidebar.write("url source document", source_url)
    with st.sidebar.form(key="chat_form"):
        chat_start_button = st.form_submit_button("Start Chat")
        if chat_start_button:
            st.session_state.messages = []
            st.session_state.start_chat = True

    if "start_chat" in st.session_state and st.session_state.start_chat:
        run_llm_api_chat()

    clear_chat_button = st.sidebar.button("Clear Chat")
    if clear_chat_button:
        st.session_state.messages = []


if __name__ == "__main__":
    main()
