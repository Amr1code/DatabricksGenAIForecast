"""
Streamlit chat interface for the Retail Demand Forecasting GenAI agent.

Run locally with:
    streamlit run app.py

Requires a local .env file (see .env.example) with:
    OPENAI_API_KEY
    DATABRICKS_SERVER_HOSTNAME
    DATABRICKS_HTTP_PATH
    DATABRICKS_TOKEN
"""

import streamlit as st
from agent import ask_agent

st.set_page_config(page_title="Retail Demand Forecasting Assistant", page_icon="📊")

st.title("📊 Retail Demand Forecasting Assistant")
st.caption(
    "Ask questions about store sales trends and forecasting accuracy. "
    "Answers are grounded in real data from the project's Gold Delta table."
)

with st.expander("Example questions"):
    st.markdown(
        "- What's the recent sales trend for store 1?\n"
        "- How accurate is our forecasting model for store 5?\n"
        "- Compare the sales performance of store 5 versus store 10 over the last 60 days\n"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask about a store's sales or forecast accuracy..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking the data..."):
            # Pass prior turns (excluding this new question) for conversational context
            history = st.session_state.messages[:-1]
            answer = ask_agent(question, history=history)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.header("About")
    st.markdown(
        "This assistant uses OpenAI function-calling to decide which data "
        "queries to run against a Databricks Gold Delta table, then explains "
        "the results in plain language.\n\n"
        "Part of the **Retail Demand Forecasting & GenAI Analytics** project."
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()