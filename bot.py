import streamlit as st
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Sequence, TypedDict
from langchain.chat_models import init_chat_model


class State(TypedDict):
    question: Annotated[Sequence[BaseMessage], ...]  

model =init_chat_model(
    model= "meta/llama-4-maverick-17b-128e-instruct",
    api_key="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",
    model_provider="nvidia",
    streaming=True
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a sentiment analyst and guardian angel therapist 
    You talk to users about their day, detect their mood, and respond with empathy and encouragement.
    Keep your tone gentle, warm, and supportive. Always try to make them feel better .
    """),
    MessagesPlaceholder(variable_name="question"),
])

def sentiment_analyst(state: State):
    """Analyzes the user's sentiment and responds kindly."""
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"question": response.content}

workflow = StateGraph(state_schema=State)
workflow.add_node("sentiment_analyst", sentiment_analyst)
workflow.add_edge(START, "sentiment_analyst")
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "guardian_001"}}


st.set_page_config(page_title="Guardian Angel ", page_icon="", layout="centered")

st.markdown(
    """
    <h2 style='text-align:center;'> How Do you Feel ?</h2>
    <p style='text-align:center; color:gray;'>
    Let check your mood
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)
if "messages" not in st.session_state:
    st.session_state["messages"] = []


for msg in st.session_state["messages"]:
    role = "assistant" if msg["role"] == "assistant" else "user"
    with st.chat_message(role):
        st.markdown(msg["content"])


if prompt := st.chat_input("How are you feeling today? "):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for chunk, metadata in app.stream(
            {"question": [HumanMessage(prompt)]}, config, stream_mode="messages"
        ):
            if hasattr(chunk, "content"):
                full_response += chunk.content
                response_placeholder.markdown(full_response + " ")
        response_placeholder.markdown(full_response)


    st.session_state["messages"].append({"role": "assistant", "content": full_response})
