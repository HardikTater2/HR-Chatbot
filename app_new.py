import streamlit as st
from core_hr1 import workflow

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <style>
    body, .stApp {
        background: linear-gradient(120deg, #f8fafc 0%, #e3e7ed 100%);
        font-family: 'Inter', Arial, sans-serif;
    }
    .chat-title {
        font-size: 1.9rem;
        font-weight: 600;
        color: #222f3e;
        margin-bottom: 0.05rem;
    }
    .chat-subtitle {
        font-size: 0.78rem;
        color: #636e72;
        margin-bottom: 0.5rem;
    }
    .stTextInput>div>div>input {
        background: rgba(255,255,255,0.7);
        border-radius: 8px;
        border: 1px solid #e0e5ec;
        font-size: 0.87rem;
        padding: 0.28rem 0.5rem;
    }
    .user-bubble-wrapper, .bot-bubble-wrapper {
        display: flex;
        align-items: flex-end;
        margin-bottom: 0.3rem;
    }
    .user-bubble-wrapper {
        justify-content: flex-end;
    }
    .bot-bubble-wrapper {
        justify-content: flex-start;
    }
    .user-bubble, .bot-bubble {
        position: relative;
        padding: 0.22rem 0.5rem;
        font-size: 1.0 rem;  /* SMALLER TEXT */
        max-width: 45vw;
        min-width: 1.5rem;
        word-break: break-word;
        box-shadow: 0 1px 4px rgba(0,184,148,0.03);
        animation: fadeIn 0.4s;
        border-radius: 9px;
        backdrop-filter: blur(6px);
        border: 1px solid #e0e5ec;
    }
    .user-bubble {
        background: rgba(108,99,255,0.11);
        color: #6c63ff !important;
        border-radius: 9px 9px 2px 9px;
        margin-left: 2vw;
        border: 1px solid #6c63ff22;
    }
    .bot-bubble {
        background: rgba(255,255,255,0.7);
        color: #222f3e !important;
        border-radius: 9px 9px 9px 2px;
        margin-right: 2vw;
        border: 1px solid #e0e5ec;
    }
    .user-icon, .bot-icon {
        width: 19px;
        height: 19px;
        border-radius: 50%;
        margin: 0 0.35rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        background: rgba(255,255,255,0.8);
        box-shadow: 0 1px 4px rgba(108,99,255,0.04);
    }
    .user-icon {
        color: #fff;
        background: linear-gradient(135deg, #6c63ff 70%, #00cec9 100%);
    }
    .bot-icon {
        color: #6c63ff;
        background: #fff;
    }
    .footer {
        font-size: 0.68rem;
        color: #b2bec3;
        text-align: center;
        margin-top: 1.1rem;
        margin-bottom: 0.1rem;
        font-family: 'Inter', Arial, sans-serif;
        letter-spacing: 0.1px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "metadata" not in st.session_state:
    st.session_state["metadata"] = {}
if "current_agent" not in st.session_state:
    st.session_state["current_agent"] = ""
if "sender" not in st.session_state:
    st.session_state["sender"] = "anonymous"

# --- Header ---
st.markdown('<div class="chat-title">Multi-Agent Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="chat-subtitle">Your smart assistant is here to help. Ask anything!</div>', unsafe_allow_html=True)

# --- Chat History Rendering (compact, small text) ---
for chat in st.session_state["chat_history"]:
    if chat["role"] == "user":
        st.markdown(
            f'''
            <div class="user-bubble-wrapper">
                <div class="user-bubble">{chat["content"]}</div>
                <div class="user-icon">🧑‍💻</div>
            </div>
            ''',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'''
            <div class="bot-bubble-wrapper">
                <div class="bot-icon">🤖</div>
                <div class="bot-bubble">{chat["content"]}</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

# --- Auto-scroll to bottom ---
st.markdown("""
    <script>
    var chatContainer = window.parent.document.querySelector('.main');
    if(chatContainer){
        chatContainer.scrollTo({top: chatContainer.scrollHeight, behavior: 'smooth'});
    }
    </script>
""", unsafe_allow_html=True)

# --- User Input ---
user_input = st.chat_input("Type your message...", key="user_input")

if user_input:
    # Prevent duplicate consecutive messages
    if not st.session_state["chat_history"] or st.session_state["chat_history"][-1]["content"] != user_input:
        # Prepare state for workflow
        state = {
            "input": user_input,
            "sender": st.session_state["sender"],
            "chat_history": st.session_state["chat_history"].copy(),
            "current_agent": st.session_state["current_agent"],
            "agent_output": "",
            "metadata": st.session_state["metadata"],
        }
        result = workflow(state)  # Call your core logic

        # Update session state
        st.session_state["chat_history"] = result["chat_history"]
        st.session_state["metadata"] = result["metadata"]
        st.session_state["current_agent"] = state["current_agent"]
        st.rerun()  # Refresh UI to show new messages


