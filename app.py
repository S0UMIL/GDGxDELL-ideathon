import streamlit as st
from pipeline import run_pipeline

st.set_page_config(
    page_title="Dell's Right Hand",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_key" not in st.session_state:
    st.session_state.input_key = 0
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

st.markdown("""
<style>
    body { background-color: #0d1117; color: #c9d1d9; }
    [data-testid="stSidebar"] { display: none; }
    .main { max-width: 800px; margin: 0 auto; }
    .title-section { text-align: center; margin-bottom: 24px; padding: 20px 0; border-bottom: 1px solid #30363d; }
    .main-title { font-size: 65px; font-weight: bold; color: #00bcd4; margin: 0; }
    .subtitle { font-size: 14px; color: #8b949e; margin-top: 8px; }
    .user-bubble { background-color: #0b5394; color: white; padding: 12px 16px; border-radius: 12px; margin: 8px 0 8px auto; width: fit-content; max-width: 75%; }
    .assistant-bubble { background-color: #37474f; color: #eceff1; padding: 12px 16px; border-radius: 12px; margin: 8px 0 8px 0; width: fit-content; max-width: 75%; }
    .metadata { font-size: 11px; color: #8b949e; margin-top: 6px; margin-left: 4px; }
    .input-hint { font-size: 11px; color: #8b949e; margin-top: 4px; text-align: right; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-section">
    <p class="main-title">Dell's Right Hand</p>
    <p class="subtitle">Your intelligent Dell knowledge assistant</p>
</div>
""", unsafe_allow_html=True)

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-bubble">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">{message["content"]}</div>', unsafe_allow_html=True)
        metadata = f"Confidence: {message.get('confidence', 'N/A')}% | Status: {message.get('status', 'N/A')} | Sources: {message.get('sources', 'N/A')}"
        st.markdown(f'<div class="metadata">{metadata}</div>', unsafe_allow_html=True)

def handle_enter():
    val = st.session_state[f"user_input_{st.session_state.input_key}"]
    if val.strip():
        st.session_state.pending_query = val.strip()
        st.session_state.input_key += 1

st.text_input(
    "Ask a question...",
    placeholder="Type your question and press Enter...",
    label_visibility="collapsed",
    key=f"user_input_{st.session_state.input_key}",
    on_change=handle_enter
)

st.markdown('<div class="input-hint">Press Enter to send</div>', unsafe_allow_html=True)

if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None

    st.session_state.messages.append({"role": "user", "content": query})

    try:
        response = run_pipeline(query)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.get("answer", "No response"),
            "confidence": response.get("confidence", 0),
            "status": response.get("status", "Complete"),
            "sources": ", ".join(response.get("sources", [])) or "N/A"
        })
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Error: {str(e)}",
            "confidence": 0,
            "status": "Error",
            "sources": "N/A"
        })
    st.rerun()
