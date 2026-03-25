import streamlit as st
from pipeline import run_pipeline

# Page config
st.set_page_config(
    page_title="Dell's Right Hand",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False

# Custom CSS for dark theme
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
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div class="title-section">
    <p class="main-title">Dell's Right Hand</p>
    <p class="subtitle">Your intelligent Dell knowledge assistant</p>
</div>
""", unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-bubble">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">{message["content"]}</div>', unsafe_allow_html=True)
        metadata = f"Confidence: {message.get('confidence', 'N/A')}% | Status: {message.get('status', 'N/A')} | Sources: {message.get('sources', 'N/A')}"
        st.markdown(f'<div class="metadata">{metadata}</div>', unsafe_allow_html=True)

# Input section
col1, col2 = st.columns([0.85, 0.15], gap="small")

with col1:
    user_input = st.text_input("Ask a question...", placeholder="Type here...", label_visibility="collapsed", key="user_input")

with col2:
    submit_button = st.button("Send", use_container_width=True)

# Handle submission
if submit_button and user_input and not st.session_state.processing:
    st.session_state.processing = True
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    try:
        # Call pipeline
        response = run_pipeline(user_input)
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.get("answer", "No response"),
            "confidence": response.get("confidence", 0),
            "status": response.get("status", "Complete"),
            "sources": response.get("sources", "N/A")
        })
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Error: {str(e)}",
            "confidence": 0,
            "status": "Error",
            "sources": "N/A"
        })
    finally:
        st.session_state.processing = False
        st.rerun()
