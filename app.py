import streamlit as st
import os
import glob
from sentence_transformers import SentenceTransformer
import faiss

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="FinanceGPT", layout="wide")

# Force a clean, executive interface
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stChatFloatingInputContainer {padding-bottom: 2rem;}
    </style>
    """, unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- LOGIN SYSTEM ---
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title("System Authorization")
    st.markdown("Please authenticate to access FinanceGPT.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Authenticate", use_container_width=True):
            if u == "admin" and p == "finance":
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Invalid credentials. Access denied.")
    st.stop()

# --- INITIALIZE SYSTEM ---
@st.cache_resource
def init_system():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    data_files = glob.glob(os.path.join(BASE_DIR, "data", "*.txt"))
    raw_text = ""
    for file in data_files:
        with open(file, "r", encoding="utf-8") as f:
            raw_text += f.read() + "\n\n"
            
    # Chunk the text carefully
    chunks = [c.strip() for c in raw_text.split("\n\n") if len(c.strip()) > 20]
    
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    return embedder, index, chunks

try:
    with st.spinner("Initializing neural architecture..."):
        embedder, index, chunks = init_system()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR (TOOLS) ---
with st.sidebar:
    st.title("FinanceGPT")
    st.markdown("A Finance Guide")
    
    if st.button("New Session", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    # --- PUSH BUTTONS TO BOTTOM ---
    # This creates the vertical space to pin the following buttons to the bottom
    for _ in range(25):
        st.write("")
        
    st.markdown("---")
    
    if st.button("Export Transcript", use_container_width=True):
        content = "\n\n".join([f"{m['role'].upper()}:\n{m['content']}" for m in st.session_state.messages])
        st.download_button("Download Transcript", content, "session_transcript.txt", use_container_width=True)

    if st.button("Terminate Session", use_container_width=True):
        st.session_state.auth = False
        st.rerun()

# --- MAIN UI ---
st.title("FinanceGPT")

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input Handling
if prompt := st.chat_input("Ask..."):
    
    # Display User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Synthesizing response..."):
            
            # --- STRICT RETRIEVAL (CEO MODE) ---
            vec = embedder.encode([prompt])
            D, I = index.search(vec, 1)
            retrieved_context = chunks[I[0][0]]
            
            # 1. Clean up "Question/Answer" labels if they exist in the raw text
            if "Answer:" in retrieved_context:
                retrieved_context = retrieved_context.split("Answer:")[-1].strip()
                
            # 2. Strict Cutoff: Prevent chunk bleed-over to the next topic
            # We use splitlines to ensure we only get the first coherent block
            if "\n" in retrieved_context:
                retrieved_context = retrieved_context.split("\n")[0].strip()
            
            # Deliver the exact, verified fact
            final_display = retrieved_context
            
            st.markdown(final_display)
            st.session_state.messages.append({"role": "assistant", "content": final_display})