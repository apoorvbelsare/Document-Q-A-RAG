import streamlit as st
from rag_pipeline import build_chat_engine, load_documents

st.set_page_config(page_title="RAG Assistant", layout="wide")

st.title("📚 Local RAG Assistant")

# -------------------------
# Session storage
# -------------------------
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------
# Upload documents
# -------------------------
uploaded_files = st.file_uploader(
    "Upload PDFs or DOCX",
    type=["pdf", "docx", "epub", "txt"],
    accept_multiple_files=True
)

if uploaded_files and st.button("Process Documents"):
    with st.spinner("Processing..."):
        docs = load_documents(uploaded_files)

        # BUILD CHAT ENGINE (with memory)
        st.session_state.chat_engine = build_chat_engine(docs)

    st.success("Documents indexed!")

# -------------------------
# Ask questions
# -------------------------
if st.session_state.chat_engine:

    query = st.text_input("Ask a question about your documents:")

    if query:
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(query)

        # store history
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("Assistant", response.response))

    # display chat history
    for role, msg in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**🧑‍💻 You:** {msg}")
        else:
            st.markdown(f"**🤖 Assistant:** {msg}")

    # show sources for last response
    if st.session_state.chat_history:
        st.subheader("Sources")
        for node in response.source_nodes:
            st.write("---")
            st.write(node.node.text[:300] + "...")