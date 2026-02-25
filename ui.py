import streamlit as st
from rag_pipeline import build_query_engine, load_documents

st.set_page_config(page_title="RAG Assistant", layout="wide")

st.title("📚 Local RAG Assistant")

# Session storage
if "engine" not in st.session_state:
    st.session_state.engine = None

# Upload documents
uploaded_files = st.file_uploader(
    "Upload PDFs or DOCX",
    type=["pdf", "docx", "epub"],
    accept_multiple_files=True
)

if uploaded_files and st.button("Process Documents"):
    with st.spinner("Processing..."):
        docs = load_documents(uploaded_files)
        st.session_state.engine = build_query_engine(docs)
    st.success("Documents indexed!")

# Ask questions
if st.session_state.engine:
    query = st.text_input("Ask a question about your documents:")

    if query:
        with st.spinner("Thinking..."):
            response = st.session_state.engine.query(query)

        st.subheader("Answer")
        st.write(response.response)

        st.subheader("Sources")
        for node in response.source_nodes:
            st.write("---")
            st.write(node.node.text[:300] + "...")