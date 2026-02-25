from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine


from llama_index.readers.file import (
    PyMuPDFReader,
    DocxReader,
    EpubReader
)

import tempfile
import os

# ---------------------------
# Local models
# ---------------------------

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

llm = Ollama(
    model="llama3",
    request_timeout=120.0
)

# ---------------------------
# Document Loader
# ---------------------------

def load_documents(uploaded_files):
    """
    Saves Streamlit uploads temporarily and
    uses proper file readers for clean extraction.
    """

    temp_dir = tempfile.mkdtemp()

    file_extractor = {
        ".pdf": PyMuPDFReader(),   # ⭐ best for PDF text extraction
        ".docx": DocxReader(),
        ".epub": EpubReader()
    }

    saved_files = []

    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        saved_files.append(file_path)

    reader = SimpleDirectoryReader(
        input_files=saved_files,
        file_extractor=file_extractor
    )

    documents = reader.load_data()

    return documents


# ---------------------------
# Query Engine Builder
# ---------------------------

def build_chat_engine(documents):
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=80
    )

    nodes = splitter.get_nodes_from_documents(documents)

    index = VectorStoreIndex(
        nodes,
        embed_model=embed_model
    )

    # memory buffer (stores conversation)
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000
    )

    chat_engine = ContextChatEngine.from_defaults(
        retriever=index.as_retriever(similarity_top_k=5),
        memory=memory,
        llm=llm
    )

    return chat_engine