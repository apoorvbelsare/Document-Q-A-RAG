from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

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

def build_query_engine(documents):
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=80   # improved context retention
    )

    nodes = splitter.get_nodes_from_documents(documents)

    index = VectorStoreIndex(
        nodes,
        embed_model=embed_model
    )

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=5
    )

    return query_engine