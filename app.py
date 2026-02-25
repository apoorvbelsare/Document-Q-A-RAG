from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import DocxReader, PyMuPDFReader, EpubReader

import streamlit as st
# -----------------------------
# Configure local models
# -----------------------------

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

Settings.llm = Ollama(
    model="llama3",
    request_timeout=120.0
)

# -----------------------------
# Load documents
# -----------------------------


file_extractor = {
    ".pdf": PyMuPDFReader(),   # ⭐ better extraction
    ".docx": DocxReader(),
    ".epub": EpubReader(),
}

documents = SimpleDirectoryReader(
    "documents",
    file_extractor=file_extractor
).load_data()
# -----------------------------
# Better chunking
# -----------------------------

splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=80,
)

nodes = splitter.get_nodes_from_documents(documents)

# -----------------------------
# Build index
# -----------------------------

index = VectorStoreIndex(nodes)

# -----------------------------
# Query engine
# -----------------------------

# reranker improves result ordering
# vector retriever
vector_retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=6,
)

# keyword retriever
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=6,
)

# combine both
fusion_retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=6,
)

# reranker (keep this!)
from llama_index.core.postprocessor import SentenceTransformerRerank

reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=2
)

# create chat memory
memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000
)

# create chat engine (replaces query_engine)
chat_engine = ContextChatEngine.from_defaults(
    retriever=fusion_retriever,
    memory=memory,
    node_postprocessors=[reranker],
)
# -----------------------------
# Ask question
# -----------------------------

response = query_engine.query("what is the second chapter about?")

print("\nAnswer:")
print(response)
print("\nRetrieved Chunks:\n")

#for node in response.source_nodes:
 #   print("----")
  #  print(node.text)