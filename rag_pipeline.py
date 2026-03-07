from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.readers.file import PyMuPDFReader, DocxReader, EpubReader
from groq import Groq
from dotenv import load_dotenv
import tempfile
import os

# ============================
# Load Environment Variables
# ============================

load_dotenv()

# ============================
# Custom Groq LLM
# ============================

from pydantic import Field
from typing import Any

class GroqLLM(CustomLLM):
    client: Any = Field(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(
            self,
            "client",
            Groq(api_key=os.getenv("GROQ_API_KEY"))
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=8192,
            num_output=1024,
            is_chat_model=True,
        )

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        chat_completion = self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
        )

        return CompletionResponse(
            text=chat_completion.choices[0].message.content
        )

    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        stream = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        def generator():
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield CompletionResponse(text=delta)

        return generator()
    def __init__(self):
        super().__init__()
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._metadata = LLMMetadata(
            context_window=8192,
            num_output=1024,
            is_chat_model=True,
        )

    @property
    def metadata(self):
        return self._metadata

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        chat_completion = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )

        return CompletionResponse(
            text=chat_completion.choices[0].message.content
        )

    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        stream = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        def generator():
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield CompletionResponse(text=delta)

        return generator()

# ============================
# Models
# ============================

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

llm = GroqLLM()

Settings.llm = llm
Settings.embed_model = embed_model

# ============================
# Document Loader
# ============================

def load_documents(uploaded_files):
    temp_dir = tempfile.mkdtemp()

    file_extractor = {
        ".pdf": PyMuPDFReader(),
        ".docx": DocxReader(),
        ".epub": EpubReader(),
    }

    saved_files = []

    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        saved_files.append(file_path)

    reader = SimpleDirectoryReader(
        input_files=saved_files,
        file_extractor=file_extractor,
    )

    documents = reader.load_data()

    return documents

# ============================
# Chat Engine Builder
# ============================

def build_chat_engine(documents):
    if not documents:
        raise ValueError("No documents provided.")

    splitter = SentenceSplitter(
        chunk_size=256,
        chunk_overlap=20,
    )

    nodes = splitter.get_nodes_from_documents(documents, include_metadata=False)
    #limit node count 
    nodes = nodes[:200]
    index = VectorStoreIndex(
        nodes,
        embed_model=embed_model,
    )

    # Vector retriever
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=6,
    )

    # BM25 retriever
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=6,
    )

    # Hybrid retriever
    fusion_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=6,
    )

    # Reranker
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=3,
    )

    memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000
    )

    chat_engine = ContextChatEngine.from_defaults(
        retriever=fusion_retriever,
        memory=memory,
        node_postprocessors=[reranker],
        llm=llm,
    )

    return chat_engine