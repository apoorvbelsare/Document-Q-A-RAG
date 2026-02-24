from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank

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

documents = SimpleDirectoryReader("documents").load_data()

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
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=2   # keep best 2 chunks
)

query_engine = index.as_query_engine(
    similarity_top_k=6,   # retrieve more candidates
    node_postprocessors=[reranker]
)

# -----------------------------
# Ask question
# -----------------------------

response = query_engine.query("What problem does RAG solve?" +
 "How does vector search work?"+
"Explain RAG like I'm a beginner")

print("\nAnswer:")
print(response)
print("\nRetrieved Chunks:\n")

for node in response.source_nodes:
    print("----")
    print(node.text)