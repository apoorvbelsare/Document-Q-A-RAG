from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# local embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# local LLM via Ollama
llm = Ollama(model="llama3")

# load documents
documents = SimpleDirectoryReader("documents").load_data()

# build index
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

# query engine using local LLM
query_engine = index.as_query_engine(llm=llm)

response = query_engine.query("who is undercover? Why?")

print("\nAnswer:")
print(response)
