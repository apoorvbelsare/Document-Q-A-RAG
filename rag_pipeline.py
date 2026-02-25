from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm = Ollama(model="llama3")

def load_documents(uploaded_files):
    import tempfile, os

    docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        docs.extend(SimpleDirectoryReader(input_files=[tmp_path]).load_data())
        os.remove(tmp_path)
    return docs

def build_query_engine(documents):
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)

    index = VectorStoreIndex(
        nodes,
        embed_model=embed_model
    )

    return index.as_query_engine(llm=llm, similarity_top_k=5)