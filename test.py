from llama_index.readers.file import PDFReader

loader = PDFReader()
docs = loader.load_data(file="documents/1.pdf")
print(f"Loaded {len(docs)} documents")

for d in docs:
    print("----")
    print(d.text[:200])
print(docs[0].text[:1000])