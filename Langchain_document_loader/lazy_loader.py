from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

# lazy load fetched documents one at a time when needed.
docs = loader.lazy_load()

for document in docs:
    print(document.metadata)