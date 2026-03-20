from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader

loader = UnstructuredPDFLoader('Unit1-Introduction.pdf')

docs = loader.load()

# print(len(docs))

print(docs[3].page_content)