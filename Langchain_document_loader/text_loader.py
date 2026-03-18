from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token=(
        os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    )
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Write a summary from the following text. \n {text}',
    input_variables=['text']
)

loader = TextLoader('langchain.txt', encoding='utf-8')

docs = loader.load()

# print(type(docs))

# # print(docs[0])

# print(docs[0].page_content)

# print(docs[0].metadata)

chain = prompt | model | parser

result = chain.invoke({'text': docs[0].page_content})

print(result)