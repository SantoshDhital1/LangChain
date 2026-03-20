from langchain_community.document_loaders import WebBaseLoader
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
    template='Answer the following question {questions} from the following text. \n {text}',
    input_variables=['questions','text']
)

# We can also pass the list of URLs.
url = 'https://www.geeksforgeeks.org/artificial-intelligence/introduction-to-langchain/'

loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({'questions': "what is langchain?", 'text': docs[0].page_content})

print(result)