from langchain_community.document_loaders import CSVLoader
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
    template='Answer the following question {question} from the following file. \n {file}',
    input_variables=['question', 'file']
)

loader = CSVLoader(file_path="car.csv")

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({'question': 'Which car gives the better mileage in lowest price?', 'file':docs[0].page_content})

print(result)