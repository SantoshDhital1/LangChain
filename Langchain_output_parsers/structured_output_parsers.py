from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_classic.output_parsers.structured import StructuredOutputParser, ResponseSchema
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


schema = [
    ResponseSchema(name="fact-1", description="fact-1 about the topic"),
    ResponseSchema(name="fact-2", description="fact-2 about the topic"),
    ResponseSchema(name="fact-3", description="fact-3 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)


template = PromptTemplate(
    template="Write 3 facts about the {topic} in short. \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"topic": "Mars"})

print(result)