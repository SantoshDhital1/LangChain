from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_classic.schema.runnable import RunnableSequence, RunnableParallel
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


prompt1 = PromptTemplate(
    template="Generate a short tweet about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a short linkedin post about {topic}",
    input_variables=['topic']
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'tweet': RunnableSequence(prompt1, model, parser),
        'linkedin': RunnableSequence(prompt2, model, parser)
    }
)

result = parallel_chain.invoke({'topic': 'AI'})

print(result)