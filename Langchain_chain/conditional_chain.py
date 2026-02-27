from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=(
        os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    )
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

feedback = """Okay so I actually had to return this item with amazon because they messed up delivery and my friend wouldn't get it in time because I am studying overseas and had to get it from Best Buy instead. However, I just have to leave a review.

I LOVE this MacBook. I am a Mac user through and through, I have had a 2020 Pro, 2021 Air, and 2023 Air before. This one is far superior. I upgraded from my 2023 Air because my cat likes to chew on my screen and I didn't want to wait to fix the screen so I am going to sell the other once fixed. Additionally, I am a hobby photographer and waiting 10 seconds for each task on my air to complete in Lightroom and photoshop was miserable.

I did a lot of research before getting this laptop and was trying to decide between a M3 pro, M4 pro and M5 chip. Ultimately I decided on this because the deal was unbeatable for Black Friday and the product reviews only say it is a little powerhouse despite the lower CPU/GPU in comparison with the pro chips.

I can't compare to those chips, but the difference between the 2023 Air and the M5 is night and day in regards to speed of complex processes. It is also very nice for the SD card slot to be in the computer and not needing an adapter. For any PHOTOGRAPHER this is an amazing laptop that will be a lovely upgrade. It runs Lightroom and photoshop seamlessly. It takes no loading time to upload 100 photos onto light room and instantly saves them. Well worth the money.

I haven't had any problems with it overheating, even though I am running a lot of tasks. The screen size is the perfect size IMO, the 13 is too small and my 15 air was too big. Brightness is great. It looks amazing in the space black color and is so sleek. Heavier than my 15 in air but that is to be expected. The battery life is good for how powerful it is.

Overall super happy with this laptop. Would buy again every time."""

class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description="give the sentiment of the feedback.")


parser1 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="classify only the sentiment of the following text into positive or negative. \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser1.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback. \n {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback. \n {feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x['sentiment' == 'positive'], prompt2 | model | parser),
    (lambda x: x['sentiment' == 'negative'], prompt3 | model | parser),
    RunnableLambda(lambda x: "cannot find sentiment")
)

chain = classifier_chain | branch_chain
 
result = chain.invoke({'feedback': feedback})

print(result)

chain.get_graph().print_ascii()