from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import JsonOutputParser
from typing import Optional
from pydantic import BaseModel, Field
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

# Define the Review schema
class Review(BaseModel):
    key_themes: list[str] = Field(description="write down all the key themes discussed in the review as a list")
    summary: str = Field(description='A brief summary of the review')
    sentiment: str = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside the list.")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside the list.")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer.")


# Initialize the JSON output parser with the Review schema
parser = JsonOutputParser(pydantic_object=Review)
format_instructions = parser.get_format_instructions()

review_text = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Santosh Dhital
"""

# Create a prompt that instructs the model to return JSON
prompt = f"""{format_instructions}

Analyze the following product review and extract the information in the specified JSON format:

{review_text}

Return ONLY the JSON object, no additional text."""

# Invoke the model with the prompt
response = model.invoke(prompt)

# Parse the JSON response
try:
    result = parser.parse(response.content)
    print("Successfully parsed review:")
    print(result)
except Exception as e:
    print(f"Error parsing response: {e}")
    print(f"Raw response:\n{response.content}")
