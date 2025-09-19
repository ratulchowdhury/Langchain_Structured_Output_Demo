from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from typing import TypedDict
from dotenv import load_dotenv  
load_dotenv()

model = ChatOpenAI(model="gpt-4o", temperature=0.5)

class entityExtractor(TypedDict):
    summary: str
    sentiment: int
    verdict: str
    
structured_model = model.with_structured_output(entityExtractor)

review = input("Enter the your movie review:")
template = PromptTemplate(template="""
                          Consider yourself a movie review analyzer. Now your task is to analyze the {review} 
                          and give a short summary, sentiment(positive, negative, mixed) and verdict (hit or flop).
                          The final output should be of the following format: 
                          "summary": short summary of the review, 
                          "sentiment": sentiment of the review , should be an of the following [positive, negative, mixed],
                          "verdict": possible public verdict of the film based on the review , should be any of the following [hit, flop]                  
                          """,
                          input_variables=["review"])

chain = template | structured_model
response = chain.invoke({"review": review})
print(type(response))
print(response)
print(response["verdict"])

