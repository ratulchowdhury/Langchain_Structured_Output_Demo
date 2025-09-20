from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, Annotated, Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv  
load_dotenv()

model = ChatOpenAI(model="gpt-4o", temperature=0.5)

class entityExtractor(BaseModel):
    
    storyline: str = Field(description="One liner storyline of the movie")
    summary: str = Field(description="short summary of the review")
    sentiment: Literal["positive", "negative", "mixed"] = Field(description="sentiment of the review")
    verdict: Literal["hit","flop"] = Field(description="possible public verdict of the film based on the review")   
    upside : Optional[list[str]] = Field(default=None, description="Possible upside of the movie")      
    drawbacks : Optional[list[str]] = Field(default=None, description="Possible drawbacks of the movie")
    rating: Optional[float] = Field(gt = 0, lt = 10,default=7.2, description="Possible imbd rating of the movie generated from the review out of 10")
    
    
structured_model = model.with_structured_output(entityExtractor)

review = input("Enter the your movie review:")
template = PromptTemplate(template="""
                          Consider yourself a movie review analyzer. Now your task is to analyze the {review} 
                          and generate a one liner storyline, short summary, sentiment(positive, negative, mixed), verdict (hit or flop), 
                          possible upsides and drawbacks of the move from the review.
                          """,
                          input_variables=["review"])

chain = template | structured_model
response = chain.invoke({"review": review})
print(type(response))
print(response)
print(response.rating)

