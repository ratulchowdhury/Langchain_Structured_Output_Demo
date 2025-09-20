from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, Annotated, Literal, Optional
from dotenv import load_dotenv  
load_dotenv()

model = ChatOpenAI(model="gpt-4o", temperature=0.5)

class entityExtractor(TypedDict):
    storyline: Annotated[str, "One liner storyline of the movie"]
    summary: Annotated[str, "short summary of the review"]
    sentiment: Annotated[Literal["positive", "negative", "mixed"], "sentiment of the review"]
    verdict: Annotated[Literal["hit","flop"], "possible public verdict of the film based on the review"]
    upside : Annotated[Optional[list[str]], "Possible upside of the movie"]
    drawbacks : Annotated[Optional[list[str]], "Possible drawbacks of the movie"]
    
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
print(response["verdict"])

