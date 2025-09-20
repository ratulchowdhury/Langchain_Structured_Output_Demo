from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, Annotated, Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv  
load_dotenv()

model = ChatOpenAI(model="gpt-4o", temperature=0.5)

json_schema = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "MovieReview",
  "type": "object",
  "properties": {
    "storyline": {
      "type": "string",
      "description": "One liner storyline of the movie"
    },
    "summary": {
      "type": "string",
      "description": "short summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["positive", "negative", "mixed"],
      "description": "sentiment of the review"
    },
    "verdict": {
      "type": "string",
      "enum": ["hit", "flop"],
      "description": "possible public verdict of the film based on the review"
    },
    "upside": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Possible upside of the movie"
    },
    "drawbacks": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Possible drawbacks of the movie"
    },
    "rating": {
      "type": "number",
      "exclusiveMinimum": 0,
      "exclusiveMaximum": 10,
      "default": 7.2,
      "description": "Possible imbd rating of the movie generated from the review out of 10"
    }
  },
  "required": ["storyline", "summary", "sentiment", "verdict"]
}

    
structured_model = model.with_structured_output(json_schema)

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
print(response["rating"])
print(response)

