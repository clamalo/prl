# gemini_demo_genai_NEW.py: Example of using Google Gemini API for various tasks


import json

# IMPORTS
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# SETUP
api_key = "AIzaSyCMGmY880U4KMeBQs9w29klNlCMRQJ8CNQ"
client = genai.Client(api_key=api_key)


# BASIC MODEL CONFIGURATION
"""
Use to initialize a Gemini model.
"""
# (No explicit object instantiation needed; calls happen directly.)


# SYSTEM INSTRUCTION
"""
Use to set a system instruction for the model.
"""
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain the theory of relativity in two sentences.",
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful assistant that provides concise and accurate information."
    ),
)
print(response.text)


# BASIC USAGE
"""
Use to generate a single response from the model.
"""
prompt = "Explain the theory of relativity in two sentences."
response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
print(response.text)


# TEMPERATURE
"""
Use to control the randomness of the model's responses.
"""
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt,
    config=types.GenerateContentConfig(temperature=0.5),
)
print(response.text)


# STREAMING RESPONSES
"""
Use to stream responses from the model.
"""
for chunk in client.models.generate_content_stream(
    model="gemini-2.0-flash", contents=prompt
):
    print(chunk.text, end="", flush=True)


# CHAT SESSIONS
"""
Use to create a chat session with the model and maintain conversation history.
"""
chat = client.chats.create(model="gemini-2.0-flash")
print("User: Hi Gemini!")
response = chat.send_message("Hi Gemini!")
print("Gemini:", response.text)
print("User: Explain the theory of relativity in two sentences.")
response = chat.send_message("Explain the theory of relativity in two sentences.")
print("Gemini:", response.text)


# STRUCTURED RESPONSES
"""
Use to define a structured response schema using Pydantic. Always define the generation_config in the call.
"""


class MuffinRecipe(BaseModel):
    name: str = Field(..., description="Name of the muffin recipe")
    servings: int = Field(..., description="Number of muffins this makes")
    ingredients: list[str] = Field(
        ..., description="List of ingredients with quantities"
    )
    steps: list[str] = Field(..., description="Step-by-step baking instructions")


response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Give me a simple blueberry muffin recipe",
    config=types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=MuffinRecipe
    ),
)
response_json = json.loads(response.text)
print(response_json)


# NESTED STRUCTURED RESPONSES
"""
Use to define a nested structured response schema using Pydantic. Always define the generation_config in the call.
"""


class Ingredient(BaseModel):
    item: str = Field(..., description="Ingredient name")
    quantity: float = Field(..., description="Amount of the ingredient")
    unit: str = Field(..., description="Measurement unit for the quantity")


class Step(BaseModel):
    order: int = Field(..., description="Step number in sequence")
    instruction: str = Field(..., description="Detailed instruction for this step")


class NestedMuffinRecipe(BaseModel):
    name: str = Field(..., description="Name of the muffin recipe")
    servings: int = Field(..., description="Number of muffins this makes")
    ingredients: list[Ingredient] = Field(..., description="List of Ingredient objects")
    steps: list[Step] = Field(..., description="Ordered Step objects")


response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Give me a simple blueberry muffin recipe",
    config=types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=NestedMuffinRecipe
    ),
)
nested_response_json = json.loads(response.text)
print(nested_response_json)


# THINKING BUDGET
"""
Use to set a thinking budget for the model.
"""
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-04-17",
    contents="Provide a list of 3 famous physicists and their key contributions",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=1024)
    ),
)
print(response.text)


# USAGE METADATA
usage = response.usage_metadata
print("Input (prompt) tokens: ", usage.prompt_token_count)
print("Thinking tokens:        ", usage.thoughts_token_count)
print("Output (response) tokens:", usage.candidates_token_count)
