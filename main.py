from openai import OpenAI
from dotenv import load_dotenv
import requests
import json
from datetime import datetime
import os
from pydantic import BaseModel, Field
from typing import Optional
load_dotenv()
client=OpenAI()

def run_command(cmd: str):
    result=os.system(cmd)
    return result
def get_weather(city: str):
    url= f"https://wttr.in/{city.lower()}?format=%C+%t"
    response=requests.get(url)

    if response.status_code==200:
        return f"The weather in {city} is {response.text}"
    return "Something went wrong"

available_tools={
    "get_weather":get_weather,
    "run_command":run_command
}

SYSTEM_PROMPT = f"""
    You are an helpfull AI Assistant who is specialized in resolving user query.
    You work on start, plan, action, observe mode.

    For the given user query and available tools, plan the step by step execution, based on the planning,
    select the relevant tool from the available tool. and based on the tool selection you perform an action to call the tool.

    Wait for the observation and based on the observation from the tool call resolve the user query.

    Rules:
    - Follow the Output JSON Format.
    - Always perform one step at a time and wait for next input
    - Carefully analyse the user query

    Output JSON Format:
    {{
        "step": "string",
        "content": "string",
        "function": "The name of function if the step is action",
        "input": "The input parameter for the function",
    }}
    Available Tools:
    - "get_weather": Takes a city name as an input and returns the current weather for the city
    - "run_command": Takes linux command as a string and executes the command and returns the output after executing it.

    Example:
    User Query: What is the weather of new york?
    Output: {{ "step": "plan", "content": "The user is interseted in weather data of new york" }}
    Output: {{ "step": "plan", "content": "From the available tools I should call get_weather" }}
    Output: {{ "step": "action", "function": "get_weather", "input": "new york" }}
    Output: {{ "step": "observe", "output": "12 Degree Cel" }}
    Output: {{ "step": "output", "content": "The weather for new york seems to be 12 degrees." }}

"""
print("\n\n\n")
class MyOutputFormat(BaseModel):
    step: str=Field(..., description="The ID of the step.Example: PLAN, OUTPUT,TOOL, etc")
    content:Optional[str]=Field(None, description="The optional string content for the ")
    tool:Optional[str]=Field(None,description="The ID of the tool to call.")
    input:Optional[str]=Field(None, description="The input params for the tools")
message_history=[
    {"role":"system", "content":SYSTEM_PROMPT}
]
while True:
    user_query=input("> ")
    message_history.append({"role":"user","content":user_query})
    while True:
        response= client.chat.completions.parse(
            model="gpt-4o",
            response_format=MyOutputFormat,
            messages=message_history
        )

        raw_result=response.choices[0].message.content
        message_history.append({"role":"assistant","content":raw_result})

        parsed_result=response.choices[0].message.parsed
        if parsed_result.step=="START":
            print(parsed_result.content)
            continue;

        if parsed_result.step=="TOOL":
            tool_to_call = parsed_result.tool 
            tool_input = parsed_result.input
            print(f"{tool_to_call} ()")
            tool_response = available_tools[tool_to_call](tool_input)
            print(f"{tool_to_call} ({tool_input})=={tool_response}")

            message_history.append({"role":"develop", "content":json.dumps({"step":"OBSERVE","tool":tool_to_call,"input":tool_input})})
            continue;

        if parsed_result.step=="PLAN":
            print(parsed_result.content)
            continue;

        if parsed_result.step=="OUTPUT":
            print(parsed_result.content)
            break
