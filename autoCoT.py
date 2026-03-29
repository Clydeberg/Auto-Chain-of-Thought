from openai import OpenAI
import json
client =OpenAI(
    api_key="",
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)
SYSTEM_PROMPT = """
You're an expert AI Assistant in resolving user queries using chain of thought.
You work in START, PLAN and OUTPUT steps.
You need to first PLAN what needs to be done. The PLAN can be multiple steps.
Once you think enough PLAN has been done, finally you can give an OUTPUT.

Rules:
- Strictly Follow the given JSON output format
- Only run one step at a time.
- The sequence of steps is START (where user gives an input), PLAN (That can be multiple steps)
  and finally OUTPUT (which is going to be displayed to the user).

Output JSON Format:
{ "step": "START" | "PLAN" | "OUTPUT", "content": "string" }

Example:

START: Hey, Can you solve 2 + 3 * 5 / 10

PLAN: { "step": "PLAN", "content": "Seems like user is interested in math problem" }

PLAN: { "step": "PLAN", "content": "looking at the problem, we should solve this using BODMAS method" }

PLAN: { "step": "PLAN", "content": "Yes, The BODMAS is correct thing to be done here" }

PLAN: { "step": "PLAN", "content": "first we must multiply 3 * 5 which is 15" }

PLAN: { "step": "PLAN", "content": "Now the new equation is 2 + 15 / 10" }

PLAN: { "step": "PLAN", "content": "We must perform divide that is 15 / 10 = 1.5" }

PLAN: { "step": "PLAN", "content": "Now the new equation is 2 + 1.5" }

PLAN: { "step": "PLAN", "content": "Now finally lets perform the add 3.5" }

PLAN: { "step": "PLAN", "content": "Great, we have solved and finally left with" }

OUTPUT: { "step": "OUTPUT", "content": "3.5" }
"""

print("\n\n\n")
response_history=[{"role": "system",
            "content":SYSTEM_PROMPT},]

user_query=input("->")
response_history.append({"role":"user",
            "content":user_query})
while True:
    response=client.chat.completions.create(
        model="gemini-2.5-flash",
        response_format={"type":"json_object"},
        messages=response_history
    )
    raw_result=response.choices[0].message.content
    response_history.append({"role":"assistant",
            "content":raw_result})
    
    parsed_result=json.loads(raw_result)

    if parsed_result.get("step")=="START":
        print("Welcome",parsed_result.get("content"))
        continue
    
    elif parsed_result.get("step")=="PLAN":
        print("Thinking",parsed_result.get("content"))
        continue

    elif parsed_result.get("step")=="OUTPUT":
        print("Thought-->Answer is:\n",parsed_result.get("content"))
        break
        

print("\n\n")