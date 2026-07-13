# day86:AI Agent - LLM that can decide WHICH TOOL to use to answer a question

import datetime
import math
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline as hf_pipeline



#  1: Define Tools 

@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression. Use this for any math calculation.
    Input should be a valid Python math expression like '5+5' or 'math.sqrt(16)'"""
    try:
        result = eval(expression, {"math": math, "__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def get_current_date(query: str) -> str:
    """Returns the current date and time. Use when asked about today's date."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def word_counter(text: str) -> str:
    """Counts the number of words in a given text."""
    count = len(text.split())
    return f"Word count: {count}"

@tool
def unit_converter(query: str) -> str:
    """Converts kilometers to miles. Input format: 'X km'
    Example: '10 km' returns miles equivalent"""
    try:
        km = float(query.replace('km','').strip())
        return f"{km} km = {km * 0.621371:.2f} miles"
    except:
        return "Please provide format: 'X km'"

tools = [calculator, get_current_date, word_counter, unit_converter]
print(f"Tools available: {[t.name for t in tools]}")

# Step 2: Simple ReAct-style Agent 

def simple_agent(question):

    q_lower = question.lower()
    if any(op in q_lower for op in ['+','-','*','/','sqrt','square','calculate','math']):
        expr = ''.join(c for c in question if c in '0123456789+-*/().sqrt math')
        result = calculator.invoke(expr.strip())
        tool_used = "calculator"
    elif any(w in q_lower for w in ['date','today','time','current']):
        result = get_current_date.invoke(question)
        tool_used = "get_current_date"
    elif any(w in q_lower for w in ['km','kilometer','miles','convert']):
        result = unit_converter.invoke(question)
        tool_used = "unit_converter"
    elif any(w in q_lower for w in ['count','words','how many words']):
        result = word_counter.invoke(question)
        tool_used = "word_counter"
    else:
        result = "I don't have a tool for that question"
        tool_used = "none"
    return result, tool_used

# Step 3: Test the Agent 
questions = [
    "What is 15 * 24 +50- 100?",
    "What is today's date?",
    "Convert 50 km to miles",
    "Count words in: machine learning is amazing and powerful",
    "Calculate sqrt(196)",
]

print("\nAgent responses:")
for q in questions:
    result, tool_used = simple_agent(q)
    print(f"\n  Q: {q}")
    print(f"  Tool used: {tool_used}")
    print(f"  Answer: {result}")
