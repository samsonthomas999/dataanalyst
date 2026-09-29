from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

def analyse(df, question):
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )
    
    summary = f"""
Columns: {list(df.columns)}
Shape: {df.shape}
Sample data:
{df.head(10).to_string()}
Stats:
{df.describe().to_string()}
"""
    
    prompt = f"""You are a data analyst. Here is the dataframe summary:

{summary}

Answer this question concisely: {question}
"""
    
    response = llm.invoke(prompt)
    return response.content