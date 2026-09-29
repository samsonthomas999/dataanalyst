from langchain_groq import ChatGroq
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

def analyse(df, question):
    llm = ChatGroq(
        model="groq/compound-mini",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )
    
    tools = [PythonREPLTool()]
    
    prompt = PromptTemplate.from_template("""
You are a data analyst. You have access to a pandas dataframe.
The dataframe is already loaded as variable `df`.

Answer the following question by writing and executing Python code.
Always use the `df` variable to access the data.
Use plotly for visualizations.
Do not include any thinking tags or reasoning blocks in your response.

You have access to the following tools:
{tools}

Tool names: {tool_names}

Question: {input}

{agent_scratchpad}
""")

    agent = create_react_agent(llm, tools, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    result = executor.invoke({
        "input": f"The dataframe 'df' has the following columns: {list(df.columns)}. Question: {question}"
    })
    return result["output"]