from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType

from langchain_experimental.agents.agent_toolkits import create_csv_agent

from typing import *
import PlotlyTool
import chainlit as cl
import plotly.express as px
import pandas as pd

file = "orders.csv"

def plotly_chart_creator() -> str:
    """Creates a plotly chart from a dataset

    Returns:
        str: the plotly chart
    """
    df = pd.read_csv(file)
    fig = px.line(df)
    cl.Message(
        content="chart",
        elements=[cl.Plotly(name="chart", figure=fig, display="inline")],
    ).send()
    return "ok"


@cl.on_chat_start
async def start():
    df = pd.read_csv(file)
    tools = [
        PlotlyTool.PlotlyPythonAstREPLTool(locals={"df": df}),
    ]
    agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo", streaming=True),
        file,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        extra_tools=tools,
    )

    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    cl.user_session.set("figure", None)
    res = await agent.ainvoke(
        message.content, callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    elements = []
    figure = cl.user_session.get("figure")
    if figure:
        elements.append(cl.Plotly(name="chart", figure=figure, display="inline"))

    await cl.Message(content=res, elements=elements).send()