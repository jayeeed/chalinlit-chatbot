from dotenv import dotenv_values

from operator import itemgetter

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory

from chainlit.types import ThreadDict

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import date

from openai import AsyncOpenAI
from google.cloud import bigquery

import chainlit as cl
from chainlit.playground.providers.openai import ChatOpenAI 
from langchain.chat_models import ChatOpenAI as ResumeChatOpenAI

from fastapi import APIRouter, Query, HTTPException
import pandas as pd
from pydantic import BaseModel
from typing import Union
from langchain.sql_database import SQLDatabase
from langchain.memory import ConversationBufferWindowMemory
from langchain_experimental.sql import SQLDatabaseChain, SQLDatabaseSequentialChain  
from pyodbc import ProgrammingError, OperationalError, NotSupportedError

import logging



config=dotenv_values(".env")

db_name=config["db_name"]
host=config["host"]
user=config["user"]
password=config["password"]
port=config["port"]

# Database connection string
DATABASE_URL = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}'
print("DATABASE_URL: {}".format(DATABASE_URL))

# Create a database engine
engine = create_engine(DATABASE_URL)

# Create a Session class
Session = sessionmaker(bind=engine)

openai_client = AsyncOpenAI()


# settings = {"model": "gpt-3.5-turbo", "temperature": 0, "stop": ["```"]}
settings = {"model": "gpt-4", "temperature": 0, "stop": ["```"]}

sql_query_prompt = """You have PostgreSQL tables named `my_reports`, `my_invoices`, `my_logistics`, `payment_reports`,  `my_purchase_orders` and 'transactions' in the database `mydata`.

Write an PostgreSQL query, but avoid MS SQL or MySQL query to retrieve the full order based on the given question:

{input}
"""

explain_query_result_prompt = """Today is {date}
You received a query from a supplier or buyer regarding these tables.
They executed a PostgreSQL query, but avoid MS SQL or MySQL query, and provided the results in Markdown table format.
Analyze the table and explain the problem to the supplier or buyer only if there is error in SQL execution, else do not show any error.

Markdown Table:

{table}


Short and concise analysis:
"""


class Text(BaseModel):
    text: str
    activeChatId: str

llm = ResumeChatOpenAI(model_name="gpt-4", temperature=0, api_key=config['OPENAI_API_KEY']) 
chat_template = """ Based on the schema given {info} write an executable Microsoft SQL Server query for the user input, avoid MySQL query. 
        Execute it in the database and get sql results. Make a response to user from mssql results based on 
        the question. 
        Input: "user input"
        SQL query: "SQL Query here"
        """
chat_prompt = ChatPromptTemplate.from_messages([
    ('system', chat_template),
    MessagesPlaceholder(variable_name='history'),
    ('human', "{input}")
])
    
db = SQLDatabase.from_uri(DATABASE_URL)
table_info = db.table_info[:10]
m1 = ConversationBufferWindowMemory(k=4,return_messages=True)
# db_chain = SQLDatabaseChain.from_llm(llm, db,verbose = True, use_query_checker=True)
db_chain = SQLDatabaseSequentialChain.from_llm(llm, db,verbose = True, use_query_checker=True)

def setup_runnable():
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    model = ResumeChatOpenAI(streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful chatbot"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    runnable = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | model
        | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)
    

@cl.password_auth_callback
def auth():
    return cl.User(identifier="test")


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    setup_runnable()    


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    memory = ConversationBufferMemory(return_messages=True)
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "USER_MESSAGE":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)

    setup_runnable()

# FROM BIGQUERY    
@cl.step(type="llm")
async def gen_query(human_query: str):
    current_step = cl.context.current_step
    current_step.generation = cl.ChatGeneration(
        provider=ChatOpenAI.id,
        messages=[
            cl.GenerationMessage(
                role="user",
                template=sql_query_prompt,
                formatted=sql_query_prompt.format(input=human_query),
            )
        ],
        settings=settings,
        inputs={"input": human_query},
    )

    # Call OpenAI and stream the message
    stream_resp = await openai_client.chat.completions.create(
        messages=[m.to_openai() for m in current_step.generation.messages],
        stream=True,
        **settings
    )
    async for part in stream_resp:
        token = part.choices[0].delta.content or ""
        if token:
            await current_step.stream_token(token)

    current_step.language = "sql"
    current_step.generation.completion = current_step.output

    return current_step.output


@cl.step
async def execute_query(query):
    # Execute the SQL query
    # query_job = client.query(query)
    
    # # Wait for the query to complete
    # query_job.result()

    # # Get the query results
    # results = query_job.to_dataframe()
    
    # markdown_table = results.to_markdown(index=False)

    # NEW PostgreSQL 
    # Create a session
    session = Session()
    
    result = session.execute(text(query))
    # result = session.execute(query)
    print("result: {}".format(result))
    
    # Convert the result to a pandas DataFrame for easier Markdown conversion
    df = pd.DataFrame(result.fetchall(), columns=result.keys())
    print("df: {}".format(df))
    
    markdown_table = df.to_markdown(index=False)
    print("markdown_table: {}".format(markdown_table))
    # Close the session
    session.close()
    
    return markdown_table
    

@cl.step(type="llm")
async def analyze(table):
    current_step = cl.context.current_step
    today = str(date.today())
    current_step.generation = cl.ChatGeneration(
        provider=ChatOpenAI.id,
        messages=[
            cl.GenerationMessage(
                role="user",
                template=explain_query_result_prompt,
                formatted=explain_query_result_prompt.format(date=today, table=table),
            )
        ],
        settings=settings,
        inputs={"date": today, "table": table},
    )

    final_answer = cl.Message(content="")
    await final_answer.send()

    # Call OpenAI and stream the message
    stream = await openai_client.chat.completions.create(
        messages=[m.to_openai() for m in current_step.generation.messages],
        stream=True,
        **settings
    )
    async for part in stream:
        token = part.choices[0].delta.content or ""
        if token:
            await final_answer.stream_token(token)

    final_answer.actions = [
        cl.Action(name="take_action", value="action", label="Take action")
    ]
    await final_answer.update()

    current_step.output = final_answer.content
    current_step.generation.completion = final_answer.content

    return current_step.output


@cl.step(type="run")
async def chain(human_query: str):
    # sql_query = await gen_query(human_query)
    # table = await execute_query(sql_query)
    # analysis = await analyze(table)
    # return analysis 
    
    res = await text(human_query)   
    return res


# @cl.step(type="run")
@cl.step(type="llm")
# async def text(text: Text, history: Union[str, None] = []): #, history: Optional[str]= Query(None, description="test")): # :str | None = []):
async def text(human_query: str, history: Union[str, None] = []): #, history: Optional[str]= Query(None, description="test")): # :str | None = []):
    history = history or [];
    try:        
        print('\033[31m' + 'human:' + '\033[0m', end='')
        # prompt = text.text # input()
        prompt = human_query # input()
        
        if prompt != '':
            chat = m1.load_memory_variables({})['history']
            formatted_prompt = chat_prompt.format(info=table_info, history=chat, input=prompt)                  
            response = db_chain.run(formatted_prompt)
            m1.save_context({'input': prompt}, {'output': response})                   
      
            return response # It is of type: <class 'str'>

    except (ProgrammingError, NotSupportedError, OperationalError, ValueError, TypeError, Exception) as e:
        print("Exception occurred: {}".format(str(e)))
        logging.error("Exception occurred: {}".format(str(e)))
        raise HTTPException(status_code=422, detail=f"Error in /text endpoint: {str(e)}")
    
    
# FROM RESUME-CHAT
@cl.on_message
async def on_message(message: cl.Message):
   
    # # await chain(message.content)
    # await text(message.content)
    
     
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory

    runnable = cl.user_session.get("runnable")  # type: Runnable

    res = cl.Message(content="")
    
    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await res.stream_token(chunk)
    
    await chain(message.content)
    # await text(message.content)   
    await res.send()
        
    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(res.content)
    
 

# FROM BIGQUERY
# @cl.on_message
# async def main(message: cl.Message):
#     await chain(message.content)  
    
    

@cl.action_callback(name="take_action")
async def take_action(action: cl.Action):
    await cl.Message(content="Contacting supplier/buyer...").send()


# @cl.oauth_callback
# def auth_callback(provider_id: str, token: str, raw_user_data, default_app_user):
#     if provider_id == "google":
#         if "@chainlit.io" in raw_user_data["email"]:
#             return default_app_user
#     return None