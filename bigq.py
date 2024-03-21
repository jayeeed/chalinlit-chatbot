from datetime import date
import chainlit as cl
from chainlit.playground.providers.openai import ChatOpenAI

from google.cloud import bigquery
from openai import AsyncOpenAI

client = bigquery.Client()
openai_client = AsyncOpenAI()


settings = {
    "model": "gpt-3.5-turbo-0125",
    "temperature": 0,
    "max_tokens": 4096,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": ["```"],
}


sql_query_prompt = """You have a BigQuery table named `orders` in the dataset `order`.
The table contains information about orders, including `order_id`, `order_date`, `estimated_delivery_date`, `status`, `tracking_id`, `customer_id`, `supplier_id`, `customer_contact` and `supplier_contact`.

Write a SQL query to retrieve the full order based on the given question:

{input}
```"""

explain_query_result_prompt = """Today is {date}
You received a query from a customer support operator regarding the orders table.
They executed a SQL query and provided the results in Markdown table format.
Analyze the table and explain the problem to the operator.

Markdown Table:

```
{table}
```

Concise analysis in 3 short bullet points just highlighting the most important points:
"""


@cl.step(type="llm")
async def gen_query(human_query: str):
    current_step = cl.context.current_step
    current_step.generation = cl.ChatGeneration(
        provider=ChatOpenAI.id,
        messages=[
            {
                "role": "user",
                "content": sql_query_prompt.format(input=human_query),
            }
        ],
        settings=settings,
        inputs={"input": human_query},
    )

    # Call OpenAI and stream the message
    stream_resp = await openai_client.chat.completions.create(
        messages = current_step.generation.messages,
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
    query_job = client.query(query)

    # Wait for the query to complete
    query_job.result()

    # Get the query results
    results = query_job.to_dataframe()
    markdown_table = results.to_markdown(index=False)

    return markdown_table


@cl.step(type="llm")
async def analyze(table):
    current_step = cl.context.current_step
    today = str(date.today())
    current_step.generation = cl.ChatGeneration(
        provider=ChatOpenAI.id,
        messages=[
            {
                "role": "user",
                "content": explain_query_result_prompt.format(date=today, table=table),
            }
        ],
        settings=settings,
        inputs={"date": today, "table": table},
    )

    final_answer = cl.Message(content="")
    await final_answer.send()

    # Call OpenAI and stream the message
    stream = await openai_client.chat.completions.create(
        messages = current_step.generation.messages,
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
    sql_query = await gen_query(human_query)
    table = await execute_query(sql_query)
    analysis = await analyze(table)
    return analysis


@cl.on_message
async def main(message: cl.Message):
    await chain(message.content)


@cl.action_callback(name="take_action")
async def take_action(action: cl.Action):
    await cl.Message(content="Contacting supplier...").send()


@cl.oauth_callback
def auth_callback(provider_id: str, token: str, raw_user_data, default_app_user):
    if provider_id == "google":
        if "@gmail.com" in raw_user_data["email"]:
            return default_app_user
    return None
