import os
from openai import AsyncOpenAI


from chainlit.playground.providers.openai import ChatOpenAI
import chainlit as cl


client = AsyncOpenAI()

template = """SQL tables (and columns):
* Customers(customer_id, signup_date)
* Streaming(customer_id, video_id, watch_date, watch_minutes)

A well-written SQL query that {input}:
```"""


settings = {
    "model": "gpt-3.5-turbo",
    "temperature": 0,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": ["```"],
}

@cl.step(type="llm", root=True, language="sql")
async def text_to_sql(text: str):
    # Create a ChatGeneration to enable the prompt playground
    generation = cl.ChatGeneration(
        provider=ChatOpenAI.id,
        messages=[
            {
                "role": "user",
                "content": template.format(input=text),
            }
        ],
        settings=settings,
        variables={"input": text},
    )

    # Call OpenAI
    stream = await client.chat.completions.create(
        messages = generation.messages, stream=True, **settings
    )

    current_step = cl.context.current_step

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await current_step.stream_token(token)

    # Convert messages to OpenAI format before passing to API
    openai_messages = generation.messages

    stream = await client.chat.completions.create(
        messages=openai_messages, stream=True, **settings
    )


    generation.completion = current_step.output

    current_step.generation = generation


@cl.on_message
async def main(message: cl.Message):
    await text_to_sql(message.content)

