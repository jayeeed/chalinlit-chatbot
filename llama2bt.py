import os
import chainlit as cl
import requests
import json


@cl.on_message
async def main(message: cl.Message):
    prompt_history = cl.user_session.get("prompt_history")
    prompt = f"{prompt_history}{message.content}"
    response = ""

    ui_msg = cl.Message(
        author="Llama 2",
        content="",
    )

    s = requests.Session()
    with s.post(
        "https://model-4w7mr0dw.api.baseten.co/production/predict",
        headers={"Authorization": "Api-Key {}".format(os.getenv("BASETEN_API_KEY"))},
        data=json.dumps({"prompt": prompt, "stream": True, "max_new_tokens": 4096}),
        stream=True,
    ) as resp:
        buffer = ""
        start_response = False
        for token in resp.iter_content(1):
            token = token.decode("utf-8")
            buffer += token
            if not start_response:
                if "[/INST]" in buffer:
                    start_response = True
            else:
                response += token
                await ui_msg.stream_token(token)

    await ui_msg.send()
    if not prompt_history:
        prompt_history = ""
    prompt_history += message.content + response
    cl.user_session.set("prompt_history", prompt_history)