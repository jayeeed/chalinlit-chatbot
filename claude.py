import asyncio
from anthropic import AsyncAnthropic

client = AsyncAnthropic(api_key="ANTHROPIC_API_KEY")

async def main() -> None:
    Prompt = "Hello, how are you?"
    completion = client.messages.stream(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": Prompt,
            }
        ],
        model="claude-3-opus-20240229",
    )
    async with completion as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)
