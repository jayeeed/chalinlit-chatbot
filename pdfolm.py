import os
from typing import List

import chainlit as cl
from chainlit.types import AskFileResponse

from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

# from langchain import hub
# prompt = hub.pull("rlm/rag-prompt")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500, separators=["\n\n", "\n", " ", ""])

embeddings = OllamaEmbeddings(model="llama2:70b-chat")

welcome_message = """Welcome to the Chainlit PDF QA! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""

def process_file(file: AskFileResponse):
    loader = UnstructuredFileLoader(file.path)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    return docs


def get_docsearch(file: AskFileResponse):
    docs = process_file(file)

    # Save data in the user session
    cl.user_session.set("docs", docs)

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=f"./db",
    )

    return vectordb


@cl.on_chat_start
async def start():
    await cl.Avatar(
        name="Chatbot",
        url="https://avatars.githubusercontent.com/u/128686189?s=400&u=a1d1553023f8ea0921fba0debbe92a8c5f840dd9&v=4",
    ).send()
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,

            accept=["text/plain", "text/csv", "text/html", "application/zip", "application/pdf", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  "application/vnd.ms-powerpoint", "application/vnd.openxmlformats-officedocument.presentationml.presentation", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"],

            max_size_mb=100,
            timeout=1800,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
    await msg.send()

    vectordb = await cl.make_async(get_docsearch)(file)

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    system_msg = """You are a chatbot. You'll receive a prompt that includes retrieved content from the vectorDB based on the user's question, and the source.
    Your task is to respond to the user's new question using the information from the vectorDB without relying on your own knowledge.
    ----------------
    {summaries}"""

    messages = [
        SystemMessagePromptTemplate.from_template(
            system_msg),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    chain_type_kwargs = {"prompt": prompt}

    llm = ChatOllama(model="llama2:70b-chat", temperature=0)

    retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6},
)

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        memory=memory,
        return_source_documents=True, 
    )

    # Let the user know that the system is ready
    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]

    text_elements = []

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()