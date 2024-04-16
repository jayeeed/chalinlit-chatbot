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

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500, separators=["\n\n", "\n", " ", ""])

embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

# welcome_message = """Welcome to the Chainlit PDF QA! To get started:
# 1. Upload a PDF or text file
# 2. Ask a question about the file
# """

# def process_file(file: AskFileResponse):
#     loader = UnstructuredFileLoader(file.path)
#     documents = loader.load()
#     docs = text_splitter.split_documents(documents)
#     for i, doc in enumerate(docs):
#         doc.metadata["source"] = f"source_{i}"
#     return docs


# def get_docsearch(file: AskFileResponse):
#     docs = process_file(file)

#     # Save data in the user session
#     cl.user_session.set("docs", docs)

#     vectordb = Chroma.from_documents(
#         documents=docs,
#         embedding=embeddings,
#         persist_directory=f"./bllm",
#     )

#     return vectordb


@cl.on_chat_start
async def start():
    vectordb = Chroma(
        persist_directory="./bllm",
        embedding_function=embeddings
        )

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    system_msg = """
    You are a helpful chatbot. Use vectorDB knowlegdebase only for understanding 'Bengali' grammar. 
    Use your existing knowledgebase for answering questions. And translate your responses with the 'Bengali' 
    grammartical knowledge from your database. I will can ask question in 'Bengali' or 'English' and you will response only in 'Bengali'.
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
        search_kwargs={"k": 3},
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
    msg = cl.Message(content="ChromaDB loaded. You can now ask questions!")
    await msg.send()

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