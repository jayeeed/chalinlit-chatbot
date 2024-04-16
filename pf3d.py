from flask import Flask, request, jsonify

# Import your existing code
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
import sys
import os

class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

app = Flask(__name__)

# Load the PDF and split it into chunks
loader = PyPDFLoader("./pf3d/Jayed Bin Jahangir_Machine Learning Engineer.pdf")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
all_splits = text_splitter.split_documents(data)
embedding = OllamaEmbeddings(model="mxbai-embed-large:latest")

with SuppressStdout():
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding, persist_directory="./pf3d")

# Define the Flask endpoint
@app.route("/query/", methods=["POST"])
def process_query():
    query = request.json.get("query")
    if not query or query.strip() == "":
        return jsonify({"error": "Empty query provided"}), 400

    # Prompt
    template = """
    You're a virtual assisant of Jayed.
    Answer questions about only Jayed.
    Answer questions on behalf of Jayed. 
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer as concise as possible. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    llm = Ollama(model="llama2:70b-chat", temperature=0, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    
    retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    result = qa_chain.invoke({"query": query})
    return jsonify({"answer": result["result"]})

# Example of how to run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
