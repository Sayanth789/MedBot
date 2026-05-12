from flask import Flask, render_template, jsonify, request
from src.helper import download_hugingface_embeddings
from langchain_pinecone import PineconeVectorStore 
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate 
from dotenv import load_dotenv 
from src.prompt import * 
import os 


load_dotenv()

app = Flask(__name__)


PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')


embeddings = download_hugingface_embeddings() 

index_name = 'medical-chatbot'

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

llm = OllamaLLM(
    model="llama3",
    temperature=0.8
)


PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k":2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route('/get', methods=['GET', 'POST'])
def chat():
    msg = request.form.get("msg")
    # using get() to avoid errors if 'msg' is missing
    result = qa.invoke({"query": msg})
    print("Response: ", result["result"])
    return str(result['result'])



if __name__ == '__main__':
    # Using 8080 as requested
<<<<<<< HEAD
    app.run(host="0.0.0.0", port=7860, debug=True)
=======
    app.run(host="0.0.0.0", port=8080, debug=True)
>>>>>>> 01fb722 (Add The Med bot (Streamlit + LangChain))
