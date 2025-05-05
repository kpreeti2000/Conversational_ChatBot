# requirements: pip install langchain-google-genai google-generativeai langchain faiss-cpu pypdf
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# Set your Google API key
os.environ["GOOGLE_API_KEY"] = "Enter google api key here"

# Configure Gemini
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# 1. Load PDF document
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(pages)
    return docs

# 2. Create embeddings and build vector store using Gemini
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 3. Set up the Gemini model for chat
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0.3
)

def chat():
    print("Gemini RAG Chatbot (type 'exit' to quit)")
    
    # Define the PDF file path
    pdf_path = "C:/Users/HP/Desktop/chatgpt/ML.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return
    
    print("Loading PDF and creating knowledge base...")
    docs = load_pdf(pdf_path)
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Initialize conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Set up the conversational retrieval chain with memory
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    
    print("Knowledge base created. You can now ask questions about the PDF.")
    print("The chatbot will remember previous conversations.")
    
    while True:
        query = input("\nYou: ").strip()
        if query.lower() == "exit":
            break
            
        try:
            # Get response from the chain using invoke instead of __call__
            result = qa.invoke({"question": query})
            answer = result.get('answer', '')
            
            print("\nBot:", answer)
            
        except Exception as e:
            print("Error:", str(e))
            print("Please try again or type 'exit' to quit")

if __name__ == "__main__":
    chat()
