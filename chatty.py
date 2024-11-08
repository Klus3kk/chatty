import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables, including OpenAI API key
load_dotenv()

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logging.info("Starting the Chatty - Spotify AI Assistant...")

# Load and process the PDF document for retrieving Spotify-related information
loader = PyPDFLoader("data/Spotify.pdf")
try:
    docs = loader.load()
    logging.info("Document loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load document: {e}")

# Define a function to split documents into smaller chunks for better handling
def split_documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return text_splitter.split_documents(documents)

# Split the loaded documents
splits = split_documents(docs)

# Create a vector store to hold document embeddings for quick retrieval
vector_store = InMemoryVectorStore(OpenAIEmbeddings())
vector_store.add_documents(documents=splits)

# Create a retriever that can search for relevant document segments
retriever = vector_store.as_retriever()

# Define system prompt template with a friendly, conversational tone
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise. "
    "Be friendly and cool, like a music-loving friend. "
    "You're thrilled about music and make jokes about big corporations sometimes."
    "\n\n"
    "{context}"
)

# Set up the language model and prompt template for responses
llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Create a retrieval-augmented generation (RAG) chain for responses
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Function to process and answer user queries
def get_answer(query):
    try:
        response = rag_chain.invoke({"input": query})
        return response["answer"]
    except Exception as e:
        logging.error(f"Error during query processing: {e}")
        return "Sorry, something went wrong. Please try again."

# Main program loop for user interaction
if __name__ == "__main__":
    print("*********************")
    print("Chatty - Your Spotify AI Assistant")
    print("*********************")
    print("Type 'exit' to end the application.")
    print("Type 'Example' to see example questions.")
    print("*********************\n\n")
    
    while True:
        user_query = input("User: ")
        if user_query.lower() == "exit":
            print("Bye bye for now!")
            break
        elif user_query == "Example":
            # Provide example questions for the user
            questions = [
                "Tell me shortly about Spotify's growth",
                "Give me some interesting news about Spotify",
                "What's the share of music industry revenues worldwide in 2014"
            ]
            for question in questions:
                print(f"User: {question}")
                answer = get_answer(question)
                print(f"Chatty: {answer}\n")
        else:
            # Answer custom user queries
            answer = get_answer(user_query)
            print(f"Chatty: {answer}\n")
