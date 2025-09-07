import tempfile
import os
import time
from typing import Dict
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict,Literal
from pydantic import BaseModel, Field
from langchain_cohere import CohereEmbeddings
from langchain_cerebras import ChatCerebras
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END, START
from langchain_community.vectorstores import FAISS
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.messages import AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load environment variables
load_dotenv()
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

class QuestionClassifierSchema(BaseModel):
    question: Literal["RAG", "Search_node"]

# Initializing cohere embeddings
cohere_embeddings = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=COHERE_API_KEY
)

# Making the Tavily search tool
tavily_search_tool = TavilySearch(max_results=10,tavily_api_key=TAVILY_API_KEY,search_depth="advanced")
all_tools=[tavily_search_tool]

# Initializing cohere LLM with tool call

llm = init_chat_model("command-a-03-2025", model_provider="cohere")
llm_for_classification = init_chat_model("command-a-03-2025", model_provider="cohere")
structured_llm_for_classification=llm_for_classification.with_structured_output(QuestionClassifierSchema)
llm_with_tools=llm.bind_tools(all_tools)

# Prompt template
rag_prompt = ChatPromptTemplate.from_template(
    """
You are a chatbot assistant , Answer the question based on the provided context and history, Answer in a concise manner.
<context>
{context}
</context>

<message_history>
{messages}
</message_history>

Question: {question}
Answer:
"""
)


search_prompt = ChatPromptTemplate.from_template(
    """
You are a chatbot assistant , use search node to ans the question and Use history for question clarification if needed, Answer in a concise manner.

Question: {question}
History: {history}
Answer:
"""
)


st.title("Mini Chat GPT")
with st.sidebar:
    # Fetching the thread ID from user input
    thread_id = st.text_input("Enter a thread ID (Change this to see the past interaction of user, format should be thread-x (x is a number))", value="thread-1")
    
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=False)
    
    # Making different user interfaces using threads
    if "THREAD_STORE" not in st.session_state:
        st.session_state["THREAD_STORE"] = {}
        
    # Langgraph configuration
    CONFIG = {'configurable': {'thread_id': thread_id}}
    
    # Fetching message_history for the given thread ID
    if thread_id not in st.session_state["THREAD_STORE"]:
        st.session_state["THREAD_STORE"][thread_id] = []

# Fetching the message history for the current thread
st.session_state["message_history"] = st.session_state["THREAD_STORE"][thread_id]
    
# Making state variable of chatbot
class ChatState(TypedDict):
    messages: str
    question: str
    answer: str
    context: str
    next: Literal["RAG", "Search_node", "Normal"]
    
    
# Loading the conversation history in streamlit UI
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

file_upload = False

retriever = None

if uploaded_files:
    all_docs = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_files.read())
        loader = PyPDFLoader(tmp.name)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = uploaded_files.name
        all_docs.extend(docs)
        os.unlink(tmp.name)

    # Making Vector Store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_splits = text_splitter.split_documents(all_docs)
    Vector_store = FAISS.from_documents(all_splits, cohere_embeddings)
    
    # Setting up the retriever based on selected document
    retriever = Vector_store.as_retriever(search_kwargs={"k": 20})
        

## Graph Node

def retrieve_context(state: ChatState):
    question = state["question"]
    if retriever:
        context_docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in context_docs])
    else:
        context = ""
    return {"context": context}

## Graph Node

def classify_question(state: ChatState):
    question = state["question"]
    history = state["messages"]
    classification_prompt = f"""
    Classify the question into one of the following:
    - "RAG" → if the answer should come from the uploaded documents.
    - "Search_node" → if the answer requires real-time information.
    - "Normal" → if it's casual conversation or general chit-chat.

    Past Conversation History: {history}
    
    Latest Question: {question}
    """
    response = structured_llm_for_classification.invoke(classification_prompt).question
    return {"next": response}

## Routing Function

def routing_node(state: ChatState) -> Literal["RAG", "Search_node", "Normal"]:
    return state["next"]
    

## Graph Node 

def Normal(state: ChatState):
    question = state["question"]
    history = state["messages"]
    result = llm.invoke(f"Conversation history:\n{history}\n\nLatest question:\n{question}")
    return {"answer": result.content}


## Graph Node

def RAG(state: ChatState):
    context = state.get("context", "")
    question = state["question"]
    messages = state["messages"]

    final_prompt = rag_prompt.format(
        context=context,
        question=question,
        messages=messages
    )
    result = llm.invoke(final_prompt)
    return {"answer": result.content}
    
## Graph Node

def Search_node(state: ChatState):
    question = state["question"]
    history = state["messages"]
    final_prompt = search_prompt.format(question=question,history=history)
    result = llm_with_tools.invoke(final_prompt)

    tool_calls = result.additional_kwargs.get("tool_calls", [])
    if tool_calls:
        if tool_calls[0]['function']['name'] == "tavily_search":
            tool_args = tool_calls[0]['function']['arguments']
            tool_result = tavily_search_tool.invoke(tool_args)
            tool_message = ToolMessage(
                content=tool_result,
                tool_call_id=tool_calls[0]['id'],
            )
            final_result = llm_with_tools.invoke([
                AIMessage(content=result.content, additional_kwargs={"tool_calls": tool_calls}),
                tool_message
            ])
            return {"answer": final_result.content}

    return {"answer": result.content}

# Making the memory of graph
checkpointer = InMemorySaver()

# Define the state graph
graph = StateGraph(ChatState)

# Making nodes to the graph
graph.add_node("classify_question", classify_question)
graph.add_node("retrieve_context", retrieve_context)
graph.add_node("RAG", RAG)
graph.add_node("Search_node", Search_node)
graph.add_node("Normal", Normal)

# Connecting nodes
graph.add_edge(START, "classify_question")
graph.add_conditional_edges("classify_question", routing_node, {
    "RAG": "retrieve_context",
    "Search_node": "Search_node",
    "Normal": "Normal"
})
graph.add_edge("retrieve_context", "RAG")
graph.add_edge("RAG", END)
graph.add_edge("Search_node", END)
graph.add_edge("Normal", END)

# Compiling the graph
rag_graph = graph.compile(checkpointer=checkpointer)

# RAG Q&A
if retriever:
    question = st.chat_input("Ask a question")
    
    if question:
        
        # first adding the message to message_history and thread store
        st.session_state['message_history'].append({'role': 'user', 'content': question})
        st.session_state["THREAD_STORE"][thread_id] = st.session_state['message_history']
    
        # displaying the user message
        with st.chat_message('user'):
            st.text(question)

        # slowing the typing speed
        def slow_stream(generator, delay=0.05):  # 50 ms delay between chunks
            for chunk ,metadata in generator:
                time.sleep(delay)
                yield chunk.content
                
        # invoking the graph if question is provided
        with st.chat_message('assistant'):
            ai_message=st.write_stream(
                slow_stream(
                    rag_graph.stream({"question": question,"messages":st.session_state['message_history']}, config=CONFIG, stream_mode='messages'),
                    delay=0.02)
                )
            # saving the message to message history and thread store
            st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
            st.session_state["THREAD_STORE"][thread_id] = st.session_state['message_history']