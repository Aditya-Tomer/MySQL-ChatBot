from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import streamlit as st
import os

# Initialize Database
def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

# SQL Chain
def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.

    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}

    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

    Your turn:

    Question: {question}
    SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

# Response Generation
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

# App Setup
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_balloon:", layout="wide")

# Session Initialization
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
    st.session_state.active_session = None
if "db" not in st.session_state:
    st.session_state.db = None

# Sidebar UI
with st.sidebar:
    st.title("Settings")

    with st.container():
        with st.expander("Database Connection", expanded=False):
            st.text_input("Host", value="localhost", key="Host")
            st.text_input("Port", value="3306", key="Port")
            st.text_input("User", value="root", key="User")
            st.text_input("Password", type="password", value="872003", key="Password")
            st.text_input("Database", value="imdb", key="Database")

            if st.button("Connect"):
                with st.spinner("Connecting to database..."):
                    try:
                        st.session_state.db = init_database(
                            st.session_state["User"],
                            st.session_state["Password"],
                            st.session_state["Host"],
                            st.session_state["Port"],
                            st.session_state["Database"]
                        )
                        st.success("Connected to database!")
                    except Exception as e:
                        st.error("Failed to connect to the database. Please check your credentials or configuration.")
                        print("DB Connection Error:", e)

    st.markdown("---")

    if st.button("+ New Chat"):
        st.session_state.active_session = None

    st.subheader("Conversations")
    session_keys = list(st.session_state.chat_sessions.keys())[::-1]
    for session_name in session_keys:
        if st.button(session_name, key=session_name):
            st.session_state.active_session = session_name

# Greeting Detection
GREETINGS = ["hi", "hello", "hey", "who are you", "what can you do", "help"]
def is_greeting(msg):
    return msg.lower().strip() in GREETINGS

# Main Chat Logic
if st.session_state.active_session is None:
    session_name = "New Chat"
    st.session_state.active_session = session_name
    st.session_state.chat_sessions[session_name] = [
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database.")
    ]

current_chat = st.session_state.chat_sessions[st.session_state.active_session]

for message in current_chat:
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query and user_query.strip():
    # Rename session if it's still "New Chat"
    if st.session_state.active_session == "New Chat" and len(current_chat) == 1:
        new_name = user_query[:20].strip()
        if new_name not in st.session_state.chat_sessions:
            st.session_state.chat_sessions[new_name] = current_chat
            del st.session_state.chat_sessions["New Chat"]
            st.session_state.active_session = new_name
            current_chat = st.session_state.chat_sessions[new_name]

    current_chat.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        try:
            if is_greeting(user_query):
                response = "Hi there! I'm here to help you with SQL queries. Ask me anything about your database."
            elif st.session_state.db:
                response = get_response(user_query, st.session_state.db, current_chat)
            else:
                response = "Please connect to a database first."
        except Exception as e:
            response = "Oops! Something went wrong. Please try again or check your input."
            print("Chat Error:", e)
        st.markdown(response)

    current_chat.append(AIMessage(content=response))
