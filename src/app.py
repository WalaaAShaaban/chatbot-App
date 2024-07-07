import sys
sys.path.append('../chatbot-App/')
import streamlit as st
from streamlit_chat import message as st_message
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain.chains import LLMMathChain
from model.ChainDB import ChainDB
from model.ChainPDF import ChainPDF
from model.Prompt import sys_msg
from langchain_google_genai import GoogleGenerativeAI




google_api_key = "AIzaSyBFbjwM2aC-P17T2vsRgpKbDJ8O4gYPSow"
llm = GoogleGenerativeAI(model="gemini-pro",google_api_key=google_api_key,temperature=0)

llm_math = LLMMathChain(llm=llm)
pdf_chain = ChainPDF().create_pdf_chain()
db_chain = ChainDB().create_sql_agent()

# Tools
llm_tool = Tool(
    name="General Knowledge",
    func=llm.invoke,
    description="Useful for answering general questions"
)

math_tool = Tool(
    name="Caculator",
    func=llm_math.run,
    description="Useful tool when you need to answer questions about math"
)

pdf_tool = Tool(
    name="pdf",
    func=pdf_chain.run,
    description="Useful tool when you need to answer questions about machine learning  or book or deep learning"
)

db_tool = Tool(
    name="Database",
    func=lambda data:  db_chain.invoke({'question':data}),
    description="Useful tool when you need to answer questions about database"
)

tools = [
    math_tool,
    llm_tool,
    pdf_tool, 
    db_tool
]



memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
)

conversational_agent = initialize_agent(
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=7,
    memory=memory,
    handle_parsing_errors=True
    
)

conversational_agent.agent.llm_chain.prompt= sys_msg


def get_response():
    user_message = st.session_state.chat_text

    st.session_state.history.append({"message": user_message, "is_user": True})
    response = conversational_agent.run(user_message)
    st.session_state.history.append({"message": response, "is_user": False})
    st.session_state.chat_text = ""

    
    
def main():
    
    st.header("Chatbot Application 📄 🛢️ 📚 ")
    
    if "history" not in st.session_state:
        st.session_state.history = []

    
    st.text_input("Enter your question ...",key="chat_text", on_change=get_response)
        
    for i, chat in enumerate(st.session_state.history):
        st_message(**chat, key=str(i))

if __name__ == "__main__":
    main()
