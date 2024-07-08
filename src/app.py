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
    description='''
"Introduction to Machine Learning with Python" by Andreas C. Müller and Sarah Guido is a comprehensive guide that introduces readers to the fundamental concepts and practical applications of machine learning using the Python programming language. The book covers a wide range of topics, including supervised and unsupervised learning, model evaluation, feature engineering, and more. Here is a detailed overview of the key aspects covered in the book:

Introduction to Machine Learning: The book starts by providing an overview of machine learning concepts, its applications, and the Python ecosystem for machine learning. Readers are introduced to the basics of machine learning and the essential tools and libraries available in Python for building machine learning models.

Supervised Learning: The authors delve into supervised learning techniques, including linear models, support vector machines, decision trees, and ensemble methods. Readers learn how to train and evaluate supervised learning models for tasks such as classification and regression.

Unsupervised Learning: The book explores unsupervised learning algorithms such as clustering, dimensionality reduction, and anomaly detection. Readers gain insights into how unsupervised learning can be used to discover patterns and structures in data without labeled outcomes.

Model Evaluation: The authors discuss the importance of evaluating machine learning models to assess their performance. Readers learn about various evaluation metrics such as accuracy, precision, recall, F1 score, and ROC curves, and how to interpret these metrics to make informed decisions about model selection.

Feature Engineering: The book emphasizes the significance of feature engineering in building effective machine learning models. Readers discover techniques for feature selection, extraction, and transformation to improve model performance and interpretability.

Text Data Processing: The book also covers working with text data, including techniques for tokenization, vectorization, and text classification using natural language processing (NLP) tools and libraries in Python.

Throughout the book, readers are provided with practical examples, code snippets, and hands-on exercises to reinforce their understanding of machine learning concepts and Python programming. The authors focus on building a solid foundation in machine learning principles while demonstrating how to apply these concepts to real-world datasets and problems.

By the end of the book, readers will have gained the knowledge and skills necessary to implement machine learning algorithms, evaluate model performance, and make data-driven decisions using Python and popular machine learning libraries like scikit-learn.
'''
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

    # st.session_state.history.append({"message": user_message, "is_user": True})
    response = conversational_agent.run(user_message)
    st.session_state.history.append({"message": response, "is_user": False})
    st.session_state.history.append({"message": user_message, "is_user": True})
    st.session_state.chat_text = ""

    
    
def main():
    
    st.header("Chatbot Application 📄 🛢️ 📚 ")
    
    if "history" not in st.session_state:
        st.session_state.history = []

    
    st.text_input("Enter your question ...",key="chat_text", on_change=get_response)
        
    for i, chat in enumerate(st.session_state.history[::-1]):
        st_message(**chat, key=str(i))

if __name__ == "__main__":
    main()
