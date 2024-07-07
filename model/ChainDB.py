from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from model.Prompt import answer_db_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

class ChainDB:
    
    def __init__(self):
        self.google_api_key = "AIzaSyBFbjwM2aC-P17T2vsRgpKbDJ8O4gYPSow"
        self.llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=self.google_api_key, temperature=0)
        self.connection_string = 'mysql+pymysql://root:root@127.0.0.1:3306/classicmodels'
        try:
            self.engine = create_engine(self.connection_string)
            # Test the connection
            with self.engine.connect() as connection:
                print("@" * 30)
                print("Connection to the database was successful")
            self.connection = self.engine
        except OperationalError as e:
            print("@" * 30)
            print(f"The error '{e}' occurred")
            self.connection = None



    def create_engine(self):
        if self.connection:
            db = SQLDatabase(self.connection)
            return db
        else:
            print("No database connection available.")
            return None
    
    def postprocess_query(self, query):
        try:
            return query.split('```sql')[1].split('```')[0]
        except:
            return query

    def create_sql_agent(self):
        db = self.create_engine()
        execute_query = QuerySQLDataBaseTool(db=db)
        write_query = create_sql_query_chain(self.llm,db)
        answer = answer_db_prompt | self.llm | StrOutputParser()
        
        chain = (
            
                RunnablePassthrough().assign(query=write_query | self.postprocess_query).assign(result=itemgetter("query") | execute_query) | answer
            )
        print("+" * 30)
        print(type(chain))
        return chain

