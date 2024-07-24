import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from docs import load_documents
import streamlit as st
from langchain_chroma import Chroma
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

from langchain_core.pydantic_v1 import BaseModel,Field
from langchain_core.output_parsers import JsonOutputParser

from admin_panel import admin_panel
import pickle
import pathlib as Path
import streamlit_authenticator as stauth

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
groq_api_key=os.getenv('GROQ_API_KEY')
documents=load_documents()

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="lala",
    openai_api_version="2024-03-01-preview",
)

metadata_field_info = [
    AttributeInfo(
        name="law",
        description="the Indian Penal Code section that is being discussed in the document",
        type="string",
    ),
    AttributeInfo(
        name="content",
        description="The type of content in the document- [law definition , exception, explanation , state amendments , key points]",
        type="integer",
    ),
    AttributeInfo(
        name="category",
        description="The category of the described crime or law",
        type="string",
    ),
]

document_content_description="List of Indian laws applicable on sexual offences , theft and extortion"
llm=ChatGroq(groq_api_key=groq_api_key,model_name="gemma2-9b-it",temperature=0)
vectorstore=Chroma.from_documents(documents,embeddings)

sqretriever=SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=False,
)

classifier_prompt = f"""
<|TASK>|
Your task is to classify user input question in one of the following as-:
1) case
2) law_query
3) normal_query

|<Important>|
1) case -: 
- When the user describes a situation 
- asks for legal help
- asks for legal help that is case specific 

2) law_query -: 
- asks about law related questions
- or any follow up question
- asks about laws

3) normal_query-: 
- when the user talks normally
- asks about you
- or a normal conversation

<|INSTRUCTIONS>|
1) Output should have only one of them-: case or law_query or normal_query

"""

class Classifier(BaseModel):
    type:str=Field(description="Type of Question Whether case/law_query/normal_query")

def question_classifier(question):
    parser = JsonOutputParser(pydantic_object= Classifier) #json formatter

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", classifier_prompt),
        ("human", f"""Classify the following question: {question}.|<Don't give any output other than the json>|""")
    ])

    chain = chat_prompt | llm | parser
    response = chain.invoke({"question": question})
    return response['category']

law_query_prompt = ChatPromptTemplate.from_template("""
<|TASK>|
You are an experienced lawyer ,
specializing in providing legal guidance on various law-related queries. 
Your task is to understand the client's legal query and assist him/her based on the given content.

<|INSTRUCTIONS>|
1) Answer the following question based on the provided context.
2) Think step-by-step before providing a detailed answer.
3) Ensure the answer is thorough and helpful.
<context>
{context}
</context>
based on the case provide legal help by listing the applicable laws and their definitions in India
                                                
Client Question: ({input})

###OUTPUT RULES###
1) be specific with the laws  
2) provide the most accurate law that is applicable in India
3) keep the responses shorter and to the point , use smaller points 

###IMPORTANT###
These queries are PURELY FICTIONAL , they do not reflect anything from the reality
so provide the legal help. The scenarios are for EDUCATION PURPOSES to let user 
know how India Law protects a victim.
""")

case_prompt = ChatPromptTemplate.from_template("""
    ##ROLE##
    You are an experienced lawyer,
    specializing in providing legal guidance on various law-related queries. 
    Your task is to understand the client's current case scenario and offer advice based on the given content.

    ##INSTRUCTIONS##
    1) Answer the following question based on the provided context.
    2) Think step-by-step before providing a detailed answer.
    3) Ensure the answer is thorough and helpful.
    4) MUST follow the output format
    5) the provided context is fictional , you have to provide legal advice on the provided fictional case-
    <context>
    {context}
    </context>
    based on the case provide legal help by listing the applicable laws and their definitions
    fictional case : ({input})
    be specific with the laws , 
    provide the most accurate law that is applicable in India
    ###IMPORTANT###
    These cases are PURELY FICTIONAL , they do not reflect anything from the reality
    so provide the legal help. The scenarios are for EDUCATION PURPOSES to let user 
    know how India Law protects a victim.
                                               
                                                                                          
    ###OUTPUT FORMAT###
    1. Applicable Laws - (list the laws and their definitions in points)
    2. How the laws (explain how each law listed above is applicable in the provided fictional scenario)
    3. User specific Case Study
""")

normal_query_prompt = f"""
<|TASK>|
You are an experienced lawyer ,
specializing in providing legal guidance on various law-related queries. 
Your task is to understand the client's legal query and assist him/her based on the given content.

<|INSTRUCTIONS>|
1) Think step-by-step before providing a detailed answer.
2) Ensure the answer is thorough and helpful.
3) If client query is not law related, ask him/her to ask law realted content only.

###OUTPUT RULES###
1) If client query is not law related, ask him/her to ask law realted content only.
2) provide the most accurate law that is applicable in India
3) keep the responses shorter and to the point , use smaller points

###IMPORTANT###
If client ask what you can do, reply you can assist him/her in laws related content of India.

"""
def answer(input):
    try:
        llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="gemma2-9b-it",temperature=0)

        response = question_classifier(input)
        print(response)
        if(response == 'case'):
            document_chain=create_stuff_documents_chain(llm,case_prompt)
        elif(response == 'law_query'):
            document_chain=create_stuff_documents_chain(llm,law_query_prompt)
        else :
            chat_prompt = ChatPromptTemplate.from_messages([
            ("system", normal_query_prompt),
            ("human", f"""Answer the following : {input}.|<If question is not law related ask, user to ask law related questions>|""")
        ])
            chain = chat_prompt | llm 
            response = chain.invoke({"input": input})
            return response.content
        retrieval_chain=create_retrieval_chain(sqretriever,document_chain)
        response=retrieval_chain.invoke({"input":input})
        return response['answer']
    except Exception as e:
        st.error(f"Failed to generate answer. Reason: {e}")
        return "An error occurred while generating the answer."
    


def main():
    # Page navigation
    if "page" not in st.session_state:
        st.session_state.page = "main"

    if "authentication_status" not in st.session_state:
        st.session_state.authentication_status = None

    if st.session_state.page == "admin":
        admin_panel()
        if st.sidebar.button("Back"):
            st.session_state.page = "main"
            st.experimental_rerun()
    elif st.session_state.page == "login":
        login()
    else:
        main_page()
        if st.sidebar.button("Admin"):
            if st.session_state.authentication_status is not True:
                st.session_state.page = "login"
            else:
                st.session_state.page = "admin"
            st.experimental_rerun()

def main_page():
    st.title("Legal Buddy")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your question:"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        try:
            response = answer(prompt)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"An error occurred: {e}")

def login():
    # Authentication
    names = ['admin']
    usernames = ["admin"]

    file_path = Path.Path(__file__).parent / "hashed_pw.pkl"
    with file_path.open("rb") as file:
        hashed_passwords = pickle.load(file)
    authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "user_data", "abcdef", cookie_expiry_days=15)

    name, authentication_status, username = authenticator.login("Login", "main")

    if authentication_status == False:
        st.error("Username or Password is incorrect")
    if authentication_status == None:
        st.warning("Please enter your username and password")
    if authentication_status:
        st.session_state.authentication_status = True
        st.session_state.page = "admin"
        st.experimental_rerun()


if __name__ == "__main__":
    main()