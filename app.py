import os
from atlassian import Confluence
import streamlit as st
from dotenv import dotenv_values
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from pre_process import get_page_ids, extract_page_history, extract_page_content

conf_connection = Confluence(
    url="https://chatmodel.atlassian.net/wiki/",
    username=dotenv_values(".env")["conf_username"],
    password=dotenv_values(".env")["conf_password"],
    api_version="cloud",

)

# Main Streamlit app
def main():
    st.title("Confluence: Productivity-Boosting Chat bot")
    with st.sidebar:
        st.title('🤗💬 Confluence Chat Assistant')
        st.markdown('''
        ## About
        The Confluence Chat Bot is a powerful and user-friendly assistant designed to enhance 
        productivity and collaboration within your Confluence workspace. It's a smart automation
        tool that streamlines tasks, such as retrieving information and managing content, 
        all through natural language interactions. Whether it's quick information 
        retrieval, the Confluence Chat Bot is a valuable addition 
        to your workspace for a seamless and productive Confluence experience.
        ''')
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

        if 'clicked' not in st.session_state:
            st.session_state.clicked = False

        def click_button():
            st.session_state.clicked = True

        st.button('Submit', on_click=click_button)

        if st.session_state.clicked:
            # The message and nested widget will remain on the page
            st.write('Key entered !')
    
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    if st.session_state.clicked:
        with st.spinner("Processing"):
            page_id=get_page_ids(confluence_api=conf_connection)
            page_history=extract_page_history(pages=page_id, confluence_api=conf_connection)
            text = ""
            for pages in page_id:
                attachments, content = extract_page_content(pages, confluence_api=conf_connection)
                text += content

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                length_function=len
            )

            # Process the PDF text and create the documents list
            documents = text_splitter.split_text(text=text)

        # # embeddings
        store_name = "bloreparking"

        if os.path.exists(f"{store_name}.pkl"):
            with st.spinner("Loading from cache"):
                with open(f"{store_name}.pkl", "rb") as f:
                    vectorstore = pickle.load(f)
        else:
            with st.spinner("Creating Embeddings"):
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                vectorstore = FAISS.from_texts(documents, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(vectorstore, f)

        st.session_state.processed_data = {
            "document_chunks": documents,
            "vectorstore": vectorstore,
        }

        
        # Load the Langchain chatbot
        with st.spinner("Fetching you the response"):
            llm = ChatOpenAI(temperature=0.2, max_tokens=1000, model_name="gpt-3.5-turbo",
                            openai_api_key=openai_api_key)
            qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

        # Initialize Streamlit chat UI
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask your questions to confluence bot?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Fetching you the response"):
                result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})
                print(prompt)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = result["answer"]
                    message_placeholder.markdown(full_response + "|")
                message_placeholder.markdown(full_response)
                print(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()