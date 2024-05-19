import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader


from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

#side bar content
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown("""
    ## about 
    this is an llm-powered chatbot built using:
    -[streamlit]............
    -[langchain].................
    -[openai]...................llm model
                """)
    add_vertical_space(5)
    st.write('Made with ❤️ by [Sanjana Suthar].............')

def main():
    st.header("Chat with PDF 💬") 
    load_dotenv()
    
    # Upload a PDF file
    
    pdf = st.file_uploader("Upload your PDF", type='pdf') 
    print("=====================================================================")
    print(pdf)
    print("=====================================================================")
    # st.write(pdf.name)

    # st.write(pdf)
    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        st.write(pdf_reader)
    
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )  
        chunks=text_splitter.split_text(text=text)
        # st.write(chunks)
        # st.write(text) 
         
        # embeddings=OpenAIEmbeddings()
        # VectorStore=FAISS.from_texts(chunks,embedding=embeddings) 
        
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings loaded from the Disk')
        else:  
            embeddings=OpenAIEmbeddings()
            VectorStore=FAISS.from_texts(chunks,embedding=embeddings)        
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(VectorStore,f)

            # st.write('Embeddings Computaion Completed')    
        # accept user query questions
        query = st.text_input("ask questions about your PDF files:")
        st.write(query)

        
        if query:
            docs = VectorStore.similarity_search(query=query,k=3)

            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm,chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question=query)
                print(cb)
            # response = chain.run(input_doxuments=docs,question=query)
            st.write(response)
            # st.write(docs)
if __name__ == '__main__':
    main()
  