import numpy as np

# %%
from langchain.vectorstores import FAISS


# %%
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# %%
import llm_models
import text_splitting

# %%
folder_path = 'D:/abir/ai_ml_projects/rag_multiple_documents/docs/'


embeddings = llm_models.open_ai_embedding()

# # %%
# if __name__ == '__main__':
#     embeddings = open_ai_embedding()
#     embeddings.embed_query('Hello world')

# %%
def get_vector_store(docs):
    faiss_embedding = FAISS.from_documents(
        docs,
        embeddings
    )
    faiss_embedding.save_local("faiss_index")
    return


prompt_template = """
Human : Use the following pieces of context to provide a concise answer,
to the question at the end. However, please use atlease 50 words with detailed 
explanation. If you don't know the answer, just say that you don't know and don't
try  to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context","question"]
)

# %%
def get_response(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vectorstore_faiss.as_retriever(
            search_type='similarity',search_kwargs={"k":6}
        ),
        return_source_documents=True,
        chain_type_kwargs={'prompt':prompt}
    )
    answer = qa({'query':query})
    return answer

# # %%
# faiss_index = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
# # llm = llm_model_openai()
# llm = llm_model()
# user_question = "What is climate change?"

# llm_response = get_response(llm,faiss_index,user_question)

# %%
def return_similarity_score(faiss_index,query,k=6):
    similarity_score = faiss_index.similarity_search_with_score(query,k=k)
    score = []
    for score_ in similarity_score:
        score.append(score_[1])
    
    return score

# %%
def process_llm_response(llm_response,faiss_index,user_question,k=6):
    print(llm_response['result'])
    print('\n\nSources:')
    score = return_similarity_score(faiss_index,user_question,k)
    i = 0
    doc_ref = ''
    for source in llm_response['source_documents']:
        doc_no = source.metadata['source'].split('\\')[-1]
        doc_ref+=f"{doc_no}     page number is          {source.metadata['page']}       and score is     {np.round(score[i],2)} \n\n"
        i+=1
    return str(llm_response['result'])+'\n\nSources:\n\n'+doc_ref

    

import streamlit as st

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with multiple PDFs using OpenAI")

    user_question = st.text_input("Ask a Question from the PDF Files, Like- What is climate change?")

    with st.sidebar:
        st.title("Menu:")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = text_splitting.data_ingestion("D:/abir/ai_ml_projects/rag_multiple_documents/docs/")
                get_vector_store(docs)
                st.success('done')
        
    if st.button("LLM output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
            llm = llm_models.llm_model()
            llm_response = get_response(llm,faiss_index,user_question)
            st.write(process_llm_response(llm_response,faiss_index,user_question))
            st.success("Done")


if __name__ == "__main__":
    main()
