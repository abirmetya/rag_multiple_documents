from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
import os
<<<<<<< HEAD
<<<<<<< HEAD
=======
# os.environ["OPENAI_API_KEY"] = "add your own key"
>>>>>>> d212e94 (remove openai keys)
=======
>>>>>>> 64331be (removed key)


def open_ai_embedding(deployment="text-embedding-3-small"):    
    # Use old version of Ada. You probably want V2 rather than this.
    embeddings = OpenAIEmbeddings(deployment=deployment)
    return embeddings

def llm_model():
    return ChatOpenAI()