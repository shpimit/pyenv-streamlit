from langchain_community.chat_models import ChatPerplexity
from langchain.chains import RetrievalQA
from langchain import hub
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma

import os
from dotenv import load_dotenv

load_dotenv()

pplx_api_key = os.getenv('PPLX_API_KEY')

def get_ai_message(user_question):
    embedding =  UpstageEmbeddings(model='embedding-query')
    # 이미 저장된 데이터를 사용할 때
    database = Chroma(collection_name='chroma-tax',  persist_directory="./chroma", embedding_function=embedding)

    llm = ChatPerplexity(
        temperature=0, pplx_api_key=pplx_api_key, model="sonar-pro"
    )

    prompt = hub.pull("rlm/rag-prompt")

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        # retriever=database.as_retriever(),
        retriever=database.as_retriever(search_kwargs={'k': 4}),
        chain_type_kwargs={"prompt": prompt}
    )

    # 강의에서는 위처럼 진행하지만 업데이트된 LangChain 문법은 `.invoke()` 활용을 권장
    ai_message = qa_chain.invoke({"query": user_question}) 

    return ai_message