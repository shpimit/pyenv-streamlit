from langchain_community.chat_models import ChatPerplexity
from langchain.chains import RetrievalQA
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma

import os
from dotenv import load_dotenv

load_dotenv()

pplx_api_key = os.getenv('PPLX_API_KEY')

def get_ai_message(user_question):
    # embedding =  UpstageEmbeddings(model='embedding-query')
    embedding = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs = {'device': 'cpu'}, # 모델이 CPU에서 실행되도록 설정. GPU를 사용할 수 있는 환경이라면 'cuda'로 설정할 수도 있음
        encode_kwargs = {'normalize_embeddings': True}, # 임베딩 정규화. 모든 벡터가 같은 범위의 값을 갖도록 함. 유사도 계산 시 일관성을 높여줌
    ) 

    # embedding = SentenceTransformer("sentence-transformers/static-similarity-mrl-multilingual-v1", device="cpu")

    # 이미 저장된 데이터를 사용할 때
    database = Chroma(collection_name='chroma-tax-768',  persist_directory="./chroma_768", embedding_function=embedding)

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

    dictionary = ["사람을 나타내는 표현 → 거주자"]

    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전: {dictionary}
        
        질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()

    tax_chain = {"query": dictionary_chain} | qa_chain    

    # 강의에서는 위처럼 진행하지만 업데이트된 LangChain 문법은 `.invoke()` 활용을 권장
    # ai_message = qa_chain.invoke({"query": user_question}) 

    ai_message = tax_chain.invoke({"question": user_question})

    return ai_message