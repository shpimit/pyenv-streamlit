# 설치

## 파일 설명

- chat_base.py는 챗팅 기본 파일일
- chat_upstage.py   upstage 연결 하는 파일
  - 내부적으로 llm_update.py는  fewshot이 빠진 llm 연결, llm_update_fewshot.py은 fewshot까지 연결하는 python파일
  - config.py는 llm_update_fewshot.py에  fewshot을 위한 파일일
- chat_pply.py는 perplexity 연결 하는 파일
  - llm_pplx_huggingembedding.py는 embedding 모델은 HuggingFaceEmbeddings을 사용함
  - llm_pplx_upstageembedding.py는 embedding 모델은 UpstageEmbedding을 사용함
- ChromDB
  - chroma는 4096 Dimension 4096인 embedding
  - chroma_768은 Dimension인 768인 embedding → HuggingFaceEmbedding에서 사용함
  - chroma_create_768.ipynb는 chromaDB 768 Dimension 데이터를 처음 저장할 때

```shell
$ pip install langchain langchain-core langchain-openai python-dotenv langchain-pinecone #강의
- 
$ pip install -qU python-dotenv langchain langchain-community langchain-text-splitters langchain-core langchain-upstage langchain-chroma # Upstage Chroma
$ pip install -qU python-dotenv langchain langchain-community langchain-text-splitters langchain-core langchain_openai langchain-chroma # OpenAI Chroma

# HuggingFaceEmbeddings 사용하기 위해서는 필요함
$ pip install -U sentence-transformers


```