# main_chroma.py
import os
import time
import numpy as np
from typing import List, Dict, Set
import re
import hashlib
import shutil
import glob
import pickle

# LangChain 및 관련 라이브러리 임포트
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
# FAISS 대신 Chroma를 임포트합니다.
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.retrievers import MultiQueryRetriever
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# 기타 유틸리티 라이브러리
import tiktoken
from sentence_transformers.cross_encoder import CrossEncoder
# Komoran 형태소 분석기 import (설치 필요: pip install konlpy)
from konlpy.tag import Komoran

# --- 1. 초기 설정 및 모델 로드 ---

# 임베딩 모델
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'}, # GPU 사용 시 'cuda'로 변경
    encode_kwargs={'normalize_embeddings': True}
)
# 재순위화 모델
reranker_model = CrossEncoder('BAAI/bge-reranker-large', device='cpu') # GPU 사용 시 'cuda'로 변경

# LLM 정의 (Ollama 사용)
llm_cheap = Ollama(model="llama3:8b")
llm_powerful = Ollama(model="qwen2:7b")

# --- ★★★ 변경점: 상수 정의 (Chroma DB 경로로 변경) ★★★ ---
CHROMA_DB_PATH = "chroma_db_bge_m3" # Chroma DB 저장 디렉토리
LARGE_CHUNK_PICKLE = f"{CHROMA_DB_PATH}_large_chunks.pkl" # large chunk pickle 파일 경로
HASH_LIST_FILE = f"{CHROMA_DB_PATH}_hashes.txt"  # 여러 파일의 해시값을 관리하는 파일
L_MAX = 4096
CHUNK_OVERLAP = 256

import jpype

jvm_path = r"C:\Program Files\Eclipse Adoptium\jdk-11.0.27.6-hotspot\bin\client\jvm.dll"
jar_path = r"C:\Users\jh100\pythonfolder\venv\Lib\site-packages\konlpy\java\*"

if not jpype.isJVMStarted():
    jpype.startJVM(jvm_path, f"-Djava.class.path={jar_path}", "-Xmx4g")

def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def extract_keywords_from_text(text: str, max_keywords: int = 10) -> List[str]:
    words = re.findall(r'[가-힣a-zA-Z0-9]+', text.lower())
    word_freq = {}
    for word in words:
        if len(word) > 1:
            word_freq[word] = word_freq.get(word, 0) + 1
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_keywords]]

# 고유명사(NNP)만 추출하는 함수
def extract_proper_nouns(text: str, max_keywords: int = 5) -> list[str]:
    komoran = Komoran()
    tagged = komoran.pos(text)
    # NNP: 고유명사
    proper_nouns = [word for word, pos in tagged if pos == 'NNP']
    # java.lang.String → Python str로 변환
    proper_nouns = [str(word) for word in proper_nouns]
    # 중복 제거 및 상위 max_keywords개만 반환
    return list(dict.fromkeys(proper_nouns))[:max_keywords]

# --- 3. 온라인 파이프라인 함수들 (검색, 답변 생성 등)만 남김 ---
def rerank_documents(query: str, retrieved_docs: List[Document], reranker_model, rerank_n: int = 2) -> List[Document]:
    if not retrieved_docs:
        return []
    pairs = [(query, doc.page_content) for doc in retrieved_docs]
    scores = reranker_model.predict(pairs)
    doc_score_pairs = list(zip(retrieved_docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in doc_score_pairs[:rerank_n]]

def compress_document_context(query: str, context: str, llm_cheap) -> str:
    """
    LLM을 이용해 large chunk 내에서 질문에 답하기 위해 필요한 핵심 문장만 그대로 추출합니다.
    요약이 아니라, 관련된 문장만 원문 그대로 뽑아냅니다.
    """
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""
        You are a helpful AI assistant that ALWAYS responds in Korean. 당신은 항상 한국어로 답변하는 AI 어시스턴트입니다.
        관련된 문장을 추출할 때, 모호한 경우에는 키워드 중심으로 컨텍스트 압축을 진행하고, 키워드 자체도 없다면 [관련 내용 없음]으로 출력해주세요.
        아래 문맥(Context)에서 다음 질문(Question)에 답하기 위해 반드시 필요한 핵심 문장들만 원문 그대로 추출하세요. 요약하지 말고, 관련된 문장만 그대로 나열하세요.
        
        Context: {context}
        
        Question: {question}
        
        Extracted Sentences:
        """
    )
    chain = LLMChain(llm=llm_cheap, prompt=prompt)
    result = chain.invoke({"context": context, "question": query})
    return result['text']

def generate_final_answer(context: str, query: str) -> str:
    # 이 함수는 예시 문서에 의존하지 않도록 일반화되었습니다.
    # 실제 사용 시에는 문서의 특성에 맞게 키워드 추출 등을 조정할 수 있습니다.
    if not context.strip():
        # 문서 기반 답변이 불가능한 경우
        prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            무조건 답변은 "한국어"로 해주세요.
           [관련 내용이 없습니다. 좀 더 구체적으로 질문해주세요.]라고 출력만 해주세요.

[사용자 질문]
{question}

[답변]"""
        )
        chain = LLMChain(llm=llm_cheap, prompt=prompt)
        result = chain.invoke({"question": query})
        return result['text']
    else:
        # 문서 기반 답변이 가능한 경우
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=""" 
            당신은 문서 기반 AI 어시스턴트입니다. 반드시 "한국어"로만 답변하세요.
            
            아래 제공된 문서 내용(Context)에서 사용자 질문(Question)에 대해 답변하세요.
            - 반드시 답변 시작에 "[문서 기반 답변입니다.]"라는 문구를 포함하세요.
            - 답변 마지막에 [파일명, 페이지번호] 형식으로 출처를 명시하세요.
            - 만약 context가 비어 있거나, [관련 내용 없음]이라면, "[문서 기반 답변입니다, 관련 내용이 없습니다.]"라고 출력해주세요.

[문서 내용]
{context}

[사용자 질문]
{question}

[답변]"""
        )
        chain = LLMChain(llm=llm_powerful, prompt=prompt) # 답변 생성은 고성능 모델 사용
        result = chain.invoke({"context": context, "question": query})
        return result['text']

def print_analysis_info(query: str, retrieved_docs: List[Document], context: str, document_path: str):
    print("\n" + "-"*50)
    print("-- 분석 정보 --")
    print("="*50)
    print(f"입력 문서: {document_path}")
    print(f"검색된 문서 수: {len(retrieved_docs)}")
    if retrieved_docs:
        print(f"--- 첫 번째 문서 미리보기: {retrieved_docs[0].page_content[:200]}...---")
    print(f"--- 컨텍스트 길이: {len(context)} 문자")
    if context:
        print(f"--- 컨텍스트 미리보기: {context[:300]}...")
    else:
        print("--- 컨텍스트가 없습니다. ---")
    print("="*50)

def hybrid_rag_pipeline(
    query: str,
    vectorstore,  # Chroma (small chunk)
    large_chunk_dict: Dict[str, Document],
    reranker_model,
    llm_cheap,
    llm_powerful,
    top_k: int = 10,
    rerank_n: int = 3
) -> str:
    print("[1/5] 1차 검색: small chunk 대상으로 벡터 검색 중...")
    retrieved_small_docs = vectorstore.similarity_search(query, k=top_k)
    print(f"[1/5] 1차 검색 완료 (검색된 small chunk 수: {len(retrieved_small_docs)})")

    keywords = extract_proper_nouns(query, max_keywords=5)
    print(f"[고유명사 추출] 쿼리에서 추출된 고유명사: {keywords}")

    # 키워드 포함 개수에 따라 점수 부여
    keyword_weight = 0.5  # 키워드 1개당 가중치
    scored_docs = []
    for doc in retrieved_small_docs:
        num_keywords = sum(1 for kw in keywords if kw in doc.page_content)
        score = 1.0 + (num_keywords * keyword_weight)
        scored_docs.append((doc, score))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    rerank_candidates = [doc for doc, _ in scored_docs[:top_k]]
    print(f"[키워드 점수 기반 정렬] 상위 {len(rerank_candidates)}개 문서 선택")

    if not rerank_candidates:
        print("[예외 처리] 키워드 관련도가 높은 문서가 없습니다. 관련 내용 없음 반환.")
        return generate_final_answer("", query)

    print("[2/5] 재순위화(CrossEncoder) 진행 중...")
    reranked_small_docs = rerank_documents(query, rerank_candidates, reranker_model, rerank_n)
    print(f"[2/5] 재순위화 완료 (선택된 small chunk 수: {len(reranked_small_docs)})")

    print("[3/5] parent large chunk 추출 중...")
    parent_large_ids = {doc.metadata["parent_large_id"] for doc in reranked_small_docs if "parent_large_id" in doc.metadata}
    print(f"[3/5] parent large chunk 추출 완료 (대상 large chunk 수: {len(parent_large_ids)})")

    print("[4/5] context 압축(핵심 문장 추출) 진행 중...")
    compressed_contexts = []
    for idx, large_id in enumerate(parent_large_ids):
        if large_id in large_chunk_dict:
            print(f"    - [{idx+1}/{len(parent_large_ids)}] large chunk({large_id}) 압축 중...")
            large_doc = large_chunk_dict[large_id]
            compressed = compress_document_context(query, large_doc.page_content, llm_cheap)
            if compressed.strip() and "관련 내용 없음" not in compressed:
                compressed_contexts.append(compressed)
                print(f"        [압축 결과]\n{compressed}\n")
            else:
                print(f"        [압축 결과 없음 또는 관련 내용 없음]")
            print(f"    - [{idx+1}/{len(parent_large_ids)}] large chunk({large_id}) 압축 완료")
    context = "\n---\n".join(compressed_contexts)
    print(f"[4/5] context 압축 완료 (압축된 context 블록 수: {len(compressed_contexts)})")

    print("[5/5] 최종 답변 생성 중...")
    answer = generate_final_answer(context, query)
    print("[5/5] 최종 답변 생성 완료!")
    return answer

# --- main 실행부 ---
if __name__ == "__main__":
    # 오프라인 DB 구축(적재)는 offline_pipeline.py에서만 수행
    # 이 파일에서는 Chroma DB가 이미 구축되어 있다고 가정하고, ONLINE 파이프라인만 실행
    if not os.path.exists(CHROMA_DB_PATH) or not os.path.exists(LARGE_CHUNK_PICKLE):
        print(f"--- 에러: Chroma DB 또는 large chunk pickle 경로를 찾을 수 없습니다: '{CHROMA_DB_PATH}', '{LARGE_CHUNK_PICKLE}' ---")
        print("--- 먼저 offline_pipeline.py를 실행하여 DB를 구축하세요. ---")
    else:
        # 1. Chroma small chunk vectorstore 로드
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embedding_model
        )
        # 2. large chunk pickle 로드
        with open(LARGE_CHUNK_PICKLE, 'rb') as f:
            large_chunk_dict: Dict[str, Document] = pickle.load(f)
        # 3. 질의 입력 및 하이브리드 RAG 실행
        query = "네뷸라' 프로젝트의 현재 전체 공정률은 몇 퍼센트입니까?"

        print("\n" + "-"*50)
        print("-- [Ollama & ChromaDB] 하이브리드 RAG 파이프라인을 시작합니다.")
        print(f"   (질문: {query})")
        print("="*50 + "\n")
        start_time = time.time()
        final_answer = hybrid_rag_pipeline(
            query,
            vectorstore,
            large_chunk_dict,
            reranker_model,
            llm_cheap,
            llm_powerful,
            top_k=10,
            rerank_n=3
        )
        end_time = time.time()
        print("Step 5/5: 파이프라인 완료!")
        print("\n" + "-"*50)
        print("-- 최종 답변 --")
        print(final_answer)
        print("="*50)
        print(f"--- 총 실행 시간: {end_time - start_time:.2f}초---")
        # 분석 정보 출력
        # (retrieved_small_docs, context 등은 hybrid_retrieve_and_generate에서 반환하도록 확장 가능)
