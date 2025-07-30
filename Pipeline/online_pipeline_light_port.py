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

# 멀티 쓰레딩 관련 라이브러리 추가
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import concurrent.futures

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
import chromadb

# 기타 유틸리티 라이브러리
import tiktoken
from sentence_transformers.cross_encoder import CrossEncoder
# Komoran 형태소 분석기 import (설치 필요: pip install konlpy)
from konlpy.tag import Komoran

# --- 1. 초기 설정 및 모델 로드 ---

# 임베딩 모델
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)
# 재순위화 모델
reranker_model = CrossEncoder('BAAI/bge-reranker-large', device='cuda')

# LLM 정의 (Ollama 사용)
llm_cheap = Ollama(model="qwen2:7b")  # 답변 생성용 (빠르고 효율적)
llm_powerful = Ollama(model="qwen2:7b")  # 컨텍스트 압축용 (정확도 우선)

# --- ★★★ 변경점: 상수 정의 (포트 기반 연결 추가) ★★★ ---
# 로컬 기반 연결 설정
CHROMA_DB_PATH = "hybrid_vector_db" # Chroma DB 저장 디렉토리
LARGE_CHUNK_PICKLE = f"{CHROMA_DB_PATH}_large_chunks.pkl" # large chunk pickle 파일 경로
HASH_LIST_FILE = f"{CHROMA_DB_PATH}_hashes.txt"  # 여러 파일의 해시값을 관리하는 파일

# 포트 기반 연결 설정
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

L_MAX = 4096
CHUNK_OVERLAP = 256

# --- 멀티 쓰레딩 성능 설정 (GPU 3대 최적화) ---
MAX_SEARCH_WORKERS = 6      # 검색용 최대 쓰레드 수 (GPU 3대 × 2)

import jpype

jvm_path = r"C:\Program Files\Java\jdk-11\bin\server\jvm.dll"
jar_path = r"C:\Users\kjmkj\workspace\project004\.venv\Lib\site-packages\konlpy\java\*"

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

# --- 멀티 쓰레딩 헬퍼 함수들 ---
def search_collection_threaded(collection_name: str, query: str, client, embedding_model, k: int = 5) -> List[Document]:
    """
    단일 컬렉션에서 검색을 수행하는 쓰레드 함수 (포트 기반 연결 지원)
    """
    try:
        vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_model
        )
        retrieved = vectorstore.similarity_search(query, k=k)
        print(f"  - '{collection_name}' 컬렉션 검색 완료 (검색된 문서: {len(retrieved)}개)")
        return retrieved
    except Exception as e:
        print(f"  - '{collection_name}' 컬렉션 검색 실패: {e}")
        return []

def parallel_search_all_collections(collection_names: List[str], query: str, client, embedding_model, k: int = 5, max_workers: int = 4) -> List[Document]:
    """
    모든 컬렉션을 병렬로 검색하는 함수
    """
    print(f"[병렬 검색] {len(collection_names)}개 컬렉션을 {max_workers}개 쓰레드로 병렬 검색 중...")
    all_retrieved_docs = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 각 컬렉션에 대해 검색 작업 제출
        future_to_collection = {
            executor.submit(search_collection_threaded, name, query, client, embedding_model, k): name 
            for name in collection_names
        }
        
        # 완료된 작업들의 결과 수집
        for future in as_completed(future_to_collection):
            collection_name = future_to_collection[future]
            try:
                retrieved_docs = future.result()
                all_retrieved_docs.extend(retrieved_docs)
            except Exception as e:
                print(f"  - '{collection_name}' 컬렉션 처리 중 오류: {e}")
    
    print(f"[병렬 검색 완료] 총 {len(all_retrieved_docs)}개의 문서 검색됨")
    return all_retrieved_docs

def compress_contexts_optimized(query: str, contexts: List[str], llm_powerful) -> List[str]:
    """
    컨텍스트 압축 최적화 함수 (배치 처리 사용)
    """
    if len(contexts) == 0:
        return []
    elif len(contexts) == 1:
        # 1개: 개별 처리 (배치 오버헤드 없음)
        print(f"[개별 압축] 1개 컨텍스트 압축 중...")
        compressed = compress_document_context(query, contexts[0], llm_powerful)
        if compressed.strip() and "관련 내용 없음" not in compressed:
            print(f"        [압축 결과]\n{compressed}\n")
            return [compressed]
        else:
            print("        [압축 결과 없음 또는 관련 내용 없음]")
            return []
    else:
        # 2개 이상: 배치 처리 (최고 효율)
        print(f"[배치 압축] {len(contexts)}개 컨텍스트를 배치로 압축 중...")
        compressed_contexts = compress_documents_batch(query, contexts, llm_powerful)
        
        # 유효한 결과만 필터링
        valid_contexts = []
        for i, compressed in enumerate(compressed_contexts):
            if compressed.strip() and "관련 내용 없음" not in compressed:
                valid_contexts.append(compressed)
                print(f"        [배치 압축 결과 {i+1}]\n{compressed}\n")
            else:
                print(f"        [배치 압축 결과 {i+1} - 관련 내용 없음]")
        
        print(f"[배치 압축 완료] {len(valid_contexts)}개 컨텍스트 압축 완료")
        return valid_contexts

def get_optimal_thread_count(gpu_count: int = 3, task_type: str = "search") -> int:
    """
    GPU 개수에 따른 최적 쓰레드 수 계산 (검색 전용)
    """
    if task_type == "search":
        # 검색 작업: GPU 개수 × 2 (I/O 대기 시간 활용)
        return min(gpu_count * 2, 8)  # 최대 8개로 제한
    else:
        return 2  # 기본값

def print_gpu_optimization_info():
    """
    GPU 최적화 정보 출력
    """
    print("\n" + "="*60)
    print("-- GPU 3대 최적화 설정 --")
    print("="*60)
    print(f"검색용 최대 쓰레드: {MAX_SEARCH_WORKERS}개 (GPU 3대 × 2)")
    print("="*60)
    print("  - 검색: I/O 대기 시간을 활용하여 GPU 개수 × 2")
    print("  - 압축: 배치 처리로 최적화")
    print("  - 안전장치: 최대 쓰레드 수 제한으로 시스템 안정성 확보")
    print("="*60)

# --- 3. 온라인 파이프라인 함수들 (검색, 답변 생성 등)만 남김 ---
def rerank_documents(query: str, retrieved_docs: List[Document], reranker_model, rerank_n: int = 2) -> List[Document]:
    if not retrieved_docs:
        return []
    pairs = [(query, doc.page_content) for doc in retrieved_docs]
    scores = reranker_model.predict(pairs)
    doc_score_pairs = list(zip(retrieved_docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in doc_score_pairs[:rerank_n]]

def compress_documents_batch(query: str, contexts: List[str], llm_powerful) -> List[str]:
    """
    LLM의 batch 기능을 이용해 여러 large chunk를 한 번에 압축합니다.
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
    chain = LLMChain(llm=llm_powerful, prompt=prompt)
    
    # batch 처리를 위한 입력 데이터 구성
    batch_inputs = [{"context": ctx, "question": query} for ctx in contexts]
    
    # batch 실행
    results = chain.batch(batch_inputs)
    
    # 결과에서 'text'만 추출하여 리스트로 반환
    return [result['text'] for result in results]

def compress_document_context(query: str, context: str, llm_powerful) -> str:
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
    chain = LLMChain(llm=llm_powerful, prompt=prompt)  # 컨텍스트 압축은 고성능 모델 사용
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
        chain = LLMChain(llm=llm_cheap, prompt=prompt) # 답변 생성은 빠른 모델 사용
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

# --- ★★★ 새로운 함수: 사실 확인용 초고속 답변 파이프라인 ★★★ ---
def fast_fact_check_pipeline(
    query: str,
    retrieved_docs: List[Document],  # 검색된 문서 목록
    reranker_model,
    llm_cheap,
    top_k: int = 5,  # 더 적은 문서 검색
    rerank_n: int = 2  # 더 적은 문서 재순위화
) -> str:
    """
    사실 확인용 초고속 답변 파이프라인
    - 빠른 응답을 위해 검색 문서 수를 줄임
    - 재순위화 단계를 간소화
    - 컨텍스트 압축 없이 직접 답변 생성
    """
    print(f"[FAST] 1/3 단계: 검색된 문서 수: {len(retrieved_docs)}")

    if not retrieved_docs:
        return generate_final_answer("", query)

    print("[FAST] 2/3 단계: 빠른 재순위화 중...")
    reranked_docs = rerank_documents(query, retrieved_docs, reranker_model, rerank_n)
    print(f"[FAST] 2/3 단계 완료 (선택된 문서 수: {len(reranked_docs)})")

    # 컨텍스트 압축 없이 직접 문서 내용 사용
    context = "\n---\n".join([doc.page_content for doc in reranked_docs])
    
    print("[FAST] 3/3 단계: 빠른 답변 생성 중...")
    # 빠른 답변을 위한 간소화된 프롬프트
    if not context.strip():
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
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=""" 
            당신은 문서 기반 AI 어시스턴트입니다. 반드시 "한국어"로만 답변하세요.
            
            아래 제공된 문서 내용(Context)에서 사용자 질문(Question)에 대해 간결하고 정확하게 답변하세요.
            - 답변 시작에 "[빠른 사실 확인 답변]"이라는 문구를 포함하세요.
            - 답변은 2-3문장으로 간결하게 작성하세요.
            - 출처 정보는 간단히 [문서]로 표시하세요.

[문서 내용]
{context}

[사용자 질문]
{question}

[답변]"""
        )
        chain = LLMChain(llm=llm_cheap, prompt=prompt)  # 빠른 답변은 가벼운 모델 사용
        result = chain.invoke({"context": context, "question": query})
        return result['text']

# --- ★★★ 새로운 함수: 추론용 일반 답변 파이프라인 ★★★ ---
def reasoning_general_pipeline(
    query: str,
    retrieved_small_docs: List[Document],  # 검색된 문서 목록
    large_chunk_dict: Dict[str, Document],
    reranker_model,
    llm_cheap,
    llm_powerful,
    top_k: int = 15,  # 더 많은 문서 검색
    rerank_n: int = 3  # 더 많은 문서 재순위화
) -> str:
    """
    추론용 일반 답변 파이프라인 (기존 hybrid_rag_pipeline과 유사하지만 더 정교함)
    - 더 많은 문서를 검색하여 포괄적인 분석
    - 컨텍스트 압축을 통한 정확한 정보 추출
    - 추론과 분석이 포함된 상세한 답변 생성
    """
    print(f"[REASONING] 1/5 단계: 검색된 문서 수: {len(retrieved_small_docs)}")

    keywords = extract_proper_nouns(query, max_keywords=5)
    print(f"[REASONING] 고유명사 추출: {keywords}")

    # 키워드 포함 개수에 따라 점수 부여
    keyword_weight = 0.5
    scored_docs = []
    for doc in retrieved_small_docs:
        num_keywords = sum(1 for kw in keywords if kw in doc.page_content)
        score = 1.0 + (num_keywords * keyword_weight)
        scored_docs.append((doc, score))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    rerank_candidates = [doc for doc, _ in scored_docs[:top_k]]
    print(f"[REASONING] 키워드 점수 기반 정렬 완료 (상위 {len(rerank_candidates)}개 문서)")

    if not rerank_candidates:
        return generate_final_answer("", query)

    print("[REASONING] 2/5 단계: 정교한 재순위화 중...")
    reranked_small_docs = rerank_documents(query, rerank_candidates, reranker_model, rerank_n)
    print(f"[REASONING] 2/5 단계 완료 (선택된 문서 수: {len(reranked_small_docs)})")

    print("[REASONING] 3/5 단계: parent large chunk 추출 중...")
    parent_large_ids = {doc.metadata["parent_large_id"] for doc in reranked_small_docs if "parent_large_id" in doc.metadata}
    print(f"[REASONING] 3/5 단계 완료 (대상 large chunk 수: {len(parent_large_ids)})")

    print("[REASONING] 4/5 단계: 정교한 컨텍스트 압축 중...")
    
    # 최적화된 압축 처리 사용
    contexts_to_compress = []
    for large_id in parent_large_ids:
        if large_id in large_chunk_dict:
            contexts_to_compress.append(large_chunk_dict[large_id].page_content)
    
    if contexts_to_compress:
        # 최적화된 압축 사용 (배치 처리)
        compressed_contexts = compress_contexts_optimized(query, contexts_to_compress, llm_powerful)
        print(f"    - 압축 완료.")
    else:
        compressed_contexts = []
        print("    - 압축할 컨텍스트가 없습니다.")
    
    context = "\n---\n".join(compressed_contexts)
    print(f"[REASONING] 4/5 단계 완료 (압축된 context 블록 수: {len(compressed_contexts)})")

    print("[REASONING] 5/5 단계: 추론 기반 상세 답변 생성 중...")
    # 추론용 상세 답변을 위한 프롬프트
    if not context.strip():
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
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=""" 
            당신은 문서 기반 AI 어시스턴트입니다. 반드시 "한국어"로만 답변하세요.
            
            아래 제공된 문서 내용(Context)에서 사용자 질문(Question)에 대해 추론과 분석을 포함한 상세한 답변을 제공하세요.
            - 답변 시작에 "[추론 기반 상세 답변]"이라는 문구를 포함하세요.
            - 관련된 배경 정보, 원인, 결과, 영향 등을 포함하여 포괄적으로 설명하세요.
            - 필요시 여러 관점에서 분석하고 비교하세요.
            - 답변 마지막에 [파일명, 페이지번호] 형식으로 출처를 명시하세요.

[문서 내용]
{context}

[사용자 질문]
{question}

[답변]"""
        )
        chain = LLMChain(llm=llm_cheap, prompt=prompt)  # 추론 답변은 빠른 모델 사용
        result = chain.invoke({"context": context, "question": query})
        return result['text']



# --- main 실행부 ---
if __name__ == "__main__":
    print("이 파일은 모듈로만 사용됩니다. 포트 연결 기능이 제거되었습니다.")
    print("auto_qa_processor.py를 사용하여 QA 처리를 진행하세요.")
