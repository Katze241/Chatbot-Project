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
llm_cheap = Ollama(model="qwen2.5:3b")  # 답변 생성용 (빠르고 효율적)
llm_powerful = Ollama(model="qwen2:7b")  # 컨텍스트 압축용 (정확도 우선)

# --- ★★★ 변경점: 상수 정의 (Chroma DB 경로로 변경) ★★★ ---
CHROMA_DB_PATH = "chroma_db_bge_m3" # Chroma DB 저장 디렉토리
LARGE_CHUNK_PICKLE = f"{CHROMA_DB_PATH}_large_chunks.pkl" # large chunk pickle 파일 경로
HASH_LIST_FILE = f"{CHROMA_DB_PATH}_hashes.txt"  # 여러 파일의 해시값을 관리하는 파일
L_MAX = 4096
CHUNK_OVERLAP = 256

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
    rerank_n: int = 5  # 더 많은 문서 재순위화
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
    
    # 무조건 배치 처리 사용
    if len(parent_large_ids) > 1:
        print(f"    - 배치 처리 모드: {len(parent_large_ids)}개의 large chunk를 한 번에 압축합니다...")
        contexts_to_compress = []
        for large_id in parent_large_ids:
            if large_id in large_chunk_dict:
                contexts_to_compress.append(large_chunk_dict[large_id].page_content)
        
        if contexts_to_compress:
            batch_results = compress_documents_batch(query, contexts_to_compress, llm_powerful)
            
            # 유효한 결과만 필터링
            compressed_contexts = []
            for idx, res in enumerate(batch_results):
                if res.strip() and "관련 내용 없음" not in res:
                    compressed_contexts.append(res)
                    print(f"        [배치 압축 결과 {idx+1}]\n{res}\n")
                else:
                    print(f"        [배치 압축 결과 {idx+1} - 관련 내용 없음]")
            print(f"    - 배치 압축 완료.")
        else:
            compressed_contexts = []
            print("    - 압축할 컨텍스트가 없습니다.")
    else:
        # large chunk가 1개인 경우 개별 처리
        compressed_contexts = []
        for large_id in parent_large_ids:
            if large_id in large_chunk_dict:
                print(f"    - large chunk({large_id}) 개별 압축 중...")
                large_doc = large_chunk_dict[large_id]
                compressed = compress_document_context(query, large_doc.page_content, llm_cheap)
                if compressed.strip() and "관련 내용 없음" not in compressed:
                    compressed_contexts.append(compressed)
                    print(f"        [압축 결과]\n{compressed}\n")
                else:
                    print(f"        [압축 결과 없음 또는 관련 내용 없음]")
                print(f"    - large chunk({large_id}) 압축 완료")
    
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
    DB_NAME = "hybrid_vector_db"
    CHROMA_DB_PATH = f"./{DB_NAME}"
    LARGE_CHUNK_PICKLE = f"{CHROMA_DB_PATH}_large_chunks.pkl"

    if not os.path.exists(CHROMA_DB_PATH) or not os.path.exists(LARGE_CHUNK_PICKLE):
        print(f"--- 에러: Chroma DB 또는 large chunk pickle 경로를 찾을 수 없습니다: '{CHROMA_DB_PATH}', '{LARGE_CHUNK_PICKLE}' ---")
        print("--- 먼저 offline_pipeline_collection.py를 실행하여 DB를 구축하세요. ---")
    else:
        # 1. DB 클라이언트 생성 및 모든 컬렉션 이름 가져오기
        client = Chroma.PersistentClient(path=CHROMA_DB_PATH)
        collection_names = [c.name for c in client.list_collections()]
        
        if not collection_names:
            print(f"--- 에러: DB에 컬렉션이 없습니다. offline_pipeline_collection.py를 실행하여 DB를 구축하세요. ---")
            exit()
            
        print("통합 검색 대상 컬렉션 목록:")
        for name in collection_names:
            print(f"  - {name}")

        # 2. 질의 입력
        query = "'ATS, ATC, ATP는 각각 어떤 차이가 있나요?"
        
        print("\n" + "="*60)
        print("-- [Ollama & ChromaDB] 2단계 대화형 RAG 시스템")
        print(f"   (질문: {query})")
        print("="*60)
        
        # --- ★★★ 1단계: 빠른 사실 확인 답변 ★★★ ---
        print("\n" + "-"*50)
        print("-- [1단계] 빠른 사실 확인 답변 생성 중...")
        print("-"*50)
        start_time_fast = time.time()
        
        # 모든 컬렉션에서 검색 수행
        print("[1단계] 모든 컬렉션 대상으로 통합 벡터 검색 중...")
        all_retrieved_docs = []
        for name in collection_names:
            vectorstore = Chroma(
                client=client,
                collection_name=name,
                embedding_function=embedding_model
            )
            retrieved = vectorstore.similarity_search(query, k=5)  # 빠른 답변용으로 적은 수
            all_retrieved_docs.extend(retrieved)
            print(f"  - '{name}' 컬렉션 검색 완료 (검색된 문서: {len(retrieved)}개)")

        # 중복 문서 제거
        unique_docs_dict = {doc.page_content: doc for doc in all_retrieved_docs}
        unique_retrieved_docs = list(unique_docs_dict.values())
        print(f"  - 중복 제거 후 총 {len(unique_retrieved_docs)}개의 후보 문서 확보")
        
        fast_answer = fast_fact_check_pipeline(
            query,
            unique_retrieved_docs,  # 검색된 문서 목록 전달
            reranker_model,
            llm_cheap,
            top_k=5,
            rerank_n=2
        )
        end_time_fast = time.time()
        
        print("\n" + "="*60)
        print("-- [1단계] 빠른 답변 결과")
        print("="*60)
        print(fast_answer)
        print("="*60)
        print(f"--- 빠른 답변 생성 시간: {end_time_fast - start_time_fast:.2f}초 ---")
        
        # --- ★★★ 사용자 선택 대기 ★★★ ---
        print("\n" + "-"*50)
        print("-- 추가 상세 분석이 필요하신가요?")
        print("-"*50)
        print("다음 중 선택해주세요:")
        print("1. 'y' 또는 'yes' - 상세한 추론 분석 답변 받기")
        print("2. 'n' 또는 'no' - 현재 답변으로 충분")
        print("3. Enter 키 - 현재 답변으로 충분")
        print("-"*50)
        
        user_choice = input("선택: ").strip().lower()
        
        # --- ★★★ 2단계: 사용자 선택에 따른 추론 답변 ★★★ ---
        if user_choice in ['y', 'yes', '1']:
            print("\n" + "-"*50)
            print("-- [2단계] 상세 추론 분석 답변 생성 중...")
            print("-"*50)
            start_time_reasoning = time.time()
            
            # 모든 컬렉션에서 검색 수행 (상세 분석용으로 더 많은 문서)
            print("[2단계] 모든 컬렉션 대상으로 통합 벡터 검색 중...")
            all_retrieved_docs_detailed = []
            for name in collection_names:
                vectorstore = Chroma(
                    client=client,
                    collection_name=name,
                    embedding_function=embedding_model
                )
                retrieved = vectorstore.similarity_search(query, k=15)  # 상세 분석용으로 더 많은 수
                all_retrieved_docs_detailed.extend(retrieved)
                print(f"  - '{name}' 컬렉션 검색 완료 (검색된 문서: {len(retrieved)}개)")

            # 중복 문서 제거
            unique_docs_dict_detailed = {doc.page_content: doc for doc in all_retrieved_docs_detailed}
            unique_retrieved_docs_detailed = list(unique_docs_dict_detailed.values())
            print(f"  - 중복 제거 후 총 {len(unique_retrieved_docs_detailed)}개의 후보 문서 확보")
            
            # large chunk pickle 로드
            with open(LARGE_CHUNK_PICKLE, 'rb') as f:
                large_chunk_dict: Dict[str, Document] = pickle.load(f)
            
            reasoning_answer = reasoning_general_pipeline(
                query,
                unique_retrieved_docs_detailed,  # 검색된 문서 목록 전달
                large_chunk_dict,
                reranker_model,
                llm_cheap,
                llm_powerful,
                top_k=15,
                rerank_n=5
            )
            end_time_reasoning = time.time()
            
            print("\n" + "="*60)
            print("-- [2단계] 상세 추론 답변 결과")
            print("="*60)
            print(reasoning_answer)
            print("="*60)
            print(f"--- 상세 답변 생성 시간: {end_time_reasoning - start_time_reasoning:.2f}초 ---")
            
            # --- 최종 요약 ---
            total_time = (end_time_fast - start_time_fast) + (end_time_reasoning - start_time_reasoning)
            print("\n" + "="*60)
            print("-- 최종 실행 요약 --")
            print(f"빠른 답변 시간: {end_time_fast - start_time_fast:.2f}초")
            print(f"상세 답변 시간: {end_time_reasoning - start_time_reasoning:.2f}초")
            print(f"총 실행 시간: {total_time:.2f}초")
            print("="*60)
        else:
            print("\n" + "-"*50)
            print("-- 빠른 답변으로 종료합니다.")
            print(f"총 실행 시간: {end_time_fast - start_time_fast:.2f}초")
            print("-"*50)
