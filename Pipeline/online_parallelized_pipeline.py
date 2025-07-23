# main_chroma.py
import os
import time
import numpy as np
from typing import List, Dict, Set, Optional, Tuple
import re
import hashlib
import shutil
import glob
import pickle
from functools import lru_cache
import concurrent.futures
from threading import Lock

# SQLite 버전 업그레이드를 위한 패치
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

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

# LLM 정의 (Ollama 사용) - 속도 최적화를 위한 설정
llm_cheap = Ollama(
    model="qwen2.5:3b",  # 더 작은 모델로 속도 향상
    temperature=0.1,     # 낮은 temperature로 빠른 응답
    timeout=300,         # 타임아웃 증가 (60 → 300)
    num_ctx=2048,        # 컨텍스트 길이 제한
    num_predict=1024     # 최대 토큰 수 제한 (512 → 1024로 증가)
)
llm_powerful = Ollama(
    model="qwen2:7b",    # 저품질 문서 -> 고성능 답변용
    temperature=0.2,     # 약간 높은 temperature
    timeout=300,
    num_ctx=2048,        # 컨텍스트 길이 제한
    num_predict=4096     # 최대 토큰 수 제한 (2048 → 4096으로 증가)
)

# --- ★★★ 변경점: 상수 정의 (Chroma DB 경로로 변경) ★★★ ---
CHROMA_DB_PATH = "chroma_db_bge_m3" # Chroma DB 저장 디렉토리
LARGE_CHUNK_PICKLE = f"{CHROMA_DB_PATH}_large_chunks.pkl" # large chunk pickle 파일 경로
HASH_LIST_FILE = f"{CHROMA_DB_PATH}_hashes.txt"  # 여러 파일의 해시값을 관리하는 파일
L_MAX = 4096
CHUNK_OVERLAP = 256

# 성능 최적화를 위한 상수
MAX_CONTEXT_LENGTH = 4000  # 컨텍스트 최대 길이 단축 (8000 → 4000)
MAX_SENTENCES = 10  # 최대 문장 수 단축 (20 → 10)
MAX_WORKERS = 4  # 병렬 처리 최대 워커 수 증가 (3 → 4)
CACHE_SIZE = 1000  # 캐시 크기
MIN_CONTEXT_QUALITY_SCORE = 0.2  # 최소 컨텍스트 품질 점수 낮춤 (0.3 → 0.2)
MAX_RETRIES = 1  # 최대 재시도 횟수 단축 (2 → 1)

# 지능형 모드 선택을 위한 상수
QUERY_COMPLEXITY_THRESHOLD = 0.3  # 질문 복잡도 임계값 (0.6 → 0.3으로 낮춤)
DOCUMENT_QUALITY_THRESHOLD = 0.45  # 문서 품질 임계값 (0.7 → 0.45로 낮춤)
REASONING_KEYWORDS = ['분석', '평가', '비교', '원인', '결과', '전략', '방안', '개선', '문제', '해결', 
                      '추천', '제안', '예측', '전망', '바탕', '추론', '추측', '이유']

# 최적 분기점을 위한 추가 상수
ULTRA_FAST_TIMEOUT = 60  # 초고속 모드 타임아웃 (초)
NORMAL_MODE_TIMEOUT = 300  # 일반 모드 타임아웃 (초)
MIN_CONTEXT_LENGTH = 300  # 최소 컨텍스트 길이
MAX_ULTRA_FAST_CONTEXT = 2500  # 초고속 모드 최대 컨텍스트

import jpype

jvm_path = r"/usr/lib/jvm/java-11-openjdk-amd64/lib/server/libjvm.so"
jar_path = r"/home/discovery/.local/lib/python3.11/site-packages/konlpy/java/*"

# JVM이 이미 시작되었는지 확인
if not jpype.isJVMStarted():
    try:
        jpype.startJVM(jvm_path, f"-Djava.class.path={jar_path}", "-Xmx4g")
    except Exception as e:
        print(f"JVM 시작 실패: {e}")
        print("JVM 없이 실행을 계속합니다...")
        # JVM 없이 실행하기 위해 Komoran 사용을 비활성화
        def extract_proper_nouns(text: str, max_keywords: int = 5) -> list[str]:
            # JVM이 없을 때는 키워드 추출만 수행
            words = re.findall(r'[가-힣a-zA-Z0-9]+', text)
            return words[:max_keywords]

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
    try:
        komoran = Komoran()
        tagged = komoran.pos(text)
        # NNP: 고유명사
        proper_nouns = [word for word, pos in tagged if pos == 'NNP']
        # java.lang.String → Python str로 변환
        proper_nouns = [str(word) for word in proper_nouns]
        # 중복 제거 및 상위 max_keywords개만 반환
        return list(dict.fromkeys(proper_nouns))[:max_keywords]
    except Exception as e:
        print(f"Komoran 사용 불가: {e}")
        print("대체 방법으로 키워드 추출을 수행합니다...")
        # JVM이 없을 때는 키워드 추출만 수행
        words = re.findall(r'[가-힣a-zA-Z0-9]+', text)
        return words[:max_keywords]

# --- 3. 온라인 파이프라인 함수들 (검색, 답변 생성 등)만 남김 ---
def rerank_documents(query: str, retrieved_docs: List[Document], reranker_model, rerank_n: int = 2) -> List[Document]:
    if not retrieved_docs:
        return []
    pairs = [(query, doc.page_content) for doc in retrieved_docs]
    scores = reranker_model.predict(pairs)
    doc_score_pairs = list(zip(retrieved_docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in doc_score_pairs[:rerank_n]]

@lru_cache(maxsize=CACHE_SIZE)
def get_compression_prompt() -> PromptTemplate:
    """컨텍스트 압축을 위한 간단한 프롬프트"""
    return PromptTemplate(
        input_variables=["question", "context"],
        template="""질문: {question}

문서: {context}

질문에 답하는 데 필요한 문장들:"""
    )

@lru_cache(maxsize=CACHE_SIZE)
def get_advanced_compression_prompt() -> PromptTemplate:
    """고급 컨텍스트 압축을 위한 간단한 프롬프트"""
    return PromptTemplate(
        input_variables=["question", "context"],
        template="""질문: {question}

문서: {context}

관련 문장들:"""
    )

@lru_cache(maxsize=CACHE_SIZE)
def get_answer_prompt() -> PromptTemplate:
    """답변 생성을 위한 간단한 프롬프트"""
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""문서: {context}

질문: {question}

답변:"""
    )

@lru_cache(maxsize=CACHE_SIZE)
def get_adaptive_answer_prompt() -> PromptTemplate:
    """적응형 답변 생성을 위한 간단한 프롬프트"""
    return PromptTemplate(
        input_variables=["context", "question", "context_quality"],
        template="""문서: {context}

질문: {question}

답변:"""
    )

@lru_cache(maxsize=CACHE_SIZE)
def get_no_context_prompt() -> PromptTemplate:
    """컨텍스트가 없을 때 사용할 간단한 프롬프트"""
    return PromptTemplate(
        input_variables=["question"],
        template="""질문: {question}

답변: 문서에서 관련 내용을 찾을 수 없습니다."""
    )

def preprocess_context_for_compression(context: str, query: str) -> str:
    """
    컨텍스트 압축 전 사전 처리를 수행하여 성능을 향상시킵니다.
    키워드 기반 필터링과 길이 제한을 적용합니다.
    """
    if len(context) <= MAX_CONTEXT_LENGTH:
        return context
    
    # 키워드 추출
    keywords = extract_proper_nouns(query, max_keywords=3) + extract_keywords_from_text(query, max_keywords=5)
    
    # 문장 분리 및 점수 계산
    sentences = re.split(r'[.!?]+', context)
    relevant_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:  # 너무 짧은 문장 제외
            continue
        
        # 키워드 매칭 점수 계산 (대소문자 무시)
        keyword_score = sum(1 for kw in keywords if kw.lower() in sentence.lower())
        if keyword_score > 0:
            relevant_sentences.append((sentence, keyword_score))
    
    # 점수 순으로 정렬하고 상위 문장들만 선택
    relevant_sentences.sort(key=lambda x: x[1], reverse=True)
    filtered_sentences = [s[0] for s in relevant_sentences[:MAX_SENTENCES]]
    
    return '. '.join(filtered_sentences) if filtered_sentences else context[:MAX_CONTEXT_LENGTH]

def compress_single_context(args: Tuple[str, str, str]) -> Tuple[str, float]:
    """
    단일 컨텍스트 압축을 수행하는 함수 (병렬 처리용)
    """
    query, context, large_id = args
    
    try:
        # 사전 처리
        filtered_context = preprocess_context_for_compression(context, query)
        
        # 프롬프트 캐싱 활용
        prompt = get_compression_prompt()
        chain = LLMChain(llm=llm_cheap, prompt=prompt)
        
        result = chain.invoke({"context": filtered_context, "question": query})
        compressed = result['text'].strip()
        
        # 결과 검증 및 후처리
        if not compressed or compressed.lower() in ['관련 내용 없음', 'none', 'no relevant content']:
            return "", 0.0
        
        # 불필요한 프롬프트 텍스트 제거
        compressed = re.sub(r'^질문에 답하는 데 필요한 문장들:\s*', '', compressed, flags=re.IGNORECASE)
        compressed = re.sub(r'^문서에서 질문에 답하는 데 필요한 문장들:\s*', '', compressed, flags=re.IGNORECASE)
        
        # 압축된 결과의 품질 재평가
        quality = calculate_context_quality_score(query, compressed)
        
        return compressed, quality
        
    except Exception as e:
        print(f"컨텍스트 압축 중 오류 발생 (large_id: {large_id}): {e}")
        return "", 0.0

def compress_document_context_parallel(
    query: str, 
    large_chunk_dict: Dict[str, Document], 
    parent_large_ids: Set[str]
) -> List[str]:
    """
    병렬 처리를 통한 컨텍스트 압축 성능 향상
    """
    if not parent_large_ids:
        return []
    
    # 병렬 처리할 작업 준비
    compression_tasks = []
    for large_id in parent_large_ids:
        if large_id in large_chunk_dict:
            large_doc = large_chunk_dict[large_id]
            compression_tasks.append((query, large_doc.page_content, large_id))
    
    compressed_contexts = []
    
    # 병렬 처리 실행
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {
            executor.submit(compress_single_context, task): task[2] 
            for task in compression_tasks
        }
        
        for future in concurrent.futures.as_completed(future_to_id):
            large_id = future_to_id[future]
            try:
                compressed, quality = future.result()
                if compressed and compressed.strip():
                    compressed_contexts.append(compressed)
                    print(f"    - large chunk({large_id}) 압축 완료")
                else:
                    print(f"    - large chunk({large_id}) 압축 결과 없음")
            except Exception as e:
                print(f"    - large chunk({large_id}) 압축 실패: {e}")
    
    return compressed_contexts

def compress_document_context(query: str, context: str, llm_cheap) -> str:
    """
    LLM을 이용해 large chunk 내에서 질문에 답하기 위해 필요한 핵심 문장만 그대로 추출합니다.
    성능 최적화: 프롬프트 캐싱, 컨텍스트 길이 제한, 키워드 기반 사전 필터링
    """
    # 사전 처리
    filtered_context = preprocess_context_for_compression(context, query)
    
    # 프롬프트 캐싱 활용
    prompt = get_compression_prompt()
    chain = LLMChain(llm=llm_cheap, prompt=prompt)
    
    try:
        result = chain.invoke({"context": filtered_context, "question": query})
        compressed = result['text'].strip()
        
        # 결과 검증 및 후처리
        if not compressed or compressed.lower() in ['관련 내용 없음', 'none', 'no relevant content']:
            return "관련 내용 없음"
        
        # 불필요한 프롬프트 텍스트 제거
        compressed = re.sub(r'^질문에 답하는 데 필요한 문장들:\s*', '', compressed, flags=re.IGNORECASE)
        compressed = re.sub(r'^문서에서 질문에 답하는 데 필요한 문장들:\s*', '', compressed, flags=re.IGNORECASE)
        
        return compressed
        
    except Exception as e:
        print(f"컨텍스트 압축 중 오류 발생: {e}")
        return "관련 내용 없음"

def generate_final_answer(context: str, query: str) -> str:
    """
    간단한 최종 답변 생성 함수
    """
    # 컨텍스트 검증
    if not context or not context.strip() or context.strip() == "관련 내용 없음":
        prompt = get_no_context_prompt()
        chain = LLMChain(llm=llm_cheap, prompt=prompt)
        
        answer = safe_llm_invoke(chain, {"question": query})
        if not answer:
            return "문서에서 관련 내용을 찾을 수 없습니다."
        return answer
    
    # 컨텍스트가 있는 경우
    prompt = get_answer_prompt()
    chain = LLMChain(llm=llm_powerful, prompt=prompt)
    
    answer = safe_llm_invoke(chain, {"context": context, "question": query})
    
    # 답변 품질 검증
    if not answer or len(answer) < 5:
        return "문서에서 명확한 답변을 찾을 수 없습니다."
    
    return answer

def calculate_context_quality_score(query: str, context: str) -> float:
    """컨텍스트 품질 점수 계산"""
    if not context or not context.strip():
        return 0.0
    
    # 1. 키워드 매칭 점수
    query_keywords = extract_keywords_from_text(query, max_keywords=5)
    context_keywords = extract_keywords_from_text(context, max_keywords=10)
    
    keyword_matches = sum(1 for qk in query_keywords if any(qk in ck for ck in context_keywords))
    keyword_score = keyword_matches / len(query_keywords) if query_keywords else 0.0
    
    # 2. 추론 키워드 점수 (추론이 필요한 질문인지 판단)
    reasoning_score = 0.0
    for keyword in REASONING_KEYWORDS:
        if keyword in query:
            reasoning_score += 0.1
    reasoning_score = min(reasoning_score, 1.0)
    
    # 3. 컨텍스트 길이 점수
    length_score = min(len(context) / 1000, 1.0)  # 1000자 기준
    
    # 4. 최종 품질 점수 (가중 평균)
    final_score = (keyword_score * 0.5 + reasoning_score * 0.3 + length_score * 0.2)
    
    return final_score

# 지능형 모드 선택 함수들
def analyze_query_complexity(query: str) -> float:
    """질문 복잡도 분석"""
    complexity_score = 0.0
    
    # 1. 추론 키워드 검사
    for keyword in REASONING_KEYWORDS:
        if keyword in query:
            complexity_score += 0.15
    
    # 2. 질문 길이 검사
    if len(query) > 50:
        complexity_score += 0.2
    
    # 3. 복합 질문 검사 (여러 개의 질문이 포함된 경우)
    question_indicators = ['그리고', '또한', '또는', '또한', '또한', '또한']
    for indicator in question_indicators:
        if indicator in query:
            complexity_score += 0.1
    
    return min(complexity_score, 1.0)

def assess_document_quality(query: str, retrieved_docs: List[Document]) -> float:
    """검색된 문서들의 품질 평가"""
    if not retrieved_docs:
        return 0.0
    
    # 1. 키워드 매칭 점수
    query_keywords = extract_keywords_from_text(query, max_keywords=5)
    total_keyword_score = 0.0
    
    for doc in retrieved_docs[:3]:  # 상위 3개 문서만 평가
        doc_keywords = extract_keywords_from_text(doc.page_content, max_keywords=10)
        matches = sum(1 for qk in query_keywords if any(qk in dk for dk in doc_keywords))
        total_keyword_score += matches / len(query_keywords) if query_keywords else 0.0
    
    avg_keyword_score = total_keyword_score / min(len(retrieved_docs), 3)
    
    # 2. 문서 다양성 점수 (다양한 문서가 검색되었는지)
    diversity_score = min(len(retrieved_docs) / 5, 1.0)
    
    # 3. 문서 길이 점수
    avg_length = sum(len(doc.page_content) for doc in retrieved_docs) / len(retrieved_docs)
    length_score = min(avg_length / 500, 1.0)  # 500자 기준
    
    # 최종 품질 점수
    final_score = (avg_keyword_score * 0.6 + diversity_score * 0.2 + length_score * 0.2)
    
    return final_score

def select_optimal_mode(query: str, retrieved_docs: List[Document]) -> str:
    """질문과 문서 품질을 분석하여 최적의 모드 선택"""
    
    # 1. 질문 복잡도 분석
    query_complexity = analyze_query_complexity(query)
    
    # 2. 문서 품질 분석
    document_quality = assess_document_quality(query, retrieved_docs)
    
    # 3. 컨텍스트 길이 예측
    estimated_context_length = sum(len(doc.page_content) for doc in retrieved_docs[:3])
    
    # 4. 모드 선택 로직 (개선된 버전)
    print(f"[지능형 모드 선택] 질문 복잡도: {query_complexity:.2f}, 문서 품질: {document_quality:.2f}, 예상 컨텍스트 길이: {estimated_context_length}")
    
    # 개선된 분기 로직
    if query_complexity > QUERY_COMPLEXITY_THRESHOLD:
        # 복잡한 질문은 일반 모드
        print(f"[지능형 모드 선택] 일반 모드 선택 (복잡한 질문)")
        return "normal"
    elif document_quality < DOCUMENT_QUALITY_THRESHOLD or estimated_context_length > MAX_ULTRA_FAST_CONTEXT:
        # 낮은 품질 또는 너무 긴 길이 = 일반 모드
        print(f"[지능형 모드 선택] 일반 모드 선택 (낮은 문서 품질 또는 너무 긴 길이)")
        return "normal"
    elif document_quality > 0.45 and estimated_context_length < 2500:
        # 높은 품질 또는 적당한 길이의 컨텍스트 = 초고속 모드
        print(f"[지능형 모드 선택] 초고속 모드 선택 (높은 문서 품질)")
        return "ultra_fast"
    else:
        # 기본적으로 초고속 모드 (속도 우선)
        print(f"[지능형 모드 선택] 초고속 모드 선택 (기본값)")
        return "ultra_fast"

@lru_cache(maxsize=CACHE_SIZE)
def get_advanced_compression_prompt() -> PromptTemplate:
    """고급 컨텍스트 압축을 위한 최적화된 프롬프트 템플릿"""
    return PromptTemplate(
        input_variables=["question", "context"],
        template="""당신은 문서 분석 전문가입니다. 주어진 질문에 답하기 위해 필요한 핵심 정보만 정확히 추출하세요.

**분석 지침:**
1. 질문의 핵심 키워드를 파악하세요
2. 문서에서 해당 키워드와 관련된 문장들을 찾으세요
3. 관련성 높은 순서로 문장들을 정렬하세요
4. 원문을 그대로 유지하세요 (요약 금지)
5. 관련성이 낮은 내용은 제외하세요

**질문:** {question}

**문서 내용:**
{context}

**추출된 관련 문장들 (관련성 순):**
"""
    )

@lru_cache(maxsize=CACHE_SIZE)
def get_adaptive_answer_prompt() -> PromptTemplate:
    """적응형 답변 생성을 위한 프롬프트 템플릿"""
    return PromptTemplate(
        input_variables=["context", "question", "context_quality"],
        template="""당신은 전문적이고 정확한 문서 기반 AI 어시스턴트입니다.

**컨텍스트 품질:** {context_quality}/1.0

**답변 규칙:**
1. 반드시 한국어로 답변
2. 제공된 문서 내용만을 기반으로 답변
3. 컨텍스트 품질이 낮으면 불확실성을 명시
4. 문서에 없는 내용은 추측하지 말고 "문서에 명시되지 않음"이라고 표시
5. 답변 시작에 "[문서 기반 답변입니다.]" 포함
6. 답변 마지막에 출처 정보 포함 (가능한 경우)

**문서 내용:**
{context}

**질문:**
{question}

**답변:**"""
    )

def adaptive_compress_context(query: str, context: str, llm_cheap) -> Tuple[str, float]:
    """
    적응형 컨텍스트 압축 함수
    컨텍스트 품질을 평가하고 필요에 따라 다른 전략을 적용합니다.
    """
    # 컨텍스트 품질 평가
    quality_score = calculate_context_quality_score(query, context)
    
    # 품질이 낮은 경우 더 적극적인 필터링 적용
    if quality_score < MIN_CONTEXT_QUALITY_SCORE:
        print(f"    - 컨텍스트 품질이 낮음 (점수: {quality_score:.2f}), 강화된 필터링 적용")
        filtered_context = preprocess_context_for_compression(context, query)
        # 추가 키워드 기반 필터링
        keywords = extract_proper_nouns(query, max_keywords=5) + extract_keywords_from_text(query, max_keywords=8)
        sentences = re.split(r'[.!?]+', filtered_context)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:  # 더 엄격한 길이 제한
                continue
            # 키워드 매칭 점수 계산
            keyword_score = sum(1 for kw in keywords if kw.lower() in sentence.lower())
            if keyword_score >= 2:  # 더 엄격한 기준
                relevant_sentences.append((sentence, keyword_score))
        
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        filtered_context = '. '.join([s[0] for s in relevant_sentences[:10]])  # 더 적은 문장 수
    
    else:
        filtered_context = preprocess_context_for_compression(context, query)
    
    # 고급 프롬프트 사용
    prompt = get_advanced_compression_prompt()
    chain = LLMChain(llm=llm_cheap, prompt=prompt)
    
    for attempt in range(MAX_RETRIES):
        try:
            result = chain.invoke({"context": filtered_context, "question": query})
            compressed = result['text'].strip()
            
            # 결과 검증
            if not compressed or compressed.lower() in ['관련 내용 없음', 'none', 'no relevant content']:
                return "관련 내용 없음", 0.0
            
            # 불필요한 프롬프트 텍스트 제거
            compressed = re.sub(r'^추출된 관련 문장들[^:]*:\s*', '', compressed, flags=re.IGNORECASE)
            compressed = re.sub(r'^관련 문장들[^:]*:\s*', '', compressed, flags=re.IGNORECASE)
            
            # 압축된 결과의 품질 재평가
            final_quality = calculate_context_quality_score(query, compressed)
            
            return compressed, final_quality
            
        except Exception as e:
            print(f"    - 컨텍스트 압축 시도 {attempt + 1} 실패: {e}")
            if attempt == MAX_RETRIES - 1:
                return "관련 내용 없음", 0.0
    
    return "관련 내용 없음", 0.0

def compress_single_context_adaptive(args: Tuple[str, str, str]) -> Tuple[str, float]:
    """
    적응형 단일 컨텍스트 압축을 수행하는 함수 (병렬 처리용)
    """
    query, context, large_id = args
    
    try:
        compressed, quality = adaptive_compress_context(query, context, llm_cheap)
        return compressed, quality
        
    except Exception as e:
        print(f"컨텍스트 압축 중 오류 발생 (large_id: {large_id}): {e}")
        return "", 0.0

def compress_document_context_parallel_adaptive(
    query: str, 
    large_chunk_dict: Dict[str, Document], 
    parent_large_ids: Set[str]
) -> List[Tuple[str, float]]:
    """
    적응형 병렬 처리를 통한 컨텍스트 압축 성능 향상
    품질 점수와 함께 결과를 반환합니다.
    """
    if not parent_large_ids:
        return []
    
    # 병렬 처리할 작업 준비
    compression_tasks = []
    for large_id in parent_large_ids:
        if large_id in large_chunk_dict:
            large_doc = large_chunk_dict[large_id]
            compression_tasks.append((query, large_doc.page_content, large_id))
    
    compressed_results = []
    
    # 병렬 처리 실행
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {
            executor.submit(compress_single_context_adaptive, task): task[2] 
            for task in compression_tasks
        }
        
        for future in concurrent.futures.as_completed(future_to_id):
            large_id = future_to_id[future]
            try:
                compressed, quality = future.result()
                if compressed and compressed.strip():
                    compressed_results.append((compressed, quality))
                    print(f"    - large chunk({large_id}) 압축 완료 (품질: {quality:.2f})")
                else:
                    print(f"    - large chunk({large_id}) 압축 결과 없음")
            except Exception as e:
                print(f"    - large chunk({large_id}) 압축 실패: {e}")
    
    # 품질 점수 순으로 정렬
    compressed_results.sort(key=lambda x: x[1], reverse=True)
    return compressed_results

def generate_adaptive_final_answer(context: str, query: str, context_quality: float = 1.0) -> str:
    """
    간단한 적응형 최종 답변 생성 함수
    """
    # 컨텍스트 검증
    if not context or not context.strip() or context.strip() == "관련 내용 없음":
        prompt = get_no_context_prompt()
        chain = LLMChain(llm=llm_cheap, prompt=prompt)
        
        answer = safe_llm_invoke(chain, {"question": query})
        if not answer:
            return "문서에서 관련 내용을 찾을 수 없습니다."
        return answer
    
    # 컨텍스트 품질에 따른 적응형 처리
    if context_quality > 0.5:
        # 품질이 높은 경우 저성능 모델 사용
        prompt = get_adaptive_answer_prompt()
        chain = LLMChain(llm=llm_cheap, prompt=prompt)
    else:
        # 품질이 낮은 경우 고성능 모델 사용
        prompt = get_adaptive_answer_prompt()
        chain = LLMChain(llm=llm_powerful, prompt=prompt)
    
    answer = safe_llm_invoke(chain, {
        "context": context, 
        "question": query, 
        "context_quality": f"{context_quality:.2f}"
    })
    
    # 답변 품질 검증
    if not answer or len(answer) < 5:
        return "문서에서 명확한 답변을 찾을 수 없습니다."
    
    return answer

def generate_fast_answer(context: str, query: str) -> str:
    """
    빠른 답변 생성을 위한 개선된 함수 - 추론 정확성 향상
    """
    # 컨텍스트 검증
    if not context or not context.strip() or context.strip() == "관련 내용 없음":
        return "문서에서 관련 내용을 찾을 수 없습니다."
    
    # 개선된 프롬프트 사용 - 추론 정확성 향상
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""다음 문서를 바탕으로 질문에 정확하고 논리적으로 답변하세요.

문서 내용:
{context}

질문: {question}

답변 시 다음 사항을 고려하세요:
1. 반드시 한국어로 답변
2. 문서에 명시된 사실만을 바탕으로 답변
3. 추론이 필요한 경우 논리적 근거 제시
4. 구체적인 예시나 데이터가 있다면 포함
5. 명확하고 구조화된 형태로 답변

답변:"""
    )
    chain = LLMChain(llm=llm_cheap, prompt=prompt)
    
    try:
        result = chain.invoke({"context": context, "question": query})
        answer = result['text'].strip()
        
        # 답변 품질 검증
        if not answer or len(answer) < 10:
            return "문서에서 명확한 답변을 찾을 수 없습니다."
        
        return answer
        
    except Exception as e:
        print(f"빠른 답변 생성 중 오류: {e}")
        return "문서에서 관련 내용을 찾을 수 없습니다."

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
    rerank_n: int = 3,
    use_parallel: bool = True,
    use_adaptive: bool = True,
    fast_mode: bool = False,  # 빠른 모드 옵션 추가
    ultra_fast_mode: bool = False,  # 초고속 모드 옵션 추가
    use_intelligent_mode: bool = True  # 지능형 모드 선택 활성화
) -> str:
    print("[1/5] 1차 검색: small chunk 대상으로 벡터 검색 중...")
    retrieved_small_docs = vectorstore.similarity_search(query, k=top_k)
    print(f"[1/5] 1차 검색 완료 (검색된 small chunk 수: {len(retrieved_small_docs)})")

    # 지능형 모드 선택 (1차 검색 후)
    if use_intelligent_mode:
        selected_mode = select_optimal_mode(query, retrieved_small_docs)
        ultra_fast_mode = (selected_mode == "ultra_fast")
        print(f"[지능형 모드 선택] 최종 선택: {'초고속 모드' if ultra_fast_mode else '일반 모드'}")

    if ultra_fast_mode:
        print("[초고속 모드] 최소한의 처리만 수행")
        # 초고속 모드: 상위 2개 문서만 사용
        if retrieved_small_docs:
            top_docs = retrieved_small_docs[:2]  # 상위 2개 사용
        else:
            return "문서에서 관련 내용을 찾을 수 없습니다."
    elif fast_mode:
        print("[빠른 모드] 간단한 키워드 기반 필터링 사용")
        # 빠른 모드: 키워드 기반 간단한 필터링
        keywords = extract_keywords_from_text(query, max_keywords=5)
        scored_docs = []
        for doc in retrieved_small_docs:
            keyword_score = sum(1 for kw in keywords if kw.lower() in doc.page_content.lower())
            scored_docs.append((doc, keyword_score))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in scored_docs[:2]]  # 상위 2개만 선택
    else:
        print("[일반 모드] 고품질 처리를 위한 전체 파이프라인 실행")
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
            return generate_fast_answer("", query) if fast_mode else generate_final_answer("", query)

        print("[2/5] 재순위화(CrossEncoder) 진행 중...")
        reranked_small_docs = rerank_documents(query, rerank_candidates, reranker_model, rerank_n)
        print(f"[2/5] 재순위화 완료 (선택된 small chunk 수: {len(reranked_small_docs)})")
        top_docs = reranked_small_docs

    print("[3/5] parent large chunk 추출 중...")
    parent_large_ids = {doc.metadata["parent_large_id"] for doc in top_docs if "parent_large_id" in doc.metadata}
    print(f"[3/5] parent large chunk 추출 완료 (대상 large chunk 수: {len(parent_large_ids)})")

    if ultra_fast_mode:
        print("[초고속 모드] 원본 텍스트 직접 사용")
        # 초고속 모드: 상위 2개 large chunk 사용 (추론 정확성 향상)
        contexts = []
        for large_id in list(parent_large_ids)[:2]:  # 상위 2개만 사용
            if large_id in large_chunk_dict:
                # 더 많은 컨텍스트 제공 (800 → 1200자)
                context_text = large_chunk_dict[large_id].page_content[:1200]
                contexts.append(context_text)
        
        context = "\n---\n".join(contexts)
        print(f"[4/5] 초고속 컨텍스트 추출 완료 (컨텍스트 블록 수: {len(contexts)})")
    elif fast_mode:
        print("[빠른 모드] 간단한 컨텍스트 압축 사용")
        # 빠른 모드: 간단한 컨텍스트 압축
        compressed_contexts = []
        for large_id in parent_large_ids:
            if large_id in large_chunk_dict:
                large_doc = large_chunk_dict[large_id]
                # 간단한 키워드 기반 필터링
                keywords = extract_keywords_from_text(query, max_keywords=5)
                sentences = re.split(r'[.!?]+', large_doc.page_content)
                relevant_sentences = []
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) < 10:
                        continue
                    if any(kw.lower() in sentence.lower() for kw in keywords):
                        relevant_sentences.append(sentence)
                
                if relevant_sentences:
                    compressed_contexts.append('. '.join(relevant_sentences[:3]))  # 최대 3문장
        
        context = "\n---\n".join(compressed_contexts)
        print(f"[4/5] 빠른 컨텍스트 압축 완료 (압축된 context 블록 수: {len(compressed_contexts)})")
    else:
        print("[4/5] context 압축(핵심 문장 추출) 진행 중...")
        
        if use_adaptive and use_parallel and len(parent_large_ids) > 1:
            print(f"    - 적응형 병렬 처리 모드 활성화 (워커 수: {MAX_WORKERS})")
            compressed_results = compress_document_context_parallel_adaptive(query, large_chunk_dict, parent_large_ids)
            compressed_contexts = [result[0] for result in compressed_results if result[0]]
            avg_quality = sum(result[1] for result in compressed_results) / len(compressed_results) if compressed_results else 0.0
            print(f"    - 평균 컨텍스트 품질: {avg_quality:.2f}")
        elif use_parallel and len(parent_large_ids) > 1:
            print(f"    - 병렬 처리 모드 활성화 (워커 수: {MAX_WORKERS})")
            compressed_contexts = compress_document_context_parallel(query, large_chunk_dict, parent_large_ids)
            avg_quality = 0.8  # 기본 품질 점수
        else:
            print("    - 순차 처리 모드")
            compressed_contexts = []
            total_quality = 0.0
            for idx, large_id in enumerate(parent_large_ids):
                if large_id in large_chunk_dict:
                    print(f"    - [{idx+1}/{len(parent_large_ids)}] large chunk({large_id}) 압축 중...")
                    large_doc = large_chunk_dict[large_id]
                    if use_adaptive:
                        compressed, quality = adaptive_compress_context(query, large_doc.page_content, llm_cheap)
                        total_quality += quality
                    else:
                        compressed = compress_document_context(query, large_doc.page_content, llm_cheap)
                        quality = 0.8
                    
                    if compressed.strip() and "관련 내용 없음" not in compressed:
                        compressed_contexts.append(compressed)
                        print(f"        [압축 결과] 품질: {quality:.2f}\n{compressed}\n")
                    else:
                        print(f"        [압축 결과 없음 또는 관련 내용 없음]")
                    print(f"    - [{idx+1}/{len(parent_large_ids)}] large chunk({large_id}) 압축 완료")
            
            avg_quality = total_quality / len(parent_large_ids) if parent_large_ids else 0.0
        
        context = "\n---\n".join(compressed_contexts)
        print(f"[4/5] context 압축 완료 (압축된 context 블록 수: {len(compressed_contexts)})")

    print("[5/5] 최종 답변 생성 중...")
    if ultra_fast_mode:
        answer = generate_enhanced_reasoning_answer(context, query)  # 향상된 추론 함수 사용
    elif fast_mode:
        answer = generate_fast_answer(context, query)
    elif use_adaptive:
        answer = generate_adaptive_final_answer(context, query, avg_quality)
    else:
        answer = generate_final_answer(context, query)
    print("[5/5] 최종 답변 생성 완료!")
    return answer

def safe_llm_invoke(chain, inputs: dict, max_retries: int = 3) -> str:
    """
    안전한 LLM 호출을 위한 래퍼 함수
    연결 오류 시 재시도 및 에러 처리를 포함합니다.
    """
    for attempt in range(max_retries):
        try:
            result = chain.invoke(inputs)
            return result['text'].strip()
        except Exception as e:
            error_msg = str(e).lower()
            if attempt < max_retries - 1:
                if "connection" in error_msg or "timeout" in error_msg or "read timed out" in error_msg:
                    print(f"    - LLM 호출 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                    print(f"    - 5초 후 재시도...")
                    time.sleep(5)
                    continue
                else:
                    print(f"    - LLM 호출 오류 (시도 {attempt + 1}/{max_retries}): {e}")
                    time.sleep(2)
                    continue
            else:
                print(f"    - 최대 재시도 횟수 초과: {e}")
                return ""
    
    return ""

def generate_enhanced_reasoning_answer(context: str, query: str) -> str:
    """
    향상된 추론 능력을 위한 Few-shot 프롬프팅 기반 답변 생성 함수
    """
    # 컨텍스트 검증
    if not context or not context.strip() or context.strip() == "관련 내용 없음":
        return "문서에서 관련 내용을 찾을 수 없습니다."
    
    # Few-shot 프롬프팅을 사용한 향상된 프롬프트
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""다음 예시를 참고하여 문서를 바탕으로 질문에 정확하고 논리적으로 답변하세요.

예시 1:
문서: 프로젝트 진행률이 65%이고, 주요 마일스톤 중 3개가 완료되었으며, 팀원들의 만족도가 높습니다.
질문: 이 프로젝트의 성공 가능성은 어떻게 평가됩니까?
답변: 문서에 따르면 이 프로젝트는 성공 가능성이 높습니다. 진행률이 65%로 중간 이상이며, 주요 마일스톤 3개가 완료되어 중요한 단계들을 성공적으로 통과했고, 팀원들의 만족도가 높아 팀워크가 원활함을 시사합니다.

예시 2:
문서: 예산 초과 15%, 일정 지연 2주, 고객 만족도 하락이 발생했습니다.
질문: 이 프로젝트의 위험 요소는 무엇입니까?
답변: 문서에 따르면 이 프로젝트는 여러 위험 요소를 가지고 있습니다. 예산 초과 15%는 재정적 부담을, 일정 지연 2주는 시간적 압박을, 고객 만족도 하락은 품질 문제를 시사합니다.

문서 내용:
{context}

질문: {question}

답변: 문서에 따르면"""
    )
    
    # 먼저 작은 모델로 시도
    try:
        chain = LLMChain(llm=llm_cheap, prompt=prompt)
        result = chain.invoke({"context": context, "question": query})
        answer = result['text'].strip()
        
        # 답변 품질 검증
        if not answer or len(answer) < 15:
            return "문서에서 명확한 답변을 찾을 수 없습니다."
        
        return answer
        
    except Exception as e:
        print(f"향상된 추론 답변 생성 중 오류 (작은 모델): {e}")
        # 오류 시 기본 함수로 폴백
        return generate_fast_answer(context, query)

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
        query = "'퀀텀 데이터베이스 소프트웨어' 유지보수 계약이 만료된 후에 재계약하려면 어떻게 해야 해?"
        
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
            rerank_n=3,
            use_parallel=True,      # 병렬 처리 활성화
            use_adaptive=True,      # 적응형 처리 활성화
            fast_mode=False,        # 빠른 모드 비활성화
            ultra_fast_mode=True,    # 초고속 모드 활성화 ← 현재 선택됨
            use_intelligent_mode=True  # 지능형 모드 선택 활성화
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
