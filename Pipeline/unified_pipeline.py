# unified_pipeline.py
import os
import time
import numpy as np
from typing import List, Dict, Set
import re
import hashlib
import shutil
import glob
import pickle
import chromadb

# LangChain 및 관련 라이브러리 임포트
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.retrievers import MultiQueryRetriever
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# 기타 유틸리티 라이브러리
import tiktoken
from sentence_transformers.cross_encoder import CrossEncoder
from konlpy.tag import Komoran

# --- 1. 초기 설정 및 모델 로드 ---

# 임베딩 모델
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
# 재순위화 모델
reranker_model = CrossEncoder('BAAI/bge-reranker-large', device='cpu')

# LLM 정의 (Ollama 사용)
llm_cheap = Ollama(model="qwen2:7b")
llm_powerful = Ollama(model="qwen2:7b")

# --- 상수 정의 ---
DB_NAME = "hybrid_vector_db"
CHROMA_DB_PATH = f"./{DB_NAME}"
LARGE_CHUNK_PICKLE = f"{CHROMA_DB_PATH}_large_chunks.pkl"
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

def extract_proper_nouns(text: str, max_keywords: int = 5) -> list[str]:
    komoran = Komoran()
    tagged = komoran.pos(text)
    proper_nouns = [word for word, pos in tagged if pos == 'NNP']
    proper_nouns = [str(word) for word in proper_nouns]
    return list(dict.fromkeys(proper_nouns))[:max_keywords]

# --- 공통 함수들 ---
def rerank_documents(query: str, retrieved_docs: List[Document], reranker_model, rerank_n: int = 2) -> List[Document]:
    if not retrieved_docs:
        return []
    pairs = [(query, doc.page_content) for doc in retrieved_docs]
    scores = reranker_model.predict(pairs)
    doc_score_pairs = list(zip(retrieved_docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in doc_score_pairs[:rerank_n]]

def generate_final_answer_simple(context: str, query: str) -> str:
    """단순 파이프라인용 답변 생성 함수"""
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
            
            질문에 대한 정답만 출력하세요.
            설명, 이유, 불필요한 문장은 절대 포함하지 마세요.
            출력에는 정답을 명확히 명시한 문장으로 작성하세요.

[문서 내용]
{context}

[사용자 질문]
{question}

[답변]"""
        )
        chain = LLMChain(llm=llm_powerful, prompt=prompt)
        result = chain.invoke({"context": context, "question": query})
        return result['text']

def generate_final_answer_complex(context: str, query: str) -> str:
    """복잡 파이프라인용 답변 생성 함수"""
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
        chain = LLMChain(llm=llm_powerful, prompt=prompt)
        result = chain.invoke({"context": context, "question": query})
        return result['text']

def compress_document_context(query: str, context: str, llm_cheap) -> str:
    """복잡 파이프라인용 문서 압축 함수"""
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""
        You are a helpful AI assistant that ALWAYS responds in Korean. 당신은 항상 한국어로 답변하는 AI 어시스턴트입니다.
        [역할]
    당신은 문맥(Context)에서 특정 질문(Question)에 대한 답변의 근거가 되는 문장을 **원문 그대로** 찾아내는 고정밀 텍스트 추출 엔진입니다. 당신은 아래 [예시]와 [절대 규칙]에 따라 [실제 작업]을 수행해야 합니다.

    [절대 규칙]
    1.  **요약 금지:** 절대 문장을 요약하거나 자신의 언어로 재작성하지 마세요.
    2.  **변경 금지:** 단어 하나도 추가하거나 빼거나 바꾸지 마세요.
    3.  **의견 금지:** 당신의 의견, 부연 설명, 인사말 등 다른 어떤 텍스트도 포함하지 마세요.
    4.  **정보 부재 시:** 만약 [문맥]에 [질문]에 답할 정보가 전혀 없다면, 다른 말 없이 정확히 **[관련 내용 없음]** 이라고만 출력해야 합니다.
    ---

    [실제 작업]
    [문맥]
    {context}

    [질문]
    {question}

    [추출된 문장]
        """
    )
    chain = LLMChain(llm=llm_cheap, prompt=prompt)
    result = chain.invoke({"context": context, "question": query})
    return result['text']

# --- 파이프라인 함수들 ---
def simple_pipeline(
    query: str,
    retrieved_small_docs: List[Document], 
    reranker_model,
    llm_powerful,
    top_k: int = 10,
    rerank_n: int = 3
) -> str:
    """단순 파이프라인 (small_non_compression.py 기반)"""
    print(f"[1/3] 1차 검색 완료 (통합 검색된 small chunk 수: {len(retrieved_small_docs)})")

    keywords = extract_proper_nouns(query, max_keywords=5)
    print(f"[고유명사 추출] 쿼리에서 추출된 고유명사: {keywords}")

    keyword_weight = 0.5
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
        return generate_final_answer_simple("", query)

    print("[2/3] 재순위화(CrossEncoder) 진행 중...")
    reranked_small_docs = rerank_documents(query, rerank_candidates, reranker_model, rerank_n)
    print(f"[2/3] 재순위화 완료 (선택된 small chunk 수: {len(reranked_small_docs)})")

    context = "\n---\n".join([doc.page_content for doc in reranked_small_docs])
    print(f"[Context 생성] 재순위화된 {len(reranked_small_docs)}개의 chunk로 context 구성")

    print("[3/3] 최종 답변 생성 중...")
    answer = generate_final_answer_simple(context, query)
    print("[3/3] 최종 답변 생성 완료!")
    return answer

def compress_documents_batch(query: str, contexts: List[str], llm_cheap) -> List[str]:
    """
    LLM의 batch 기능을 이용해 여러 large chunk를 한 번에 압축합니다.
    """
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""
        You are a helpful AI assistant that ALWAYS responds in Korean. 당신은 항상 한국어로 답변하는 AI 어시스턴트입니다.
        [역할]
    당신은 문맥(Context)에서 특정 질문(Question)에 대한 답변의 근거가 되는 문장을 **원문 그대로** 찾아내는 고정밀 텍스트 추출 엔진입니다. 당신은 아래 [예시]와 [절대 규칙]에 따라 [실제 작업]을 수행해야 합니다.

    [절대 규칙]
    1.  **요약 금지:** 절대 문장을 요약하거나 자신의 언어로 재작성하지 마세요.
    2.  **변경 금지:** 단어 하나도 추가하거나 빼거나 바꾸지 마세요.
    3.  **의견 금지:** 당신의 의견, 부연 설명, 인사말 등 다른 어떤 텍스트도 포함하지 마세요.
    4.  **정보 부재 시:** 만약 [문맥]에 [질문]에 답할 정보가 전혀 없다면, 다른 말 없이 정확히 **[관련 내용 없음]** 이라고만 출력해야 합니다.
    ---

    [실제 작업]
    [문맥]
    {context}

    [질문]
    {question}

    [추출된 문장]
        """
    )
    chain = LLMChain(llm=llm_cheap, prompt=prompt)
    
    # batch 처리를 위한 입력 데이터 구성
    batch_inputs = [{"context": ctx, "question": query} for ctx in contexts]
    
    # batch 실행
    results = chain.batch(batch_inputs)
    
    # 결과에서 'text'만 추출하여 리스트로 반환
    return [result['text'] for result in results]

def complex_pipeline(
    query: str,
    retrieved_small_docs: List[Document], 
    large_chunk_dict: Dict[str, Document],
    reranker_model,
    llm_cheap,
    llm_powerful,
    top_k: int = 10,
    rerank_n: int = 3,
    use_batch: bool = False
) -> str:
    """복잡 파이프라인 (online_pipeline_collection.py 기반) + 배치 처리 옵션"""
    print(f"[1/5] 1차 검색 완료 (통합 검색된 small chunk 수: {len(retrieved_small_docs)})")

    keywords = extract_proper_nouns(query, max_keywords=5)
    print(f"[고유명사 추출] 쿼리에서 추출된 고유명사: {keywords}")

    keyword_weight = 0.5
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
        return generate_final_answer_complex("", query)

    print("[2/5] 재순위화(CrossEncoder) 진행 중...")
    reranked_small_docs = rerank_documents(query, rerank_candidates, reranker_model, rerank_n)
    print(f"[2/5] 재순위화 완료 (선택된 small chunk 수: {len(reranked_small_docs)})")

    print("[3/5] parent large chunk 추출 중...")
    parent_large_ids = {doc.metadata["parent_large_id"] for doc in reranked_small_docs if "parent_large_id" in doc.metadata}
    print(f"[3/5] parent large chunk 추출 완료 (대상 large chunk 수: {len(parent_large_ids)})")

    print("[4/5] context 압축(핵심 문장 추출) 진행 중...")
    
    if use_batch and len(parent_large_ids) > 1:
        # 배치 처리 모드
        print(f"    - 배치 처리 모드: {len(parent_large_ids)}개의 large chunk를 한 번에 압축합니다...")
        contexts_to_compress = []
        for large_id in parent_large_ids:
            if large_id in large_chunk_dict:
                contexts_to_compress.append(large_chunk_dict[large_id].page_content)
        
        if contexts_to_compress:
            batch_results = compress_documents_batch(query, contexts_to_compress, llm_cheap)
            
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
        # 개별 처리 모드 (기존 방식)
        print(f"    - 개별 처리 모드: {len(parent_large_ids)}개의 large chunk를 순차적으로 압축합니다...")
        compressed_contexts = []
        for idx, large_id in enumerate(parent_large_ids):
            if large_id in large_chunk_dict:
                print(f"      - [{idx+1}/{len(parent_large_ids)}] large chunk({large_id}) 압축 중...")
                large_doc = large_chunk_dict[large_id]
                compressed = compress_document_context(query, large_doc.page_content, llm_cheap)
                if compressed.strip() and "관련 내용 없음" not in compressed:
                    compressed_contexts.append(compressed)
                    print(f"        [압축 결과]\n{compressed}\n")
                else:
                    print(f"        [압축 결과 없음 또는 관련 내용 없음]")
                print(f"      - [{idx+1}/{len(parent_large_ids)}] large chunk({large_id}) 압축 완료")
    
    context = "\n---\n".join(compressed_contexts)
    print(f"[4/5] context 압축 완료 (압축된 context 블록 수: {len(compressed_contexts)})")

    print("[5/5] 최종 답변 생성 중...")
    answer = generate_final_answer_complex(context, query)
    print("[5/5] 최종 답변 생성 완료!")
    return answer

def print_pipeline_menu():
    """파이프라인 선택 메뉴 출력"""
    print("\n" + "="*60)
    print("RAG 파이프라인 선택")
    print("="*60)
    print("1️. 단순 파이프라인 (빠른 응답)")
    print("   - 3단계 처리")
    print("   - Small chunk 내용 직접 사용")
    print("   - 빠른 응답 속도")
    print("   - 기본적인 답변 품질")
    print()
    print("2️. 복잡 파이프라인 (정확한 답변)")
    print("   - 5단계 처리")
    print("   - Large chunk 압축 및 핵심 문장 추출")
    print("   - 상대적으로 느린 응답 속도")
    print("   - 더 정확하고 집중된 답변")
    print("="*60)

def print_compression_menu():
    """압축 방식 선택 메뉴 출력"""
    print("\n" + "-"*50)
    print("압축 처리 방식 선택")
    print("-"*50)
    print("1️. 개별 처리 (안정적)")
    print("   - 각 large chunk를 순차적으로 처리")
    print("   - 메모리 사용량 적음")
    print("   - 안정적인 처리")
    print()
    print("2️. 배치 처리 (빠름)")
    print("   - 여러 large chunk를 한 번에 처리")
    print("   - 처리 속도 향상")
    print("   - 메모리 사용량 증가")
    print("-"*50)

def get_user_choice() -> int:
    """사용자 선택 입력 받기"""
    while True:
        try:
            choice = int(input("\n파이프라인을 선택하세요 (1 또는 2): "))
            if choice in [1, 2]:
                return choice
            else:
                print("1 또는 2만 입력해주세요.")
        except ValueError:
            print("숫자를 입력해주세요.")

def get_user_query() -> str:
    """사용자 질문 설정 (코드 내에서 직접 설정)"""
    # 여기서 질문을 직접 설정하세요
    query = "테러가 의심되는 '코드 블랙' 상황에 대응하기 위해 매뉴얼이 개정되었다고 하는데, 언제 누가 개정한 건가요?"
    
    print("\n" + "-"*50)
    print(f"설정된 질문: {query}")
    return query.strip()

# --- main 실행부 ---
if __name__ == "__main__":
    if not os.path.exists(CHROMA_DB_PATH) or not os.path.exists(LARGE_CHUNK_PICKLE):
        print(f"--- 에러: Chroma DB 또는 large chunk pickle 경로를 찾을 수 없습니다: '{CHROMA_DB_PATH}', '{LARGE_CHUNK_PICKLE}' ---")
        print("--- 먼저 offline_pipeline.py를 실행하여 DB를 구축하세요. ---")
        exit()

    # 1. DB 클라이언트 생성 및 모든 컬렉션 이름 가져오기
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection_names = [c.name for c in client.list_collections()]
    
    if not collection_names:
        print(f"--- 에러: DB에 컬렉션이 없습니다. offline_pipeline.py를 실행하여 DB를 구축하세요. ---")
        exit()
        
    print("통합 검색 대상 컬렉션 목록:")
    for name in collection_names:
        print(f"  - {name}")

    # 2. 파이프라인 선택
    print_pipeline_menu()
    pipeline_choice = get_user_choice()
    
    # 3. 질문 입력
    query = get_user_query()
    
    if not query:
        print("[오류] 질문을 입력해주세요.")
        exit()

    # 4. 파이프라인별 실행
    if pipeline_choice == 1:
        print("\n" + "-"*50)
        print("-- [단순 파이프라인] 하이브리드 RAG 파이프라인을 시작합니다.")
        print(f"   (질문: {query})")
        print("="*50 + "\n")
        
        start_time = time.time()
        
        # 모든 컬렉션에서 검색 수행
        print("[1/3] 1차 검색: 모든 컬렉션 대상으로 통합 벡터 검색 중...")
        all_retrieved_docs = []
        for name in collection_names:
            vectorstore = Chroma(
                client=client,
                collection_name=name,
                embedding_function=embedding_model
            )
            retrieved = vectorstore.similarity_search(query, k=10)
            all_retrieved_docs.extend(retrieved)
            print(f"  - '{name}' 컬렉션 검색 완료 (검색된 문서: {len(retrieved)}개)")

        # 중복 문서 제거
        unique_docs_dict = {doc.page_content: doc for doc in all_retrieved_docs}
        unique_retrieved_docs = list(unique_docs_dict.values())
        print(f"  - 중복 제거 후 총 {len(unique_retrieved_docs)}개의 후보 문서 확보")
        
        # 단순 파이프라인 실행
        final_answer = simple_pipeline(
            query=query,
            retrieved_small_docs=unique_retrieved_docs,
            reranker_model=reranker_model,
            llm_powerful=llm_powerful,
            top_k=10, 
            rerank_n=3
        )
        
    else:  # pipeline_choice == 2
        print("\n" + "-"*50)
        print("-- [복잡 파이프라인] 하이브리드 RAG 파이프라인을 시작합니다.")
        print(f"   (질문: {query})")
        print("="*50 + "\n")
        
        # 압축 방식 선택 (복잡 파이프라인에서만)
        print_compression_menu()
        compression_choice = get_user_choice()
        use_batch = (compression_choice == 2)
        
        start_time = time.time()
        
        # 모든 컬렉션에서 검색 수행
        print("[1/5] 1차 검색: 모든 컬렉션 대상으로 통합 벡터 검색 중...")
        all_retrieved_docs = []
        for name in collection_names:
            vectorstore = Chroma(
                client=client,
                collection_name=name,
                embedding_function=embedding_model
            )
            retrieved = vectorstore.similarity_search(query, k=10)
            all_retrieved_docs.extend(retrieved)
            print(f"  - '{name}' 컬렉션 검색 완료 (검색된 문서: {len(retrieved)}개)")

        # 중복 문서 제거
        unique_docs_dict = {doc.page_content: doc for doc in all_retrieved_docs}
        unique_retrieved_docs = list(unique_docs_dict.values())
        print(f"  - 중복 제거 후 총 {len(unique_retrieved_docs)}개의 후보 문서 확보")
        
        # large chunk pickle 로드
        with open(LARGE_CHUNK_PICKLE, 'rb') as f:
            large_chunk_dict: Dict[str, Document] = pickle.load(f)
        
        # 복잡 파이프라인 실행 (배치 처리 옵션 포함)
        final_answer = complex_pipeline(
            query=query,
            retrieved_small_docs=unique_retrieved_docs,
            large_chunk_dict=large_chunk_dict,
            reranker_model=reranker_model,
            llm_cheap=llm_cheap,
            llm_powerful=llm_powerful,
            top_k=10, 
            rerank_n=3,
            use_batch=use_batch
        )
    
    end_time = time.time()
    
    print("\n" + "-"*50)
    print("-- 최종 답변 --")
    print(final_answer)
    print("="*50)
    print(f"--- 총 실행 시간: {end_time - start_time:.2f}초---") 
