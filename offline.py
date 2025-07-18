import os
import glob
import hashlib
import pickle
from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# --- 상수 정의 ---
CHROMA_DB_PATH = "chroma_db_bge_m3"  # project 폴더 기준 상위에 DB 생성
HASH_LIST_FILE = f"{CHROMA_DB_PATH}_hashes.txt"
LARGE_CHUNK_PICKLE = f"{CHROMA_DB_PATH}_large_chunks.pkl"
L_MAX = 4096  # large chunk size
S_MAX = 500   # small chunk size
CHUNK_OVERLAP = 256
SMALL_OVERLAP = 50

# --- 임베딩 모델 (필요시 경로/옵션 수정) ---
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

def get_document_text(file_path: str) -> str:
    """지정된 경로의 파일을 읽어 내용을 반환합니다."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"--- 외부 파일 읽기 실패: {e} ---")
            return ""
    print(f"--- 지정된 파일 경로를 찾을 수 없습니다: {file_path} ---")
    return ""

def hybrid_split_document(
    text: str, file_path: str
) -> Tuple[List[Document], List[Document], Dict[str, Document]]:
    """
    문서를 large/small chunk로 분할하고, small chunk → large chunk 매핑 정보를 반환합니다.
    Returns:
        small_chunks: Small chunk Document 리스트 (parent_large_id 포함)
        large_chunks: Large chunk Document 리스트 (large_id 포함)
        large_chunk_dict: large_id → large chunk Document dict
    """
    # 1. Large chunk 분할
    large_splitter = RecursiveCharacterTextSplitter(
        chunk_size=L_MAX, chunk_overlap=CHUNK_OVERLAP
    )
    large_chunks = large_splitter.create_documents([text])
    small_chunks = []
    large_chunk_dict = {}
    for i, large_doc in enumerate(large_chunks):
        large_id = f"{os.path.basename(file_path)}_large_{i}"
        large_doc.metadata["large_id"] = large_id
        large_doc.metadata["source_file"] = file_path
        large_chunk_dict[large_id] = large_doc
        # 2. 각 large chunk 내에서 small chunk 분할
        small_splitter = RecursiveCharacterTextSplitter(
            chunk_size=S_MAX, chunk_overlap=SMALL_OVERLAP
        )
        sub_small_chunks = small_splitter.create_documents([large_doc.page_content])
        for j, small_doc in enumerate(sub_small_chunks):
            small_id = f"{os.path.basename(file_path)}_small_{i}_{j}"
            small_doc.metadata["parent_large_id"] = large_id
            small_doc.metadata["small_id"] = small_id
            small_doc.metadata["source_file"] = file_path
            small_chunks.append(small_doc)
    return small_chunks, list(large_chunk_dict.values()), large_chunk_dict

def build_hybrid_vector_store_for_folder(folder_path: str):
    """
    폴더 내 모든 .md 파일을 하이브리드 방식으로 chunking하여 Chroma DB(small chunk)와 large chunk pickle을 생성합니다.
    """
    md_files = glob.glob(os.path.join(folder_path, "*.md"))
    if not md_files:
        print(f"--- 폴더 내에 md 파일이 없습니다: {folder_path} ---")
        return
    print(f"--- 폴더 내 md 파일 {len(md_files)}개를 순차적으로 적재합니다 ---")
    hashes = {}
    if os.path.exists(HASH_LIST_FILE):
        with open(HASH_LIST_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if ':' in line:
                    path, h = line.strip().split(":", 1)
                    hashes[path] = h
    all_small_chunks = []
    all_large_chunk_dict = {}
    for file_path in md_files:
        print(f"[적재 시작] {file_path}")
        document_text = get_document_text(file_path)
        if not document_text:
            print(f"--- 문서를 찾을 수 없거나 내용이 없습니다: {file_path} ---")
            continue
        current_hash = hashlib.md5(document_text.encode()).hexdigest()
        if file_path in hashes and hashes[file_path] == current_hash:
            print(f"--- 이미 적재된 파일입니다(중복 방지): {file_path} ---")
            continue
        # 하이브리드 chunk 분할
        small_chunks, large_chunks, large_chunk_dict = hybrid_split_document(document_text, file_path)
        all_small_chunks.extend(small_chunks)
        all_large_chunk_dict.update(large_chunk_dict)
        hashes[file_path] = current_hash
        print(f"--- 문서가 하이브리드 chunk로 분할되었습니다: {file_path} ---")
        print(f"    large chunk: {len(large_chunks)}개, small chunk: {len(small_chunks)}개")
        print(f"[적재 완료] {file_path}\n")
    # Chroma DB에 small chunk 저장
    if all_small_chunks:
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )
        vectorstore.add_documents(all_small_chunks)
        print(f"--- 전체 small chunk {len(all_small_chunks)}개가 Chroma DB에 저장되었습니다 ---")
    # large chunk pickle 저장
    with open(LARGE_CHUNK_PICKLE, 'wb') as f:
        pickle.dump(all_large_chunk_dict, f)
    print(f"--- 전체 large chunk dict가 pickle로 저장되었습니다: {LARGE_CHUNK_PICKLE} ---")
    # 해시 파일 저장
    with open(HASH_LIST_FILE, 'w', encoding='utf-8') as f:
        for path, h in hashes.items():
            f.write(f"{path}:{h}\n")

if __name__ == "__main__":
    # 사용 예시: project 폴더 내에서 실행
    FOLDER_PATH = r"C:\Users\osh74\Downloads\testcase_script"
    build_hybrid_vector_store_for_folder(FOLDER_PATH) 
