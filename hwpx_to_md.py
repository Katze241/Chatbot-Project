"""
hwpx_to_md.py

HWPX(한글 XML) 문서를 Markdown(.md)으로 변환하는 스크립트 (HWPX 전용 버전)
- HWPX XML 구조에 최적화
- 올바른 네임스페이스 사용
- 개별 섹션 파일 처리
- 셀 병합 처리 개선
- 빈 행(empty rows) 적절히 표시
- 문단 구조 인식

Usage:
    python hwpx_to_md.py input.hwpx output.md

Requirements:
    - Python 3.7+ (no external dependencies)
"""
import os
import sys
import zipfile
import argparse
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple
import re

# 실제 HWPX 네임스페이스
NAMESPACES = {
    'hp': 'http://www.hancom.co.kr/hwpml/2011/paragraph',
    'hs': 'http://www.hancom.co.kr/hwpml/2011/section'
}


def safe_spacing(text: str) -> str:
    """줄바꿈 제거 → 다중 공백 축소"""
    if not text or not text.strip():
        return ""
    # 줄바꿈을 공백으로 변환하고 여러 공백을 하나로 축소
    return re.sub(r'\s+', ' ', text.strip())


def clean_cell_text(text: str) -> str:
    """셀 텍스트 정리 - 빈 셀 처리 개선"""
    if not text:
        return "&nbsp;"
    
    cleaned = safe_spacing(text)
    if not cleaned or cleaned.lower() == 'none':
        return "&nbsp;"
    
    return cleaned


def extract_sections_from_hwpx(hwpx_path: str) -> List[ET.Element]:
    """ZIP 형식 .hwpx에서 각 섹션 XML을 개별적으로 파싱하여 Element 리스트 반환"""
    sections = []
    with zipfile.ZipFile(hwpx_path, 'r') as z:
        section_files = [name for name in z.namelist() 
                        if name.startswith("Contents/section") and name.endswith('.xml')]
        # 섹션 파일들을 번호순으로 정렬 (문자열 정렬이 아닌 숫자 정렬)
        section_files.sort(key=lambda x: int(x.split('section')[1].split('.')[0]))
        
        for section_file in section_files:
            try:
                xml_content = z.read(section_file).decode('utf-8')
                root = ET.fromstring(xml_content)
                sections.append(root)
            except ET.ParseError as e:
                print(f"Warning: 섹션 {section_file} 파싱 오류: {e}")
            except Exception as e:
                print(f"Warning: 섹션 {section_file} 처리 오류: {e}")
    
    print(f"총 {len(sections)}개 섹션을 로드했습니다.")
    return sections


def extract_table_data(table_elem: ET.Element) -> List[List[str]]:
    """표 요소에서 데이터를 추출하여 2차원 리스트로 반환 (강화된 셀 병합 처리)"""
    # 표의 행들을 찾기 (더 강력한 방식)
    rows = []
    for row_elem in table_elem.iter():
        row_tag = row_elem.tag
        if '}' in row_tag:
            row_tag = row_tag.split('}')[1]
        if row_tag.lower() in ['tr', 'row']:
            rows.append(row_elem)
    
    if not rows:
        return []
    
    # 표의 전체 구조를 먼저 파악
    max_cols = 0
    for row in rows:
        col_count = 0
        for cell_elem in row.iter():
            cell_tag = cell_elem.tag
            if '}' in cell_tag:
                cell_tag = cell_tag.split('}')[1]
            if cell_tag.lower() in ['tc', 'cell']:
                # cellSpan 태그에서 colSpan 찾기 (더 강력한 탐색)
                col_span = 1
                for child in cell_elem.iter():
                    child_tag = child.tag
                    if '}' in child_tag:
                        child_tag = child_tag.split('}')[1]
                    if child_tag.lower() == 'cellspan':
                        col_span = int(child.get('colSpan', '1'))
                        break
                col_count += col_span
        max_cols = max(max_cols, col_count)
    
    # 2차원 배열로 표 데이터 채우기
    n_rows = len(rows)
    grid: List[List[Optional[str]]] = [[None for _ in range(max_cols)] for _ in range(n_rows)]
    
    # 각 행의 셀들을 처리
    for row_idx, row in enumerate(rows):
        col_idx = 0
        for cell_elem in row.iter():
            cell_tag = cell_elem.tag
            if '}' in cell_tag:
                cell_tag = cell_tag.split('}')[1]
            
            if cell_tag.lower() in ['tc', 'cell']:
                # 다음 빈 칸 찾기 (이미 채워진 칸은 건너뛰기)
                while col_idx < max_cols and grid[row_idx][col_idx] is not None:
                    col_idx += 1
                
                # cellSpan 태그에서 colSpan과 rowSpan 찾기 (강화된 탐색)
                col_span = 1
                row_span = 1
                for child in cell_elem.iter():
                    child_tag = child.tag
                    if '}' in child_tag:
                        child_tag = child_tag.split('}')[1]
                    if child_tag.lower() == 'cellspan':
                        col_span = int(child.get('colSpan', '1'))
                        row_span = int(child.get('rowSpan', '1'))
                        break
                
                # 셀 텍스트 추출
                cell_text = extract_cell_text(cell_elem)
                
                # 셀을 grid에 채우기 (병합된 셀의 모든 위치에 동일한 텍스트 복사)
                for dr in range(row_span):
                    for dc in range(col_span):
                        r, c = row_idx + dr, col_idx + dc
                        if r < n_rows and c < max_cols:
                            # 병합된 셀의 모든 위치에 동일한 텍스트 복사
                            grid[r][c] = cell_text if cell_text else ""
                
                col_idx += col_span
    
    # None 값을 빈 문자열로 변환
    result = []
    for row in grid:
        result_row = []
        for cell in row:
            if cell is None:
                result_row.append("&nbsp;")
            else:
                result_row.append(clean_cell_text(cell))
        result.append(result_row)
    
    return result


def extract_cell_text(cell_elem: ET.Element) -> str:
    """셀 요소에서 텍스트 추출"""
    texts = []
    
    # 모든 텍스트 요소 찾기
    for text_elem in cell_elem.findall('.//hp:t', NAMESPACES):
        if text_elem.text:
            texts.append(text_elem.text.strip())
    
    # 텍스트가 없으면 빈 문자열 반환
    if not texts:
        return ""
    
    # 여러 텍스트를 공백으로 연결
    return ' '.join(texts)


def format_table_as_markdown(table_data: List[List[str]], table_num: int = 0) -> str:
    """표 데이터를 마크다운 형식으로 변환"""
    if not table_data:
        return ""
    
    lines = []
    
    # 테이블 제목 추가
    if table_num > 0:
        lines.append(f"### 표 {table_num} ({len(table_data)}행 x {len(table_data[0]) if table_data else 0}열)")
        lines.append("")
    
    # 테이블 헤더
    if table_data:
        header_row = table_data[0]
        lines.append("| " + " | ".join(header_row) + " |")
        
        # 구분선
        separator = "| " + " | ".join(["---"] * len(header_row)) + " |"
        lines.append(separator)
        
        # 데이터 행들
        for row in table_data[1:]:
            # 빈 행도 포함 (모든 셀이 &nbsp;이어도 표시)
            lines.append("| " + " | ".join(row) + " |")
    
    lines.append("")  # 테이블 후 빈 줄
    return '\n'.join(lines)


def extract_paragraph_text(elem: ET.Element) -> str:
    """문단 요소에서 텍스트 추출"""
    texts = []
    for text_elem in elem.findall('.//hp:t', NAMESPACES):
        if text_elem.text:
            texts.append(text_elem.text.strip())
    
    result = ' '.join(texts)
    return safe_spacing(result)


def process_section_in_order(section: ET.Element, table_counter: Dict[str, int]) -> List[str]:
    """섹션을 순서대로 처리하여 원본 위치에 표 삽입"""
    result_lines = []
    processed_tables = set()  # 이미 처리된 테이블 추적
    
    def get_element_id(elem: ET.Element) -> str:
        """요소의 고유 ID 생성"""
        return f"{elem.tag}_{id(elem)}"
    
    def process_elements_sequentially(parent_elem: ET.Element) -> None:
        """요소들을 순차적으로 처리 (중복 제거 로직 포함)"""
        for child in parent_elem:
            tag = child.tag
            if '}' in tag:
                tag = tag.split('}')[1]
            
            if tag.lower() == 'p':
                # 문단 내 표들 먼저 확인
                tables_in_paragraph = child.findall('.//hp:tbl', NAMESPACES)
                if tables_in_paragraph:
                    # 표가 있는 문단: 표만 처리하고 텍스트는 추출하지 않음 (중복 방지)
                    for table_elem in tables_in_paragraph:
                        table_id = get_element_id(table_elem)
                        if table_id not in processed_tables:
                            table_data = extract_table_data(table_elem)
                            if table_data:
                                table_counter['count'] += 1
                                table_md = format_table_as_markdown(table_data, table_counter['count'])
                                if table_md.strip():
                                    result_lines.append(table_md)
                            processed_tables.add(table_id)
                else:
                    # 표가 없는 문단만 텍스트로 추출
                    para_text = extract_paragraph_text(child)
                    if para_text and para_text.strip() and len(para_text.strip()) > 2:
                        # 의미 있는 텍스트만 추가 (2글자 이상)
                        result_lines.append(para_text)
                        result_lines.append("")  # 문단 간 빈 줄
                
                # 하위 요소들도 재귀적으로 처리
                process_elements_sequentially(child)
            
            elif tag.lower() == 'tbl':
                # 독립적인 표 처리 (중복 방지)
                table_id = get_element_id(child)
                if table_id not in processed_tables:
                    table_data = extract_table_data(child)
                    if table_data:
                        table_counter['count'] += 1
                        table_md = format_table_as_markdown(table_data, table_counter['count'])
                        if table_md.strip():
                            result_lines.append(table_md)
                    processed_tables.add(table_id)
            
            else:
                # 기타 요소들 재귀 처리
                process_elements_sequentially(child)
    
    # 섹션의 모든 하위 요소들을 순차 처리
    process_elements_sequentially(section)
    
    return result_lines


def process_section(section: ET.Element) -> Tuple[List[str], List[List[List[str]]]]:
    """섹션에서 문단과 표를 추출 (레거시 함수 - 하위 호환성)"""
    paragraphs = []
    tables = []
    
    # 문단 추출
    for para in section.findall('.//hp:p', NAMESPACES):
        para_text = extract_paragraph_text(para)
        if para_text.strip():
            paragraphs.append(para_text)
    
    # 표 추출
    for table in section.findall('.//hp:tbl', NAMESPACES):
        table_data = extract_table_data(table)
        if table_data:
            tables.append(table_data)
    
    return paragraphs, tables


def convert_to_markdown(hwpx_path: str) -> str:
    """HWPX 파일을 마크다운으로 변환"""
    sections = extract_sections_from_hwpx(hwpx_path)
    
    md_lines = []
    table_counter = {'count': 0}
    
    for section in sections:
        section_content = process_section_in_order(section, table_counter)
        md_lines.extend(section_content)
    
    return '\n'.join(md_lines)


def convert_hwpx_to_markdown(input_path: str, output_path: str):
    """HWPX 파일을 마크다운으로 변환"""
    ext = os.path.splitext(input_path)[1].lower()
    
    if ext not in ['.hwpx', '.zip']:
        raise ValueError("HWPX 파일(.hwpx 또는 .zip)만 지원합니다.")

    md_content = convert_to_markdown(input_path)

    # 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"변환 완료: {input_path} → {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='HWPX to Markdown 변환기 (HWPX 전용)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python hwpx_to_md.py document.hwpx output.md
  python hwpx_to_md.py document.zip output.md

주요 특징:
  - HWPX 파일 전용 (HWP 변환 불필요)
  - 빈 행(all "none" cells) 적절히 표시
  - 셀 병합 처리 개선
  - 표 구조 완전성 유지
  - 외부 라이브러리 의존성 없음
        """
    )
    parser.add_argument('input', help='입력 .hwpx 또는 .zip 파일')
    parser.add_argument('output', help='출력 .md 파일')
    
    args = parser.parse_args()
    
    try:
        convert_hwpx_to_markdown(args.input, args.output)
    except Exception as e:
        print(f"오류 발생: {e}")
        sys.exit(1) 
