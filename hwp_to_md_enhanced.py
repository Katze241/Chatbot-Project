"""
hwp_to_md_enhanced.py

한글(.hwp/.hwpx) 문서를 Markdown(.md)으로 변환하는 스크립트 (빈 행 처리 개선 버전)
- 실제 HWPX XML 구조에 맞춰 수정
- 올바른 네임스페이스 사용
- 개별 섹션 파일 처리
- 셀 병합 처리 개선
- 빈 행(empty rows) 적절히 표시
- 문단 구조 인식

Usage:
    python hwp_to_md_enhanced.py input.hwp output.md
    python hwp_to_md_enhanced.py input.hwpx output.md

Requirements:
    pip install pyhwp
"""
import os
import sys
import zipfile
import argparse
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple
import re

try:
    from hwp5.hwp5 import HWP5File
    HAS_PYHWP = True
except ImportError:
    HAS_PYHWP = False

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
        return ""
    
    text = text.strip()
    
    # "none"은 빈 셀로 처리
    if text.lower() == "none":
        return ""
    
    # 줄바꿈과 탭을 공백으로 변환
    text = re.sub(r'[\r\n\t]+', ' ', text)
    # 연속된 공백을 하나로 축소
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


def extract_xml_from_hwp(hwp_path: str) -> str:
    """pyhwp를 이용해 .hwp 이진에서 HWPML(XML) 문자열을 반환"""
    if not HAS_PYHWP:
        raise RuntimeError("pyhwp 라이브러리가 필요합니다. pip install pyhwp")
    doc = HWP5File(hwp_path)
    return doc.restore_xml()


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
                continue
    return sections


def extract_table_data(table_elem: ET.Element) -> List[List[str]]:
    """표 요소에서 데이터를 추출하여 2차원 리스트로 반환 (셀 병합 처리 포함)"""
    # 표의 행들을 찾기
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
                # cellSpan 태그에서 colSpan 찾기
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
                
                # cellSpan 태그에서 colSpan과 rowSpan 찾기
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
                result_row.append("")
            else:
                result_row.append(clean_cell_text(cell))
        result.append(result_row)
    
    return result


def extract_cell_text(cell_elem: ET.Element) -> str:
    """셀에서 텍스트 추출"""
    texts = []
    
    # 셀 내부의 모든 텍스트 요소 찾기
    for text_elem in cell_elem.iter():
        text_tag = text_elem.tag
        if '}' in text_tag:
            text_tag = text_tag.split('}')[1]
        
        if text_tag.lower() in ['t', 'text']:
            if text_elem.text:
                texts.append(text_elem.text)
    
    # 텍스트 연결
    cell_text = ' '.join(texts)
    return safe_spacing(cell_text)


def format_table_as_markdown(table_data: List[List[str]], table_num: int = 0) -> str:
    """2차원 표 데이터를 마크다운 표 형식으로 변환 (병합된 헤더 처리 개선)"""
    if not table_data or len(table_data) < 1:
        return ""
    
    # 표의 최대 열 수 계산
    max_cols = max(len(row) for row in table_data) if table_data else 0
    if max_cols == 0:
        return ""
    
    # 모든 행의 열 수를 맞춤
    normalized_table = []
    for row in table_data:
        # 부족한 열은 빈 문자열로 채움
        normalized_row = row + [""] * (max_cols - len(row))
        normalized_table.append(normalized_row)
    
    lines = []
    
    # 표 제목 추가
    if table_num > 0:
        lines.append(f"### 표 {table_num} ({len(normalized_table)}행 x {max_cols}열)")
        lines.append("")
    
    # 첫 번째 행을 헤더로 처리
    if normalized_table:
        header = normalized_table[0]
        # 헤더 셀 처리 (이제 셀 병합이 제대로 처리되므로 단순화)
        processed_header = []
        
        for cell in header:
            if cell and cell.strip():
                processed_header.append(cell.strip())
            else:
                processed_header.append("---")  # 빈 셀만 ---
        
        lines.append("| " + " | ".join(processed_header) + " |")
        
        # 구분자 행
        separator = ["---"] * max_cols
        lines.append("| " + " | ".join(separator) + " |")
        
        # 데이터 행들
        for row in normalized_table[1:]:
            # 빈 셀은 그대로 유지 (빈 행 표시)
            formatted_row = []
            for cell in row:
                # 완전히 빈 셀은 공백으로 표시
                if not cell or cell.strip() == "":
                    formatted_row.append("&nbsp;")
                else:
                    formatted_row.append(cell)
            lines.append("| " + " | ".join(formatted_row) + " |")
    
    lines.append("")  # 표 뒤 빈 줄
    return "\n".join(lines)


def extract_paragraph_text(elem: ET.Element) -> str:
    """문단 요소에서 텍스트 추출"""
    texts = []
    for text_elem in elem.findall('.//hp:t', NAMESPACES):
        if text_elem.text:
            texts.append(text_elem.text)
    
    paragraph = ''.join(texts)
    return safe_spacing(paragraph)


def process_section_in_order(section: ET.Element, table_counter: Dict[str, int]) -> List[str]:
    """섹션에서 문단과 표를 원래 순서대로 처리 (중복 방지, 순서 수정)"""
    content_lines = []
    processed_tables = set()  # 이미 처리된 표 추적
    
    def get_element_id(elem: ET.Element) -> str:
        """요소의 고유 ID 생성"""
        return str(id(elem))
    
    def process_elements_sequentially(parent_elem: ET.Element) -> None:
        """부모 요소의 직접적인 자식들을 순서대로 처리"""
        for child in parent_elem:
            tag = child.tag
            if '}' in tag:
                tag = tag.split('}')[1]
            
            if tag.lower() == 'p':
                # 문단 내 표들 먼저 처리
                tables_in_paragraph = child.findall('.//hp:tbl', NAMESPACES)
                if tables_in_paragraph:
                    for table_elem in tables_in_paragraph:
                        table_id = get_element_id(table_elem)
                        if table_id not in processed_tables:
                            table_data = extract_table_data(table_elem)
                            if table_data:
                                table_counter['count'] += 1
                                table_md = format_table_as_markdown(table_data, table_counter['count'])
                                if table_md:
                                    content_lines.append(table_md)
                                processed_tables.add(table_id)
                else:
                    # 표가 없는 문단만 텍스트로 추출
                    para_text = extract_paragraph_text(child)
                    if para_text and para_text.strip():
                        content_lines.append(para_text)
                        content_lines.append("")  # 문단 간 빈 줄
                
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
                        if table_md:
                            content_lines.append(table_md)
                        processed_tables.add(table_id)
            
            else:
                # 기타 요소들은 하위 요소 계속 처리
                process_elements_sequentially(child)
    
    # 섹션의 모든 직접 자식 요소들을 순서대로 처리
    process_elements_sequentially(section)
    
    return content_lines


def process_section(section: ET.Element) -> Tuple[List[str], List[List[List[str]]]]:
    """기존 호환성을 위한 래퍼 함수 (더 이상 사용되지 않음)"""
    paragraphs = []
    tables = []
    
    # 문단 처리
    for p_elem in section.findall('.//hp:p', NAMESPACES):
        para_text = extract_paragraph_text(p_elem)
        if para_text and para_text.strip():
            paragraphs.append(para_text)
    
    # 표 처리  
    for tbl_elem in section.findall('.//hp:tbl', NAMESPACES):
        table_data = extract_table_data(tbl_elem)
        if table_data:
            tables.append(table_data)
    
    return paragraphs, tables


def convert_to_markdown(hwpx_path: str) -> str:
    """HWPX 파일을 마크다운으로 변환 (원래 순서 보존)"""
    sections = extract_sections_from_hwpx(hwpx_path)
    
    if not sections:
        raise ValueError("섹션을 찾을 수 없습니다.")
    
    all_content_lines = []
    table_counter = {'count': 0}  # 전체 문서에서 표 번호 카운터
    
    # 각 섹션을 순서대로 처리
    for section in sections:
        section_content = process_section_in_order(section, table_counter)
        all_content_lines.extend(section_content)
    
    return '\n'.join(all_content_lines)


def convert_hwp_to_markdown(input_path: str, output_path: str):
    """HWP/HWPX 파일을 마크다운으로 변환"""
    ext = os.path.splitext(input_path)[1].lower()
    
    if ext == '.hwp':
        if not HAS_PYHWP:
            raise RuntimeError("HWP 파일 처리를 위해 pyhwp가 필요합니다: pip install pyhwp")
        xml_str = extract_xml_from_hwp(input_path)
        # HWP는 단일 XML이므로 별도 처리 필요
        root = ET.fromstring(xml_str)
        paragraphs, tables = process_section(root)
        
        md_lines = []
        for para in paragraphs:
            md_lines.append(para)
            md_lines.append("")
        
        if tables:
            md_lines.extend(["", "## 표 목록", ""])
            for i, table_data in enumerate(tables, 1):
                table_md = format_table_as_markdown(table_data, i)
                if table_md:
                    md_lines.append(table_md)
        
        md_content = '\n'.join(md_lines)
        
    elif ext == '.hwpx':
        md_content = convert_to_markdown(input_path)
    else:
        raise ValueError("지원하지 않는 확장자입니다. .hwp 또는 .hwpx만 가능합니다.")

    # 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"변환 완료: {input_path} → {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='HWP/HWPX to Markdown 변환기 (빈 행 처리 개선)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python hwp_to_md_enhanced.py document.hwpx output.md
  python hwp_to_md_enhanced.py document.hwp output.md

주요 개선사항:
  - 빈 행(all "none" cells) 적절히 표시
  - 셀 병합 처리 개선
  - 표 구조 완전성 유지
        """
    )
    parser.add_argument('input', help='입력 .hwp 또는 .hwpx 파일')
    parser.add_argument('output', help='출력 .md 파일')
    
    args = parser.parse_args()
    
    try:
        convert_hwp_to_markdown(args.input, args.output)
    except Exception as e:
        print(f"오류 발생: {e}")
        sys.exit(1) 
