"""
hwp_to_md_fixed.py

한글(.hwp/.hwpx) 문서를 Markdown(.md)으로 변환하는 스크립트 (수정된 버전)
- 실제 HWPX XML 구조에 맞춰 수정
- 올바른 네임스페이스 사용
- 개별 섹션 파일 처리
- 셀 병합 처리
- 문단 구조 인식

Usage:
    python hwp_to_md_fixed.py input.hwp output.md
    python hwp_to_md_fixed.py input.hwpx output.md

Requirements:
    pip install pyhwp
"""
import os
import sys
import zipfile
import argparse
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
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
    text = text.replace('\n', ' ').replace('\r', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def extract_xml_from_hwp(hwp_path: str) -> str:
    """pyhwp를 이용해 .hwp 이진에서 HWPML(XML) 문자열을 반환"""
    if not HAS_PYHWP:
        raise RuntimeError("pyhwp 라이브러리가 필요합니다. pip install pyhwp")
    doc = HWP5File(hwp_path)
    return doc.restore_xml()


def extract_sections_from_hwpx(hwpx_path: str) -> List[ET.Element]:
    """HWPX에서 각 섹션을 개별적으로 파싱하여 Element 리스트 반환"""
    sections = []
    
    with zipfile.ZipFile(hwpx_path, 'r') as z:
        # section 파일들을 찾아서 정렬
        section_files = []
        for name in z.namelist():
            if name.startswith("Contents/section") and name.endswith('.xml'):
                section_files.append(name)
        
        # 섹션 번호순으로 정렬
        section_files.sort(key=lambda x: int(x.split('section')[1].split('.')[0]))
        
        # 각 섹션을 개별적으로 파싱
        for section_file in section_files:
            try:
                xml_content = z.read(section_file).decode('utf-8')
                root = ET.fromstring(xml_content)
                sections.append(root)
            except Exception as e:
                print(f"Warning: {section_file} 파싱 실패: {e}")
                continue
    
    return sections


def extract_text_from_sections(sections: List[ET.Element]) -> str:
    """섹션들에서 텍스트 추출"""
    texts = []
    
    for section in sections:
        # 문단들을 찾기
        for p in section.findall('.//hp:p', NAMESPACES):
            paragraph_texts = []
            
            # 각 문단 내의 run들을 처리
            for run in p.findall('.//hp:run', NAMESPACES):
                # run 내의 텍스트 요소들 찾기
                for t in run.findall('.//hp:t', NAMESPACES):
                    if t.text:
                        paragraph_texts.append(t.text)
            
            # 문단 텍스트 조합
            if paragraph_texts:
                paragraph = ' '.join(paragraph_texts).strip()
                if paragraph:
                    texts.append(paragraph + '\n')
    
    return '\n'.join(texts)


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
                cell_texts = []
                for text_elem in cell_elem.iter():
                    text_tag = text_elem.tag
                    if '}' in text_tag:
                        text_tag = text_tag.split('}')[1]
                    
                    if text_tag.lower() in ['t', 'text']:
                        if text_elem.text:
                            cell_texts.append(text_elem.text)
                
                cell_text = safe_spacing(' '.join(cell_texts))
                if not cell_text:
                    cell_text = ""
                
                # 셀을 grid에 채우기 (병합된 셀의 모든 위치에 동일한 텍스트)
                for dr in range(row_span):
                    for dc in range(col_span):
                        r, c = row_idx + dr, col_idx + dc
                        if r < n_rows and c < max_cols:
                            grid[r][c] = cell_text
                
                col_idx += col_span
    
    # None 값을 빈 문자열로 변환하고 최종 결과 생성
    result: List[List[str]] = []
    for row in grid:
        result_row: List[str] = []
        for cell in row:
            result_row.append(cell if cell is not None else "")
        result.append(result_row)
    
    return result


def extract_tables_from_sections(sections: List[ET.Element]) -> List[Dict[str, Any]]:
    """섹션들에서 표 추출"""
    tables = []
    
    for section in sections:
        # 표들을 찾기
        for tbl in section.findall('.//hp:tbl', NAMESPACES):
            table_data = extract_table_data(tbl)
            if table_data and len(table_data) >= 1:  # 최소 1행은 있어야 함
                tables.append({
                    'data': table_data,
                    'rows': len(table_data),
                    'columns': len(table_data[0]) if table_data else 0
                })
    
    return tables


def table_to_markdown(table_data: List[List[str]]) -> str:
    """표 데이터를 마크다운 형식으로 변환"""
    if not table_data:
        return ""
    
    # 빈 행 제거
    filtered_data = []
    for row in table_data:
        if any(cell.strip() for cell in row):
            filtered_data.append(row)
    
    if not filtered_data:
        return ""
    
    # 열 개수가 일치하도록 조정
    max_cols = max(len(row) for row in filtered_data)
    for row in filtered_data:
        while len(row) < max_cols:
            row.append("")
    
    # 마크다운 테이블 생성
    lines = []
    
    # 헤더 (첫 번째 행)
    header = filtered_data[0]
    lines.append('| ' + ' | '.join(cell.replace('|', '\\|') for cell in header) + ' |')
    
    # 구분선
    lines.append('| ' + ' | '.join(['---'] * len(header)) + ' |')
    
    # 데이터 행들
    for row in filtered_data[1:]:
        lines.append('| ' + ' | '.join(cell.replace('|', '\\|') for cell in row) + ' |')
    
    return '\n'.join(lines)


def sections_to_markdown(sections: List[ET.Element]) -> str:
    """섹션들에서 텍스트와 표를 Markdown으로 변환"""
    md_parts = []
    
    # 텍스트 추출
    text_content = extract_text_from_sections(sections)
    if text_content.strip():
        md_parts.append("## 문서 내용\n\n" + text_content)
    
    # 표 추출
    tables = extract_tables_from_sections(sections)
    if tables:
        md_parts.append("\n## 표 목록\n")
        for i, table in enumerate(tables, 1):
            md_parts.append(f"\n### 표 {i} ({table['rows']}행 x {table['columns']}열)\n")
            table_md = table_to_markdown(table['data'])
            if table_md:
                md_parts.append(table_md + "\n")
    
    return '\n'.join(md_parts)


def convert(input_path: str, output_path: str):
    """HWP/HWPX 파일을 Markdown으로 변환"""
    ext = os.path.splitext(input_path)[1].lower()
    
    if ext == '.hwp':
        # HWP 파일 처리 (기존 방식 유지)
        xml_str = extract_xml_from_hwp(input_path)
        try:
            root = ET.fromstring(xml_str)
            sections = [root]  # HWP는 단일 XML
        except ET.ParseError as e:
            print(f"HWP XML 파싱 오류: {e}")
            sys.exit(1)
    elif ext == '.hwpx':
        # HWPX 파일 처리 (수정된 방식)
        sections = extract_sections_from_hwpx(input_path)
        if not sections:
            print("HWPX 파일에서 섹션을 찾을 수 없습니다.")
            sys.exit(1)
    else:
        print("지원하지 않는 확장자입니다. .hwp 또는 .hwpx만 가능")
        sys.exit(1)
    
    # Markdown 변환
    md_content = sections_to_markdown(sections)
    
    # 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"Converted {input_path} → {output_path}")
    print(f"섹션 수: {len(sections)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HWP/HWPX to Markdown 변환기')
    parser.add_argument('input', help='입력 .hwp 또는 .hwpx 파일')
    parser.add_argument('output', help='출력 .md 파일')
    args = parser.parse_args()
    convert(args.input, args.output) 
