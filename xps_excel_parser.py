"""XPS workbook importer. Reads numerical values from XLSX XML (stdlib only).

For the supplied instrument workbook: A = imported energy, C = measured CPS;
B is a user-added *shifted* BE axis for C1s/F1s. E is an existing
instrument background: it is deliberately NEVER used as measured intensity.
"""
from __future__ import annotations

import io
import re
import zipfile
import posixpath
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

CORE_ORDER = ('C1s','O1s','F1s','P2p','Li1s','N1s','Survey')
NS = {'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
      'p':'http://schemas.openxmlformats.org/package/2006/relationships'}
MAX_UPLOAD = 30*1024*1024
MAX_XML_TOTAL = 100*1024*1024

@dataclass
class Spectrum:
    name: str
    x: np.ndarray
    y: np.ndarray
    sheet: str
    energy_column: str
    intensity_column: str
    embedded_shift_eV: Optional[float]
    note: str


def _core_name(s: str) -> Optional[str]:
    norm = re.sub(r'[^A-Za-z0-9]','',str(s)).lower()
    for name in CORE_ORDER:
        if name.lower() in norm:
            return name
    return None


def _col_number(cell_ref: str) -> int:
    letters = re.match(r'[A-Z]+', cell_ref)
    if letters is None:
        return -1
    val=0
    for ch in letters.group(0): val=26*val + ord(ch)-64
    return val-1


def _to_float(val):
    if isinstance(val,(int,float)):
        f=float(val)
    elif val is not None:
        try: f=float(str(val).replace(',','').strip())
        except (TypeError, ValueError): return None
    else: return None
    return f if np.isfinite(f) else None


def _parse_xlsx_cells(xml: bytes, strings: list[str]) -> list[dict[int,object]]:
    result=[]
    for el in ET.fromstring(xml).findall('.//m:sheetData/m:row',NS):
        row={}
        for cell in el.findall('m:c',NS):
            col=_col_number(cell.attrib.get('r',''))
            if not 0 <= col < 80: continue
            kind=cell.attrib.get('t','')
            v=cell.find('m:v',NS)
            if kind=='inlineStr':
                inline=cell.find('m:is',NS)
                text=''.join(t.text or '' for t in inline.findall('.//m:t',NS)) if inline is not None else ''
                row[col]=text
            elif v is not None and v.text is not None:
                if kind=='s':
                    i=int(v.text); row[col]=strings[i] if i<len(strings) else ''
                elif kind in ('str','e','d'):
                    row[col]=v.text
                else:
                    row[col]=_to_float(v.text)
        result.append(row)
    return result


def _normalize_data(x, y) -> tuple[np.ndarray, np.ndarray]:
    xa=np.asarray(x,float); ya=np.asarray(y,float)
    keep=np.isfinite(xa)&np.isfinite(ya)
    xa=xa[keep]; ya=ya[keep]
    if len(xa)<20: raise ValueError('측정 데이터 포인트가 20개 미만')
    ux, inv, counts=np.unique(xa,return_inverse=True,return_counts=True)
    uy=np.bincount(inv,weights=ya)/counts
    if ux[-1]-ux[0]<1: raise ValueError('BE 범위가 너무 좁음')
    return ux,uy


def _find_data_columns(rows: list[dict[int,object]], core: str):
    # Instrument XLSX: row 16+ energy in A, measured counts in C.
    # The BE column B is an optional manually corrected axis (no heading).
    # Other XLSX layouts: find energy and CPS headers, with fallback A,B.
    if len(rows)>30:
        common=[]
        for row in rows[:60]:
            if _to_float(row.get(0)) is not None and _to_float(row.get(2)) is not None:
                common.append(row)
        if len(common)>=20 and any('Binding Energy' in str(row.get(0,'')) for row in rows[:25]):
            return 0,2,1
    for i,row in enumerate(rows[:45]):
        energy_cols=[c for c,v in row.items() if c<20 and isinstance(v,str) and ('binding' in v.lower() or re.match(r'^be\s*\(?ev\)?$',v.strip(),re.I))]
        intensity_cols=[c for c,v in row.items() if c<20 and isinstance(v,str) and ('intensity' in v.lower() or 'counts / s' in v.lower() or 'raw data' in v.lower() or v.strip().lower()=='cps')]
        if energy_cols and intensity_cols:
            return energy_cols[0],intensity_cols[0],None
    # A / B for simple two-column clean tables.
    for ca,cb in [(0,1),(0,2)]:
        n=sum(1 for row in rows[:100] if _to_float(row.get(ca)) is not None and _to_float(row.get(cb)) is not None)
        if n>=20: return ca,cb,None
    raise ValueError('BE 및 실측 intensity 열을 인식하지 못함')


def _extract_spectrum(rows: list[dict[int,object]], sheet: str, name: str) -> Spectrum:
    energy_col, cps_col, optional_shifted = _find_data_columns(rows,name)
    rawx=[]; y=[]; alt=[]
    for row in rows:
        xx=_to_float(row.get(energy_col)); yy=_to_float(row.get(cps_col))
        if xx is not None and yy is not None:
            rawx.append(xx); y.append(yy)
            alt.append(_to_float(row.get(optional_shifted)) if optional_shifted is not None else None)
    chosenx=rawx; energy_letter=chr(65+energy_col); shift=None
    if optional_shifted is not None and len(rawx)>20:
        pairs=[(a,b) for a,b in zip(rawx,alt) if b is not None]
        if len(pairs)>=0.95*len(rawx):
            ds=np.array([b-a for a,b in pairs])
            if (abs(float(np.median(ds)))>1e-5 and abs(float(np.median(ds)))<=5
                    and float(np.std(ds))<0.001):
                shift=float(np.median(ds)); chosenx=[a+shift for a in rawx]
                energy_letter=chr(65+optional_shifted)
    x,yarr=_normalize_data(chosenx,y)
    if shift is not None:
        note=f'{sheet}: {energy_letter}열에서 기존 BE 이동 {shift:+.2f} eV 감지. 추가 보정 0.00 eV.'
    else:
        note=f'{sheet}: {energy_letter}열 BE 사용. 별도 보정 이력은 파일에서 확정할 수 없음; 추가 보정 0.00 eV.'
    return Spectrum(name,x,yarr,sheet,energy_letter,chr(65+cps_col),shift,note)


def read_xlsx(payload: bytes) -> dict[str,Spectrum]:
    if len(payload)>MAX_UPLOAD: raise ValueError('Excel 업로드 크기 제한 30 MB')
    try: z=zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc: raise ValueError('유효한 .xlsx 파일이 아님') from exc
    with z:
        if sum(info.file_size for info in z.infolist())>MAX_XML_TOTAL:
            raise ValueError('압축 해제된 Excel 파일이 허용 크기를 초과함')
        shared=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            root=ET.fromstring(z.read('xl/sharedStrings.xml'))
            shared=[''.join(t.text or '' for t in si.findall('.//m:t',NS)) for si in root.findall('m:si',NS)]
        wb=ET.fromstring(z.read('xl/workbook.xml'))
        rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        rmap={el.attrib['Id']:el.attrib['Target'] for el in rels}
        out={}
        for el in wb.findall('m:sheets/m:sheet',NS):
            sheet=el.attrib.get('name','')
            core=_core_name(sheet)
            if core is None: continue
            rel_id=el.attrib.get('{'+NS['r']+'}id')
            target=rmap.get(rel_id,'')
            path=target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('xl',target))
            if path not in z.namelist(): continue
            rows=_parse_xlsx_cells(z.read(path),shared)
            try: out[core]=_extract_spectrum(rows,sheet,core)
            except ValueError: continue
    if not out: raise ValueError('C1s, O1s, F1s, P2p, Li1s, N1s 등의 스펙트럼을 찾지 못함')
    return dict(sorted(out.items(),key=lambda pair:CORE_ORDER.index(pair[0])))


def parse_input(payload: bytes, filename: str) -> dict[str,Spectrum]:
    ext=filename.rsplit('.',1)[-1].lower()
    if ext=='xlsx': return read_xlsx(payload)
    from xps_engine import parse_csv, extract_zip_csv, zip_csv_names
    if ext=='csv':
        x,y=parse_csv(payload)
        name=_core_name(filename) or 'Unknown'
        return {name:Spectrum(name,x,y,filename,'CSV BE','CSV CPS',None,'업로드 CSV에 이미 적용된 BE 보정 여부 미확인; 추가 보정 0.00 eV')}
    if ext=='zip':
        result={}
        for path in zip_csv_names(payload):
            try:
                x,y=parse_csv(extract_zip_csv(payload,path))
                name=_core_name(path)
                if name is None or name in result: continue
                result[name]=Spectrum(name,x,y,path,'CSV BE','CSV CPS',None,'CSV에 이미 적용된 BE 보정 여부 미확인; 추가 보정 0.00 eV')
            except ValueError: continue
        if result:return dict(sorted(result.items(),key=lambda p:CORE_ORDER.index(p[0])))
        raise ValueError('ZIP에서 유효한 Core Level CSV를 찾지 못함')
    raise ValueError('지원 파일 형식은 .xlsx, .csv, .zip')
