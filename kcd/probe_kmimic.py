#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
probe_kmimic.py — K-MIMIC 데이터 디렉터리 구조 탐색 (선행 코드)

목적: 파이프라인을 실제 데이터에 맞추기 위해 파일·컬럼·형식·사전(label) 구조만 파악한다.
개별 환자 행(raw row)은 출력하지 않는다. 출력은 구조·형식·집계 통계·사전(label) 목록뿐이다.

사용:
  python probe_kmimic.py /path/to/datasets/kmimic-emr-001/268
  python probe_kmimic.py /path/to/dir --sample-rows 300000 --full      # 대용량 테이블 전체 스캔(ITEMID 빈도)
  python probe_kmimic.py /path/to/dir --out kmimic_probe               # kmimic_probe.txt / .json 생성

의존성: pandas만 필요.
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from collections import Counter

import pandas as pd

HANGUL = re.compile(r"[가-힣]")
DATE_PATTERNS = [
    ("YYYY-MM-DD HH:MM:SS", re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")),
    ("YYYY-MM-DD HH:MM", re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}$")),
    ("YYYY-MM-DD", re.compile(r"^\d{4}-\d{2}-\d{2}$")),
    ("YYYYMMDDHHMMSS", re.compile(r"^\d{14}$")),
    ("YYYYMMDD", re.compile(r"^\d{8}$")),
    ("YYYY/MM/DD HH:MM:SS", re.compile(r"^\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}")),
]
DICT_TABLES = {"D_ITEMS", "D_LABITEMS", "D_TESTITEMS", "HOSPITAL"}
EVENT_ITEM_TABLES = {
    "CHARTEVENTS": "D_ITEMS", "DATETIMEEVENTS": "D_ITEMS", "INPUTEVENTS": "D_ITEMS",
    "OUTPUTEVENTS": "D_ITEMS", "PROCEDUREEVENTS": "D_ITEMS",
    "LABEVENTS": "D_LABITEMS", "TESTEVENTS": "D_TESTITEMS",
}
CATEGORICAL_HINT = {
    "ADMISSIONS": ["ADMISSION_TYPE", "ADMISSION_LOCATION", "DISCHARGE_LOCATION", "INSURANCE", "HOSPITAL_ID", "NATIONALITY"],
    "PATIENTS": ["SEX", "ANCHOR_YEAR_GROUP"],
    "TRANSFERS": ["EVENTTYPE", "CAREUNIT"],
    "ICUSTAYS": ["FIRST_CAREUNIT", "LAST_CAREUNIT", "OP_FLAG"],
    "PRESCRIPTIONS": ["DRUG_TYPE", "ROUTE", "DOSE_UNIT_RX", "FORM_UNIT_DISP"],
    "EMAR": ["EMAR_TYPE", "EVENT_TXT"],
    "EMAR_DETAIL": ["ON_OFF"],
    "LABEVENTS": ["FLAG", "VALUEUOM"],
    "MICROBIOLOGYEVENTS": ["SPEC_TYPE_DESC", "TEST_NAME", "ORG_NAME", "AB_NAME", "INTERPRETATION", "DILUTION_COMPARISON"],
    "DIAGNOSES_ICD": ["ICD_VERSION", "IS_ICU"],
    "PROCEDURES_ICD": ["ICD_VERSION"],
    "SERVICES": ["PREV_SERVICE", "CURR_SERVICE"],
    "CHARTEVENTS": ["VALUEUOM", "WARNING"],
    "INPUTEVENTS": ["AMOUNTUOM", "RATEUOM"],
    "OUTPUTEVENTS": ["VALUEUOM"],
    "PROCEDUREEVENTS": ["VALUEUOM", "LOCATION", "LOCATIONCATEGORY", "ORDERCATEGORYNAME", "ORDERCATEGORYDESCRIPTION"],
    "EDSTAY": [], "TRIAGE": [], "MEDRECON": [], "HOSPITAL": [],
}
TEXT_LIKE_HINT = ["NOTE_TXT", "COMMENTS", "ORDER_COMMENTS", "EVENT_TXT", "VALUE", "DRUG", "PROD_STRENGTH", "LABEL", "CHIEFCOMPLAINT", "CHIEF_COMPLAINT"]


def human(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return "%.1f%s" % (n, unit)
        n /= 1024.0
    return "%.1fPB" % n


def table_name(path):
    base = os.path.basename(path)
    for suf in [".csv.gz", ".csv", ".parquet"]:
        if base.lower().endswith(suf):
            return base[: -len(suf)].upper()
    return None


def read_sample(path, nrows):
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
        return df.head(nrows) if nrows else df, len(df)
    kw = dict(low_memory=False, encoding="utf-8")
    if nrows:
        kw["nrows"] = nrows
    try:
        return pd.read_csv(path, **kw), None
    except UnicodeDecodeError:
        kw["encoding"] = "cp949"
        return pd.read_csv(path, **kw), None


def estimate_rows(path, sample_lines=20000):
    """파일 크기 / 앞부분 평균 줄 길이 → 대략의 행 수 (gz는 추정 불가)."""
    if path.lower().endswith(".gz") or path.lower().endswith(".parquet"):
        return None
    size = os.path.getsize(path)
    total = 0
    n = 0
    with open(path, "rb") as f:
        header = f.readline()
        for line in f:
            total += len(line)
            n += 1
            if n >= sample_lines:
                break
    if n == 0:
        return 0
    return int((size - len(header)) / (total / n))


def count_rows_full(path):
    if path.lower().endswith(".parquet"):
        return len(pd.read_parquet(path, columns=[]))
    opener = open
    if path.lower().endswith(".gz"):
        import gzip
        opener = gzip.open
    n = 0
    with opener(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            n += chunk.count(b"\n")
    return max(n - 1, 0)


def detect_date_format(series):
    s = series.dropna().astype(str).head(2000)
    if len(s) == 0:
        return None, 0.0
    best, best_rate = None, 0.0
    for name, pat in DATE_PATTERNS:
        rate = s.map(lambda v: bool(pat.match(v))).mean()
        if rate > best_rate:
            best, best_rate = name, rate
    return (best, float(best_rate)) if best_rate >= 0.8 else (None, float(best_rate))


def column_profile(df, col, max_cats=30):
    s = df[col]
    n = len(s)
    nn = s.notna().sum()
    prof = {
        "dtype": str(s.dtype),
        "null_pct": round(100.0 * (1 - nn / n), 2) if n else None,
        "nunique_sample": int(s.nunique(dropna=True)),
    }
    if nn == 0:
        return prof
    if s.dtype == object:
        st = s.dropna().astype(str)
        prof["avg_len"] = round(float(st.str.len().mean()), 1)
        prof["max_len"] = int(st.str.len().max())
        prof["hangul_pct"] = round(100.0 * st.head(5000).map(lambda v: bool(HANGUL.search(v))).mean(), 1)
        num_rate = pd.to_numeric(st.head(5000), errors="coerce").notna().mean()
        prof["numeric_parse_pct"] = round(100.0 * float(num_rate), 1)
        fmt, rate = detect_date_format(st)
        if fmt:
            prof["date_format"] = fmt
            prof["date_parse_pct"] = round(100.0 * rate, 1)
            dt = pd.to_datetime(st, errors="coerce")
            prof["date_min"] = str(dt.min())[:19]
            prof["date_max"] = str(dt.max())[:19]
        if prof["nunique_sample"] <= max_cats:
            vc = st.value_counts().head(max_cats)
            prof["categories"] = {str(k): int(v) for k, v in vc.items()}
        if prof["avg_len"] > 30 and prof["nunique_sample"] > max_cats:
            prof["free_text_like"] = True
    else:
        if pd.api.types.is_numeric_dtype(s):
            d = s.describe()
            prof["stats"] = {k: (round(float(d[k]), 3) if pd.notna(d[k]) else None) for k in ["mean", "std", "min", "25%", "50%", "75%", "max"] if k in d}
            if prof["nunique_sample"] <= max_cats:
                vc = s.value_counts().head(max_cats)
                prof["categories"] = {str(k): int(v) for k, v in vc.items()}
    return prof


def id_columns(cols):
    return [c for c in cols if c.upper() in ("SUBJECT_ID", "HADM_ID", "STAY_ID", "ICUSTAY_ID", "ITEMID", "HOSPITAL_ID")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--sample-rows", type=int, default=200000, help="대용량 테이블에서 앞부분 n행만 프로파일")
    ap.add_argument("--full", action="store_true", help="전체 스캔으로 정확한 행 수와 ITEMID 빈도 계산(느림)")
    ap.add_argument("--out", default="kmimic_probe", help="리포트 파일 접두사")
    ap.add_argument("--top", type=int, default=80, help="빈도 상위 label 출력 개수")
    args = ap.parse_args()

    d = os.path.abspath(args.data_dir)
    if not os.path.isdir(d):
        print("디렉터리가 없습니다:", d)
        sys.exit(1)
    lines = []

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        lines.append(s)

    P("=" * 78)
    P("K-MIMIC PROBE  dir=%s  time=%s" % (d, time.strftime("%Y-%m-%d %H:%M:%S")))
    P("python=%s pandas=%s" % (sys.version.split()[0], pd.__version__))
    P("=" * 78)

    parent = os.path.dirname(d)
    try:
        sib = sorted(os.listdir(parent))
        P("[parent dir] %s  entries(%d): %s" % (parent, len(sib), ", ".join(sib[:40])))
    except Exception as e:
        P("[parent dir] 읽기 실패:", e)

    entries = sorted(os.listdir(d))
    P("[dir entries] %d" % len(entries))
    files = {}
    others = []
    for name in entries:
        p = os.path.join(d, name)
        if os.path.isdir(p):
            others.append(name + "/")
            continue
        t = table_name(p)
        if t:
            files[t] = p
        else:
            others.append(name)
    P("  tables(%d): %s" % (len(files), ", ".join(sorted(files))))
    P("  other files(%d): %s" % (len(others), ", ".join(others[:60])))

    report = {"dir": d, "tables": {}, "other_files": others}
    samples = {}
    for t in sorted(files):
        p = files[t]
        size = os.path.getsize(p)
        est = count_rows_full(p) if args.full else estimate_rows(p)
        P("-" * 78)
        P("[%s] file=%s size=%s rows%s=%s" % (t, os.path.basename(p), human(size), "" if args.full else "(est)", est))
        t0 = time.time()
        is_dict = t in DICT_TABLES
        nrows = None if is_dict else args.sample_rows
        try:
            df, _ = read_sample(p, nrows)
        except Exception as e:
            P("  읽기 실패:", repr(e))
            report["tables"][t] = {"error": repr(e), "size": size}
            continue
        df.columns = [str(c) for c in df.columns]
        samples[t] = df
        P("  sampled_rows=%d  read=%.1fs  columns(%d)=%s" % (len(df), time.time() - t0, len(df.columns), ", ".join(df.columns)))
        P("  id_columns=%s" % id_columns(df.columns))
        tinfo = {"file": os.path.basename(p), "size_bytes": size, "rows_est": est, "sampled_rows": len(df),
                 "columns": list(df.columns), "profiles": {}}
        for c in df.columns:
            prof = column_profile(df, c)
            tinfo["profiles"][c] = prof
            flag = []
            if prof.get("date_format"):
                flag.append("DATE[%s %s~%s]" % (prof["date_format"], prof.get("date_min", "")[:10], prof.get("date_max", "")[:10]))
            if prof.get("free_text_like"):
                flag.append("FREE_TEXT?")
            if prof.get("hangul_pct", 0) > 0:
                flag.append("hangul=%s%%" % prof["hangul_pct"])
            cat = ""
            if "categories" in prof and (c.upper() in CATEGORICAL_HINT.get(t, []) or prof["nunique_sample"] <= 12):
                cat = " cats=" + json.dumps(prof["categories"], ensure_ascii=False)[:300]
            P("    %-26s %-8s null=%5s%% nuniq=%-7d %s%s" % (c, prof["dtype"], prof.get("null_pct"), prof["nunique_sample"], " ".join(flag), cat))
        report["tables"][t] = tinfo

    # ---- 사전 테이블 전체 label 출력 (환자 데이터 아님) ----
    for t in sorted(DICT_TABLES & set(samples)):
        df = samples[t]
        P("-" * 78)
        P("[DICT %s] rows=%d" % (t, len(df)))
        cols = [c for c in df.columns if c.upper() in ("ITEMID", "LABEL", "ABBREVIATION", "CATEGORY", "FLUID", "LINKSTO", "UNITNAME", "PARAM_TYPE", "EDI_CODE", "LOWNORMALVALUE", "HIGHNORMALVALUE", "HOSPITAL_ID")] or list(df.columns)
        cap = 700
        for _, row in df[cols].head(cap).iterrows():
            P("    " + " | ".join("%s=%s" % (c, row[c]) for c in cols))
        if len(df) > cap:
            P("    ... (%d more rows; JSON 리포트에 전체 수록)" % (len(df) - cap))
        report["tables"][t]["dictionary"] = df[cols].astype(str).to_dict(orient="records")

    # ---- 이벤트 테이블 ITEMID 빈도 + label (표본 또는 전체) ----
    for t, dt in EVENT_ITEM_TABLES.items():
        if t not in samples:
            continue
        df = samples[t]
        icol = next((c for c in df.columns if c.upper() == "ITEMID"), None)
        if icol is None:
            continue
        if args.full:
            cnt = Counter()
            p = files[t]
            for chunk in pd.read_csv(p, usecols=[icol], chunksize=2000000, low_memory=False):
                cnt.update(chunk[icol].value_counts().to_dict())
            vc = pd.Series(cnt).sort_values(ascending=False)
            basis = "full"
        else:
            vc = df[icol].value_counts()
            basis = "sample"
        lab = None
        if dt in samples:
            ddf = samples[dt]
            dic = next((c for c in ddf.columns if c.upper() == "ITEMID"), None)
            dlab = next((c for c in ddf.columns if c.upper() == "LABEL"), None)
            if dic and dlab:
                lab = ddf.set_index(dic)[dlab].to_dict()
                extra_cols = [c for c in ddf.columns if c.upper() in ("CATEGORY", "LINKSTO", "UNITNAME", "PARAM_TYPE", "FLUID")]
                extra = {c: ddf.set_index(dic)[c].to_dict() for c in extra_cols}
        P("-" * 78)
        P("[ITEM FREQ %s] basis=%s distinct_itemids=%d  top %d:" % (t, basis, len(vc), args.top))
        top = []
        for itemid, n in vc.head(args.top).items():
            try:
                key = int(itemid)
            except Exception:
                key = itemid
            label = lab.get(key, lab.get(itemid, "?")) if lab else "?"
            ex = ""
            if lab:
                ex = " ".join("%s=%s" % (c, extra[c].get(key, extra[c].get(itemid, ""))) for c in extra)
            P("    %-10s n=%-9d %s  %s" % (itemid, int(n), label, ex))
            top.append({"itemid": str(itemid), "n": int(n), "label": str(label)})
        report["tables"][t]["item_freq_top"] = top
        report["tables"][t]["item_freq_basis"] = basis

    # ---- 약제·배양·진단 카탈로그 빈도 (집계) ----
    def topcat(t, col, k=60):
        if t in samples:
            df = samples[t]
            c = next((x for x in df.columns if x.upper() == col), None)
            if c:
                vc = df[c].value_counts().head(k)
                P("-" * 78)
                P("[TOP %s.%s] distinct=%d (sample basis)" % (t, col, df[c].nunique()))
                for v, n in vc.items():
                    P("    n=%-8d %s" % (int(n), v))
                report["tables"][t].setdefault("top_values", {})[col] = {str(v): int(n) for v, n in vc.items()}

    topcat("PRESCRIPTIONS", "DRUG", 80)
    topcat("PRESCRIPTIONS", "GSN", 30)
    topcat("PRESCRIPTIONS", "ROUTE", 30)
    topcat("EMAR", "MEDICATION", 40)
    topcat("MICROBIOLOGYEVENTS", "SPEC_TYPE_DESC", 30)
    topcat("MICROBIOLOGYEVENTS", "ORG_NAME", 30)
    topcat("MICROBIOLOGYEVENTS", "AB_NAME", 30)
    topcat("MICROBIOLOGYEVENTS", "TEST_NAME", 20)
    topcat("DIAGNOSES_ICD", "ICD_CODE", 40)
    topcat("PROCEDURES_ICD", "ICD_CODE", 30)
    topcat("TRANSFERS", "CAREUNIT", 30)
    topcat("SERVICES", "CURR_SERVICE", 30)
    topcat("CHARTEVENTS", "VALUEUOM", 40)
    topcat("INPUTEVENTS", "RATEUOM", 20)
    topcat("INPUTEVENTS", "AMOUNTUOM", 20)
    topcat("OUTPUTEVENTS", "VALUEUOM", 10)
    topcat("LABEVENTS", "VALUEUOM", 40)
    topcat("LABEVENTS", "FLAG", 10)

    # ---- 키 연결 점검 ----
    P("-" * 78)
    P("[LINKAGE] (sample basis)")
    if "ICUSTAYS" in samples:
        icu = samples["ICUSTAYS"]
        sc = next((c for c in icu.columns if c.upper() == "STAY_ID"), None)
        if sc:
            stay_ids = set(icu[sc].dropna().astype(int).tolist()) if pd.api.types.is_numeric_dtype(icu[sc]) else set(icu[sc].dropna())
            for t, df in samples.items():
                c = next((x for x in df.columns if x.upper() in ("STAY_ID", "ICUSTAY_ID")), None)
                if c and t != "ICUSTAYS":
                    col = df[c]
                    nn = col.notna().mean()
                    try:
                        inter = col.dropna().astype(int).isin(stay_ids).mean()
                    except Exception:
                        inter = col.dropna().isin(stay_ids).mean()
                    P("    %-20s %-11s nonnull=%5.1f%%  in_ICUSTAYS=%5.1f%%" % (t, c, 100 * nn, 100 * inter))
    for t, df in samples.items():
        c = next((x for x in df.columns if x.upper() == "HADM_ID"), None)
        if c:
            P("    %-20s HADM_ID nonnull=%5.1f%% nuniq=%d" % (t, 100 * df[c].notna().mean(), df[c].nunique()))

    with open(args.out + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    P("=" * 78)
    P("리포트 저장: %s.txt, %s.json  (텍스트 리포트를 그대로 붙여넣어 주면 파이프라인을 맞춥니다)" % (args.out, args.out))


if __name__ == "__main__":
    main()
