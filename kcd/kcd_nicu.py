#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kcd_nicu.py — K-MIMIC 형식 데이터로 NICU 1일 요약지(의료진용)와 부모님 설명지를 생성(A)하고,
두 문서의 할루시네이션·누락을 검증(B)하며, 신생아 CDSS 알림(C)을 붙인다.
병원 프로파일(JSON)로 코호트·문서 양식·필수 항목·신생아 기준값·CDSS 를 병원별로 수정한다.

단일 파일, 의존성 pandas만 필요(LLM은 선택: OpenAI 호환 엔드포인트, 표준 라이브러리 urllib 사용).

사용 (터미널 또는 Jupyter의 %run):
  python kcd_nicu.py list          --data DIR                          # NICU 코호트 stay·일수
  python kcd_nicu.py run           --data DIR --stay auto --out out    # 1건(마지막 날) 요약지 + 부모님 설명지
  python kcd_nicu.py run           --data DIR --stay 123 --day 2 --out out
  python kcd_nicu.py batch         --data DIR --n 50 --out out         # stay-일 50건 + 집계표
  python kcd_nicu.py perturb       --data DIR --n 50 --out out         # B3 검증기 자체 평가(두 문서)
  python kcd_nicu.py arms          --data DIR --n 10 --out out         # A0~A3 비교(LLM 필요)
  python kcd_nicu.py init-profile  --out my_hospital.json              # 병원 프로파일 초안
  python kcd_nicu.py profile-check --data DIR --profile my_hospital.json   # 항목별 데이터 자동 채움 비율
  python kcd_nicu.py evaluate-note --data DIR --stay 123 --day 2 --note note.txt --doc summary|parent

공통 옵션: --profile 병원.json, --cohort nicu|all(all = 전체 ICU, 코드 점검용), --time-unit D|h|min
Jupyter 셀에 통째로 붙여넣은 경우: kcd_main("run --data datasets/kmimic-emr-001/268 --stay auto --out out")
LLM(선택): KCD_LLM_BASE_URL=http://localhost:8000/v1  KCD_LLM_MODEL=모델명  KCD_LLM_API_KEY=(선택)
실제 환자 데이터 출력물은 분석환경 밖으로 반출하지 말 것. 외부 공유는 summary*.csv 집계 지표만.
"""
from __future__ import print_function

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import OrderedDict, defaultdict

import pandas as pd

VERSION = "0.3.0"
PROMPT_VERSION = "p2"

# =============================================================================
# 0. 공통 유틸
# =============================================================================


def fmt_num(v):
    """기록지 표기용 숫자 포맷. 검증기도 같은 함수를 기준으로 비교한다."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "?"
    v = float(v)
    if abs(v - round(v)) < 1e-9 and abs(v) < 1e7:
        return str(int(round(v)))
    if abs(v) >= 100:
        return "%.0f" % v
    if abs(v) >= 10:
        s = "%.1f" % v
    else:
        s = "%.2f" % v
    s = s.rstrip("0").rstrip(".")
    return s


def fmt_t(t):
    if t is None or pd.isna(t):
        return "시각 미상"
    return pd.Timestamp(t).strftime("%m-%d %H:%M")


def norm_id(s):
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s.where(~s.isin(["nan", "None", "NaT", ""]), None)


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


TIME_UNIT = os.environ.get("KCD_TIME_UNIT", "D")  # 상대시간(숫자)일 때 단위: D/h/min
_EPOCH = pd.Timestamp("2000-01-01")


def _parse_dt(s):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = None
        if int(pd.__version__.split(".")[0]) >= 2:
            try:
                iso = pd.to_datetime(s, errors="coerce", format="ISO8601")
                if len(s) == 0 or iso.notna().mean() >= 0.9:
                    return iso
            except (TypeError, ValueError):
                iso = None
            plain = pd.to_datetime(s, errors="coerce")
            return iso if (iso is not None and iso.notna().sum() >= plain.notna().sum()) else plain
        return pd.to_datetime(s, errors="coerce")


def to_dt(s):
    """ISO 문자열이면 datetime, 숫자(상대시간)면 기준일 + offset 으로 변환.
    가명화로 연도가 2262년(pandas 한계)을 넘으면 400년 단위로 당겨 파싱한다(윤년·요일 주기 보존, stay 내부 간격 불변)."""
    if s is None:
        return s
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    dt = _parse_dt(s)
    bad = dt.isna() & s.notna()
    if bad.any():
        ss = s[bad].astype(str).str.strip()
        yr = pd.to_numeric(ss.str.slice(0, 4), errors="coerce")
        ok = yr.notna() & (yr > 2261) & ss.str.match(r"^\d{4}-")
        if ok.any():
            k = ((yr[ok] - 2261) // 400 + 1) * 400
            fixed = (yr[ok] - k).astype(int).astype(str).str.zfill(4) + ss[ok].str.slice(4)
            dt.loc[fixed.index] = _parse_dt(fixed)
    if len(s) and dt.notna().mean() < 0.5:
        num = to_num(s)
        if num.notna().mean() >= 0.5:
            return _EPOCH + pd.to_timedelta(num, unit=TIME_UNIT)
    return dt


def sha(obj):
    return hashlib.sha1(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


def in_notebook_paste():
    """Jupyter 셀에 붙여넣어 실행된 경우(커널 인자가 sys.argv에 있음) CLI를 자동 실행하지 않는다. %run 은 정상 실행."""
    if "ipykernel" not in sys.modules:
        return False
    a0 = sys.argv[0] if sys.argv else ""
    return "ipykernel_launcher" in a0 or any(x.endswith(".json") for x in sys.argv[1:]) or not a0.endswith(".py")


# =============================================================================
# 1. 데이터 로더 (대소문자·확장자·syn_ 접두사 흡수, 대용량 테이블은 청크 필터)
# =============================================================================

TABLES = ["ADMISSIONS", "PATIENTS", "TRANSFERS", "DIAGNOSES_ICD", "D_LABITEMS", "D_TESTITEMS", "EMAR", "EMAR_DETAIL",
          "LABEVENTS", "MICROBIOLOGYEVENTS", "PRESCRIPTIONS", "PROCEDURES_ICD", "SERVICES", "TESTEVENTS", "CHARTEVENTS",
          "DATETIMEEVENTS", "D_ITEMS", "ICUSTAYS", "INPUTEVENTS", "OUTPUTEVENTS", "PROCEDUREEVENTS", "EDSTAY", "MEDRECON",
          "TRIAGE", "HOSPITAL"]
SMALL = {"ADMISSIONS", "PATIENTS", "TRANSFERS", "ICUSTAYS", "D_ITEMS", "D_LABITEMS", "D_TESTITEMS", "HOSPITAL",
         "DIAGNOSES_ICD", "PROCEDURES_ICD", "SERVICES", "EDSTAY", "TRIAGE"}
ID_COLS = ["subject_id", "hadm_id", "stay_id", "icustay_id", "itemid", "emar_id", "pharmacy_id", "hospital_id"]


class KData(object):
    def __init__(self, data_dir, verbose=True):
        self.dir = data_dir
        self.verbose = verbose
        self.files = {}
        self.cache = {}
        if not os.path.isdir(data_dir):
            raise SystemExit("데이터 디렉터리가 없습니다: %s" % data_dir)
        for name in os.listdir(data_dir):
            low = name.lower()
            for suf in (".csv.gz", ".csv", ".parquet", ".xlsx"):
                if low.endswith(suf):
                    base = name[: -len(suf)].upper()
                    if base.startswith("SYN_"):
                        base = base[4:]
                    prev = self.files.get(base)
                    # csv 우선, xlsx는 최후
                    rank = {".csv": 0, ".csv.gz": 1, ".parquet": 2, ".xlsx": 3}[suf]
                    if prev is None or rank < prev[1]:
                        self.files[base] = (os.path.join(data_dir, name), rank)
        if verbose:
            have = [t for t in TABLES if t in self.files]
            miss = [t for t in TABLES if t not in self.files]
            print("[load] %s: 테이블 %d개 발견, 없음: %s" % (data_dir, len(have), ", ".join(miss) or "-"))

    def has(self, t):
        return t in self.files

    def _post(self, df):
        df.columns = [str(c).strip().lower() for c in df.columns]
        for c in ID_COLS:
            if c in df.columns:
                df[c] = norm_id(df[c])
        return df

    def _read_all(self, path):
        if path.endswith(".parquet"):
            return pd.read_parquet(path).astype(str)
        if path.endswith(".xlsx"):
            return pd.read_excel(path, dtype=str)
        try:
            return pd.read_csv(path, dtype=str, low_memory=False, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(path, dtype=str, low_memory=False, encoding="cp949")

    def get(self, t, hadm_ids=None):
        """테이블 로드. hadm_ids가 주어지면 해당 입원만 필터(대용량은 청크)."""
        if t not in self.files:
            return pd.DataFrame()
        key = (t, None if hadm_ids is None else tuple(sorted(hadm_ids)))
        if key in self.cache:
            return self.cache[key]
        path = self.files[t][0]
        t0 = time.time()
        if t in SMALL or hadm_ids is None or not (path.endswith(".csv") or path.endswith(".csv.gz")):
            df = self._post(self._read_all(path))
            if hadm_ids is not None and "hadm_id" in df.columns:
                df = df[df["hadm_id"].isin(set(hadm_ids))]
        else:
            ids = set(hadm_ids)
            parts = []
            enc = "utf-8"
            for attempt in range(2):
                try:
                    parts = []
                    for ch in pd.read_csv(path, dtype=str, chunksize=1000000, low_memory=False, encoding=enc):
                        ch = self._post(ch)
                        if "hadm_id" in ch.columns:
                            ch = ch[ch["hadm_id"].isin(ids)]
                        parts.append(ch)
                    break
                except UnicodeDecodeError:
                    enc = "cp949"
            df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if self.verbose and time.time() - t0 > 2:
            print("[load] %-20s %8d행  %.1fs" % (t, len(df), time.time() - t0))
        self.cache[key] = df
        return df


def nicu_stays(kd, cohort="nicu"):
    """프로파일 cohort 기준(병동명 정규식·나이)으로 NICU stay 선택. cohort='all' 이면 전체 ICU(코드 점검용)."""
    icu = kd.get("ICUSTAYS").copy()
    if icu.empty:
        raise SystemExit("ICUSTAYS가 없습니다.")
    icu["intime_dt"] = to_dt(icu["intime"])
    icu["outtime_dt"] = to_dt(icu["outtime"])
    icu = icu[icu["intime_dt"].notna() & icu["outtime_dt"].notna() & (icu["outtime_dt"] > icu["intime_dt"])]
    pats = kd.get("PATIENTS")
    if not pats.empty and "anchor_age" in pats.columns:
        icu = icu.merge(pats[["subject_id", "anchor_age"]].drop_duplicates("subject_id"), on="subject_id", how="left")
    icu["_unit_match"] = False
    rx = PROFILE["cohort"].get("careunit_regex")
    for col in ("first_careunit", "last_careunit"):
        if rx and col in icu.columns:
            icu["_unit_match"] = icu["_unit_match"] | icu[col].astype(str).str.contains(rx, case=False, regex=True, na=False)
    age = to_num(icu["anchor_age"]) if "anchor_age" in icu.columns else pd.Series(float("nan"), index=icu.index)
    icu["_young"] = age.notna() & (age <= float(PROFILE["cohort"].get("max_age_years", 0)))
    icu["_neo"] = icu["_unit_match"] | icu["_young"]
    if cohort != "all":
        any_ = PROFILE["cohort"].get("match", "any") == "any"
        icu = icu[(icu["_unit_match"] | icu["_young"]) if any_ else (icu["_unit_match"] & icu["_young"])]
    return icu.sort_values("stay_id").reset_index(drop=True)


def day_t0s(row):
    """1일 요약 기준 시각 목록: 재실 중 매일 cutoff_hour 시각. 마지막 cutoff 이후 12시간 넘게 남으면 퇴실 시각도 포함."""
    D = PROFILE["daily"]
    a, b = pd.Timestamp(row["intime_dt"]), pd.Timestamp(row["outtime_dt"])
    t = a.normalize() + pd.Timedelta(hours=float(D.get("cutoff_hour", 7)))
    if t <= a:
        t += pd.Timedelta(days=1)
    out = []
    while t <= b:
        out.append(t)
        t += pd.Timedelta(days=1)
    if not out or (D.get("include_discharge_day", True) and (b - out[-1]).total_seconds() >= 12 * 3600):
        out.append(b)
    return out


# =============================================================================
# 2. 개념 사전 (SYN-ICU에서 확인한 SNUH K-MIMIC 라벨 형식 + MIMIC 영문 라벨)
# =============================================================================

# (concept, display, regex on "label | abbreviation", exclude regex)
CHART_CONCEPTS = [
    ("paw_mean", "평균기도압", r"mean airway|평균기도", None),
    ("pip", "최고기도압", r"\bPIP\b|peak airway", None),
    ("map", "MAP", r"mean ?A?BP|평균혈압|blood pressure mean|\bMAP\b|\bABPm\b|\bNBPm\b", r"airway"),
    ("sbp", "SBP", r"\bA?SBP\b|수축기|systolic|\bABPs\b|\bNBPs\b", None),
    ("dbp", "DBP", r"\bA?DBP\b|이완기|diastolic|\bABPd\b|\bNBPd\b", None),
    ("hr", "HR", r"\bHR\b|심박|heart rate", None),
    ("rr", "RR", r"\bRR\b|호흡수|respiratory rate", None),
    ("spo2", "SpO2", r"SpO2|산소포화도", None),
    ("temp", "BT", r"\bBT\b|체온|temperature", None),
    ("gcs_e", "GCS-E", r"coma scale\(eye\)|GCS.*eye|eye.*GCS|eye opening", None),
    ("gcs_v", "GCS-V", r"coma scale\(verbal\)|GCS.*verbal|verbal.*GCS|verbal response", None),
    ("gcs_m", "GCS-M", r"coma scale\(motor\)|GCS.*motor|motor.*GCS|motor response", None),
    ("gcs", "GCS", r"GCS total|Glasgow.*total", None),
    ("fio2", "FiO2", r"FIO2|inspired O2", None),
    ("peep", "PEEP", r"PEEP", None),
    ("vent_mode", "인공호흡기 모드", r"vent.*모드|ventilator.*mode|ventilator 모드|vent mode", None),
    ("o2_flow", "산소유량", r"flow rate|산소량|O2 flow|O2\(L/min\)", None),
    ("vt", "일회호흡량", r"\bVT\b|흡기량|호기량|tidal", None),
    ("ve", "분당환기량", r"spont VE|minute vol", None),
    ("compliance", "폐탄성", r"compliance", None),
    ("bst", "BST", r"\bBST\b|혈당|glucometer|POCT glucose", None),
    ("weight", "체중", r"weight|체중|몸무게", None),
]
VENT_CONCEPTS = {"vent_mode", "peep", "pip", "paw_mean", "vt", "ve", "compliance"}

OUTPUT_CONCEPTS = [
    ("crrt", r"CVVHDF|CRRT|투석|dialysis|ultrafiltrat"),
    ("urine_stool", r"소변\+대변|대소변"),
    ("stool", r"^대변|대변량|stool|feces|배변|meconium|태변"),
    ("urine", r"도뇨|배뇨|foley|urine|소변|void"),
    ("drain", r"tube|drain|흉관|PCD|J-P|Jackson|C-tube|배액|hemovac|EVD"),
]
INPUT_CONCEPTS = [("in_iv", r"정맥|\bIV\b|intravenous|TPN"), ("in_tube", r"Levin|L-tube|feeding tube|경관|\bOG\b|\bNG\b|gavage"),
                  ("in_oral", r"경구|oral|\bPO\b|수유|모유|분유|breast|formula|feeding")]

# lab: (concept, display, regex on label, low_crit, high_crit)
LAB_CONCEPTS = [
    ("egfr", "eGFR", r"eGFR", None, None),
    ("cr", "Cr", r"^Creatinine|크레아티닌", None, None),
    ("bun", "BUN", r"^BUN|Urea Nitrogen|요소질소", None, None),
    ("na", "Na", r"^Sodium|나트륨|^Na\b", 120, 160),
    ("k", "K", r"^Potassium|칼륨|^K\b", 2.5, 6.0),
    ("cl", "Cl", r"^Chloride|^Cl\b", None, None),
    ("tco2", "tCO2", r"CO2, total|^Bicarbonate|총이산화탄소", None, None),
    ("hco3", "HCO3", r"^HCO3", None, None),
    ("glucose", "Glucose", r"^Glucose|혈당", 50, 400),
    ("lactate", "Lactate", r"^Lactate|젖산", None, 4.0),
    ("wbc", "WBC", r"^WBC|White Blood", None, None),
    ("anc", "ANC", r"^ANC\b", 0.5, None),
    ("hb", "Hb", r"^Hb\b|^Hb\(|Hemoglobin|혈색소", 7.0, None),
    ("plt", "PLT", r"^PLT|Platelet Count|혈소판", 50, None),
    ("inr", "INR", r"INR", None, 3.0),
    ("aptt", "aPTT", r"aPTT|^PTT", None, None),
    ("fibrinogen", "Fibrinogen", r"^Fibrinogen", 100, None),
    ("tbil", "T.bil", r"Bilirubin, ?total|^Bilirubin|총빌리루빈", None, None),
    ("ast", "AST", r"\bAST\b|GOT", None, None),
    ("alt", "ALT", r"\bALT\b|GPT", None, None),
    ("albumin", "Albumin", r"^Albumin", None, None),
    ("crp", "CRP", r"CRP|C-Reactive", None, None),
    ("procalcitonin", "Procalcitonin", r"procalcitonin", None, None),
    ("ph", "pH", r"^pH$|^pH\b", 7.2, 7.6),
    ("pco2", "pCO2", r"^pCO|^pCO₂", None, None),
    ("po2", "pO2", r"^pO|^pO₂", None, None),
    ("mg", "Mg", r"^Magnesium|^Mg\b", None, None),
    ("phos", "P", r"^Phosph", None, None),
    ("ica", "iCa", r"Calcium, ionized|이온화칼슘", None, None),
    ("tni", "Troponin I", r"Troponin", None, None),
    ("ckmb", "CK-MB", r"CKMB|CK-MB", None, None),
    ("vanco_level", "Vancomycin 농도", r"^Vancomycin", None, None),
    ("genta_level", "Gentamicin 농도", r"^Gentamicin", None, None),
    ("amika_level", "Amikacin 농도", r"^Amikacin", None, None),
]
LAB_DISPLAY = {c: d for c, d, _, _, _ in LAB_CONCEPTS}
LAB_CRIT = {c: (lo, hi) for c, _, _, lo, hi in LAB_CONCEPTS}
LAB_FALLBACK_REF = {"cr": (0.5, 1.2), "bun": (8, 23), "na": (135, 145), "k": (3.5, 5.1), "glucose": (70, 140), "lactate": (0.5, 2.0),
                    "wbc": (4.0, 10.0), "hb": (12.0, 16.0), "plt": (150, 400), "inr": (0.8, 1.2), "tbil": (0.2, 1.2), "ph": (7.35, 7.45),
                    "pco2": (35, 45), "po2": (80, 100), "mg": (1.6, 2.6), "crp": (0, 0.5), "albumin": (3.3, 5.2)}
KEY_TREND_LABS = ["cr", "lactate", "wbc", "hb", "plt", "tbil", "k", "na"]

# 약효군: (class, regex on name/ingredient, ATC prefix list)
MED_CLASSES = [
    ("vasopressor", r"norepinephrine|norpin|vasopressin|epinephrine|adrenalin|dopamine|dobutamine|phenylephrine|milrinone", ["C01CA", "H01BA01", "C01CE"]),
    ("sedation", r"propofol|fresofol|dexmedetomidine|precedex|midazolam|remifentanil|remiva|fentanyl|morphine|ketamine|lorazepam", ["N01AX10", "N05CM18", "N05CD08", "N01AH", "N02A"]),
    ("insulin", r"insulin|humulin|novorapid|lantus|levemir", ["A10A"]),
    ("anticoagulant", r"heparin|enoxaparin|clexane|crexane|warfarin|apixaban|rivaroxaban|edoxaban|dabigatran|argatroban|nafamostat", ["B01AA", "B01AB", "B01AE", "B01AF"]),
    ("antiplatelet", r"aspirin|clopidogrel|ticagrelor|prasugrel|cilostazol", ["B01AC"]),
    ("antibiotic", r"vancomycin|teicoplanin|piperacillin|tazobactam|meropenem|imipenem|ertapenem|ceftriaxone|cefazolin|cefepime|ceftazidime|cefotaxime|cefuroxime|ampicillin|amoxicillin|ciprofloxacin|levofloxacin|moxifloxacin|metronidazole|linezolid|colistin|amikacin|gentamicin|tigecycline|clindamycin|netilmicin|azithromycin|clarithromycin|trimethoprim|sulfamethoxazole|doxycycline|daptomycin", ["J01"]),
    ("antifungal", r"fluconazole|voriconazole|caspofungin|micafungin|amphotericin|itraconazole", ["J02"]),
    ("diuretic", r"furosemide|lasix|torsemide|spironolactone|bumetanide", ["C03"]),
    ("respiratory_stimulant", r"caffeine|카페인|aminophylline|theophylline", ["N06BC01", "R03DA"]),
    ("surfactant", r"poractant|beractant|calfactant|curosurf|surfacten|survanta|surfactant", ["R07AA"]),
    ("nutrition", r"TPN|parenteral|amino ?acid|trophamine|aminoven|primene|lipid|smoflipid|intralipid|omegaven|soluvit|vitalipid|calcium gluconate|glycophos", ["B05BA", "A11"]),
    ("anticonvulsant", r"phenobarbital|levetiracetam|phenytoin|fosphenytoin", ["N03A"]),
    ("fluid", r"saline|hartmann|dextrose|plasma solution|ringer|D5W|albumin 5|NaCl", ["B05"]),
]
HIGH_ALERT = {"vasopressor", "sedation", "insulin", "anticoagulant"}
DUR_PAIRS_DEFAULT = []  # 프로파일 dur_pairs 로 채움

# 시술/사건 어휘(검증기의 언급 탐지용). 기록에 없으면 Not Addressed 로 판정됨.
EVENT_VOCAB = [
    ("ev_intubation", r"삽관|intubation"),
    ("ev_vent", r"인공호흡|ventilat"),
    ("ev_trach", r"기관절개|tracheostomy"),
    ("ev_crrt", r"CRRT|CVVHDF|지속적 신대체|투석"),
    ("ev_ecmo", r"ECMO|에크모|체외막산소"),
    ("ev_bronch", r"기관지경|bronchoscopy"),
    ("ev_eeg", r"뇌파|\bEEG\b"),
    ("ev_echo", r"심초음파|echocardiograph"),
    ("ev_transfusion", r"수혈|transfusion"),
    ("ev_mrsa", r"MRSA|VRE|CRE\b"),
    ("ev_cpr", r"심폐소생|\bCPR\b"),
    ("ev_bloodcx", r"혈액배양|blood culture"),
    ("ev_surgery", r"수술|operation"),
]

KCD_NAMES = {  # 자주 쓰이는 KCD-8 코드 표시명(사전에 없으면 코드만 표시)
    "P070": "극저체중 출생아", "P071": "기타 저체중 출생아", "P072": "극도의 미성숙", "P073": "기타 조산아", "P220": "신생아의 호흡곤란증후군",
    "P221": "신생아의 일과성 빈호흡", "P271": "출생 전후기에 발생한 기관지폐형성이상", "P293": "신생아의 지속성 태아순환", "P369": "신생아의 상세불명 세균성 패혈증",
    "P590": "조산과 관련된 신생아 황달", "P599": "상세불명의 신생아 황달", "P704": "기타 신생아 저혈당증", "P77": "태아 및 신생아의 괴사성 장염",
    "P0730": "기타 조산아", "Q210": "심실중격결손", "Q250": "동맥관 개존",
    "A419": "패혈증, 상세불명", "C220": "간세포암종", "E149": "당뇨병, 상세불명", "E7808": "순수고콜레스테롤혈증",
    "E789": "지질단백질대사장애, 상세불명", "I109": "본태성 고혈압", "I209": "협심증, 상세불명", "I251": "죽상경화성 심장병",
    "I350": "대동맥판 협착", "I4890": "심방세동 및 조동, 상세불명", "I509": "심부전, 상세불명", "I839": "하지의 정맥류",
    "J189": "폐렴, 상세불명", "J9600": "급성 호흡부전", "K219": "식도염이 없는 위-식도역류병", "N179": "급성 신부전, 상세불명",
    "N185": "만성 신장병 5기", "N189": "만성 신장병, 상세불명", "Z944": "간이식 상태", "I214": "급성 심내막하 심근경색증",
    "I639": "뇌경색증, 상세불명", "I619": "뇌내출혈, 상세불명", "J440": "급성 하기도감염을 동반한 만성 폐쇄성 폐질환",
}
ICD9P_NAMES = {
    "01.24": "개두술(기타)", "01.59": "뇌 병변 절제/파괴(기타)", "31.1": "일시적 기관절개술", "32.4": "폐엽절제술",
    "35.1": "개방 심장판막성형술", "35.2": "심장판막 치환술", "36.12": "관상동맥우회술(2개 혈관)", "38.44": "복부대동맥 절제·치환",
    "39.28": "두개외-두개내 혈관 우회술", "39.51": "동맥류 결찰", "39.65": "체외막산소공급(ECMO)", "39.98": "출혈 조절(기타)",
    "42.4": "식도절제술", "50.59": "간이식(기타)", "54.11": "시험적 개복술", "86.28": "절제적 변연절제술", "86.89": "피부 복원(기타)",
}


def norm_icd9p(code):
    code = str(code).strip()
    m = re.match(r"^(\d)\.(\d+)$", code)
    if m:
        code = "0%s.%s" % (m.group(1), m.group(2))
    return code


def classify_med(name, atc=None):
    n = (name or "").lower()
    a = (atc or "").upper()
    for cls, rx, atcs in MED_CLASSES:
        if re.search(rx, n, re.I) or (a and any(a.startswith(p) for p in atcs)):
            return cls
    return "other"


def med_generic(name):
    """'Norpin 4mg/4ml inj(Norepinephrine)' → 'Norepinephrine'. 괄호 속 성분명 우선."""
    s = str(name or "")
    m = re.findall(r"\(([A-Za-z][A-Za-z \-/,]+)\)", s)
    if m:
        return m[-1].strip()
    for cls, rx, _ in MED_CLASSES:
        mm = re.search(rx, s, re.I)
        if mm:
            return mm.group(0).capitalize()
    w = re.match(r"([A-Za-z가-힣][A-Za-z가-힣\-]+)", s)
    return w.group(1) if w else s[:20]


def map_items(d_items):
    """D_ITEMS → itemid: (concept, label, linksto)."""
    out = {}
    if d_items is None or d_items.empty:
        return out
    for _, r in d_items.iterrows():
        label = str(r.get("label", ""))
        short = label.split(" > ")[-1] if " > " in label else label
        abbr = str(r.get("abbreviation", ""))
        txt = short + " | " + abbr if " > " in label else label + " | " + abbr
        link = str(r.get("linksto", "")).lower()
        concept = None
        if "output" in link:
            for c, rx in OUTPUT_CONCEPTS:
                if re.search(rx, txt, re.I):
                    concept = c
                    break
            concept = concept or "output_other"
        elif "input" in link:
            if re.search(MED_CLASSES[0][1], txt, re.I):
                concept = "in_vasopressor"
            else:
                for c, rx in INPUT_CONCEPTS:
                    if re.search(rx, txt, re.I):
                        concept = c
                        break
            concept = concept or "in_other"
        elif "procedure" in link:
            concept = "procedure"
        elif "datetime" in link:
            concept = "datetime"
        else:
            for c, disp, rx, ex in CHART_CONCEPTS:
                if re.search(rx, txt, re.I) and not (ex and re.search(ex, txt, re.I)):
                    concept = c
                    break
        out[r["itemid"]] = (concept, label.split(" > ")[-1] if " > " in label else label, link, str(r.get("category", "")))
    return out


def map_labs(d_lab):
    out = {}
    if d_lab is None or d_lab.empty:
        return out
    for _, r in d_lab.iterrows():
        label = str(r.get("label", ""))
        concept = None
        for c, disp, rx, lo, hi in LAB_CONCEPTS:
            if re.search(rx, label, re.I):
                concept = c
                break
        out[r["itemid"]] = (concept, label)
    return out


# =============================================================================
# 2-b. 병원 프로파일 (코호트·두 문서의 섹션·필수 체크리스트·신생아 기준값·라벨·CDSS 를 병원별로 수정)
#   python kcd_nicu.py init-profile --out my_hospital.json  → 수정 → --profile my_hospital.json
#   병원 JSON 은 바꿀 키만 쓰면 됨(나머지는 기본값). 목록 키는 통째 교체.
#   체크리스트 부분 수정: documents.<summary|parent>.checklist_add / checklist_remove / checklist_override
#   기본값의 신생아 기준·목표 범위·참고범위는 예시이며 병원 NICU 프로토콜과 검사실 기준으로 교체해 사용.
# =============================================================================


def _ck(key, label, section, required, sources, **kw):
    d = {"key": key, "label": label, "section": section, "required": required, "sources": sources}
    d.update(kw)
    return d


def _f(*types, **kw):
    d = {"type": "facts", "fact_types": list(types)}
    d.update(kw)
    return d


LINE_RX = "UVC|UAC|PICC|umbilical|제대|central|중심정맥|CVC|arterial line|동맥관|A-line"

DEFAULT_PROFILE = {
    "profile_name": "default-nicu",
    "basis": "NICU 회진·인계 요약과 부모 면담에서 일반적으로 다루는 항목을 바탕으로 한 예시 양식. 병원 서식과 NICU 프로토콜에 맞게 수정해 사용.",
    "cohort": {"careunit_regex": "NICU|신생아|neonat|NBICU", "max_age_years": 0, "match": "any"},
    "daily": {"cutoff_hour": 7, "window_hours": 24, "include_discharge_day": True},
    "documents": {
        "summary": {
            "title": "NICU 1일 요약지 (자동 생성 초안, 의료진 검토 필요)",
            "sections": [{"id": "D1", "title": "1. 환아 정보"}, {"id": "D2", "title": "2. 진단·문제 목록"},
                         {"id": "D3", "title": "3. 지난 24시간 경과"}, {"id": "D4", "title": "4. 영양·수액"},
                         {"id": "D5", "title": "5. 배설·배액"}, {"id": "D6", "title": "6. 검사"},
                         {"id": "D7", "title": "7. 투약"}, {"id": "D8", "title": "8. 주의사항 (CDSS 규칙 기반)"}],
            "cdss_section": "D8",
            "section_cap": {"D1": 8, "D2": 4, "D3": 16, "D4": 6, "D5": 8, "D6": 16, "D7": 16},
            "checklist": [
                _ck("patient_info", "성별·입원일수·NICU 재실일수", "D1", True, [_f("patient_info")]),
                _ck("weight", "체중(현재·전일·입실 시 대비)", "D1", True, [_f("weight")]),
                _ck("gestational_age", "재태주수·교정주수", "D1", True, [{"type": "chart_regex", "regex": "재태|gestational|\\bGA\\b|주수|교정"}, {"type": "manual"}],
                    mention_regex="재태|주수|\\bGA\\b|교정"),
                _ck("birth_weight", "출생체중", "D1", True, [{"type": "chart_regex", "regex": "출생 ?체중|birth ?weight"}, {"type": "manual"}],
                    mention_regex="출생 ?체중|birth ?weight"),
                _ck("service", "담당 진료과", "D1", False, [{"type": "service"}]),
                _ck("isolation", "격리·다제내성균", "D1", True,
                    [{"type": "micro_org_regex", "regex": "MRSA|VRE|CRE|CRAB|CRPA|carbapenem|methicillin.resistant|vancomycin.resistant"},
                     {"type": "chart_regex", "regex": "격리|isolation"}, {"type": "manual"}], mention_regex="격리|isolation|MRSA|VRE|CRE|다제내성"),
                _ck("dx_main", "주진단", "D2", True, [_f("diagnosis_main")]),
                _ck("dx_other", "기타 진단", "D2", False, [_f("diagnosis_other")]),
                _ck("prematurity", "미숙아·저체중 관련 진단", "D2", False, [_f("prematurity")]),
                _ck("vitals", "활력징후 범위", "D3", True, [_f("vitals_24h")]),
                _ck("events", "무호흡·서맥·산소포화도 저하", "D3", True, [_f("neo_events")], mention_regex="무호흡|서맥|산소포화도 저하|apnea|brady|desat"),
                _ck("respiration", "호흡 보조", "D3", True, [_f("resp_support")]),
                _ck("resp_change", "호흡 보조 변경", "D3", False, [_f("resp_change")]),
                _ck("lines", "정맥관·카테터(UVC·UAC·PICC 등)", "D3", True,
                    [{"type": "proc_regex", "regex": LINE_RX, "note": "제거 여부는 데이터로 확인되지 않음"},
                     {"type": "chart_regex", "regex": "insertion|삽입|catheter|카테터|UVC|UAC|PICC"}],
                    mention_regex="UVC|UAC|PICC|제대|중심정맥|카테터|catheter|정맥관"),
                _ck("infusions", "지속주입(승압·진정)·CRRT", "D3", False, [_f("infusion_episode", "crrt_episode")]),
                _ck("procedures", "시술·영상검사", "D3", False, [_f("procedure", "imaging")]),
                _ck("cranial_us", "뇌초음파(뇌실내출혈 선별)", "D3", False,
                    [{"type": "facts", "fact_types": ["procedure", "imaging"], "text_regex": "brain|cranial|head|뇌"}, {"type": "proc_regex", "regex": "brain|cranial|head|뇌"}],
                    mention_regex="뇌초음파|brain|cranial|뇌실내출혈|IVH"),
                _ck("pending", "미결 배양·검사", "D3", True, [_f("micro_pending")], gap_text="미결 배양·검사: 데이터상 미보고 배양 없음, 미결 검사 확인 필요.",
                    mention_regex="미보고|미결|pending|결과 대기"),
                _ck("intake", "섭취(정맥·장관)와 mL/kg/일", "D4", True, [_f("intake_24h")]),
                _ck("feeding_route", "수유·영양 경로", "D4", True,
                    [{"type": "input_regex", "regex": "경구|oral|Levin|\\bOG\\b|\\bNG\\b|feeding|경관|수유|모유|분유"}, {"type": "med_class", "classes": ["nutrition"]}],
                    mention_regex="수유|경관|경구|모유|분유|정맥영양|TPN|영양"),
                _ck("urine", "소변량(mL/kg/h)", "D5", True, [_f("urine_24h")]),
                _ck("stool", "대변", "D5", False, [_f("stool")]),
                _ck("drains", "배액관", "D5", False, [_f("drain")]),
                _ck("labs", "최근 검사", "D6", True, [_f("lab_latest", "lab_normal", "lab_none")]),
                _ck("bilirubin", "빌리루빈", "D6", True, [{"type": "facts", "concepts": ["tbil"]}], gap_text="빌리루빈: 최근 24시간 검사 기록 없음, 필요 시 확인.",
                    mention_regex="빌리루빈|bilirubin|T\\.bil|황달"),
                _ck("glucose", "혈당", "D6", True, [{"type": "facts", "concepts": ["glucose", "bst"]}], gap_text="혈당: 최근 24시간 검사 기록 없음, 필요 시 확인.",
                    mention_regex="혈당|glucose|BST"),
                _ck("meds_active", "투여 중 약물", "D7", True, [_f("med_active", "fluids", "infusion_episode", status=None)]),
                _ck("high_alert", "고위험 약물(승압제·진정제·인슐린·항응고제)", "D7", True,
                    [{"type": "med_class", "classes": ["vasopressor", "sedation", "insulin", "anticoagulant"]}], gap_text="고위험 약물: 기준 시각 투여 중 기록 없음.",
                    mention_regex="승압|진정|인슐린|항응고|dopamine|epinephrine|insulin|heparin"),
                _ck("antibiotics", "항생제", "D7", False, [{"type": "med_class", "classes": ["antibiotic", "antifungal"]}]),
                _ck("meds_changed", "오늘 중단한 약물", "D7", False, [_f("med_stopped")]),
            ],
        },
        "parent": {
            "title": "오늘 아기의 하루 (부모님 설명지)",
            "notice": "이 설명지는 진료 기록을 바탕으로 자동 작성된 초안이며, 담당 의료진이 확인한 뒤 전달합니다.",
            "sections": [{"id": "P1", "title": "오늘 아기의 상태"}, {"id": "P2", "title": "숨쉬기"}, {"id": "P3", "title": "먹는 양과 몸무게"},
                         {"id": "P4", "title": "소변과 대변"}, {"id": "P5", "title": "오늘 받은 검사와 처치"}, {"id": "P6", "title": "사용 중인 약"},
                         {"id": "P7", "title": "의료진이 살펴보고 있는 점"}, {"id": "P8", "title": "궁금하신 점이 있으면"}],
            "cdss_section": "P7",
            "closing_section": "P8",
            "section_of": {"patient_info": "P1", "resp_support": "P2", "resp_change": "P2", "neo_events": "P2", "weight": "P3", "intake_24h": "P3",
                           "urine_24h": "P4", "stool": "P4", "lab_latest": "P5", "lab_normal": "P5", "lab_change": "P5", "lab_none": "P5",
                           "procedure": "P5", "imaging": "P5", "micro_pending": "P5", "micro_positive": "P5", "micro_negative": "P5", "crrt_episode": "P5",
                           "med_active": "P6", "med_stopped": "P6", "infusion_episode": "P6", "fluids": "P6"},
            "show_lab_values": False,
            "parent_visible_rules": ["NEO_HYPOGLYCEMIA", "NEO_HYPERGLYCEMIA", "NEO_BILI_HIGH", "NEO_BILI_RISE", "NEO_TEMP_INSTABILITY",
                                     "NEO_WEIGHT_LOSS", "NEO_SEPSIS_MARKERS", "NEO_OLIGURIA"],
            "alert_text": {
                "NEO_HYPOGLYCEMIA": "혈당이 낮게 측정된 적이 있어 의료진이 혈당을 자주 확인하고 있습니다.",
                "NEO_HYPERGLYCEMIA": "혈당이 높게 측정된 적이 있어 의료진이 혈당을 확인하고 있습니다.",
                "NEO_BILI_HIGH": "황달 수치(빌리루빈)가 높게 나와 의료진이 치료가 필요한지 확인하고 있습니다.",
                "NEO_BILI_RISE": "황달 수치(빌리루빈)가 빠르게 오르고 있어 의료진이 자주 확인하고 있습니다.",
                "NEO_TEMP_INSTABILITY": "체온이 정상 범위를 벗어난 적이 있어 체온을 자주 확인하고 있습니다.",
                "NEO_WEIGHT_LOSS": "입원했을 때보다 몸무게가 많이 줄어 먹는 양을 함께 살펴보고 있습니다.",
                "NEO_SEPSIS_MARKERS": "감염이 있는지 확인하기 위한 검사 결과를 살펴보고 있습니다.",
                "NEO_OLIGURIA": "소변 양이 적어 의료진이 수분 공급을 살펴보고 있습니다.",
            },
            "empty_text": "오늘 기록에서 따로 말씀드릴 내용은 없습니다.",
            "gap_text": "{label}: 오늘 기록에서는 확인되지 않았습니다. 궁금하시면 담당 의료진에게 물어봐 주세요.",
            "closing": ["궁금한 점은 면회나 전화 상담 때 담당 의료진에게 편하게 물어봐 주세요.",
                        "아기의 상태는 하루 중에도 바뀔 수 있어, 자세한 설명은 담당 의료진이 직접 드립니다."],
            "checklist": [
                _ck("p_day", "입원 일수", "P1", True, [_f("patient_info")]),
                _ck("p_resp", "숨쉬기", "P2", True, [_f("resp_support", "resp_change")], mention_regex="숨|호흡|인공호흡기|산소"),
                _ck("p_events", "숨 멈춤·느린 맥박", "P2", False, [_f("neo_events", status="present")], mention_regex="산소포화도|심장 박동|숨 멈춤"),
                _ck("p_weight", "몸무게", "P3", True, [_f("weight")], mention_regex="몸무게|체중"),
                _ck("p_feeding", "먹는 양", "P3", True, [_f("intake_24h")], mention_regex="먹|수유|영양|수분"),
                _ck("p_output", "소변과 대변", "P4", False, [_f("urine_24h", "stool")], mention_regex="소변|대변"),
                _ck("p_tests", "오늘 받은 검사", "P5", True,
                    [_f("lab_latest", "lab_normal", "lab_change", "lab_none", "imaging", "procedure", "micro_pending", "micro_positive", "micro_negative")],
                    mention_regex="검사"),
                _ck("p_meds", "사용 중인 약", "P6", True, [_f("med_active", "fluids", "infusion_episode")], mention_regex="약|주사|수액"),
            ],
        },
    },
    "section_of": {},            # fact_type → 요약지 섹션 재배치. 예: {"drain": "D3"}
    "exclude_fact_types": [],    # 병원 양식에서 쓰지 않는 fact_type
    "salience": {},              # fact_type → 중요도(1~3) 재지정. 예: {"lab_normal": 2}
    "must_min_salience": 2,      # 이 값 이상이면 요약지 필수 포함 + 누락 검증 대상
    "windows": {"labs_hours": 24, "resp_recent_hours": 12, "infusion_episode_gap_hours": 6, "infusion_active_hours": 4,
                "med_active_hours": 24, "tdm_level_hours": 48, "crrt_active_hours": 12, "crrt_episode_gap_hours": 12},
    "neonatal": {                # 신생아 판단 기준(예시값 — 병원 NICU 프로토콜로 교체)
        "vitals_normal": {"hr": [100, 180], "rr": [30, 60], "spo2": [90, 100], "temp": [36.5, 37.5]},
        "desat_spo2": 85, "brady_hr": 100, "glucose_low": 45, "glucose_high": 180,
        "weight_loss_pct": 10, "weight_loss_max_day": 14, "urine_low_ml_kg_h": 1.0,
        "bili_rise_mg_dl_h": 0.2, "tsb_alert_mg_dl": 15, "crp_high_mg_dl": 1.0, "line_days_alert": 7,
        "fluid_ml_kg_day_by_day": {"1": [60, 90], "2": [80, 110], "3": [100, 130], "4": [120, 150], "5": [130, 160], "6": [140, 170], "7": [150, 180]},
        "fluid_default": [140, 180],
    },
    "critical_values": {"na": [125, 155], "k": [2.5, 6.5], "hb": [8.0, None], "plt": [50, None], "ph": [7.2, 7.5], "lactate": [None, 4.0],
                        "tbil": [None, 20], "ica": [0.8, None], "anc": [0.5, None]},
    "ref_ranges": {"glucose": [45, 150], "na": [133, 146], "k": [3.5, 6.0], "cr": [0.2, 1.0], "hb": [13, 22], "plt": [150, 450], "wbc": [5, 30],
                   "crp": [0, 1.0], "tbil": [0, 12], "bun": [3, 20], "ica": [1.0, 1.4]},
    "prefer_profile_ref": True,  # True 면 데이터의 (성인) 참고범위보다 위 ref_ranges 를 우선
    "trend_labs": ["tbil", "crp", "glucose", "na", "k", "cr", "hb", "plt", "wbc"],
    "line_regex": LINE_RX,
    "tdm": {"vancomycin": "vanco_level", "gentamicin": "genta_level", "amikacin": "amika_level"},
    "concepts": {"chart_extra": [], "lab_extra": [], "med_class_extra": []},   # 병원 고유 라벨 추가(우선 적용)
    "cdss": {"disabled": [], "severity_override": {}},   # 규칙 끄기·심각도 조정. rule_id 접두어 허용(예: "CRITICAL_LAB")
    "dur_pairs": [],             # [성분A, 성분B, 사유]
}

PROFILE = None
_BASE = {}


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_profile(path=None):
    p = copy.deepcopy(DEFAULT_PROFILE)
    if path:
        with open(path, encoding="utf-8") as fh:
            user = json.load(fh)
        ops = {}
        for dn, d in (user.get("documents") or {}).items():
            ops[dn] = (d.pop("checklist_add", []), set(d.pop("checklist_remove", [])), d.pop("checklist_override", {}))
        p = _merge(p, user)
        for dn, (add, rem, ovr) in ops.items():
            doc = p["documents"][dn]
            keys = [c["key"] for c in doc["checklist"]]
            for item in add:
                if item["key"] in keys:
                    doc["checklist"][keys.index(item["key"])] = item
                else:
                    doc["checklist"].append(item)
            doc["checklist"] = [c for c in doc["checklist"] if c["key"] not in rem]
            for c in doc["checklist"]:
                if c["key"] in ovr:
                    c.update(ovr[c["key"]])
    validate_profile(p)
    return p


SOURCE_TYPES = ("facts", "med_class", "chart_regex", "proc_regex", "output_regex", "input_regex", "service", "micro_org_regex", "manual")


def validate_profile(p):
    errs = []
    for dn in ("summary", "parent"):
        doc = p["documents"].get(dn)
        if not doc:
            errs.append("documents.%s 가 없음" % dn)
            continue
        sec = {s["id"] for s in doc["sections"]}
        if doc.get("cdss_section") not in sec:
            errs.append("%s.cdss_section %s 가 sections 에 없음" % (dn, doc.get("cdss_section")))
        for c in doc["checklist"]:
            if c.get("section") not in sec:
                errs.append("%s 체크리스트 %s 의 section %s 가 sections 에 없음" % (dn, c["key"], c.get("section")))
            for s in c.get("sources", []):
                if s.get("type") not in SOURCE_TYPES:
                    errs.append("%s 체크리스트 %s 의 source type %s 를 알 수 없음" % (dn, c["key"], s.get("type")))
                if "regex" in s:
                    try:
                        re.compile(s["regex"])
                    except re.error as e:
                        errs.append("%s 체크리스트 %s 정규식 오류: %s" % (dn, c["key"], e))
        if dn == "parent":
            for ft, s in doc.get("section_of", {}).items():
                if s not in sec:
                    errs.append("parent.section_of %s → %s 가 sections 에 없음" % (ft, s))
    ssec = {s["id"] for s in p["documents"]["summary"]["sections"]}
    for ft, s in p.get("section_of", {}).items():
        if s not in ssec:
            errs.append("section_of %s → %s 가 요약지 sections 에 없음" % (ft, s))
    if errs:
        raise SystemExit("프로파일 오류:\n  - " + "\n  - ".join(errs))


def CDSS_SEC():
    return PROFILE["documents"]["summary"]["cdss_section"] if PROFILE else "D8"


def PCDSS_SEC():
    return PROFILE["documents"]["parent"]["cdss_section"] if PROFILE else "P7"


def W(key):
    return float(PROFILE["windows"][key])


def NEO(key):
    return PROFILE["neonatal"][key]


def apply_profile(p):
    """프로파일을 전역 설정에 반영(섹션, 개념 사전, 약물 목록)."""
    global PROFILE
    PROFILE = p
    if not _BASE:
        _BASE.update(chart=list(CHART_CONCEPTS), lab=list(LAB_CONCEPTS), med=list(MED_CLASSES))
    CHART_CONCEPTS[:] = _BASE["chart"]
    LAB_CONCEPTS[:] = _BASE["lab"]
    MED_CLASSES[:] = _BASE["med"]
    SECTIONS.clear()
    for s in p["documents"]["summary"]["sections"]:
        SECTIONS[s["id"]] = s["title"]
    PARENT_SECTIONS.clear()
    for s in p["documents"]["parent"]["sections"]:
        PARENT_SECTIONS[s["id"]] = s["title"]
    for c, disp, rx in reversed(p["concepts"].get("chart_extra", [])):
        CHART_CONCEPTS.insert(0, (c, disp, rx, None))
    for row in reversed(p["concepts"].get("lab_extra", [])):
        c, disp, rx = row[:3]
        lo, hi = (list(row[3:5]) + [None, None])[:2]
        LAB_CONCEPTS.insert(0, (c, disp, rx, lo, hi))
        LAB_DISPLAY.setdefault(c, disp)
        ALIAS.setdefault(c, r"(?-i:(?<![A-Za-z0-9/])%s(?![A-Za-z0-9/]))" % re.escape(disp))
    for cls, rx in reversed(p["concepts"].get("med_class_extra", [])):
        MED_CLASSES.insert(0, (cls, rx, []))
    DUR_PAIRS_DEFAULT[:] = [tuple(x) for x in p.get("dur_pairs", [])]


def crit_of(c, lo, hi):
    v = PROFILE["critical_values"].get(c)
    return (v[0], v[1]) if v else (lo, hi)


# --------------------------------------------------------------- 체크리스트 평가


def _labelize(ctx, df):
    if df is not None and not df.empty and "full_label" not in df.columns and "itemid" in df.columns:
        df["full_label"] = df["itemid"].map(lambda i: " ".join(str(x) for x in ctx.item_map.get(i, (None, str(i), "", ""))[1:4:2]))
    return df


def eval_source(ctx, facts, item, src, doc):
    t = src["type"]
    sec = item["section"]
    pre = "chk:" if doc == "summary" else "pchk:"
    key = item["key"]
    if t == "facts":
        st = src.get("status")
        types = src.get("fact_types")
        cons = set(src.get("concepts", []))
        trx = src.get("text_regex")
        return [f for f in facts if (not types or f.ftype in types) and (not cons or (f.concepts & cons)) and (st is None or f.status == st)
                and f.ftype not in ("checklist_gap", "parent_gap") and (not trx or re.search(trx, f.text, re.I))]
    if t == "med_class":
        cl = set(src["classes"])
        return [f for f in facts if f.ftype in ("med_active", "infusion_episode") and f.status == "active" and f.extra.get("cls") in cl]
    if t == "micro_org_regex":
        return [f for f in facts if f.ftype == "micro_positive" and re.search(src["regex"], f.text, re.I)]
    if t == "service":
        sv = ctx.services
        if sv is None or sv.empty or "curr_service" not in sv.columns:
            return []
        sv = sv.copy()
        sv["t"] = to_dt(sv["transfertime"]) if "transfertime" in sv.columns else pd.NaT
        sv = sv[(sv["t"].isna() | (sv["t"] <= ctx.t0)) & sv["curr_service"].notna() & ~sv["curr_service"].astype(str).isin(["nan", ""])].sort_values("t")
        if sv.empty:
            return []
        return [Fact("checklist_item", sec, "%s: %s." % (item["label"], sv.iloc[-1]["curr_service"]), 2, concepts={pre + key, "service"},
                     prov=[{"table": "SERVICES", "row_id": ctx.hadm_id}])]
    rx = src.get("regex")
    if t == "chart_regex":
        out = []
        for df, table, idc in ((ctx.chart, "CHARTEVENTS", "chartevent_id"), (ctx.dte, "DATETIMEEVENTS", "datetimeevent_id")):
            if df is None or df.empty or "itemid" not in df.columns:
                continue
            _labelize(ctx, df)
            if "t" not in df.columns:
                df["t"] = to_dt(df["charttime"])
            m = df[df["full_label"].str.contains(rx, case=False, regex=True, na=False) & (df["t"] <= ctx.t0)]
            for lab, g in m.sort_values("t").groupby("full_label"):
                r = g.iloc[-1]
                val = r.get("value")
                val = fmt_num(float(val)) if pd.notna(pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]) else str(val)
                short = ctx.item_map.get(r["itemid"], (None, lab))[1]
                out.append(Fact("checklist_item", sec, "%s: %s %s(%s 기록)." % (item["label"], short, val, fmt_t(r["t"])), 2, concepts={pre + key},
                                time=r["t"], prov=_prov(table, g.tail(1), idc)))
        return out[:3]
    if t == "proc_regex":
        p = ctx.proc
        if p.empty:
            return []
        p = p[(p["t"] <= ctx.t0) & p["label"].astype(str).str.contains(rx, case=False, regex=True, na=False)]
        if "value" in p.columns:
            vv = to_num(p["value"])
            p = p[vv.isna() | (vv != 0)]
        if p.empty:
            return []
        parts = ["%s(첫 기록 %s)" % (lab, fmt_t(g["t"].min())) for lab, g in p.groupby("label")]
        note = (", " + src["note"]) if src.get("note") else ""
        return [Fact("checklist_item", sec, "%s: %s%s." % (item["label"], ", ".join(parts[:4]), note), 2,
                     concepts={pre + key, "procedure"}, time=p["t"].min(), prov=_prov("PROCEDUREEVENTS", p.head(3), "procedureevent_id"))]
    if t in ("output_regex", "input_regex"):
        df = ctx.out if t == "output_regex" else ctx.inp
        if df.empty:
            return []
        d = ctx.win(df)
        d = d[d["label"].astype(str).str.contains(rx, case=False, regex=True, na=False)]
        if d.empty:
            return []
        labs = sorted(d["label"].astype(str).unique())
        kind = "배설" if t == "output_regex" else "섭취"
        return [Fact("checklist_item", sec, "%s: 지난 %s시간 %s 기록 경로 %s." % (item["label"], fmt_num(round(hours_of(ctx))), kind, ", ".join(labs[:4])), 2,
                     concepts={pre + key}, prov=_prov("OUTPUTEVENTS" if t == "output_regex" else "INPUTEVENTS", d.head(2),
                                                     "outputevent_id" if t == "output_regex" else "inputevent_id"))]
    return []  # manual


def apply_checklist(ctx, facts, doc):
    added = []
    ctx.checklist[doc] = []
    D = PROFILE["documents"][doc]
    pre = "chk:" if doc == "summary" else "pchk:"
    for item in D["checklist"]:
        got, used = [], None
        for src in item.get("sources", []):
            got = eval_source(ctx, facts + added, item, src, doc)
            if got:
                used = src["type"]
                break
        req = bool(item.get("required"))
        if got:
            for f in got:
                if f not in facts and f not in added:
                    added.append(f)
                if req and doc == "summary":
                    f.salience = max(f.salience, 2)
                f.concepts.add(pre + item["key"])
            ctx.checklist[doc].append({"item": item, "facts": got, "source": used, "status": "data"})
        elif req:
            if doc == "summary":
                txt = item.get("gap_text") or "%s: 데이터에 기록 없음, 확인 필요." % item["label"]
                g = Fact("checklist_gap", item["section"], txt, 2, concepts={pre + item["key"]}, status="gap", extra={"checklist": item["key"]})
            else:
                txt = item.get("gap_text") or D.get("gap_text", "{label}: 기록에서 확인되지 않았습니다.").format(label=item["label"])
                g = Fact("parent_gap", item["section"], txt, 1, concepts={pre + item["key"]}, status="gap", extra={"checklist": item["key"]})
            added.append(g)
            ctx.checklist[doc].append({"item": item, "facts": [g], "source": "gap", "status": "gap"})
        else:
            ctx.checklist[doc].append({"item": item, "facts": [], "source": None, "status": "absent"})
    return added


def checklist_report(ctx, doc):
    rows = []
    for c in ctx.checklist.get(doc, []):
        it = c["item"]
        rows.append({"key": it["key"], "label": it["label"], "section": it["section"], "required": bool(it.get("required")),
                     "status": c["status"], "source": c["source"], "fact_ids": [f.fid for f in c["facts"]]})
    return rows


def checklist_in_note(V, sents, verdicts, report, doc="summary"):
    """기록지(생성·외부 작성)가 체크리스트 항목을 다루는지. 인용 또는 항목 언급으로 판정."""
    vd = {v["sid"]: v for v in verdicts}
    ok_s = [s for s in sents if vd.get(s["sid"], {}).get("label") not in ("Contradicted", "NotAddressed") and not s.get("fixed")]
    pre = "chk:" if doc == "summary" else "pchk:"
    out = []
    for r in report:
        if r["status"] == "absent" and not r["required"]:
            out.append(dict(r, in_note="n/a"))
            continue
        fids = set(r["fact_ids"])
        hit = any(fids & set(s.get("cites", [])) for s in ok_s)
        if not hit:
            rx = V.chk_alias.get(pre + r["key"])
            facts = [V.F[f] for f in fids if f in V.F]
            for s in ok_s:
                if rx and re.search(rx, s["text"], re.I):
                    hit = True
                    break
                if facts and any(V._num_match(x, f.key_nums) for f in facts if f.key_nums for x in extract_nums(s["text"])) and \
                        (V.mentions(s["text"]) & set().union(*[f.concepts for f in facts])):
                    hit = True
                    break
        out.append(dict(r, in_note="covered" if hit else "missing"))
    return out


# =============================================================================
# 3. Fact Sheet 생성 (기준 시각 t0 = 매일 cutoff 시각, 직전 24시간 창)
# =============================================================================

SECTIONS = OrderedDict()  # 요약지 섹션(프로파일에서 채움)
PARENT_SECTIONS = OrderedDict()  # 부모님 설명지 섹션(프로파일에서 채움)
NUM_RX = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9_])")
TIME_RX = re.compile(r"(?:(\d{4})-)?(\d{2})-(\d{2}) (\d{2}):(\d{2})")
CITE_RX = re.compile(r"\[(F\d{3}(?:\s*,\s*F\d{3})*)\]")
CODE_RX = re.compile(r"\b[A-Z]\d{2,4}\b|(?<=ICD-9 )\d{2}\.\d{1,2}")


def extract_times(text):
    return ["%s-%s %s:%s" % (m.group(2), m.group(3), m.group(4), m.group(5)) for m in TIME_RX.finditer(text)]


def extract_codes(text):
    return CODE_RX.findall(CITE_RX.sub(" ", text))


def strip_for_nums(text):
    t = CITE_RX.sub(" ", text)
    t = TIME_RX.sub(" ", t)
    t = re.sub(r"\b\d{2}-\d{2}\b", " ", t)
    t = CODE_RX.sub(" ", t)
    t = re.sub(r"\(규칙 [^)]*\)", " ", t)
    return t


def strip_keep(text):
    """위치를 보존하면서 인용·시각·날짜·코드를 같은 길이의 공백으로 가림."""
    out = text
    for rx in (CITE_RX, TIME_RX, re.compile(r"\b\d{2}-\d{2}\b"), CODE_RX, re.compile(r"\(규칙 [^)]*\)")):
        out = rx.sub(lambda m: " " * len(m.group(0)), out)
    return out


def extract_nums(text):
    return [float(x) for x in NUM_RX.findall(strip_for_nums(text))]


class Fact(object):
    def __init__(self, ftype, section, text, salience, concepts=(), key_nums=(), status=None, prov=(), time=None, extra=None):
        self.fid = None
        self.ftype = ftype
        self.section = section
        self.text = text
        self.salience = salience
        self.concepts = set(concepts)
        self.key_nums = [float(fmt_num(x)) for x in key_nums if x is not None and not (isinstance(x, float) and math.isnan(x))]
        self.status = status
        self.prov = list(prov)[:20]
        self.time = time
        self.extra = extra or {}
        self.allowed_nums = extract_nums(text)
        self.allowed_nums += [abs(x) for x in self.allowed_nums if x < 0]
        self.allowed_times = extract_times(text)
        self.codes = extract_codes(text)

    @property
    def must(self):
        return self.salience >= (PROFILE["must_min_salience"] if PROFILE else 2)

    def to_dict(self):
        return {"fact_id": self.fid, "fact_type": self.ftype, "section": self.section, "text_ko": self.text, "salience": self.salience,
                "must_include": self.must, "concepts": sorted(self.concepts), "key_nums": self.key_nums, "status": self.status,
                "time": None if self.time is None or pd.isna(self.time) else str(self.time), "provenance": self.prov, "extra": self.extra}


def _prov(table, df, idcol):
    if df is None or len(df) == 0:
        return []
    col = idcol if idcol in df.columns else None
    vals = df[col].astype(str).tolist() if col else [str(i) for i in df.index.tolist()]
    return [{"table": table, "row_id": v} for v in vals[:10]]


class StayContext(object):
    """한 ICU stay의 모든 원천 데이터와 정규화 이벤트."""

    def __init__(self, kd, icu_row, item_map, lab_map, d_test=None, t0=None, win_hours=24.0):
        self.kd = kd
        self.row = icu_row
        self.subject_id = icu_row["subject_id"]
        self.hadm_id = icu_row["hadm_id"]
        self.stay_id = icu_row["stay_id"]
        self.intime = pd.Timestamp(icu_row["intime_dt"])
        self.outtime = pd.Timestamp(icu_row["outtime_dt"])
        self.t0 = pd.Timestamp(t0) if t0 is not None else self.outtime
        self.win_start = max(self.intime, self.t0 - pd.Timedelta(hours=win_hours))
        self.is_neonate = False
        self.infusion_generics = set()
        self.checklist = {}
        self.day_index, self.n_days = 1, 1
        self.item_map = item_map
        self.lab_map = lab_map
        self.d_test = d_test
        self.events_concepts = set()
        self.raw_values = defaultdict(list)  # concept -> [values] (검증기 전역 조회용)
        h = [self.hadm_id]

        def sub(t):
            df = kd.get(t, None if t in SMALL else h)
            if df.empty or "hadm_id" not in df.columns:
                return df.iloc[0:0] if not df.empty else df
            return df[df["hadm_id"] == self.hadm_id].copy()

        self.adm = sub("ADMISSIONS")
        pats = kd.get("PATIENTS")
        self.pat = pats[pats["subject_id"] == self.subject_id] if not pats.empty else pats
        self.transfers = sub("TRANSFERS")
        self.dx = sub("DIAGNOSES_ICD")
        self.px = sub("PROCEDURES_ICD")
        self.chart = sub("CHARTEVENTS")
        self.lab = sub("LABEVENTS")
        self.inp = sub("INPUTEVENTS")
        self.out = sub("OUTPUTEVENTS")
        self.proc = sub("PROCEDUREEVENTS")
        self.emar = sub("EMAR")
        self.emar_d = sub("EMAR_DETAIL")
        self.rx = sub("PRESCRIPTIONS")
        self.micro = sub("MICROBIOLOGYEVENTS")
        self.test = sub("TESTEVENTS")
        self.dte = sub("DATETIMEEVENTS")
        self.services = sub("SERVICES")
        self._prep()

    def _prep(self):
        if not self.chart.empty:
            c = self.chart
            c["t"] = to_dt(c["charttime"])
            c["v"] = to_num(c["valuenum"]) if "valuenum" in c.columns else to_num(c["value"])
            if "value" in c.columns:
                c["v"] = c["v"].fillna(to_num(c["value"]))
            c["concept"] = c["itemid"].map(lambda i: self.item_map.get(i, (None,))[0])
            self.chart = c
            for k, g in c.groupby("concept"):
                self.raw_values[k].extend(g["v"].dropna().tolist())
                self.events_concepts.add(k)
        if not self.lab.empty:
            l = self.lab
            l["t"] = to_dt(l["charttime"])
            l["v"] = to_num(l["valuenum"]) if "valuenum" in l.columns else to_num(l["value"])
            if "value" in l.columns:
                l["v"] = l["v"].fillna(to_num(l["value"]))
            l["concept"] = l["itemid"].map(lambda i: self.lab_map.get(i, (None,))[0])
            l["label"] = l["itemid"].map(lambda i: self.lab_map.get(i, (None, str(i)))[1])
            for c in ("ref_range_lower", "ref_range_upper"):
                l[c] = to_num(l[c]) if c in l.columns else float("nan")
            self.lab = l
            for k, g in l.groupby("concept"):
                self.raw_values[k].extend(g["v"].dropna().tolist())
                self.events_concepts.add(k)
        for df, tcol in ((self.inp, "starttime"), (self.out, "charttime"), (self.proc, "starttime")):
            if not df.empty:
                df["t"] = to_dt(df[tcol])
                df["concept"] = df["itemid"].map(lambda i: self.item_map.get(i, (None,))[0])
                df["label"] = df["itemid"].map(lambda i: self.item_map.get(i, (None, str(i)))[1])
                df["cat"] = df["itemid"].map(lambda i: self.item_map.get(i, (None, None, None, ""))[3])
                vcol = "amount" if "amount" in df.columns else "value"
                df["v"] = to_num(df[vcol]) if vcol in df.columns else float("nan")
        if not self.inp.empty and "endtime" in self.inp.columns:
            self.inp["t_end"] = to_dt(self.inp["endtime"])
        if not self.proc.empty and "endtime" in self.proc.columns:
            self.proc["t_end"] = to_dt(self.proc["endtime"])
        if not self.emar.empty:
            e = self.emar
            e["t"] = to_dt(e["charttime"])
            e["generic"] = e["medication"].map(med_generic)
            e["cls"] = e["medication"].map(lambda n: classify_med(n))
            if not self.emar_d.empty and "emar_id" in self.emar_d.columns:
                cols = [c for c in ["emar_id", "infusion_rate_hr", "infusion_rate_min", "infusion_rate_day", "note_txt"] if c in self.emar_d.columns]
                e = e.merge(self.emar_d[cols].drop_duplicates("emar_id"), on="emar_id", how="left")
            self.emar = e
            for g in e["generic"].dropna().unique():
                self.events_concepts.add("med:" + g.lower())
        if not self.rx.empty:
            r = self.rx
            r["t"] = to_dt(r["starttime"])
            r["t_stop"] = to_dt(r["stoptime"]) if "stoptime" in r.columns else pd.NaT
            r["generic"] = r["drug"].map(med_generic)
            r["cls"] = [classify_med(n, a) for n, a in zip(r["drug"], r.get("gsn", pd.Series([None] * len(r), index=r.index)))]
            self.rx = r
            for g in r["generic"].dropna().unique():
                self.events_concepts.add("med:" + g.lower())
        if not self.micro.empty:
            m = self.micro
            m["t"] = to_dt(m["charttime"]) if "charttime" in m.columns else to_dt(m.get("chartdate"))
            if "chartdate" in m.columns:
                m["t"] = m["t"].fillna(to_dt(m["chartdate"]))
            m["t_store"] = to_dt(m["storetime"]) if "storetime" in m.columns else pd.NaT
            self.micro = m
            self.events_concepts.add("ev_bloodcx")
        # 사건 어휘 존재 여부(검증기 전역 조회)
        labels = []
        for df in (self.proc, self.out, self.inp):
            if not df.empty and "label" in df.columns:
                labels += df["label"].dropna().astype(str).unique().tolist()
        if not self.px.empty:
            labels += [ICD9P_NAMES.get(norm_icd9p(c), "") for c in self.px["icd_code"].astype(str)]
        if not self.chart.empty and self.chart["concept"].isin(VENT_CONCEPTS).any():
            labels.append("인공호흡")
        if not self.out.empty and (self.out["concept"] == "crrt").any():
            labels.append("CRRT")
        if not self.px.empty and (self.px["icd_code"].astype(str).str.len() > 0).any():
            labels.append("수술")
        for ev, rx in EVENT_VOCAB:
            if any(re.search(rx, s, re.I) for s in labels):
                self.events_concepts.add(ev)

    # ---------------------------------------------------------------- helpers
    def icu_slice(self, df, pre_hours=0):
        if df.empty or "t" not in df.columns:
            return df
        return df[(df["t"] >= self.intime - pd.Timedelta(hours=pre_hours)) & (df["t"] <= self.t0)]

    def win(self, df):
        """요약 창(직전 24시간: win_start < t <= t0)."""
        if df is None or df.empty or "t" not in df.columns:
            return df
        return df[(df["t"] > self.win_start) & (df["t"] <= self.t0)]

    def last_window(self, df, hours):
        if df.empty or "t" not in df.columns:
            return df
        return df[(df["t"] > self.t0 - pd.Timedelta(hours=hours)) & (df["t"] <= self.t0)]

    def weight(self):
        for df in (self.inp, self.proc):
            if not df.empty and "patientweight" in df.columns:
                w = to_num(df["patientweight"]).dropna()
                w = w[(w > 2) & (w < 300)]
                if len(w):
                    return float(w.median())
        if not self.chart.empty:
            w = self.chart[self.chart["concept"] == "weight"]["v"].dropna()
            if len(w):
                return float(w.iloc[-1])
        return None


def kg(v):
    """체중 값 → kg. 300 초과는 g 단위 기록으로 간주(신생아 g, 성인 kg 기록 모두 처리)."""
    v = float(v)
    return v / 1000.0 if v > 300 else v


def fmt_w(kgv):
    return ("%s g" % fmt_num(round(kgv * 1000))) if kgv < 10 else ("%s kg" % fmt_num(round(kgv, 1)))


def w_disp(kgv):
    return float(round(kgv * 1000)) if kgv < 10 else round(kgv, 1)


def hours_of(ctx):
    return max(1.0, (ctx.t0 - ctx.win_start).total_seconds() / 3600.0)


def HH(ctx):
    return fmt_num(round(hours_of(ctx)))


def build_facts(ctx):
    facts = []
    facts += info_facts(ctx)
    facts += problem_facts(ctx)
    facts += course24_facts(ctx)
    facts += nutrition_facts(ctx)
    facts += output_facts(ctx)
    facts += lab_facts(ctx)
    facts += med_facts(ctx)
    excl = set(PROFILE.get("exclude_fact_types", []))
    facts = [f for f in facts if f.ftype not in excl]
    for f in facts:
        if f.ftype in PROFILE.get("section_of", {}):
            f.section = PROFILE["section_of"][f.ftype]
        if f.ftype in PROFILE.get("salience", {}):
            f.salience = int(PROFILE["salience"][f.ftype])
    facts += apply_checklist(ctx, facts, "summary")
    facts += apply_checklist(ctx, facts, "parent")
    order = list(SECTIONS.keys()) + list(PARENT_SECTIONS.keys())
    facts.sort(key=lambda f: (order.index(f.section) if f.section in order else len(order), 1 if f.ftype in ("checklist_gap", "parent_gap") else 0,
                              0 if f.time is None or pd.isna(f.time) else 1, f.time if f.time is not None and not pd.isna(f.time) else pd.Timestamp.min))
    for i, f in enumerate(facts, 1):
        f.fid = "F%03d" % i
    return facts


def weight_series(ctx):
    rows = []
    if not ctx.chart.empty:
        w = ctx.chart[(ctx.chart["concept"] == "weight") & (ctx.chart["t"] <= ctx.t0)].dropna(subset=["v"])
        for _, r in w.iterrows():
            rows.append((r["t"], kg(r["v"]), "CHARTEVENTS", r.get("chartevent_id")))
    if not rows:
        for df, tab, idc in ((ctx.inp, "INPUTEVENTS", "inputevent_id"), (ctx.proc, "PROCEDUREEVENTS", "procedureevent_id")):
            if not df.empty and "patientweight" in df.columns:
                d = df[df["t"] <= ctx.t0].copy()
                d["w"] = to_num(d["patientweight"])
                for _, r in d.dropna(subset=["w"]).drop_duplicates("w").iterrows():
                    rows.append((r["t"], kg(r["w"]), tab, r.get(idc)))
                if rows:
                    break
    rows = [r for r in rows if r[1] and 0.2 <= r[1] <= 300]
    rows.sort(key=lambda x: x[0])
    return rows


def info_facts(ctx):
    facts = []
    pat = ctx.pat.iloc[0] if len(ctx.pat) else {}
    adm = ctx.adm.iloc[0] if len(ctx.adm) else {}
    age = to_num(pd.Series([pat.get("anchor_age") if hasattr(pat, "get") else None])).iloc[0]
    sex = str(pat.get("sex")) if hasattr(pat, "get") else ""
    ctx.age = float(age) if pd.notna(age) else None
    ctx.sex = sex
    neo = ctx.is_neonate
    sex_ko = {"M": "남아" if neo else "남성", "F": "여아" if neo else "여성"}.get(sex, "성별 미상")
    admit = to_dt(pd.Series([adm.get("admittime") if hasattr(adm, "get") else None])).iloc[0]
    unit = ctx.row.get("first_careunit") or "ICU"
    icu_day = (ctx.t0.normalize() - ctx.intime.normalize()).days + 1
    parts = [("%s세 %s" % (fmt_num(ctx.age), sex_ko)) if (not neo and ctx.age is not None) else sex_ko]
    nums = []
    if pd.notna(admit):
        ctx.admit = pd.Timestamp(admit)
        ctx.hosp_day = (ctx.t0.normalize() - ctx.admit.normalize()).days + 1
        parts.append("입원 %d일째(입원 %s)" % (ctx.hosp_day, fmt_t(admit)))
        nums.append(ctx.hosp_day)
    else:
        ctx.admit, ctx.hosp_day = ctx.intime, icu_day
    parts.append("%s 입실 %d일째(입실 %s)" % (unit, icu_day, fmt_t(ctx.intime)))
    nums.append(icu_day)
    txt = "%s. 요약 기준 시각 %s, 지난 %s시간 기록 기준." % (", ".join(parts), fmt_t(ctx.t0), HH(ctx))
    facts.append(Fact("patient_info", "D1", txt, 3, concepts={"demo", "patient_info"}, key_nums=nums,
                      prov=[{"table": "ICUSTAYS", "row_id": ctx.stay_id}], extra={"sex_ko": sex_ko, "hosp_day": ctx.hosp_day, "icu_day": icu_day}))
    ws = weight_series(ctx)
    ctx.wt_kg = ctx.wt_first_kg = ctx.wt_pct = None
    if ws:
        cur, first = ws[-1], ws[0]
        prev = [w for w in ws if w[0] <= ctx.win_start]
        prev = prev[-1] if prev else None
        ctx.wt_kg, ctx.wt_first_kg = cur[1], first[1]
        stale = ", 지난 %s시간 내 새 측정 없음" % HH(ctx) if cur[0] <= ctx.win_start else ""
        parts = ["체중 %s(%s 측정%s)" % (fmt_w(cur[1]), fmt_t(cur[0]), stale)]
        ex = {"cur": w_disp(cur[1]), "unit": "g" if cur[1] < 10 else "kg"}
        if prev is not None and prev[0] != cur[0]:
            parts.append("전일 %s" % fmt_w(prev[1]))
            ex["prev"] = w_disp(prev[1])
        if first[0] != cur[0]:
            pct = (cur[1] - first[1]) / first[1] * 100.0
            ctx.wt_pct = round(pct, 1)
            parts.append("입실 후 첫 기록 %s 대비 %s%s%%" % (fmt_w(first[1]), "+" if pct >= 0 else "", fmt_num(round(pct, 1))))
            ex.update(first=w_disp(first[1]), pct=round(pct, 1))
        facts.append(Fact("weight", "D1", ", ".join(parts) + ".", 3, concepts={"weight"}, key_nums=[w_disp(cur[1])], time=cur[0],
                          prov=[{"table": cur[2], "row_id": cur[3]}], extra=ex))
    return facts


def problem_facts(ctx):
    facts = []
    if not ctx.dx.empty:
        dx = ctx.dx.copy()
        dx["seq"] = to_num(dx.get("seq_num", pd.Series(range(len(dx)), index=dx.index)))
        dx = dx[~dx["icd_code"].astype(str).isin(["Others", "nan", "None", ""])].sort_values("seq").drop_duplicates("icd_code")
        codes = dx["icd_code"].astype(str).tolist()
        items = [("%s %s" % (c, KCD_NAMES.get(c.replace(".", "").upper(), ""))).strip() for c in codes]
        if items:
            facts.append(Fact("diagnosis_main", "D2", "주진단(입원 진단 코드): %s." % items[0], 3, concepts={"dx"}, prov=_prov("DIAGNOSES_ICD", dx.head(1), "icd_code")))
        if len(items) > 1:
            facts.append(Fact("diagnosis_other", "D2", "기타 진단: %s." % ", ".join(items[1:8]), 2, concepts={"dx"}, prov=_prov("DIAGNOSES_ICD", dx.iloc[1:8], "icd_code")))
        pre = [c for c in codes if c.replace(".", "").upper().startswith("P07")]
        if pre:
            facts.append(Fact("prematurity", "D2", "미숙아·저체중 출생 관련 진단 코드: %s." % ", ".join(pre), 2, concepts={"preterm", "dx"}))
    if not ctx.px.empty:
        px = ctx.px[~ctx.px["icd_code"].astype(str).isin(["Others", "nan", "None", ""])].copy()
        if len(px):
            px["code"] = px["icd_code"].map(norm_icd9p)
            px["d"] = to_dt(px["chartdate"]) if "chartdate" in px.columns else pd.NaT
            px = px[px["d"].isna() | (px["d"] <= ctx.t0)].sort_values("d").drop_duplicates("code")
            if len(px):
                items = ["ICD-9 %s %s(%s)" % (r["code"], ICD9P_NAMES.get(r["code"], ""), pd.Timestamp(r["d"]).strftime("%m-%d") if pd.notna(r["d"]) else "날짜 미상")
                         for _, r in px.iterrows()]
                facts.append(Fact("procedure_icd", "D2", "수술·시술 코드: %s." % ", ".join(i.replace(" (", "(") for i in items[:6]), 2,
                                  concepts={"px", "ev_surgery"}, prov=_prov("PROCEDURES_ICD", px, "icd_code")))
    return facts


def episodes(times, gap_hours):
    times = sorted([t for t in times if pd.notna(t)])
    eps = []
    for t in times:
        if eps and (t - eps[-1][1]).total_seconds() <= gap_hours * 3600:
            eps[-1][1] = t
        else:
            eps.append([t, t])
    return eps


INVASIVE_RX = r"SIMV|IMV|\bA/?C\b|\bPC\b|\bVC\b|PRVC|SIPPV|IPPV|CMV|PCV|VCV|HFOV|HFO\b|HFJV|oscill|volume guarantee|\bVG\b|PSV|\bPS\b|\bSPONT\b"
NONINV_RX = r"CPAP|NIPPV|\bNIV\b|BiPAP|BIPAP|HFNC|high ?flow|nasal|NCPAP|SiPAP|DuoPAP|N-CPAP|비침습"


def resp_kind(mode, cats):
    if mode and re.search(NONINV_RX, mode, re.I):
        return "noninvasive"
    if mode and re.search(INVASIVE_RX, mode, re.I):
        return "invasive"
    if any(re.search(NONINV_RX, c or "", re.I) for c in cats):
        return "noninvasive"
    return "vent_unknown"


def resp_facts(ctx):
    facts = []
    c = ctx.chart
    RH = W("resp_recent_hours")
    rec = ctx.last_window(c, RH)
    vent = rec[rec["concept"].isin(VENT_CONCEPTS)]
    modes = rec[(rec["concept"] == "vent_mode") & rec["value"].notna() & ~rec["value"].astype(str).isin(["nan", ""])].sort_values("t")
    fi = c[(c["concept"] == "fio2") & (c["t"] <= ctx.t0)].dropna(subset=["v"]).sort_values("t")
    fio2 = None
    if len(fi) and fi["t"].iloc[-1] > ctx.t0 - pd.Timedelta(hours=RH):
        fv = float(fi["v"].iloc[-1])
        fio2 = fv if fv > 1 else fv * 100
    ctx.fio2 = fio2 / 100.0 if fio2 else None
    ctx.on_vent = ctx.on_o2 = False
    if len(vent):
        mode = str(modes["value"].iloc[-1]) if len(modes) else None
        cats = [ctx.item_map.get(i, (None, None, None, ""))[3] for i in vent["itemid"].unique()]
        kind = resp_kind(mode, cats)
        ctx.on_vent = kind != "noninvasive"
        ctx.resp_kind = kind
        name = {"invasive": "침습적 인공호흡", "noninvasive": "비침습 호흡 보조", "vent_unknown": "인공호흡기 관련 기록(방식 미상)"}[kind]
        parts = []
        if mode:
            parts.append("모드 %s" % mode)
        if fio2 is not None:
            parts.append("FiO2 %s%%" % fmt_num(fio2))
        peep = rec[rec["concept"] == "peep"]["v"].dropna()
        if len(peep):
            parts.append("PEEP %s cmH2O" % fmt_num(peep.iloc[-1]))
        txt = "호흡 보조(최근 %s시간): %s%s." % (fmt_num(RH), name, (" (" + ", ".join(parts) + ")") if parts else "")
        facts.append(Fact("resp_support", "D3", txt, 3, concepts={"resp"} | ({"ev_vent"} if kind != "noninvasive" else set()), status="active",
                          time=vent["t"].max(), prov=_prov("CHARTEVENTS", vent.tail(2), "chartevent_id"), extra={"kind": kind, "mode": mode, "fio2": fio2}))
    else:
        o2 = rec[rec["concept"] == "o2_flow"].dropna(subset=["v"]).sort_values("t")
        if len(o2) and o2["v"].iloc[-1] > 0:
            ctx.on_o2 = True
            ctx.resp_kind = "o2"
            fl = float(o2["v"].iloc[-1])
            txt = "호흡 보조(최근 %s시간): 산소 %s L/min(%s)%s." % (fmt_num(RH), fmt_num(fl), fmt_t(o2["t"].iloc[-1]), (", FiO2 %s%%" % fmt_num(fio2)) if fio2 is not None else "")
            facts.append(Fact("resp_support", "D3", txt, 2, concepts={"resp", "o2_flow"}, key_nums=[fl], status="active", time=o2["t"].iloc[-1],
                              prov=_prov("CHARTEVENTS", o2.tail(1), "chartevent_id"), extra={"kind": "o2", "flow": fl}))
        else:
            ctx.resp_kind = "none"
            facts.append(Fact("resp_support", "D3", "호흡 보조(최근 %s시간): 인공호흡기·산소유량 기록 없음." % fmt_num(RH), 1, concepts={"resp"},
                              status="none", extra={"kind": "none"}))
    wm = ctx.win(c)
    wm = wm[(wm["concept"] == "vent_mode") & wm["value"].notna() & ~wm["value"].astype(str).isin(["nan", ""])].sort_values("t")
    if len(wm) >= 2:
        vals, tt = wm["value"].astype(str).tolist(), wm["t"].tolist()
        ch = [(vals[i - 1], vals[i], tt[i]) for i in range(1, len(vals)) if vals[i] != vals[i - 1]]
        if ch:
            txt = "지난 %s시간 호흡 보조 모드 변경: %s → %s(마지막 변경 %s%s)." % (HH(ctx), ch[0][0], ch[-1][1], fmt_t(ch[-1][2]), ", 변경 %d회" % len(ch) if len(ch) > 1 else "")
            facts.append(Fact("resp_change", "D3", txt, 3, concepts={"resp", "vent_mode"}, time=ch[-1][2], prov=_prov("CHARTEVENTS", wm.tail(2), "chartevent_id"),
                              extra={"from": ch[0][0], "to": ch[-1][1]}))
    return facts


def course24_facts(ctx):
    facts = []
    ctx.vit = {}
    ctx.n_desat = ctx.n_brady = 0
    if not ctx.chart.empty:
        c = ctx.win(ctx.chart)
        rng, rows, nums = [], [], []
        for k, d, u in (("hr", "HR", "회/분"), ("rr", "RR", "회/분"), ("spo2", "SpO2", "%"), ("temp", "BT", "℃"), ("map", "MAP", " mmHg"), ("sbp", "SBP", " mmHg")):
            s = c[c["concept"] == k].dropna(subset=["v"]).sort_values("t")
            if not len(s):
                continue
            lo, hi = float(s["v"].min()), float(s["v"].max())
            ctx.vit[k] = {"min": lo, "max": hi, "last": float(s["v"].iloc[-1]), "n": len(s)}
            rng.append(("%s %s~%s%s" % (d, fmt_num(lo), fmt_num(hi), u)) if fmt_num(lo) != fmt_num(hi) else ("%s %s%s" % (d, fmt_num(lo), u)))
            nums += [lo, hi]
            rows.append(s.tail(1))
        if rng:
            facts.append(Fact("vitals_24h", "D3", "지난 %s시간 활력징후 범위: %s." % (HH(ctx), ", ".join(rng)), 3, concepts=set(ctx.vit) | {"vitals"},
                              key_nums=nums, prov=sum([_prov("CHARTEVENTS", r, "chartevent_id") for r in rows], []), extra={"vit": ctx.vit}))
        else:
            facts.append(Fact("vitals_24h", "D3", "지난 %s시간 활력징후 기록 없음." % HH(ctx), 2, concepts={"vitals"}))
        sp = c[(c["concept"] == "spo2") & (c["v"] < float(NEO("desat_spo2")))]
        hr = c[(c["concept"] == "hr") & (c["v"] < float(NEO("brady_hr")))]
        ctx.n_desat, ctx.n_brady = len(sp), len(hr)
        if "spo2" in ctx.vit or "hr" in ctx.vit:
            if len(sp) or len(hr):
                txt = "지난 %s시간 기록상 SpO2 %s%% 미만 %d회, HR %s회/분 미만 %d회." % (HH(ctx), fmt_num(NEO("desat_spo2")), len(sp), fmt_num(NEO("brady_hr")), len(hr))
                facts.append(Fact("neo_events", "D3", txt, 3, concepts={"spo2", "hr", "events"}, key_nums=[len(sp), len(hr)], status="present",
                                  prov=_prov("CHARTEVENTS", pd.concat([sp, hr]).head(5), "chartevent_id"), extra={"n_desat": len(sp), "n_brady": len(hr)}))
            else:
                txt = "지난 %s시간 기록상 SpO2 %s%% 미만·HR %s회/분 미만 기록 없음." % (HH(ctx), fmt_num(NEO("desat_spo2")), fmt_num(NEO("brady_hr")))
                facts.append(Fact("neo_events", "D3", txt, 1, concepts={"events"}, status="none"))
        facts += resp_facts(ctx)
    facts += infusion_facts(ctx)
    if not ctx.proc.empty:
        p = ctx.win(ctx.proc)
        if "value" in p.columns:
            vv = to_num(p["value"])
            p = p[vv.isna() | (vv != 0)]
        for lab, g in p.groupby("label"):
            ev = {"procedure"}
            for evn, rx in EVENT_VOCAB:
                if re.search(rx, lab, re.I):
                    ev.add(evn)
            txt = "%s 시행 기록(%s%s)." % (lab, fmt_t(g["t"].min()), (", %d회" % len(g)) if len(g) > 1 else "")
            facts.append(Fact("procedure", "D3", txt, 2, concepts=ev, time=g["t"].min(), prov=_prov("PROCEDUREEVENTS", g.head(3), "procedureevent_id"),
                              extra={"label": lab}))
    if not ctx.test.empty:
        te = ctx.test.copy()
        te["t"] = to_dt(te["charttime"])
        te = te[(te["t"] > ctx.win_start) & (te["t"] <= ctx.t0)]
        if len(te):
            dm = dict(zip(ctx.d_test["itemid"], ctx.d_test["label"])) if (ctx.d_test is not None and not ctx.d_test.empty) else {}
            names = sorted(set(te["itemid"].map(lambda i: dm.get(i, i)).astype(str)))
            facts.append(Fact("imaging", "D3", "영상검사 시행: %s(판독문은 데이터에 없음)." % ", ".join(names)[:160], 2, concepts={"imaging"}, time=te["t"].min(),
                              prov=_prov("TESTEVENTS", te.head(3), "testevent_id"), extra={"names": names}))
    if not ctx.micro.empty:
        m = ctx.micro[ctx.micro["t"] <= ctx.t0]
        key = "micro_specimen_id" if "micro_specimen_id" in m.columns else "t"
        for sid_, g in m.groupby(key):
            spec = str(g["spec_type_desc"].iloc[0]) if "spec_type_desc" in g.columns else "검체"
            tc = g["t"].min()
            ts = g["t_store"].max() if "t_store" in g.columns else pd.NaT
            reported = pd.notna(ts) and ts <= ctx.t0
            in_win = tc > ctx.win_start or (reported and ts > ctx.win_start)
            orgs = [o for o in g.get("org_name", pd.Series(dtype=str)).dropna().astype(str).unique() if o not in ("nan", "")]
            if orgs and reported:
                res = g[g["interpretation"].astype(str).isin(["R"])]["ab_name"].dropna().astype(str).unique().tolist() if "interpretation" in g.columns else []
                txt = "%s(%s 채취) 배양 %s 동정%s, 보고 %s%s." % (spec, fmt_t(tc), ", ".join(orgs), ("; 내성: " + ", ".join(res)) if res else "", fmt_t(ts), "" if in_win else "(이전 보고)")
                facts.append(Fact("micro_positive", "D3", txt, 3 if in_win else 2, concepts={"ev_bloodcx", "micro"}, time=tc,
                                  prov=_prov("MICROBIOLOGYEVENTS", g.head(3), "microevent_id"), extra={"orgs": orgs, "resistant": res}))
            elif not reported:
                facts.append(Fact("micro_pending", "D3", "%s(%s 채취) 배양 결과 미보고." % (spec, fmt_t(tc)), 2, concepts={"ev_bloodcx", "micro"}, status="pending",
                                  time=tc, prov=_prov("MICROBIOLOGYEVENTS", g.head(2), "microevent_id")))
            elif in_win:
                facts.append(Fact("micro_negative", "D3", "%s(%s 채취) 배양 동정균 없음, 보고 %s." % (spec, fmt_t(tc), fmt_t(ts)), 1, concepts={"micro"}, time=tc,
                                  prov=_prov("MICROBIOLOGYEVENTS", g.head(1), "microevent_id")))
    if not ctx.lab.empty:
        ctx.aki = aki_check(ctx)
        if ctx.aki:
            a = ctx.aki
            new = a["peak_t"] > ctx.win_start
            txt = "Cr %s(%s) → %s(%s) 상승%s." % (fmt_num(a["base"]), fmt_t(a["base_t"]), fmt_num(a["peak"]), fmt_t(a["peak_t"]), "" if new else "(이전 기록)")
            facts.append(Fact("aki", "D3", txt, 3 if new else 2, concepts={"cr", "aki"}, key_nums=[a["base"], a["peak"]], time=a["peak_t"], prov=a["prov"],
                              extra={"criterion": a["criterion"], "new": new}))
    return facts


def infusion_facts(ctx):
    facts = []
    t0 = ctx.t0
    if not ctx.emar.empty:
        e = ctx.icu_slice(ctx.emar)
        for cls in ("vasopressor", "sedation"):
            for g, gg in e[e["cls"] == cls].groupby("generic"):
                for a, b in episodes(gg["t"].tolist(), W("infusion_episode_gap_hours")):
                    if b <= ctx.win_start:
                        continue
                    seg = gg[(gg["t"] >= a) & (gg["t"] <= b)]
                    rtxt = ""
                    for col in ("infusion_rate_min", "infusion_rate_hr"):
                        if col in seg.columns:
                            r = to_num(seg[col]).dropna()
                            if len(r):
                                rtxt = ", 최대 주입속도 기록값 %s(%s)" % (fmt_num(r.max()), "/min" if col.endswith("min") else "/hr")
                                break
                    st = "active" if (t0 - b).total_seconds() <= W("infusion_active_hours") * 3600 else "stopped"
                    kind = "승압·강심제" if cls == "vasopressor" else "진정·진통제"
                    stxt = "기준 시각 투여 중" if st == "active" else "%s 마지막 투여 후 중단" % fmt_t(b)
                    txt = "%s %s 지속주입 %s ~ %s%s, %s." % (kind, g, fmt_t(a), fmt_t(b), rtxt, stxt)
                    ctx.infusion_generics.add(g.lower())
                    facts.append(Fact("infusion_episode", "D3", txt, 3, concepts={"med:" + g.lower(), "cls:" + cls}, status=st, time=a,
                                      prov=_prov("EMAR", seg.head(3), "emar_id"), extra={"cls": cls, "last": str(b), "first": str(a), "generic": g}))
    if not ctx.inp.empty and (ctx.inp["concept"] == "in_vasopressor").any():
        iv = ctx.icu_slice(ctx.inp)
        iv = iv[iv["concept"] == "in_vasopressor"]
        for g, gg in iv.groupby("label"):
            for a, b in episodes(gg["t"].tolist(), W("infusion_episode_gap_hours")):
                seg = gg[(gg["t"] >= a) & (gg["t"] <= b)]
                if "t_end" in seg.columns and seg["t_end"].notna().any():
                    b = max(b, min(seg["t_end"].max(), t0))
                if b <= ctx.win_start:
                    continue
                rt = to_num(seg["rate"]).dropna() if "rate" in seg.columns else pd.Series(dtype=float)
                ru = str(seg["rateuom"].dropna().iloc[0]) if ("rateuom" in seg.columns and seg["rateuom"].notna().any()) else ""
                st = "active" if (t0 - b).total_seconds() <= W("infusion_active_hours") * 3600 else "stopped"
                gen = med_generic(g)
                txt = "승압·강심제 %s 지속주입 %s ~ %s%s, %s." % (g, fmt_t(a), fmt_t(b), (", 최대 %s %s" % (fmt_num(rt.max()), ru)) if len(rt) else "",
                                                          "기준 시각 투여 중" if st == "active" else "이후 중단")
                ctx.infusion_generics.add(gen.lower())
                facts.append(Fact("infusion_episode", "D3", txt, 3, concepts={"med:" + gen.lower(), "cls:vasopressor"}, status=st, time=a,
                                  prov=_prov("INPUTEVENTS", seg.head(3), "inputevent_id"), extra={"cls": "vasopressor", "last": str(b), "first": str(a), "generic": g}))
    if not ctx.out.empty:
        cr = ctx.icu_slice(ctx.out)
        cr = cr[cr["concept"] == "crrt"]
        for a, b in episodes(cr["t"].tolist(), W("crrt_episode_gap_hours")):
            if b <= ctx.win_start:
                continue
            seg = cr[(cr["t"] > ctx.win_start) & (cr["t"] <= b)]
            st = "active" if (t0 - b).total_seconds() <= W("crrt_active_hours") * 3600 else "stopped"
            txt = "CRRT(%s) 기록 %s ~ %s, 지난 %s시간 제거량 %s mL%s." % (seg["label"].iloc[0] if len(seg) else "", fmt_t(a), fmt_t(b), HH(ctx),
                                                                fmt_num(round(seg["v"].sum())), ", 기준 시각까지 시행" if st == "active" else "")
            facts.append(Fact("crrt_episode", "D3", txt, 3, concepts={"ev_crrt"}, status=st, time=a, key_nums=[round(seg["v"].sum())],
                              prov=_prov("OUTPUTEVENTS", seg.head(3), "outputevent_id")))
    return facts


def nutrition_facts(ctx):
    facts = []
    ctx.intake_total = None
    ctx.intake_ml_kg_day = None
    if ctx.inp.empty:
        return facts
    d = ctx.win(ctx.inp)
    d = d[d["v"].notna()]
    if d.empty:
        return facts
    H = hours_of(ctx)
    sums = {k: float(d[d["concept"] == k]["v"].sum()) for k in ("in_iv", "in_oral", "in_tube")}
    other = float(d[~d["concept"].isin(["in_iv", "in_oral", "in_tube"])]["v"].sum())
    total = float(d["v"].sum())
    ent = sums["in_oral"] + sums["in_tube"]
    parts = []
    if sums["in_iv"] > 0:
        parts.append("정맥 %s mL" % fmt_num(round(sums["in_iv"])))
    if ent > 0:
        sub = [x for x in (("경구", sums["in_oral"]), ("경관", sums["in_tube"])) if x[1] > 0]
        parts.append("장관 %s mL" % fmt_num(round(ent)) + (("[%s]" % ", ".join("%s %s mL" % (a, fmt_num(round(b))) for a, b in sub)) if len(sub) > 1 else "(%s)" % sub[0][0]))
    if other > 0:
        parts.append("기타 %s mL" % fmt_num(round(other)))
    txt = "지난 %s시간 섭취 총 %s mL(%s)" % (HH(ctx), fmt_num(round(total)), ", ".join(parts))
    nums = [round(total)]
    ctx.intake_total = total
    if ctx.wt_kg:
        mkd = total / ctx.wt_kg * 24.0 / H
        ctx.intake_ml_kg_day = round(mkd)
        txt += ", %s mL/kg/일(체중 %s 기준%s)" % (fmt_num(round(mkd)), fmt_w(ctx.wt_kg), ", 24시간 환산" if H < 23 else "")
        nums.append(round(mkd))
    facts.append(Fact("intake_24h", "D4", txt + ".", 3, concepts={"intake", "io"}, key_nums=nums, prov=_prov("INPUTEVENTS", d.head(3), "inputevent_id"),
                      extra={"total": round(total), "iv": round(sums["in_iv"]), "enteral": round(ent), "mlkgday": ctx.intake_ml_kg_day}))
    return facts


def output_facts(ctx):
    facts = []
    ctx.urine_rate = None
    if ctx.out.empty:
        return facts
    d = ctx.win(ctx.out)
    d = d[d["v"].notna()]
    H = hours_of(ctx)
    ur = d[d["concept"] == "urine"]
    if len(ur):
        ml = float(ur["v"].sum())
        txt = "지난 %s시간 소변 %s mL" % (HH(ctx), fmt_num(round(ml)))
        nums = [round(ml)]
        if ctx.wt_kg:
            ctx.urine_rate = round(ml / ctx.wt_kg / H, 2)
            txt += ", %s mL/kg/h" % fmt_num(ctx.urine_rate)
            nums.append(ctx.urine_rate)
        low = ctx.urine_rate is not None and ctx.urine_rate < float(NEO("urine_low_ml_kg_h"))
        facts.append(Fact("urine_24h", "D5", txt + ".", 3 if low else 2, concepts={"urine", "io"}, key_nums=nums, prov=_prov("OUTPUTEVENTS", ur.head(3), "outputevent_id"),
                          extra={"ml": round(ml), "rate": ctx.urine_rate}))
    st = d[d["concept"].isin(["stool", "urine_stool"])]
    if len(st):
        txt = "지난 %s시간 대변·대소변 기록 %d회, 총 %s mL." % (HH(ctx), len(st), fmt_num(round(st["v"].sum())))
        facts.append(Fact("stool", "D5", txt, 2, concepts={"stool"}, key_nums=[len(st)], prov=_prov("OUTPUTEVENTS", st.head(2), "outputevent_id"),
                          extra={"n": len(st), "ml": round(st["v"].sum())}))
    for lab, g in d[d["concept"] == "drain"].groupby("label"):
        facts.append(Fact("drain", "D5", "배액관 %s 지난 %s시간 배액 %s mL." % (lab, HH(ctx), fmt_num(round(g["v"].sum()))), 2,
                          concepts={"drain", "drain:" + lab.lower()}, key_nums=[round(g["v"].sum())], status="active", prov=_prov("OUTPUTEVENTS", g.head(2), "outputevent_id"),
                          extra={"label": lab}))
    if ctx.intake_total is not None and len(d):
        tout = float(d["v"].sum())
        bal = round(ctx.intake_total - tout)
        facts.append(Fact("io_balance", "D5", "지난 %s시간 섭취·배설 균형 %s%s mL(배설 총 %s mL)." % (HH(ctx), "+" if bal >= 0 else "", fmt_num(bal), fmt_num(round(tout))), 1,
                          concepts={"io"}, key_nums=[bal]))
    return facts


def ref_range(s):
    c = s["concept"].iloc[0]
    if PROFILE and PROFILE.get("prefer_profile_ref") and c in PROFILE.get("ref_ranges", {}):
        lo, hi = PROFILE["ref_ranges"][c]
        return lo, hi
    lo = s["ref_range_lower"].dropna() if "ref_range_lower" in s.columns else pd.Series(dtype=float)
    hi = s["ref_range_upper"].dropna() if "ref_range_upper" in s.columns else pd.Series(dtype=float)
    fb = tuple(PROFILE["ref_ranges"][c]) if (PROFILE and c in PROFILE.get("ref_ranges", {})) else LAB_FALLBACK_REF.get(c, (None, None))
    return (float(lo.median()) if len(lo) else fb[0], float(hi.median()) if len(hi) else fb[1])


def aki_check(ctx):
    L = ctx.lab
    s = L[(L["concept"] == "cr") & (L["t"] >= ctx.intime - pd.Timedelta(days=7)) & (L["t"] <= ctx.t0)].dropna(subset=["v"]).sort_values("t")
    if len(s) < 2:
        return None
    best = None
    vals = s[["t", "v"]].values.tolist()
    for i, (tj, vj) in enumerate(vals):
        if tj < ctx.intime:
            continue
        prior = [(ti, vi) for ti, vi in vals[:i] if ti >= tj - pd.Timedelta(days=7)]
        if not prior:
            continue
        p48 = [(ti, vi) for ti, vi in prior if ti >= tj - pd.Timedelta(hours=48)]
        if p48:
            tm, vm = min(p48, key=lambda x: x[1])
            if vj - vm >= 0.3 - 1e-9:
                cand = {"base": vm, "base_t": tm, "peak": vj, "peak_t": tj, "criterion": "KDIGO: 48시간 내 0.3 mg/dL 이상 상승"}
                if best is None or vj - vm > best["peak"] - best["base"]:
                    best = cand
        tm, vm = min(prior, key=lambda x: x[1])
        if vm > 0 and vj >= 1.5 * vm:
            cand = {"base": vm, "base_t": tm, "peak": vj, "peak_t": tj, "criterion": "KDIGO: 7일 내 기저치 1.5배 이상"}
            if best is None or vj - vm > best["peak"] - best["base"]:
                best = cand
    if best:
        best["prov"] = _prov("LABEVENTS", s[s["t"].isin([best["base_t"], best["peak_t"]])], "labevent_id")
    return best


def lab_facts(ctx):
    facts = []
    ctx.latest_labs = {}
    ctx.glucose_rows = []
    abn_set = set()
    if not ctx.lab.empty:
        LH = W("labs_hours")
        L = ctx.last_window(ctx.lab, LH)
        normal, ncon = [], set()
        for c, disp, _, lo_c, hi_c in LAB_CONCEPTS:
            lo_c, hi_c = crit_of(c, lo_c, hi_c)
            s = L[L["concept"] == c].dropna(subset=["v"]).sort_values("t")
            if not len(s):
                continue
            r = s.iloc[-1]
            v = float(r["v"])
            ctx.latest_labs[c] = (v, r["t"])
            lo, hi = ref_range(s)
            unit = str(r.get("valueuom", "")) if pd.notna(r.get("valueuom", None)) else ""
            unit = "" if unit in ("nan", "None") else unit
            crit = (lo_c is not None and v <= lo_c) or (hi_c is not None and v >= hi_c)
            abn = (lo is not None and v < lo) or (hi is not None and v > hi)
            if c == "glucose":
                ctx.glucose_rows += [(float(x), t, "LABEVENTS", i) for x, t, i in zip(s["v"], s["t"], s.get("labevent_id", s.index))]
            if crit or abn:
                hol = ""
                if c == "tbil" and ctx.is_neonate and getattr(ctx, "admit", None) is not None:
                    hol = ", 입원 후 %s시간" % fmt_num(round((r["t"] - ctx.admit).total_seconds() / 3600.0))
                unit_txt = (" " + unit) if unit and unit.lower() != disp.lower() else ""
                direction = "높음" if (hi is not None and v > hi) else "낮음"
                txt = "%s %s%s(%s%s, %s)." % (disp, fmt_num(v), unit_txt, fmt_t(r["t"]), hol, "위험값" if crit else "참고범위보다 " + direction)
                facts.append(Fact("lab_latest", "D6", txt, 3 if crit else 2, concepts={c}, key_nums=[v], time=r["t"], prov=_prov("LABEVENTS", s.tail(1), "labevent_id"),
                                  extra={"critical": bool(crit), "ref": [lo, hi], "concept": c, "high": bool(hi is not None and v > hi)}))
                abn_set.add(c)
            else:
                normal.append((disp, v))
                ncon.add(c)
        if normal:
            facts.append(Fact("lab_normal", "D6", ("최근 %s시간 참고범위 내 검사: " % fmt_num(LH)) + "%s." % ", ".join("%s %s" % (d, fmt_num(v)) for d, v in normal[:16]),
                              1, concepts={"labs_normal"} | ncon))
        if not ctx.latest_labs:
            lt = ctx.lab[ctx.lab["t"] <= ctx.t0]["t"].max()
            facts.append(Fact("lab_none", "D6", "최근 %s시간 내 검사 기록 없음(마지막 검사 %s)." % (fmt_num(LH), fmt_t(lt)), 2, concepts={"labs_none"}, time=lt))
        for c in PROFILE.get("trend_labs", []):
            s = ctx.lab[(ctx.lab["concept"] == c) & (ctx.lab["t"] <= ctx.t0)].dropna(subset=["v"]).sort_values("t")
            if len(s) < 2 or s["t"].iloc[-1] <= ctx.win_start:
                continue
            last = s.iloc[-1]
            prev = s[(s["t"] < last["t"]) & (s["t"] >= last["t"] - pd.Timedelta(hours=72))]
            if prev.empty:
                continue
            p = prev.iloc[-1]
            if fmt_num(p["v"]) == fmt_num(last["v"]):
                continue
            disp = LAB_DISPLAY.get(c, c)
            dh = (last["t"] - p["t"]).total_seconds() / 3600.0
            txt = "%s 변화: %s(%s) → %s(%s)." % (disp, fmt_num(p["v"]), fmt_t(p["t"]), fmt_num(last["v"]), fmt_t(last["t"]))
            facts.append(Fact("lab_change", "D6", txt, 2 if c in abn_set else 1, concepts={c}, key_nums=[p["v"], last["v"]], time=last["t"],
                              prov=_prov("LABEVENTS", pd.concat([prev.tail(1), s.tail(1)]), "labevent_id"),
                              extra={"concept": c, "from": float(p["v"]), "to": float(last["v"]), "hours": dh}))
    if not ctx.chart.empty:
        b = ctx.win(ctx.chart)
        b = b[b["concept"] == "bst"].dropna(subset=["v"]).sort_values("t")
        if len(b):
            ctx.glucose_rows += [(float(x), t, "CHARTEVENTS", i) for x, t, i in zip(b["v"], b["t"], b.get("chartevent_id", b.index))]
            lo, hi = float(b["v"].min()), float(b["v"].max())
            low = lo < float(NEO("glucose_low")) or hi > float(NEO("glucose_high"))
            rng = ("%s~%s" % (fmt_num(lo), fmt_num(hi))) if fmt_num(lo) != fmt_num(hi) else fmt_num(lo)
            facts.append(Fact("bst_24h", "D6", "지난 %s시간 BST(현장 혈당) %s mg/dL, %d회 측정." % (HH(ctx), rng, len(b)), 3 if low else 2,
                              concepts={"bst", "glucose"}, key_nums=[lo, hi], prov=_prov("CHARTEVENTS", b.head(3), "chartevent_id")))
    return facts


CLS_KO = {"vasopressor": "승압·강심제", "sedation": "진정·진통제", "insulin": "인슐린", "anticoagulant": "항응고제", "antiplatelet": "항혈소판제",
          "antibiotic": "항생제", "antifungal": "항진균제", "diuretic": "이뇨제", "respiratory_stimulant": "호흡자극제", "surfactant": "폐표면활성제",
          "nutrition": "영양", "anticonvulsant": "항경련제"}


def med_facts(ctx):
    facts = []
    t0, ws = ctx.t0, ctx.win_start
    ctx.active_meds = []
    rows = []
    if not ctx.emar.empty:
        e = ctx.emar[ctx.emar["t"] <= t0]
        for g, gg in e.groupby("generic"):
            gg = gg.sort_values("t")
            rate = None
            for col in ("infusion_rate_min", "infusion_rate_hr"):
                if col in gg.columns:
                    r = to_num(gg[col]).dropna()
                    if len(r):
                        rate = (float(r.iloc[-1]), "/min" if col.endswith("min") else "/hr")
                        break
            rows.append(dict(generic=g, cls=gg["cls"].iloc[-1], name=gg["medication"].iloc[-1], first=gg["t"].min(), last=gg["t"].max(), source="EMAR",
                             prov=_prov("EMAR", gg.tail(2), "emar_id"), rate=rate))
    if not ctx.rx.empty:
        for _, x in ctx.rx[ctx.rx["t"] <= t0].iterrows():
            active = pd.isna(x["t_stop"]) or x["t_stop"] > t0
            dv = to_num(pd.Series([x.get("dose_val_rx")])).iloc[0]
            rows.append(dict(generic=x["generic"], cls=x["cls"], name=x["drug"], first=x["t"], last=(x["t"] if active else x["t_stop"]), source="PRESCRIPTIONS",
                             rx_active=active, open_end=pd.isna(x["t_stop"]), prov=[{"table": "PRESCRIPTIONS", "row_id": x.get("pharmacy_id")}],
                             dose=("%s %s" % (fmt_num(dv), x.get("dose_unit_rx", ""))).strip() if pd.notna(dv) else "", route=x.get("route")))
    by = OrderedDict()
    for r in rows:
        k = r["generic"].lower()
        if k in by:
            r["first"] = min(r["first"], by[k]["first"])
        if k not in by or (r["source"] == "EMAR" and by[k]["source"] != "EMAR") or (r["source"] == by[k]["source"] and r["last"] > by[k]["last"]):
            by[k] = r
    fluids = []
    for k, r in by.items():
        if r["source"] == "PRESCRIPTIONS":
            active = r.get("rx_active", False)
        else:
            win = W("infusion_active_hours") if r["cls"] in ("vasopressor", "sedation") else W("med_active_hours")
            active = (t0 - r["last"]).total_seconds() <= win * 3600
        new = r["first"] > ws
        stopped_today = (not active) and r["last"] > ws
        if not active and not stopped_today:
            continue
        if r["cls"] == "fluid":
            if active:
                fluids.append(r)
            continue
        if r["cls"] in ("vasopressor", "sedation") and k in ctx.infusion_generics:
            if active:
                ctx.active_meds.append((r["generic"], r["cls"], r["name"], r["last"], r.get("route")))
            continue
        ck = CLS_KO.get(r["cls"], "")
        extra_txt = ""
        if r.get("rate"):
            extra_txt += ", 마지막 주입속도 기록값 %s(%s)" % (fmt_num(r["rate"][0]), r["rate"][1])
        if r.get("dose"):
            extra_txt += ", %s" % r["dose"]
        if r.get("route") and str(r["route"]) not in ("nan", "None"):
            extra_txt += " %s" % r["route"]
        ex = {"cls": r["cls"], "generic": r["generic"], "new": bool(new)}
        if active:
            ctx.active_meds.append((r["generic"], r["cls"], r["name"], r["last"], r.get("route")))
            newtxt = ", 오늘 시작(첫 기록 %s)" % fmt_t(r["first"]) if new else ""
            if r["source"] == "PRESCRIPTIONS":
                txt = "%s%s(%s) 처방 유지 중%s%s, 처방 시작 %s%s." % ((ck + " ") if ck else "", r["generic"], r["name"], extra_txt, newtxt if new else "",
                                                              fmt_t(r["first"]), ", 종료 기록 없음" if r.get("open_end") else "")
            else:
                txt = "%s%s(%s) 투여 중%s%s, 마지막 기록 %s." % ((ck + " ") if ck else "", r["generic"], r["name"], extra_txt, newtxt, fmt_t(r["last"]))
            facts.append(Fact("med_active", "D7", txt, 3 if r["cls"] in HIGH_ALERT else 2, concepts={"med:" + r["generic"].lower(), "cls:" + r["cls"]},
                              status="active", time=r["last"], prov=r["prov"], extra=ex))
        else:
            txt = "%s%s(%s) 오늘 중단, 마지막 기록 %s." % ((ck + " ") if ck else "", r["generic"], r["name"], fmt_t(r["last"]))
            facts.append(Fact("med_stopped", "D7", txt, 2, concepts={"med:" + r["generic"].lower(), "cls:" + r["cls"]}, status="stopped", time=r["last"],
                              prov=r["prov"], extra=ex))
    if fluids:
        facts.append(Fact("fluids", "D7", "수액 투여 중: %s." % ", ".join(sorted(set(f["name"] for f in fluids))), 1,
                          concepts={"cls:fluid"} | {"med:" + f["generic"].lower() for f in fluids}, status="active"))
    return facts


# =============================================================================
# 4. CDSS — NICU 규칙(결정론). 기준값은 프로파일 neonatal/critical_values/tdm 에서 읽음.
#    알림 문장은 근거 fact 값만 사용하고, 계산값은 출처가 남는 파생 fact로 등록해 검증기 대조 대상에 포함.
# =============================================================================


def run_cdss(ctx, facts, dur_pairs=None):
    alerts = []
    neo = ctx.is_neonate
    vis = set(PROFILE["documents"]["parent"].get("parent_visible_rules", []))

    def fid_of(pred):
        return [f.fid for f in facts if pred(f)]

    def derived(ftype, text, inputs, nums=(), concepts=()):
        f = Fact(ftype, CDSS_SEC(), text, 1, concepts=set(concepts) | {ftype}, key_nums=nums, prov=[{"derived_from": inputs}])
        f.fid = "F%03d" % (len(facts) + 1)
        facts.append(f)
        return f.fid

    def add(rule, sev, title, text, trig, ref):
        if any(rule == d or rule.startswith(d) for d in PROFILE["cdss"].get("disabled", [])):
            return
        for k, v in PROFILE["cdss"].get("severity_override", {}).items():
            if rule == k or rule.startswith(k):
                sev = v
        alerts.append({"alert_id": "AL%02d" % (len(alerts) + 1), "rule_id": rule, "severity": sev, "title": title, "text": text,
                       "trigger_fact_ids": [t for t in trig if t], "reference": ref, "parent_visible": rule in vis})

    N = PROFILE["neonatal"]
    H = HH(ctx)
    vit = getattr(ctx, "vit", {})
    vf = fid_of(lambda f: f.ftype == "vitals_24h")
    if neo:
        t = vit.get("temp")
        lo, hi = N["vitals_normal"]["temp"]
        if t and (t["min"] < lo or t["max"] > hi):
            rng = ("%s~%s" % (fmt_num(t["min"]), fmt_num(t["max"]))) if fmt_num(t["min"]) != fmt_num(t["max"]) else fmt_num(t["min"])
            add("NEO_TEMP_INSTABILITY", "medium", "체온 불안정", "지난 %s시간 BT %s℃, 저체온·발열 여부와 보온 환경 확인 필요." % (H, rng), vf, "WHO 신생아 정상 체온 범위")
        h = vit.get("hr")
        if h and h["max"] > N["vitals_normal"]["hr"][1]:
            add("NEO_TACHYCARDIA", "medium", "빈맥", "지난 %s시간 HR 최고 %s회/분, 원인(발열·통증·저혈량 등) 확인 필요." % (H, fmt_num(h["max"])), vf, "신생아 활력징후 기준(프로파일)")
        r = vit.get("rr")
        if r and r["max"] > N["vitals_normal"]["rr"][1]:
            add("NEO_TACHYPNEA", "medium", "빈호흡", "지난 %s시간 RR 최고 %s회/분, 호흡 상태 재평가 필요." % (H, fmt_num(r["max"])), vf, "신생아 활력징후 기준(프로파일)")
        ev = fid_of(lambda f: f.ftype == "neo_events" and f.status == "present")
        if ev:
            add("NEO_DESAT_BRADY", "medium", "산소포화도 저하·서맥 기록", "지난 %s시간 기록상 SpO2 저하 %d회, 서맥 %d회, 무호흡 동반 여부와 치료 조정 검토 필요." % (H, ctx.n_desat, ctx.n_brady),
                ev, "NICU 무호흡·서맥 관찰 원칙")
        g = getattr(ctx, "glucose_rows", [])
        g = [x for x in g if x[1] > ctx.win_start]
        if g:
            gmin = min(g, key=lambda x: x[0])
            gmax = max(g, key=lambda x: x[0])
            inputs = fid_of(lambda f: f.ftype in ("lab_latest", "lab_normal", "lab_change", "bst_24h") and f.concepts & {"glucose", "bst"})
            if gmin[0] < float(N["glucose_low"]):
                did = derived("glucose_min", "지난 %s시간 최저 혈당 %s mg/dL(%s, %s)." % (H, fmt_num(gmin[0]), fmt_t(gmin[1]), "검사실" if gmin[2] == "LABEVENTS" else "BST"),
                              inputs, [gmin[0]], concepts={"glucose"})
                add("NEO_HYPOGLYCEMIA", "high", "저혈당", "최저 혈당 %s mg/dL, 재측정과 당 공급 확인 필요." % fmt_num(gmin[0]), [did] + inputs, "신생아 저혈당 관리 기준(프로파일)")
            if gmax[0] > float(N["glucose_high"]):
                did = derived("glucose_max", "지난 %s시간 최고 혈당 %s mg/dL(%s)." % (H, fmt_num(gmax[0]), fmt_t(gmax[1])), inputs, [gmax[0]], concepts={"glucose"})
                add("NEO_HYPERGLYCEMIA", "medium", "고혈당", "최고 혈당 %s mg/dL, 당 주입 속도 확인 필요." % fmt_num(gmax[0]), [did] + inputs, "신생아 고혈당 기준(프로파일)")
        wf = fid_of(lambda f: f.ftype == "weight")
        if ctx.wt_pct is not None and ctx.wt_pct <= -float(N["weight_loss_pct"]) and ctx.hosp_day <= int(N["weight_loss_max_day"]):
            add("NEO_WEIGHT_LOSS", "medium", "체중 감소", "입실 후 첫 기록 대비 체중 %s%%, 수유·수액 계획 점검 필요." % fmt_num(ctx.wt_pct), wf, "신생아 체중 감소 관찰 기준(프로파일)")
        inf = fid_of(lambda f: f.ftype == "intake_24h")
        if ctx.intake_ml_kg_day is not None and inf:
            lo, hi = N["fluid_ml_kg_day_by_day"].get(str(ctx.hosp_day), N["fluid_default"])
            if not (lo <= ctx.intake_ml_kg_day <= hi):
                did = derived("fluid_target", "입원 %d일째 수분·영양 목표 범위(병원 프로파일) %s~%s mL/kg/일." % (ctx.hosp_day, fmt_num(lo), fmt_num(hi)), inf, [lo, hi])
                add("NEO_FLUID_RANGE", "medium", "수분·영양 공급량 범위 밖", "섭취 %s mL/kg/일로 목표 범위 밖, 수액·수유 계획 확인 필요." % fmt_num(ctx.intake_ml_kg_day),
                    inf + [did], "병원 NICU 수액 프로토콜(프로파일)")
        if ctx.urine_rate is not None and ctx.urine_rate < float(N["urine_low_ml_kg_h"]):
            add("NEO_OLIGURIA", "medium", "소변량 감소", "지난 %s시간 소변 %s mL/kg/h, 수분 상태와 신기능 확인 필요." % (H, fmt_num(ctx.urine_rate)),
                fid_of(lambda f: f.ftype == "urine_24h"), "신생아 핍뇨 기준(프로파일)")
        tb = getattr(ctx, "latest_labs", {}).get("tbil")
        tbf = fid_of(lambda f: f.ftype in ("lab_latest", "lab_normal", "lab_change") and "tbil" in f.concepts)
        if tb and tb[0] >= float(N["tsb_alert_mg_dl"]):
            add("NEO_BILI_HIGH", "high", "고빌리루빈혈증", "T.bil %s mg/dL, 재태주수·생후시간별 광선치료 기준표 확인 필요." % fmt_num(tb[0]), tbf, "AAP 2022 고빌리루빈혈증 지침")
        for f in facts:
            if f.ftype == "lab_change" and f.extra.get("concept") == "tbil" and f.extra.get("hours"):
                rate = (f.extra["to"] - f.extra["from"]) / f.extra["hours"]
                if rate >= float(N["bili_rise_mg_dl_h"]):
                    did = derived("bili_rate", "T.bil 상승 속도 %s mg/dL/h(%s → %s)." % (fmt_num(round(rate, 2)), fmt_num(f.extra["from"]), fmt_num(f.extra["to"])),
                                  [f.fid], [round(rate, 2)], concepts={"tbil"})
                    add("NEO_BILI_RISE", "medium", "빌리루빈 빠른 상승", "T.bil 상승 속도 %s mg/dL/h, 용혈 여부와 추적 검사 간격 확인 필요." % fmt_num(round(rate, 2)),
                        [f.fid, did], "AAP 2022 고빌리루빈혈증 지침")
        crp = getattr(ctx, "latest_labs", {}).get("crp")
        if crp and crp[0] > float(N["crp_high_mg_dl"]):
            add("NEO_SEPSIS_MARKERS", "medium", "염증 수치 상승", "CRP %s mg/dL, 감염 평가(배양·항생제) 확인 필요." % fmt_num(crp[0]),
                fid_of(lambda f: f.ftype in ("lab_latest", "lab_change") and "crp" in f.concepts) + fid_of(lambda f: f.ftype == "micro_pending"), "신생아 패혈증 평가 원칙")
    for f in facts:
        if f.ftype == "lab_latest" and f.extra.get("critical"):
            c = f.extra.get("concept")
            add("CRITICAL_LAB_%s" % str(c).upper(), "high", "%s 위험값" % LAB_DISPLAY.get(c, c), "%s %s 위험값, 재검·교정 확인 필요." % (LAB_DISPLAY.get(c, c), fmt_num(f.key_nums[0])),
                [f.fid], "검사 위험값 보고 원칙(프로파일)")
    aki = fid_of(lambda f: f.ftype == "aki" and f.extra.get("new"))
    if aki:
        a = ctx.aki
        add("AKI_KDIGO", "high", "급성신손상(Cr 기준)", "Cr %s → %s, 신독성 약물과 수분 상태 점검 필요." % (fmt_num(a["base"]), fmt_num(a["peak"])), aki, "KDIGO(신생아 수정 기준 포함)")
    for f in facts:
        if f.ftype == "infusion_episode" and f.extra.get("cls") == "vasopressor" and f.status == "active":
            add("PRESSOR_ACTIVE", "medium", "승압·강심제 투여 중", "%s 기준 시각 투여 중, 혈압·관류 지표 확인 필요." % f.extra["generic"], [f.fid], "NICU 순환 보조 관찰 원칙")
    act = getattr(ctx, "active_meds", [])
    med_fid = {}
    for f in facts:
        if f.ftype in ("med_active", "infusion_episode") and f.status == "active":
            for c in f.concepts:
                if c.startswith("med:"):
                    med_fid[c[4:]] = f.fid
    for drug, lvl in PROFILE.get("tdm", {}).items():
        users = [g for g, cls, name, last, route in act if re.search(drug, g + " " + name, re.I)]
        if not users:
            continue
        L = ctx.lab
        has = (not L.empty) and ((L["concept"] == lvl) & (L["t"] > ctx.t0 - pd.Timedelta(hours=W("tdm_level_hours"))) & (L["t"] <= ctx.t0)).any()
        if not has:
            trig = [med_fid.get(u.lower()) for u in users]
            did = derived("tdm_gap", "최근 %s시간 %s 혈중농도 기록 없음." % (fmt_num(W("tdm_level_hours")), drug), trig)
            add("TDM_MISSING", "medium", "혈중농도 감시 필요", "%s 투여 중이나 혈중농도 기록 없음, TDM 시행 여부 확인 필요." % users[0], trig + [did], "아미노글리코사이드·반코마이신 TDM 원칙")
    if not ctx.proc.empty:
        p = ctx.proc[(ctx.proc["t"] <= ctx.t0) & ctx.proc["label"].astype(str).str.contains(PROFILE.get("line_regex", LINE_RX), case=False, regex=True, na=False)]
        if "value" in p.columns:
            vv = to_num(p["value"])
            p = p[vv.isna() | (vv != 0)]
        for lab, g in p.groupby("label"):
            days = int((ctx.t0 - g["t"].min()).total_seconds() // 86400)
            if days >= int(N["line_days_alert"]):
                src = fid_of(lambda f: f.ftype in ("checklist_item", "procedure") and lab in f.text)
                did = derived("line_days", "%s 첫 기록 후 %d일." % (lab, days), src, [days])
                add("LINE_DAYS", "medium", "카테터 유지 기간", "%s 첫 기록 후 %d일, 유지 필요성과 감염 위험 확인 필요." % (lab, days), src + [did], "중심정맥관 관련 혈류감염 예방 원칙")
    for f in facts:
        if f.ftype == "micro_pending":
            add("MICRO_PENDING", "medium", "배양 결과 미보고", "배양 결과 미보고 상태, 결과 확인 필요.", [f.fid], "인계 원칙(I-PASS)")
        if f.ftype == "micro_positive" and f.extra.get("resistant"):
            res = [x.lower() for x in f.extra["resistant"]]
            abx = [g for g, cls, n, last, route in act if cls == "antibiotic" and any(r.split("/")[0][:6] in (g + n).lower() for r in res)]
            if abx:
                add("ABX_MISMATCH", "high", "항생제-감수성 불일치", "%s 투여 중이나 분리균 내성 보고, 항생제 재검토 필요." % ", ".join(abx), [f.fid] + [med_fid.get(g.lower()) for g in abx], "IDSA 원칙")
    names = " ".join(g.lower() + " " + str(n).lower() for g, _, n, _, _ in act)
    for a, b, why in (dur_pairs or DUR_PAIRS_DEFAULT):
        if re.search(a, names, re.I) and re.search(b, names, re.I):
            add("DUR_PAIR", "high", "병용금기 조합", "%s + %s 병용 중: %s." % (a, b, why), [fid for g, fid in med_fid.items() if re.search(a, g, re.I) or re.search(b, g, re.I)],
                "DUR 병용금기(프로파일)")
    return alerts


# =============================================================================
# 5. 생성 모델 A — 플래너 + 템플릿 렌더러(1일 요약지·부모님 설명지) + 선택적 LLM
# =============================================================================


def plan_sections(facts):
    D = PROFILE["documents"]["summary"]
    secs = [s for s in SECTIONS if s != CDSS_SEC()]
    plan = OrderedDict((s, []) for s in secs)
    for s in secs:
        fs = [f for f in facts if f.section == s]
        must = [f for f in fs if f.must]
        rest = sorted([f for f in fs if not f.must], key=lambda f: -f.salience)
        cap = int(D.get("section_cap", {}).get(s, 10))
        plan[s] = must + rest[: max(0, cap - len(must))]
    return plan


def _alert_sent(a, sec):
    if a["trigger_fact_ids"]:
        txt = "[%s] %s [%s] (규칙 %s)" % (a["severity"].upper(), a["text"], ", ".join(a["trigger_fact_ids"]), a["rule_id"])
    else:
        txt = "[%s] %s (규칙 %s)" % (a["severity"].upper(), a["text"], a["rule_id"])
    return {"section": sec, "text": txt, "cites": list(a["trigger_fact_ids"]), "alert": a["alert_id"]}


def render_template(ctx, facts, alerts):
    """1일 요약지 = 문장 리스트. 문장마다 section, text, cites."""
    sents = []
    for s, fs in plan_sections(facts).items():
        if not fs:
            sents.append({"section": s, "text": "해당 사항 없음.", "cites": []})
        for f in fs:
            sents.append({"section": s, "text": "%s [%s]" % (f.text, f.fid), "cites": [f.fid]})
    sents += [_alert_sent(a, CDSS_SEC()) for a in alerts] or [{"section": CDSS_SEC(), "text": "규칙 기반 알림 없음.", "cites": []}]
    for i, x in enumerate(sents, 1):
        x["sid"] = "S%03d" % i
    return sents


# ---- 부모님 설명지: 쉬운 말 템플릿(숫자는 근거 fact 값 그대로). 문구는 이 사전에서만 나오므로 검증기 어휘 사전에도 쓰인다.
LAY_T = {
    "day": "아기는 오늘 입원 %s일째입니다.",
    "intro": "아래에 오늘 하루 아기의 상태를 항목별로 정리했습니다.",
    "w_cur": "아기 몸무게는 %s %s입니다.",
    "w_prev": "어제는 %s %s이었습니다.",
    "w_pct": "입원 후 처음 잰 몸무게(%s %s)와 비교하면 %s%% %s.",
    "w_up": "늘었습니다", "w_down": "줄었습니다",
    "in_total": "지난 하루 동안 아기가 받은 수분과 영양은 모두 %s mL입니다.",
    "in_iv": "주사로 %s mL", "in_ent": "입이나 관으로 먹은 양 %s mL", "in_tail": "입니다.",
    "r_inv": "아기는 숨쉬기를 돕는 인공호흡기를 사용하고 있습니다%s.",
    "r_mode": "(방식: %s)",
    "r_niv": "아기는 코를 통해 압력이나 공기를 넣어 주는 장치%s로 숨쉬기 도움을 받고 있습니다.",
    "r_o2": "아기는 코로 산소를 조금씩 받고 있습니다(분당 %s L).",
    "r_none": "최근 기록에서는 숨쉬기 보조 장치 사용이 확인되지 않았습니다.",
    "r_change": "오늘 숨쉬기 도움 방식이 바뀌었습니다(%s → %s).",
    "events": "오늘 산소포화도가 잠시 낮아진 기록이 %d번, 심장 박동이 느려진 기록이 %d번 있어 의료진이 지켜보고 있습니다.",
    "urine": "지난 하루 동안 소변은 %s mL 나왔습니다.",
    "stool": "대변 기록이 %d번 있습니다.",
    "proc": "오늘 %s 했습니다.",
    "img": "오늘 %s 했습니다.",
    "mpend": "감염 여부를 확인하는 배양 검사 결과를 기다리고 있습니다.",
    "mpos": "배양 검사에서 균이 자랐다는 결과가 있습니다. 자세한 내용은 담당 의료진에게 설명을 들으실 수 있습니다.",
    "mneg": "배양 검사에서 균이 자라지 않았다는 결과가 나왔습니다.",
    "crrt": "콩팥 기능을 돕는 투석 치료(CRRT)를 받았습니다.",
    "med_on": "%s %s사용하고 있습니다.",
    "med_new": "오늘부터 ",
    "med_off": "%s 오늘 중단했습니다.",
    "inf_on": "%s 주사로 계속 받고 있습니다.",
    "inf_off": "%s 주사는 오늘 중단했습니다.",
    "fluids": "수액을 맞고 있습니다.",
    "labs": "오늘 혈액검사를 했습니다.",
    "labs_abn": "이 중 %s 결과는 기준 범위를 벗어나 의료진이 자세히 확인하고 있습니다.",
    "labs_ok": "결과는 기준 범위 안에 있었습니다.",
    "labs_none": "최근 하루 동안 새로 한 혈액검사 기록은 없습니다.",
}
LAY_CLASS = {"vasopressor": "혈압과 심장 기능을 돕는 약", "sedation": "편안하게 해 주는 진정·진통제", "insulin": "혈당을 조절하는 인슐린",
             "anticoagulant": "피가 굳는 것을 막는 약", "antibiotic": "감염을 치료하거나 막는 항생제", "antifungal": "곰팡이 감염을 치료하는 약",
             "diuretic": "소변이 잘 나오게 하는 약", "respiratory_stimulant": "숨 멈춤(무호흡)을 줄여 주는 약", "surfactant": "폐가 잘 펴지도록 돕는 약",
             "nutrition": "주사로 주는 영양", "anticonvulsant": "경련을 막는 약", "antiplatelet": "혈소판 작용을 줄이는 약", "fluid": "수액", "other": "약"}
LAY_LAB = {"tbil": "황달 수치(빌리루빈)", "glucose": "혈당", "bst": "혈당", "crp": "염증 수치(CRP)", "na": "나트륨", "k": "칼륨", "cl": "염소",
           "hb": "혈색소(빈혈 수치)", "plt": "혈소판", "wbc": "백혈구", "anc": "호중구 수", "cr": "콩팥 기능 수치(크레아티닌)", "bun": "요소질소(BUN)",
           "ph": "혈액 산도(pH)", "pco2": "혈액 이산화탄소", "po2": "혈액 산소", "lactate": "젖산", "ica": "칼슘", "mg": "마그네슘", "phos": "인",
           "albumin": "알부민", "inr": "혈액 응고 수치", "aptt": "혈액 응고 수치", "egfr": "콩팥 기능 수치", "tco2": "혈액 중탄산", "hco3": "혈액 중탄산",
           "ast": "간 수치", "alt": "간 수치", "fibrinogen": "혈액 응고 수치"}
LAY_PROC = [(r"brain|cranial|head|뇌", "머리 초음파 검사"), (r"echo|심초음파", "심장 초음파 검사"), (r"EEG|뇌파", "뇌파 검사"),
            (r"intubat|삽관", "숨관(기관 튜브) 넣기"), (r"UVC|umbilical vein|제대정맥", "배꼽 정맥관 넣기"), (r"UAC|umbilical art|제대동맥", "배꼽 동맥관 넣기"),
            (r"PICC|central|중심정맥", "중심정맥관(굵은 주사관) 넣기"), (r"transfus|수혈", "수혈"), (r"chest|흉부|X-?ray|\bAP\b", "가슴 엑스레이 검사"),
            (r"abdomen|복부", "배 엑스레이 검사"), (r"MRI", "MRI 검사"), (r"sono|ultrasound|초음파|doppler", "초음파 검사"), (r"bronch|기관지경", "기관지 내시경 검사")]


def _final_char(word):
    w = re.sub(r"\([^)]*\)\s*$", "", str(word)).strip()
    return w[-1] if w else ""


def josa(word, pair):
    """받침 유무로 조사 선택. pair: '을를', '은는', '이가', '과와'. 한글이 아니면 뒤 글자."""
    ch = _final_char(word)
    if "가" <= ch <= "힣":
        return word + (pair[0] if (ord(ch) - 0xAC00) % 28 else pair[1])
    return word + pair[1]


def lay_proc(label):
    for rx, name in LAY_PROC:
        if re.search(rx, str(label), re.I):
            return name
    return "'%s'" % label


def lay_sentence(ctx, f):
    """fact → 부모님용 쉬운 문장. None 이면 부모님 설명지에 쓰지 않음."""
    e, t = f.extra, f.ftype
    if t == "patient_info":
        return LAY_T["day"] % e.get("hosp_day")
    if t == "weight":
        u = e.get("unit", "g")
        s = LAY_T["w_cur"] % (fmt_num(e["cur"]), u)
        if "prev" in e:
            s += " " + LAY_T["w_prev"] % (fmt_num(e["prev"]), u)
        if e.get("pct"):
            s += " " + LAY_T["w_pct"] % (fmt_num(e["first"]), u, fmt_num(abs(e["pct"])), LAY_T["w_up"] if e["pct"] > 0 else LAY_T["w_down"])
        return s
    if t == "intake_24h":
        s = LAY_T["in_total"] % fmt_num(e["total"])
        sub = ([LAY_T["in_iv"] % fmt_num(e["iv"])] if e.get("iv") else []) + ([LAY_T["in_ent"] % fmt_num(e["enteral"])] if e.get("enteral") else [])
        return s + ((" " + ", ".join(sub) + LAY_T["in_tail"]) if len(sub) > 1 else "")
    if t == "resp_support":
        k = e.get("kind")
        if k in ("invasive", "vent_unknown"):
            return LAY_T["r_inv"] % ((" " + LAY_T["r_mode"] % e["mode"]) if e.get("mode") else "")
        if k == "noninvasive":
            return LAY_T["r_niv"] % (("(%s)" % e["mode"]) if e.get("mode") else "")
        if k == "o2":
            return LAY_T["r_o2"] % fmt_num(e["flow"])
        return LAY_T["r_none"]
    if t == "resp_change":
        return LAY_T["r_change"] % (e["from"], e["to"])
    if t == "neo_events":
        return LAY_T["events"] % (e["n_desat"], e["n_brady"]) if f.status == "present" else None
    if t == "urine_24h":
        return LAY_T["urine"] % fmt_num(e["ml"])
    if t == "stool":
        return LAY_T["stool"] % e["n"]
    if t == "procedure":
        return LAY_T["proc"] % josa(lay_proc(e.get("label", "")), "을를")
    if t == "imaging":
        return LAY_T["img"] % josa(", ".join(sorted(set(lay_proc(n) for n in e.get("names", [])))), "을를")
    if t == "micro_pending":
        return LAY_T["mpend"]
    if t == "micro_positive":
        return LAY_T["mpos"]
    if t == "micro_negative":
        return LAY_T["mneg"]
    if t == "crrt_episode":
        return LAY_T["crrt"]
    if t in ("med_active", "med_stopped", "infusion_episode"):
        nm = "%s(%s)" % (LAY_CLASS.get(e.get("cls"), "약"), e["generic"])
        if t == "med_active":
            return LAY_T["med_on"] % (josa(nm, "을를"), LAY_T["med_new"] if e.get("new") else "")
        if t == "med_stopped":
            return LAY_T["med_off"] % josa(nm, "은는")
        return (LAY_T["inf_on"] % josa(nm, "을를")) if f.status == "active" else (LAY_T["inf_off"] % nm)
    if t == "fluids":
        return LAY_T["fluids"]
    if t == "parent_gap":
        return f.text
    return None


def lay_labs(facts, show_values):
    labs = [f for f in facts if f.ftype in ("lab_latest", "lab_normal", "lab_none", "bst_24h", "lab_change")]
    if not labs:
        return []
    if all(f.ftype in ("lab_none", "lab_change") for f in labs) and any(f.ftype == "lab_none" for f in labs):
        return [(LAY_T["labs_none"], [labs[0].fid])]
    abn = [f for f in labs if f.ftype == "lab_latest" or (f.ftype == "bst_24h" and f.salience >= 3)]
    names = []
    for f in abn:
        c = f.extra.get("concept") or ("bst" if f.ftype == "bst_24h" else None)
        nm = LAY_LAB.get(c, LAB_DISPLAY.get(c, c))
        if show_values and f.key_nums:
            nm = "%s %s" % (nm, fmt_num(f.key_nums[0]))
        if nm not in names:
            names.append(nm)
    s = LAY_T["labs"] + " " + ((LAY_T["labs_abn"] % ", ".join(names)) if names else LAY_T["labs_ok"])
    return [(s, [f.fid for f in labs if f.ftype != "lab_none"])]


def render_parent(ctx, facts, alerts):
    P = PROFILE["documents"]["parent"]
    smap = P.get("section_of", {})
    secs = list(PARENT_SECTIONS.keys())
    psec, csec = P.get("cdss_section", "P7"), P.get("closing_section", "P8")
    by = OrderedDict((s, []) for s in secs)
    first = secs[0]
    for f in facts:
        s = f.section if f.ftype == "parent_gap" else smap.get(f.ftype)
        if s not in by or f.ftype in ("lab_latest", "lab_change", "lab_normal", "lab_none", "bst_24h"):
            continue
        txt = lay_sentence(ctx, f)
        if txt:
            by[s].append({"section": s, "text": "%s [%s]" % (txt, f.fid), "cites": [f.fid]})
    lsec = smap.get("lab_latest", "P5")
    if lsec in by:
        for txt, cites in lay_labs(facts, P.get("show_lab_values", False)):
            by[lsec].insert(0, {"section": lsec, "text": "%s [%s]" % (txt, ", ".join(cites)), "cites": cites})
    if first in by:
        by[first].append({"section": first, "text": LAY_T["intro"], "cites": [], "fixed": True})
    lay = P.get("alert_text", {})
    seen = set()
    for a in alerts:
        r = a["rule_id"]
        if a.get("parent_visible") and r in lay and r not in seen and psec in by:
            seen.add(r)
            by[psec].append({"section": psec, "text": "%s [%s]" % (lay[r], ", ".join(a["trigger_fact_ids"])), "cites": list(a["trigger_fact_ids"])})
    for s in secs:
        if s == csec:
            by[s] = [{"section": s, "text": t, "cites": [], "fixed": True} for t in P.get("closing", [])]
        elif not by[s]:
            by[s].append({"section": s, "text": P.get("empty_text", "오늘 기록에서 따로 말씀드릴 내용은 없습니다."), "cites": [], "fixed": True})
    sents = [x for s in secs for x in by[s]]
    for i, x in enumerate(sents, 1):
        x["sid"] = "S%03d" % i
    return sents


def note_markdown(ctx, sents, verdicts=None, coverage=None, title=None, checklist=None, doc="summary", show_cites=True, extra_metrics=None):
    D = PROFILE["documents"][doc]
    title = title or D["title"]
    secmap = SECTIONS if doc == "summary" else PARENT_SECTIONS
    vd = {v["sid"]: v for v in (verdicts or [])}
    if doc == "summary":
        lines = ["# " + title, "", "- stay_id %s / hadm_id %s / 기준 시각 %s (%d/%d일째 요약)" % (ctx.stay_id, ctx.hadm_id, fmt_t(ctx.t0), ctx.day_index, ctx.n_days),
                 "- 생성: kcd_nicu.py v%s, 프로파일 %s, %s" % (VERSION, PROFILE.get("profile_name"), time.strftime("%Y-%m-%d %H:%M")), ""]
    else:
        lines = ["# " + title, "", "%s 기준" % pd.Timestamp(ctx.t0).strftime("%m월 %d일 %H시"), "", "> " + D.get("notice", ""), ""]
    cur = None
    for x in sents:
        if x["section"] != cur:
            cur = x["section"]
            lines += ["", "## " + secmap.get(cur, cur)]
        text = x["text"] if show_cites else re.sub(r"\s*\[F\d{3}(?:\s*,\s*F\d{3})*\]", "", re.sub(r"\s*\(규칙 [^)]*\)", "", x["text"]))
        mark = ""
        if show_cites and x["sid"] in vd and vd[x["sid"]]["label"] != "Supported":
            mark = "  ⚠️검증: %s" % vd[x["sid"]]["label"]
        lines.append("- " + text + mark)
    if show_cites and coverage is not None:
        miss = [c for c in coverage if c["status"] == "Missing"]
        part = [c for c in coverage if c["status"] == "Partial"]
        lines += ["", "## 검증 요약"]
        nsup = sum(1 for v in vd.values() if v["label"] == "Supported")
        lines.append("- 문장 %d개 중 근거 일치 %d개, 불일치 %d개, 근거 없음 %d개, 판정 불가 %d개" % (
            len(vd), nsup, sum(1 for v in vd.values() if v["label"] == "Contradicted"), sum(1 for v in vd.values() if v["label"] == "NotAddressed"),
            sum(1 for v in vd.values() if v["label"] == "Unverifiable")))
        lines.append("- 필수 정보 %d개 중 누락 %d개, 부분 전달 %d개" % (len(coverage), len(miss), len(part)))
        for c in miss[:10]:
            lines.append("  - 누락: %s %s" % (c["fact_id"], c["text"]))
        if extra_metrics:
            lines.append("- 읽기 쉬움: 문장 평균 %s자, 최장 %s자, 영문 약어 포함 문장 %s%%" % (extra_metrics.get("read_avg_chars"), extra_metrics.get("read_max_chars"),
                                                                     fmt_num(100 * (extra_metrics.get("read_english_sent_ratio") or 0))))
    if show_cites and checklist:
        gaps = [c for c in checklist if c["status"] == "gap"]
        lines += ["", "## 필수 항목 체크리스트 (%s)" % PROFILE.get("profile_name", ""), "", "| 항목 | 필수 | 데이터 | 문서 반영 | 근거 |", "|---|---|---|---|---|"]
        lab = {"data": "있음", "gap": "없음(확인 필요)", "absent": "없음"}
        inl = {"covered": "반영", "missing": "누락", "n/a": "-"}
        for c in checklist:
            lines.append("| %s | %s | %s | %s | %s |" % (c["label"], "필수" if c["required"] else "선택", lab.get(c["status"], c["status"]),
                                                       inl.get(c.get("in_note", "-"), c.get("in_note", "-")), ", ".join(c["fact_ids"][:4])))
        if gaps:
            lines += ["", "- 데이터에 없어 직접 확인할 항목: " + ", ".join(c["label"] for c in gaps)]
    return "\n".join(lines) + "\n"


class LLM(object):
    def __init__(self):
        self.base = os.environ.get("KCD_LLM_BASE_URL")
        self.model = os.environ.get("KCD_LLM_MODEL")
        self.key = os.environ.get("KCD_LLM_API_KEY", "EMPTY")
        self.ok = bool(self.base and self.model)

    def chat(self, system, user, temperature=0.0, max_tokens=1500):
        import urllib.request
        body = json.dumps({"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                           "temperature": temperature, "max_tokens": max_tokens}).encode("utf-8")
        req = urllib.request.Request(self.base.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key})
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read().decode("utf-8"))
        return out["choices"][0]["message"]["content"]


SYS_GROUNDED = ("당신은 NICU 환아의 1일 요약지를 작성하는 보조자다. 주어진 facts만 사용한다.\n"
                "규칙: 1) facts에 없는 정보·추정·권고를 쓰지 않는다 2) 문장마다 끝에 사용한 fact_id를 [F001] 형식으로 인용한다\n"
                "3) 수치·단위·시각(MM-DD HH:MM)은 fact 표기를 그대로 복사한다 4) 개조식(~함/~임)으로 쓴다\n"
                "5) 출력은 '- '로 시작하는 문장 목록만 쓴다. 제목·설명은 쓰지 않는다.")
SYS_PARENT = ("당신은 NICU에 입원한 아기의 부모님께 오늘 상태를 설명하는 글을 쓰는 보조자다. 주어진 facts만 사용한다.\n"
              "규칙: 1) facts에 없는 정보, 예후, 추측, 약속을 쓰지 않는다 2) 쉬운 우리말로, 한 문장은 60자 안팎으로 쓴다\n"
              "3) 의학 용어·약어는 괄호 안에 두고 쉬운 말을 먼저 쓴다 4) 숫자는 facts의 값을 그대로 쓰고, 검사 수치는 쓰지 않는다\n"
              "5) 존댓말(~습니다)로 쓴다 6) 문장마다 끝에 사용한 fact_id를 [F001] 형식으로 인용한다 7) '- '로 시작하는 문장 목록만 쓴다.")


def parse_bullets(text):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("- ", "* ", "• ")):
            out.append(line[2:].strip())
        elif re.match(r"^\d+[.)] ", line):
            out.append(re.sub(r"^\d+[.)] ", "", line))
    return [x for x in out if x]


def cites_of(text):
    ids = []
    for m in CITE_RX.finditer(text):
        ids += [x.strip() for x in m.group(1).split(",")]
    return ids


def _doc_plan(facts, doc):
    if doc == "summary":
        return plan_sections(facts)
    smap = PROFILE["documents"]["parent"].get("section_of", {})
    secs = [s for s in PARENT_SECTIONS if s not in (PCDSS_SEC(), PROFILE["documents"]["parent"].get("closing_section", "P8"))]
    plan = OrderedDict((s, []) for s in secs)
    for f in facts:
        s = f.section if f.ftype == "parent_gap" else smap.get(f.ftype)
        if s in plan:
            plan[s].append(f)
    return plan


def render_llm(ctx, facts, alerts, llm, audit, repair=True, max_rounds=2, doc="summary"):
    plan = _doc_plan(facts, doc)
    F = {f.fid: f for f in facts}
    secmap = SECTIONS if doc == "summary" else PARENT_SECTIONS
    sysp = SYS_GROUNDED if doc == "summary" else SYS_PARENT
    sents = []
    for s, fs in plan.items():
        if not fs:
            sents.append({"section": s, "text": "해당 사항 없음." if doc == "summary" else PROFILE["documents"]["parent"].get("empty_text"), "cites": [], "fixed": doc != "summary"})
            continue
        user = "섹션: %s\nfacts:\n%s" % (secmap[s], "\n".join("%s: %s" % (f.fid, f.text) for f in fs))
        t1 = time.time()
        try:
            out = llm.chat(sysp, user)
        except Exception as e:
            out = ""
            audit.log("llm_error", {"section": s}, {"error": repr(e)})
        audit.log("llm_render_" + doc, {"section": s, "facts": [f.fid for f in fs]}, {"n_chars": len(out)}, time.time() - t1, llm.model)
        bl = parse_bullets(out) or [("%s [%s]" % (f.text, f.fid)) for f in fs]
        for b in bl:
            sents.append({"section": s, "text": b, "cites": cites_of(b)})
    if doc == "summary":
        sents += [_alert_sent(a, CDSS_SEC()) for a in alerts]
    else:
        tmpl = render_parent(ctx, facts, alerts)
        sents += [x for x in tmpl if x["section"] in (PCDSS_SEC(), PROFILE["documents"]["parent"].get("closing_section", "P8"))]
    for i, x in enumerate(sents, 1):
        x["sid"] = "S%03d" % i
    stats = {"repaired": 0, "replaced": 0, "appended": 0}
    if not repair:
        return sents, stats
    V = Verifier(ctx, facts, alerts=alerts, doc=doc)
    skip = {CDSS_SEC()} if doc == "summary" else {PCDSS_SEC(), PROFILE["documents"]["parent"].get("closing_section", "P8")}
    for rnd in range(max_rounds):
        bad = [(x, V.check(x)) for x in sents if x["section"] not in skip and not x.get("fixed")]
        bad = [(x, v) for x, v in bad if v["label"] in ("Contradicted", "NotAddressed")]
        if not bad:
            break
        for x, v in bad:
            ev = [F[c] for c in x["cites"] if c in F] or plan.get(x["section"], [])
            user = "다음 문장이 근거 facts와 일치하지 않는다(사유: %s). facts만 사용해 한 문장으로 다시 쓰고 인용을 붙여라.\n문장: %s\nfacts:\n%s" % (
                v["reason"], x["text"], "\n".join("%s: %s" % (f.fid, f.text) for f in ev))
            try:
                b = parse_bullets(llm.chat(sysp, user))
                if b:
                    x["text"], x["cites"] = b[0], cites_of(b[0])
                    stats["repaired"] += 1
            except Exception as e:
                audit.log("llm_error", {"sid": x["sid"]}, {"error": repr(e)})
    for x in sents:
        if x["section"] in skip or x.get("fixed"):
            continue
        v = V.check(x)
        if v["label"] in ("Contradicted", "NotAddressed"):
            ev = [F[c] for c in x["cites"] if c in F]
            if ev:
                parts = [(f.text if doc == "summary" else (lay_sentence(ctx, f) or f.text)) for f in ev]
                x["text"] = " ".join("%s [%s]" % (p, f.fid) for p, f in zip(parts, ev))
                x["cites"] = [f.fid for f in ev]
            else:
                x["text"], x["cites"] = "(삭제됨: 근거 없는 문장)", []
            stats["replaced"] += 1
    only = required_fids(ctx, doc)
    cov = V.coverage(sents, only=only)
    for c in cov:
        if c["status"] == "Missing":
            f = F[c["fact_id"]]
            sec = f.section if (doc == "summary" or f.ftype == "parent_gap") else PROFILE["documents"]["parent"].get("section_of", {}).get(f.ftype, f.section)
            txt = f.text if doc == "summary" else (lay_sentence(ctx, f) or f.text)
            idx = max([i for i, x in enumerate(sents) if x["section"] == sec] or [len(sents) - 1])
            sents.insert(idx + 1, {"section": sec, "text": "%s [%s]" % (txt, f.fid), "cites": [f.fid]})
            stats["appended"] += 1
    for i, x in enumerate(sents, 1):
        x["sid"] = "S%03d" % i
    return sents, stats


def naive_prompt(ctx, max_lines=350):
    """A0: 접지 없는 비교군. 원시 행을 텍스트로 덤프."""
    lines = []

    def dump(name, df, cols, n):
        if df is None or df.empty:
            return
        cols = [c for c in cols if c in df.columns]
        lines.append("## %s" % name)
        for _, r in df.tail(n).iterrows():
            lines.append(" | ".join(str(r[c]) for c in cols))
    lines.append("환아: 성별 %s, NICU 입실 %s, 요약 기준 시각 %s(지난 24시간)" % (getattr(ctx, "sex", "?"), ctx.intime, ctx.t0))
    dump("DIAGNOSES_ICD", ctx.dx, ["icd_code", "seq_num"], 10)
    dump("CHARTEVENTS", ctx.win(ctx.chart), ["charttime", "itemid", "value", "valueuom"], 100)
    dump("LABEVENTS", ctx.win(ctx.lab), ["charttime", "label", "value", "valueuom"], 80)
    dump("EMAR", ctx.win(ctx.emar), ["charttime", "medication", "infusion_rate_hr", "infusion_rate_min"], 60)
    dump("PRESCRIPTIONS", ctx.rx, ["starttime", "stoptime", "drug", "dose_val_rx", "dose_unit_rx", "route"], 40)
    dump("INPUTEVENTS", ctx.win(ctx.inp), ["starttime", "label", "amount"], 40)
    dump("OUTPUTEVENTS", ctx.win(ctx.out), ["charttime", "label", "value"], 40)
    dump("PROCEDUREEVENTS", ctx.win(ctx.proc), ["starttime", "label"], 20)
    return "\n".join(lines[:max_lines])


def render_naive(ctx, llm, audit, doc="summary"):
    secmap = SECTIONS if doc == "summary" else PARENT_SECTIONS
    who = "NICU 의사로서 아래 환아 데이터로 1일 요약지를" if doc == "summary" else "아래 NICU 아기 데이터로 부모님께 드릴 오늘 상태 설명지를 쉬운 말로"
    sys_ = "%s 작성하라. '- '로 시작하는 문장 목록으로, 다음 섹션 제목을 '## '로 붙여 쓴다: %s" % (who, ", ".join(secmap.values()))
    t1 = time.time()
    out = llm.chat(sys_, naive_prompt(ctx), max_tokens=2000)
    audit.log("llm_naive_" + doc, {"stay": ctx.stay_id}, {"n_chars": len(out)}, time.time() - t1, llm.model)
    sents, sec = [], list(secmap.keys())[0]
    for line in out.splitlines():
        l = line.strip()
        if l.startswith("#"):
            h = l.lstrip("#").strip()
            for k, v in secmap.items():
                core = re.sub(r"^\d+\.\s*", "", v).split(" (")[0]
                if core and (core in h or h in v):
                    sec = k
        elif l.startswith(("- ", "* ")):
            sents.append({"section": sec, "text": l[2:], "cites": []})
    for i, x in enumerate(sents, 1):
        x["sid"] = "S%03d" % i
    return sents


def required_fids(ctx, doc):
    """문서별 누락 검증 대상 fact. 요약지는 None(=must fact 전체), 부모 설명지는 필수 체크리스트 항목의 fact."""
    if doc == "summary":
        return None
    return {f.fid for c in ctx.checklist.get("parent", []) if c["item"].get("required") for f in c["facts"]}


# =============================================================================
# 6. 검증 모델 B — B1 할루시네이션, B2 누락
# =============================================================================

ALIAS = {  # concept -> 언급 탐지 정규식
    "hr": r"\bHR\b|심박", "sbp": r"\bSBP\b|수축기|\bBP\b|혈압", "dbp": r"\bDBP\b|이완기|\bBP\b|혈압", "map": r"\bMAP\b|평균동맥압|평균혈압",
    "rr": r"\bRR\b|호흡수", "spo2": r"SpO2|산소포화도", "temp": r"\bBT\b|체온", "gcs": r"GCS|의식", "o2_flow": r"산소 ?\d|L/min",
    "urine": r"소변|urine", "io": r"섭취|배설|I/O", "drain": r"배액", "ev_crrt": EVENT_VOCAB[3][1], "ev_vent": EVENT_VOCAB[1][1],
}
for c, disp, rx, _, _ in LAB_CONCEPTS:
    ALIAS.setdefault(c, r"(?-i:(?<![A-Za-z0-9/])%s(?![A-Za-z0-9/]))" % re.escape(disp))
ALIAS["k"] = r"(?-i:(?<![A-Za-z0-9/])K(?![A-Za-z0-9/]))|칼륨|포타슘"
ALIAS["na"] = r"(?-i:(?<![A-Za-z0-9/])Na(?![A-Za-z0-9/]))|나트륨"
ALIAS["cr"] = r"(?-i:\bCr\b)|크레아티닌|creatinine"
ALIAS["mg"] = r"(?-i:(?<![A-Za-z/])Mg(?![A-Za-z/]))|마그네슘"
ALIAS["hr"] = r"(?-i:(?<![A-Za-z/])HR(?![A-Za-z]))|심박"
ALIAS["rr"] = r"(?-i:(?<![A-Za-z/])RR(?![A-Za-z]))|호흡수"
ALIAS["temp"] = r"(?-i:\bBT\b)|체온"
ALIAS["sbp"] = r"(?-i:\bSBP\b|\bBP\b)|수축기|혈압"
ALIAS["dbp"] = r"(?-i:\bDBP\b|\bBP\b)|이완기|혈압"
ALIAS["map"] = r"(?-i:\bMAP\b)|평균동맥압|평균혈압"
for ev, rx in EVENT_VOCAB:
    ALIAS.setdefault(ev, rx)
ALIAS["weight"] = r"체중|몸무게"
ALIAS["glucose"] = ALIAS["glucose"] + r"|혈당"
ALIAS["bst"] = r"(?-i:\bBST\b)|현장 혈당"
ALIAS["tbil"] = ALIAS["tbil"] + r"|빌리루빈|황달"
ALIAS["crp"] = ALIAS["crp"] + r"|염증 수치"
ALIAS["hb"] = ALIAS["hb"] + r"|혈색소|빈혈 수치"
ALIAS["plt"] = ALIAS["plt"] + r"|혈소판"
ALIAS["wbc"] = ALIAS["wbc"] + r"|백혈구"
ALIAS["cr"] = ALIAS["cr"] + r"|크레아티닌"
ALIAS["urine"] = r"소변|urine"
ALIAS["stool"] = r"대변|배변|stool"
KNOWN_MEDS = sorted(set(re.findall(r"[a-z]{4,}", "|".join(rx for _, rx, _ in MED_CLASSES))) - {"insulin", "saline", "dextrose", "solution", "plasma", "ringer"})
TEMPLATE_LEX = set("""최종 활력징후 기준 최근 시간 범위 의식 기록 호흡 전동 인공호흡기 산소유량 없음 있음 참고범위 검사 위험값 높음 낮음 섭취 배설 소변 균형 배액관 유지 배액
투여 마지막 중단 사용 승압제 진정 진통제 인슐린 항응고제 항혈소판제 항생제 항진균제 이뇨제 수액 입실 퇴실 재실 입원 경로 병동 이동 주진단 기타 진단 수술 시술 코드 관련 경과
최고 최저 상승 배양 동정 미보고 결과 내성 동정균 영상검사 판독문 데이터 해당 사항 규칙 알림 필요 확인 주입속도 기록값 후 직전 까지 중 이후 지속 시행 회 총 제거량 남성 여성 나이 미상
모드 날짜 명세 상태 관찰 주의 계획 측정 간격 단축 검토 재확인 적합성 가능 여부 교정 재검 추적 점검 신독성 약물 전환 혈당 피하 저칼륨 저마그네슘 전해질 심전도 정상화 재저하
혈압 이상 미만 증가 집중 위험 재입실 신대체요법 투석 접근로 범위로 신기능 용량 경고 계산 감수성 불일치 재검토 분리균 보고 위치 관리 개 검증기 보완 삭제됨 문장 근거""".split())


TEMPLATE_LEX |= set("""환아 남아 여아 일째 요약 지난 기록상 미만 회 측정 전일 첫 대비 장관 경관 경구 정맥 기타 환산 배변 대소변 혼합 변경 마지막 침습적 비침습 보조
방식 미상 지속주입 기준 시각 승압 강심제 오늘 시작 중단 처방 유지 현장 혈당 목표 재태주수 생후시간별 광선치료 기준표 용혈 간격 무호흡 서맥 저하 보온 환경 온도
빈맥 빈호흡 원인 발열 통증 저혈량 재평가 당 공급 주입 속도 수유 계획 신기능 감염 평가 혈중농도 시행 여부 카테터 유지 필요성 혈류감염 관류 지표 미결 조정""".split())


def parent_lex():
    """부모님 설명지 어휘(쉬운 말 템플릿·병원 프로파일 문구). 이 어휘 밖 표현은 근거 확인 대상."""
    P = PROFILE["documents"]["parent"] if PROFILE else {}
    texts = list(LAY_T.values()) + list(LAY_CLASS.values()) + list(LAY_LAB.values()) + [v for _, v in LAY_PROC]
    texts += list(P.get("alert_text", {}).values()) + list(P.get("closing", [])) + [P.get("empty_text", ""), P.get("gap_text", ""), P.get("notice", "")]
    texts += [c.get("gap_text", "") + " " + c.get("label", "") for c in P.get("checklist", [])]
    return set(re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}|[가-힣]{2,}", " ".join(texts)))


class Verifier(object):
    def __init__(self, ctx, facts, judge=None, alerts=None, doc="summary"):
        self.doc = doc
        self.lex = TEMPLATE_LEX | (parent_lex() if doc == "parent" else set())
        self.cdss_secs = (CDSS_SEC(), PCDSS_SEC())
        self.alert_lex = set()
        self.alert_texts = [x.get("text", "") for x in (alerts or [])]
        for a in (alerts or []):
            self.alert_lex |= set(re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}|[가-힣]{2,}", a.get("text", "") + " " + a.get("title", "")))
        self.ctx = ctx
        self.facts = facts
        self.F = {f.fid: f for f in facts}
        self.judge = judge
        self.chk_alias = {}
        for dn, pre in (("summary", "chk:"), ("parent", "pchk:")):
            for it in (PROFILE["documents"][dn]["checklist"] if PROFILE else []):
                self.chk_alias[pre + it["key"]] = it.get("mention_regex") or re.escape(it["label"].split("(")[0].split("·")[0].strip())
        self.record_concepts = set(ctx.events_concepts)
        self.concept_vals = defaultdict(list)
        self.concept_times = defaultdict(set)
        for f in facts:
            self.record_concepts |= f.concepts
            for c in f.concepts:
                self.concept_vals[c] += f.allowed_nums
                self.concept_times[c] |= set(f.allowed_times)
        for c, vals in ctx.raw_values.items():
            self.concept_vals[c] += vals
        rec_meds = set(c[4:] for c in self.record_concepts if c.startswith("med:"))
        self.med_names = sorted(set(KNOWN_MEDS) | rec_meds, key=len, reverse=True)
        self.all_nums = [n for f in facts for n in f.allowed_nums]
        self.all_times = set(t for f in facts for t in f.allowed_times)

    # 언급된 개념
    def mentions(self, text, with_chk=False, cited=None):
        t = CITE_RX.sub(" ", text)
        if self.doc == "parent":  # 쉬운 말 약효 설명(예: '혈압과 심장 기능을 돕는 약')은 측정값 언급이 아님
            for ph in LAY_CLASS.values():
                t = t.replace(ph, " ")
            for c in (cited or ()):  # 인용한 검사의 쉬운 말 이름(예: '백혈구(호중구)')은 그 검사 언급으로 처리
                if c in LAY_LAB:
                    t = t.replace(LAY_LAB[c], " ")
        t = re.sub(r"\(규칙 [^)]*\)", " ", t)
        found = set()
        if with_chk:
            for c, rx in self.chk_alias.items():
                if re.search(rx, t, re.I):
                    found.add(c)
        for c, rx in ALIAS.items():
            if re.search(rx, t, re.I):
                found.add(c)
        low = t.lower()
        for m in self.med_names:
            if len(m) >= 4 and re.search(r"(?<![a-z])%s(?![a-z])" % re.escape(m), low):
                found.add("med:" + m)
        # 약물 이름은 rec_meds 의 첫 단어로도 매칭
        return found

    @staticmethod
    def _num_match(x, vals):
        for v in vals:
            if abs(x - v) <= max(0.051, 0.01 * abs(v)):
                return True
        return False

    def check(self, s):
        text = s["text"]
        if s.get("fixed"):
            return {"sid": s["sid"], "label": "Supported", "method": "fixed", "reason": "", "citation_error": False}
        if re.match(r"^(해당 사항 없음|규칙 기반 알림 없음)", text) or text.startswith("(삭제됨"):
            return {"sid": s["sid"], "label": "Supported", "method": "empty", "reason": "", "citation_error": False}
        cites = [c for c in s.get("cites", []) if c in self.F]
        bad_cite = [c for c in s.get("cites", []) if c not in self.F]
        cf = [self.F[c] for c in cites]
        nums = extract_nums(text)
        times = extract_times(text)
        ment = self.mentions(text, cited=set().union(*[f.concepts for f in cf]) if cf else None)
        problems = []
        not_addr = []
        cit_err = bool(bad_cite)
        cited_concepts = set().union(*[f.concepts for f in cf]) if cf else set()
        cited_nums = [n for f in cf for n in f.allowed_nums]
        cited_times = set(t for f in cf for t in f.allowed_times)
        cited_raw = " ".join(f.text for f in cf)
        cited_text = cited_raw.lower()
        # 1) 개념(약물·사건·검사) 언급
        for c in ment:
            if c in cited_concepts or (c.startswith("med:") and c[4:] in cited_text):
                continue
            if cf and c in ALIAS and re.search(ALIAS[c], cited_raw, re.I):
                continue  # 근거 fact 문구 자체에 같은 표현이 있음(예: 진단명 속 '혈압')
            if s.get("section") in self.cdss_secs and c in ALIAS and any(re.search(ALIAS[c], t, re.I) for t in self.alert_texts):
                continue  # CDSS 고정 문구
            if c in ("sbp", "dbp") and ({"sbp", "dbp"} & cited_concepts):
                continue
            if c in self.record_concepts or (c.startswith("med:") and any(c[4:] in x for x in self.record_concepts)):
                if cf:
                    cit_err = True
            else:
                # 검사/활력 등 약어 단독 언급은 기록 부재 시 근거 없음
                not_addr.append(c)
        # 2) 수치
        pool_global = []
        for c in ment:
            pool_global += self.concept_vals.get(c, [])
        for x in nums:
            if cf and self._num_match(x, cited_nums):
                continue
            if self._num_match(x, pool_global if pool_global else self.all_nums):
                if cf:  # 인용한 근거와 다름 → 인용 근거 기준으로는 지지되지 않음(VeriFact 기준)
                    problems.append("수치 %s 인용 근거와 불일치(다른 기록 값과 일치)" % fmt_num(x))
                continue
            problems.append("수치 %s 근거 불일치" % fmt_num(x))
        # 3) 시각
        for t in times:
            if cf and t in cited_times:
                continue
            if t in self.all_times:
                if cf:
                    problems.append("시각 %s 인용 근거와 불일치(다른 기록 시각과 일치)" % t)
                continue
            problems.append("시각 %s 근거 불일치" % t)
        # 4) 상태(투여 중/중단)
        st_active = bool(re.search(r"투여 중|유지 중|적용 중|지속 중|투여중|사용 중|처방 유지|사용하고 있|계속 받고 있", text))
        st_stop = bool(re.search(r"중단|종료|중지", text))
        for f in cf:
            if f.status == "active" and st_stop and not st_active:
                problems.append("상태 불일치(기록: 투여 중)")
            if f.status == "stopped" and st_active and not st_stop:
                problems.append("상태 불일치(기록: 중단)")
        # 5) 코드
        for code in extract_codes(text):
            if not any(code in f.codes for f in (cf or self.facts)):
                problems.append("코드 %s 근거 없음" % code)
        if problems:
            lab, reason = "Contradicted", "; ".join(problems)
        elif not_addr:
            lab, reason = "NotAddressed", "기록에 없는 항목 언급: %s" % ", ".join(sorted(not_addr))
        else:
            novel = self.novel_terms(text, cf)
            if s.get("section") in self.cdss_secs:
                novel = [t for t in novel if t not in self.alert_lex]
            if novel and self.judge is None:
                lab, reason = "Unverifiable", "근거 텍스트에 없는 용어: %s" % ", ".join(novel[:5])
            elif novel:
                lab, reason = self.judge_claim(text, cf)
            elif not cf and not nums and not ment:
                lab, reason = "Unverifiable", "검증 가능한 요소 없음"
            else:
                lab, reason = "Supported", ""
        return {"sid": s["sid"], "label": lab, "method": "cited" if cf else "global", "reason": reason, "citation_error": cit_err,
                "mentions": sorted(ment)}

    def novel_terms(self, text, cf):
        src = " ".join(f.text for f in (cf or [])) if cf else ""
        toks = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}|[가-힣]{2,}", CITE_RX.sub(" ", re.sub(r"\(규칙 [^)]*\)", " ", text)))
        src_toks = set(re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}|[가-힣]{2,}", src))
        src_stems = set(t[:2] for t in src_toks if re.match(r"[가-힣]", t))
        out = []
        for t in toks:
            if t in src_toks or t in self.lex or t.lower() in self.med_names:
                continue
            if re.match(r"[가-힣]", t) and (t[:2] in src_stems or t[:-1] in self.lex or t[:-2] in self.lex or t.rstrip("함임됨음") in self.lex):
                continue
            if re.match(r"[A-Za-z]", t) and t.upper() in ("HIGH", "MEDIUM", "LOW", "INFO"):
                continue
            out.append(t)
        return out

    def judge_claim(self, text, cf):
        sys_ = ("당신은 임상 기록 사실 검증자다. claim이 evidence로 뒷받침되는지 판정한다. 라벨: Supported / Contradicted / NotAddressed. "
                "evidence에 없는 의학 지식으로 추론하지 않는다. JSON으로만 답한다: {\"label\": ..., \"reason\": ...}")
        ev = "\n".join("%s: %s" % (f.fid, f.text) for f in (cf or self.facts[:40]))
        try:
            out = self.judge.chat(sys_, "claim: %s\nevidence:\n%s" % (text, ev), max_tokens=200)
            m = re.search(r"\{.*\}", out, re.S)
            j = json.loads(m.group(0)) if m else {}
            lab = j.get("label", "Unverifiable")
            if lab not in ("Supported", "Contradicted", "NotAddressed"):
                lab = "Unverifiable"
            return lab, "judge: " + str(j.get("reason", ""))[:200]
        except Exception as e:
            return "Unverifiable", "judge 오류: %r" % e

    def verify(self, sents):
        return [self.check(s) for s in sents]

    def coverage(self, sents, verdicts=None, only=None):
        vd = {v["sid"]: v for v in (verdicts or self.verify(sents))}
        res = []
        for f in self.facts:
            if only is not None:
                if f.fid not in only:
                    continue
            elif not f.must or f.ftype == "parent_gap":
                continue
            keys = f.key_nums
            if self.doc == "parent":
                hide = f.ftype in ("lab_latest", "lab_change", "lab_normal", "bst_24h") and not PROFILE["documents"]["parent"].get("show_lab_values")
                keys = [] if hide else f.key_nums[:1]
            cand = []
            mention_only = False
            for s in sents:
                v = vd.get(s["sid"])
                if v and v["label"] in ("Contradicted", "NotAddressed"):
                    continue
                if f.fid in s.get("cites", []):
                    cand.append(s)
                    continue
                m = self.mentions(s["text"], with_chk=True, cited=set().union(*[self.F[c].concepts for c in s.get("cites", []) if c in self.F]) or None)
                key_c = {c for c in f.concepts if not c.startswith("cls:")}
                if m & key_c:
                    if f.ftype == "checklist_gap" or (not f.key_nums and any(c.startswith("chk:") for c in (m & key_c))):
                        ok = True  # 체크리스트 항목 언급(예: 외부 기록지의 '알레르기 없음')
                    elif keys:
                        ok = any(self._num_match(x, keys) for x in extract_nums(s["text"]))
                    elif f.allowed_times:  # 수치 없는 사건 fact: 같은 시각 언급이 있어야 전달로 인정(부정문 오인 방지)
                        ok = bool(set(extract_times(s["text"])) & set(f.allowed_times))
                    else:
                        ok = not re.search(r"없음|없다|기록 없|no record", s["text"])
                    if ok:
                        cand.append(s)
                    else:
                        mention_only = True
            if cand:
                if keys:
                    got = [n for n in keys if any(self._num_match(x, [n]) for s in cand for x in extract_nums(s["text"]))]
                    frac = len(got) / float(len(keys))
                else:
                    frac = 1.0
                    if f.status:
                        cited_c = [s for s in cand if f.fid in s.get("cites", [])] or cand
                        txt = " ".join(s["text"] for s in cited_c)
                        if f.status == "active" and re.search(r"중단", txt) and not re.search(r"투여 중|유지|지속|직전까지|사용하고 있|받고 있|맞고 있", txt):
                            frac = 0.5
                st = "Covered" if frac >= 0.999 else "Partial"
            else:
                st, frac = ("Partial", 0.0) if mention_only else ("Missing", 0.0)
            res.append({"fact_id": f.fid, "fact_type": f.ftype, "salience": f.salience, "status": st, "frac": round(frac, 2), "text": f.text})
        return res


def readability(sents):
    """부모님 설명지 읽기 쉬움 지표(고정 문구 제외): 문장 길이, 영문 약어 포함 문장 비율, 문장당 숫자 수."""
    xs = [re.sub(r"\s*\[F\d{3}(?:\s*,\s*F\d{3})*\]", "", s["text"]) for s in sents if not s.get("fixed")]
    if not xs:
        return {"read_avg_chars": 0, "read_max_chars": 0, "read_english_sent_ratio": 0, "read_nums_per_sent": 0}
    units = {"mL", "kg", "L", "g", "min"}
    eng = [x for x in xs if [w for w in re.findall(r"[A-Za-z]{2,}", x) if w not in units]]
    return {"read_avg_chars": round(sum(len(x) for x in xs) / float(len(xs)), 1), "read_max_chars": max(len(x) for x in xs),
            "read_english_sent_ratio": round(len(eng) / float(len(xs)), 3), "read_nums_per_sent": round(sum(len(extract_nums(x)) for x in xs) / float(len(xs)), 2)}


def metrics(verdicts, coverage, alerts=None, checklist=None):
    n = len(verdicts) or 1
    lab = defaultdict(int)
    for v in verdicts:
        lab[v["label"]] += 1
    cite_err = sum(1 for v in verdicts if v.get("citation_error"))
    must = len(coverage) or 1
    miss = [c for c in coverage if c["status"] == "Missing"]
    part = [c for c in coverage if c["status"] == "Partial"]
    sal_all = sum(c["salience"] for c in coverage) or 1
    out = {"n_sentences": len(verdicts), "supported": lab["Supported"], "contradicted": lab["Contradicted"], "not_addressed": lab["NotAddressed"],
           "unverifiable": lab["Unverifiable"], "hallucination_rate": round((lab["Contradicted"] + lab["NotAddressed"]) / float(n), 4),
           "contradiction_rate": round(lab["Contradicted"] / float(n), 4), "citation_error_rate": round(cite_err / float(n), 4),
           "n_must": len(coverage), "missing": len(miss), "partial": len(part), "omission_rate": round(len(miss) / float(must), 4),
           "weighted_omission": round(sum(c["salience"] for c in miss) / float(sal_all), 4), "critical_missing": sum(1 for c in miss if c["salience"] >= 3)}
    if checklist is not None:
        req = [c for c in checklist if c["required"]]
        out["checklist_required"] = len(req)
        out["checklist_from_data"] = sum(1 for c in req if c["status"] == "data")
        out["checklist_gap"] = sum(1 for c in req if c["status"] == "gap")
        out["checklist_required_missing_in_note"] = sum(1 for c in req if c.get("in_note") == "missing")
    if alerts is not None:
        out["alerts_high"] = sum(1 for a in alerts if a["severity"] == "high")
        out["alerts_total"] = len(alerts)
    return out


# =============================================================================
# 7. B3 — 합성 오류 주입으로 검증기 자체 평가
# =============================================================================

SWAP_MEDS = ["Dopamine", "Vancomycin", "Meropenem", "Heparin", "Midazolam", "Amiodarone", "Dobutamine", "Ceftriaxone"]
FABRICATIONS = ["기관절개술 시행함.", "ECMO 적용 중임.", "혈액배양에서 MRSA 동정됨.", "농축적혈구 2단위 수혈함.", "심폐소생술 시행 이력 있음."]


PARENT_FABRICATIONS = ["아기는 오늘 수혈을 받았습니다.", "아기는 오늘 ECMO 치료를 시작했습니다.", "혈액배양에서 MRSA가 자랐습니다.",
                       "아기는 오늘 기관절개술을 받았습니다.", "심폐소생술을 받은 적이 있습니다."]


def perturb(sents, facts, rng, kind, doc="summary", must_fids=None):
    """반환: 새 문장 리스트, 정답 {'kind', 'sid' 또는 'fact_id'}"""
    S = [dict(x) for x in sents]
    skip = (CDSS_SEC(),) if doc == "summary" else (PCDSS_SEC(), PROFILE["documents"]["parent"].get("closing_section", "P8"))
    body = [i for i, x in enumerate(S) if x["section"] not in skip and x["cites"] and not x.get("fixed")]
    if not body:
        return None
    F = {f.fid: f for f in facts}
    if kind == "num_change":
        cand = [i for i in body if extract_nums(S[i]["text"])]
        if not cand:
            return None
        i = rng.choice(cand)
        t = S[i]["text"]
        stripped = strip_for_nums(t)
        ms = list(NUM_RX.finditer(stripped))
        ms = [m for m in ms if float(m.group(0)) != 0]
        if not ms:
            return None
        m = rng.choice(ms)
        old = m.group(0)
        v = float(old)
        new = fmt_num(v * rng.choice([1.6, 2.0, 2.5, 0.4]) if abs(v) > 1 else v + rng.choice([1.5, 2.0, 3.0]))
        # 원문에서 같은 위치 근사: 첫 번째 일치 토큰 치환(시간/인용 제외)
        pos = [mm for mm in re.finditer(r"(?<![A-Za-z0-9_.:\-\[])%s(?![A-Za-z0-9_])" % re.escape(old), strip_keep(t))]
        if not pos:
            return None
        p = pos[0]
        S[i]["text"] = t[:p.start()] + new + t[p.end():]
        return S, {"kind": kind, "sid": S[i]["sid"]}
    if kind == "drug_swap":
        cand = [i for i in body if any(c.startswith("med:") for f in [F.get(x) for x in S[i]["cites"]] if f for c in f.concepts)]
        if not cand:
            return None
        i = rng.choice(cand)
        f = F[S[i]["cites"][0]]
        g = f.extra.get("generic")
        if not g or g not in S[i]["text"]:
            return None
        rec = " ".join(x.text for x in facts).lower()
        choices = [m for m in SWAP_MEDS if m.lower() not in rec]
        if not choices:
            return None
        S[i]["text"] = S[i]["text"].replace(g, rng.choice(choices))
        return S, {"kind": kind, "sid": S[i]["sid"]}
    if kind == "time_shift":
        cand = [i for i in body if extract_times(S[i]["text"])]
        if not cand:
            return None
        i = rng.choice(cand)
        m = TIME_RX.search(S[i]["text"])
        hh = (int(m.group(4)) + rng.choice([3, 5, 7])) % 24
        new = "%s-%s %02d:%s" % (m.group(2), m.group(3), hh, m.group(5))
        S[i]["text"] = S[i]["text"][:m.start()] + new + S[i]["text"][m.end():]
        return S, {"kind": kind, "sid": S[i]["sid"]}
    if kind == "status_flip":
        cand = [i for i in body if re.search(r"투여 중|중단|사용하고 있|계속 받고 있", S[i]["text"]) and any(F[c].status in ("active", "stopped") for c in S[i]["cites"] if c in F)]
        if not cand:
            return None
        i = rng.choice(cand)
        t = S[i]["text"]
        if re.search(r"사용하고 있습니다|계속 받고 있습니다", t):
            t = re.sub(r"(오늘부터 )?사용하고 있습니다|주사로 계속 받고 있습니다", "오늘 중단했습니다", t, count=1)
        elif "중단했습니다" in t:
            t = re.sub(r"(은|는) 오늘 중단했습니다|주사는 오늘 중단했습니다", "를 계속 사용하고 있습니다", t, count=1)
        elif "투여 중" in t:
            t = re.sub(r"전동 직전까지 투여 중|투여 중", "중단", t, count=1)
        else:
            t = re.sub(r"[^,]*마지막 투여 후 중단|사용 후 중단|중단", "투여 중", t, count=1)
        S[i]["text"] = t
        return S, {"kind": kind, "sid": S[i]["sid"]}
    if kind == "fabricate":
        rec_ev = set(c for f in facts for c in f.concepts)
        pool = FABRICATIONS if doc == "summary" else PARENT_FABRICATIONS
        fab = [x for x in pool if not any(re.search(rx, x, re.I) and ev in rec_ev for ev, rx in EVENT_VOCAB)]
        if not fab:
            return None
        i = rng.choice(body)
        new = {"section": S[i]["section"], "text": "%s [%s]" % (rng.choice(fab), S[i]["cites"][0]), "cites": [S[i]["cites"][0]], "sid": "SFAB"}
        S.insert(i + 1, new)
        return S, {"kind": kind, "sid": "SFAB"}
    if kind == "delete":
        must = [f for f in facts if (f.fid in must_fids if must_fids is not None else (f.must and f.ftype != "parent_gap")) and any(f.fid in x["cites"] for x in S)]
        if not must:
            return None
        f = rng.choice(must)
        S = [x for x in S if f.fid not in x["cites"]]
        return S, {"kind": kind, "fact_id": f.fid}
    return None


# =============================================================================
# 8. 실행·감사 로그
# =============================================================================


class Audit(object):
    def __init__(self, path):
        self.path = path
        self.f = open(path, "a", encoding="utf-8") if path else None

    def log(self, step, inp, out, secs=0.0, model=None):
        if self.f:
            self.f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "step": step, "input_hash": sha(inp), "input": inp, "output": out,
                                     "secs": round(secs, 3), "model": model, "prompt_version": PROMPT_VERSION, "code_version": VERSION,
                                     "profile": PROFILE.get("profile_name") if PROFILE else None}, ensure_ascii=False, default=str) + "\n")

    def close(self):
        if self.f:
            self.f.close()


DOCS = ("summary", "parent")


class Pipeline(object):
    def __init__(self, data_dir, dur_csv=None, verbose=True, profile=None, cohort="nicu"):
        apply_profile(load_profile(profile))
        self.verbose = verbose
        if verbose:
            print("[profile] %s (요약지 체크리스트 %d항목, 부모 설명지 %d항목)" % (PROFILE["profile_name"], len(PROFILE["documents"]["summary"]["checklist"]),
                                                                   len(PROFILE["documents"]["parent"]["checklist"])))
        self.kd = KData(data_dir, verbose=verbose)
        self.item_map = map_items(self.kd.get("D_ITEMS"))
        self.lab_map = map_labs(self.kd.get("D_LABITEMS"))
        self.d_test = self.kd.get("D_TESTITEMS")
        self.cohort = cohort
        self.cands = nicu_stays(self.kd, cohort)
        if self.cands.empty:
            raise SystemExit("NICU 코호트가 0건입니다. ICUSTAYS 병동명(first_careunit)과 PATIENTS 나이(anchor_age)를 확인하고 프로파일 cohort.careunit_regex / "
                             "max_age_years 를 수정하세요. 코드 점검만 할 때는 --cohort all.")
        self.dur = None
        if dur_csv:
            d = pd.read_csv(dur_csv, dtype=str)
            self.dur = list(zip(d.iloc[:, 0].str.lower(), d.iloc[:, 1].str.lower(), d.iloc[:, 2] if d.shape[1] > 2 else ["병용금기"] * len(d)))
        if verbose:
            um = sum(1 for v in self.item_map.values() if v[0] is None)
            ul = sum(1 for v in self.lab_map.values() if v[0] is None)
            print("[map] D_ITEMS %d개 중 개념 미매핑 %d, D_LABITEMS %d개 중 미매핑 %d" % (len(self.item_map), um, len(self.lab_map), ul))
            print("[cohort] %s: stay %d건 (신생아 판정 %d건)" % (cohort, len(self.cands), int(self.cands["_neo"].sum())))

    def preload(self, stays):
        h = self.cands[self.cands["stay_id"].isin([str(s) for s in stays])]["hadm_id"].tolist()
        for t in ["CHARTEVENTS", "LABEVENTS", "INPUTEVENTS", "OUTPUTEVENTS", "PROCEDUREEVENTS", "EMAR", "EMAR_DETAIL", "PRESCRIPTIONS",
                  "MICROBIOLOGYEVENTS", "TESTEVENTS", "DATETIMEEVENTS"]:
            if self.kd.has(t):
                df = self.kd.get(t, h)
                for hid, g in (df.groupby("hadm_id") if (not df.empty and "hadm_id" in df.columns) else []):
                    self.kd.cache[(t, (hid,))] = g
                for hid in h:
                    self.kd.cache.setdefault((t, (hid,)), df.iloc[0:0] if not df.empty else df)

    def row(self, stay_id):
        r = self.cands[self.cands["stay_id"] == str(stay_id)]
        if r.empty:
            raise SystemExit("stay_id %s 가 코호트에 없습니다(--cohort all 로 전체 ICU 확인 가능)." % stay_id)
        return r.iloc[0]

    def days(self, stay_id):
        return day_t0s(self.row(stay_id))

    def context(self, stay_id, day=None):
        row = self.row(stay_id)
        t0s = day_t0s(row)
        if day in (None, "last"):
            idx = len(t0s)
        elif day == "first":
            idx = 1
        else:
            idx = int(day)
            if not 1 <= idx <= len(t0s):
                raise SystemExit("stay %s 는 %d일 요약까지 있습니다(--day 1~%d)." % (stay_id, len(t0s), len(t0s)))
        ctx = StayContext(self.kd, row, self.item_map, self.lab_map, self.d_test, t0=t0s[idx - 1], win_hours=float(PROFILE["daily"]["window_hours"]))
        ctx.day_index, ctx.n_days = idx, len(t0s)
        ctx.is_neonate = bool(row.get("_neo", False))
        return ctx

    def render(self, ctx, facts, alerts, doc, mode, llm, audit):
        if mode == "template":
            return (render_template(ctx, facts, alerts) if doc == "summary" else render_parent(ctx, facts, alerts)), {}
        if mode in ("grounded", "repair"):
            return render_llm(ctx, facts, alerts, llm, audit, repair=(mode == "repair"), doc=doc)
        if mode == "naive":
            return render_naive(ctx, llm, audit, doc=doc), {}
        raise ValueError(mode)

    def process(self, stay_id, day=None, out_dir=None, mode="template", llm=None, write=True, docs=DOCS):
        t1 = time.time()
        ctx = self.context(stay_id, day)
        sd = os.path.join(out_dir, "stay_%s_d%d" % (stay_id, ctx.day_index)) if out_dir else None
        if sd and write:
            os.makedirs(sd, exist_ok=True)
        audit = Audit(os.path.join(sd, "audit.jsonl") if (sd and write) else None)
        facts = build_facts(ctx)
        audit.log("build_facts", {"stay_id": stay_id, "t0": str(ctx.t0), "day": ctx.day_index}, {"n_facts": len(facts), "n_must": sum(f.must for f in facts)}, time.time() - t1)
        alerts = run_cdss(ctx, facts, self.dur)
        audit.log("cdss", {"facts": [f.fid for f in facts]}, {"alerts": [a["rule_id"] for a in alerts]})
        judge = llm if (llm is not None and llm.ok) else None
        res = {"ctx": ctx, "facts": facts, "alerts": alerts}
        allm = {"stay_id": stay_id, "day": ctx.day_index, "mode": mode, "n_facts": len(facts), "neonate": ctx.is_neonate}
        for doc in docs:
            sents, stats = self.render(ctx, facts, alerts, doc, mode, llm, audit)
            V = Verifier(ctx, facts, judge=judge, alerts=alerts, doc=doc)
            vd = V.verify(sents)
            cov = V.coverage(sents, vd, only=required_fids(ctx, doc))
            chk = checklist_in_note(V, sents, vd, checklist_report(ctx, doc), doc=doc)
            m = metrics(vd, cov, alerts if doc == "summary" else None, chk)
            if doc == "parent":
                m.update(readability(sents))
            m.update({"repair_" + k: v for k, v in stats.items()})
            audit.log("verify_" + doc, {"n_sent": len(sents)}, m)
            res[doc] = {"sents": sents, "verdicts": vd, "coverage": cov, "checklist": chk, "metrics": m}
            pre = "" if doc == "summary" else "parent_"
            allm.update({pre + k: v for k, v in m.items()})
            if sd and write:
                if doc == "summary":
                    with open(os.path.join(sd, "summary_%s.md" % mode), "w", encoding="utf-8") as f:
                        f.write(note_markdown(ctx, sents, vd, cov, checklist=chk, doc="summary"))
                else:
                    with open(os.path.join(sd, "parent_%s.md" % mode), "w", encoding="utf-8") as f:
                        f.write(note_markdown(ctx, sents, doc="parent", show_cites=False))
                    with open(os.path.join(sd, "parent_%s_verify.md" % mode), "w", encoding="utf-8") as f:
                        f.write(note_markdown(ctx, sents, vd, cov, checklist=chk, doc="parent", extra_metrics=m))
                with open(os.path.join(sd, "verify_%s_%s.json" % (doc, mode)), "w", encoding="utf-8") as f:
                    json.dump({"metrics": m, "verdicts": vd, "coverage": cov, "checklist": chk, "sentences": sents}, f, ensure_ascii=False, indent=1, default=str)
        allm["secs"] = round(time.time() - t1, 2)
        if sd and write:
            with open(os.path.join(sd, "factsheet.json"), "w", encoding="utf-8") as f:
                json.dump({"stay_id": stay_id, "hadm_id": ctx.hadm_id, "t0": str(ctx.t0), "window_start": str(ctx.win_start), "day": ctx.day_index,
                           "facts": [x.to_dict() for x in facts]}, f, ensure_ascii=False, indent=1, default=str)
            with open(os.path.join(sd, "alerts.json"), "w", encoding="utf-8") as f:
                json.dump(alerts, f, ensure_ascii=False, indent=1, default=str)
        audit.close()
        res["metrics"] = allm
        return res


def pick_stay_days(pipe, stay, day, n, seed):
    """(stay_id, day) 목록. stay 지정 시 그 stay, 아니면 무작위 stay n개와 각 stay의 무작위 하루."""
    rnd = random.Random(seed)
    if stay and stay != "auto" and day != "all":
        return [(str(stay), day)]
    if day == "all":
        ids = [str(stay)] if (stay and stay != "auto") else pipe.cands["stay_id"].tolist()
        ids = ids if n >= len(ids) else sorted(rnd.sample(ids, n))
        return [(s, k) for s in ids for k in range(1, len(day_t0s(pipe.row(s))) + 1)]
    ids = pipe.cands["stay_id"].tolist()
    if stay == "auto":
        return [(ids[0], day)]
    ids = ids if n >= len(ids) else sorted(rnd.sample(ids, n))
    out = []
    for s in ids:
        k = len(day_t0s(pipe.row(s)))
        out.append((s, day if day else rnd.randint(1, k)))
    return out


def write_summary(rows, out_dir, name):
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, name + ".csv"), index=False)
    agg = df.select_dtypes("number").mean().round(4)
    with open(os.path.join(out_dir, name + ".md"), "w", encoding="utf-8") as f:
        f.write("# %s (n=%d)\n\n| 지표 | 평균 |\n|---|---|\n" % (name, len(df)))
        for k, v in agg.items():
            f.write("| %s | %s |\n" % (k, v))
    return df, agg


def mk(a):
    return Pipeline(a.data, a.dur_csv, profile=a.profile, cohort=a.cohort)


def cmd_list(a):
    p = mk(a)
    n_days = [len(day_t0s(r)) for _, r in p.cands.iterrows()]
    print("코호트 stay 수: %d, 요약 가능한 stay-일 수: %d" % (len(p.cands), sum(n_days)))
    for col in ("first_careunit",):
        if col in p.cands.columns:
            print("병동 분포:", p.cands[col].value_counts().head(10).to_dict())
    print("앞쪽 stay_id 20개:", ", ".join(p.cands["stay_id"].head(20).tolist()))


def cmd_run(a):
    p = mk(a)
    os.makedirs(a.out, exist_ok=True)
    llm = LLM()
    if a.mode != "template" and not llm.ok:
        raise SystemExit("LLM 모드는 KCD_LLM_BASE_URL, KCD_LLM_MODEL 환경변수가 필요합니다.")
    for sid, day in pick_stay_days(p, a.stay or "auto", a.day, 1, a.seed):
        p.preload([sid])
        r = p.process(sid, day, a.out, mode=a.mode, llm=llm if llm.ok else None)
        s, q = r["summary"], r["parent"]
        print("\n" + note_markdown(r["ctx"], s["sents"], s["verdicts"], s["coverage"], checklist=s["checklist"], doc="summary"))
        print("\n" + note_markdown(r["ctx"], q["sents"], doc="parent", show_cites=False))
        print("[metrics]", json.dumps(r["metrics"], ensure_ascii=False))
        print("[out] %s" % os.path.join(a.out, "stay_%s_d%d" % (sid, r["ctx"].day_index)))


def cmd_batch(a):
    p = mk(a)
    os.makedirs(a.out, exist_ok=True)
    sd = pick_stay_days(p, None, a.day, a.n, a.seed)
    p.preload([s for s, _ in sd])
    rows, rule_counts = [], defaultdict(int)
    for i, (sid, day) in enumerate(sd, 1):
        try:
            r = p.process(sid, day, a.out, mode="template", write=not a.no_write)
            rows.append(r["metrics"])
            for al in set(x["rule_id"].split("_LAB_")[0] if "CRITICAL_LAB" in x["rule_id"] else x["rule_id"] for x in r["alerts"]):
                rule_counts[al] += 1
        except SystemExit:
            raise
        except Exception as e:
            print("[warn] stay %s day %s 실패: %r" % (sid, day, e))
        if i % 20 == 0:
            print("  %d/%d" % (i, len(sd)))
    df, agg = write_summary(rows, a.out, "summary_template")
    with open(os.path.join(a.out, "cdss_rule_counts.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rule_counts), f, ensure_ascii=False, indent=1)
    print(agg.to_string())
    print("CDSS 규칙별 발화 stay-일 수:", dict(rule_counts))


def cmd_perturb(a):
    p = mk(a)
    os.makedirs(a.out, exist_ok=True)
    sd = pick_stay_days(p, None, a.day, a.n, a.seed)
    p.preload([s for s, _ in sd])
    rng = random.Random(a.seed)
    kinds = ["num_change", "drug_swap", "time_shift", "status_flip", "fabricate", "delete"]
    docs = DOCS if a.doc == "both" else (a.doc,)
    res = defaultdict(lambda: {"n": 0, "detected": 0})
    fp = defaultdict(lambda: [0, 0])
    for sid, day in sd:
        try:
            r = p.process(sid, day, None, mode="template", write=False)
        except Exception as e:
            print("[warn] stay %s 실패: %r" % (sid, e))
            continue
        for doc in docs:
            V = Verifier(r["ctx"], r["facts"], alerts=r["alerts"], doc=doc)
            S0 = r[doc]["sents"]
            cv = V.verify(S0)
            fp[(doc, "clean")][0] += len(cv)
            fp[(doc, "clean")][1] += sum(1 for v in cv if v["label"] != "Supported")
            only = required_fids(r["ctx"], doc)
            for k in kinds:
                for _ in range(a.k):
                    out = perturb(S0, r["facts"], rng, k, doc=doc, must_fids=only)
                    if out is None:
                        continue
                    S, truth = out
                    vd = V.verify(S)
                    res[(doc, k)]["n"] += 1
                    if k == "delete":
                        st = [c for c in V.coverage(S, vd, only=only) if c["fact_id"] == truth["fact_id"]]
                        if st and st[0]["status"] in ("Missing", "Partial"):
                            res[(doc, k)]["detected"] += 1
                    else:
                        tv = [v for v in vd if v["sid"] == truth["sid"]]
                        if tv and tv[0]["label"] != "Supported":
                            res[(doc, k)]["detected"] += 1
                        others = [v for v in vd if v["sid"] != truth["sid"]]
                        fp[(doc, "others")][0] += len(others)
                        fp[(doc, "others")][1] += sum(1 for v in others if v["label"] != "Supported")
    rows = [{"doc": d, "perturbation": k, "n": v["n"], "detected": v["detected"], "recall": round(v["detected"] / float(v["n"]), 4) if v["n"] else None}
            for (d, k), v in sorted(res.items())]
    for (d, k), (n, b) in sorted(fp.items()):
        rows.append({"doc": d, "perturbation": "false_flag_" + k, "n": n, "detected": b, "recall": round(b / float(n), 4) if n else None})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "perturb_eval.csv"), index=False)
    print(df.to_string(index=False))


def cmd_arms(a):
    llm = LLM()
    if not llm.ok:
        raise SystemExit("arms 비교에는 LLM이 필요합니다(KCD_LLM_BASE_URL, KCD_LLM_MODEL).")
    p = mk(a)
    os.makedirs(a.out, exist_ok=True)
    sd = pick_stay_days(p, None, a.day, a.n, a.seed)
    p.preload([s for s, _ in sd])
    rows = []
    for sid, day in sd:
        for mode in ("naive", "grounded", "repair", "template"):
            try:
                rows.append(p.process(sid, day, a.out, mode=mode, llm=llm)["metrics"])
            except Exception as e:
                print("[warn] stay %s %s 실패: %r" % (sid, mode, e))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "arms.csv"), index=False)
    cols = [c for c in df.columns if any(c.endswith(x) for x in ("hallucination_rate", "contradiction_rate", "omission_rate", "weighted_omission",
                                                                    "checklist_required_missing_in_note", "read_avg_chars", "read_english_sent_ratio"))]
    print(df.groupby("mode")[cols].mean().round(4).to_string())


def cmd_init_profile(a):
    path = a.out if a.out.endswith(".json") else os.path.join(a.out, "hospital_profile.json")
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(load_profile(a.profile) if a.profile else DEFAULT_PROFILE, f, ensure_ascii=False, indent=2)
    print("프로파일 저장: %s (병원 양식에 맞게 수정 후 --profile %s 로 사용)" % (path, path))


def cmd_profile_check(a):
    """체크리스트 항목별로 이 병원 데이터에서 자동으로 채워지는 비율과 매칭되는 사전 라벨을 보여줌(환자 값은 출력하지 않음)."""
    p = mk(a)
    want = {"chart_regex": ("chart", "datetime"), "proc_regex": ("procedure",), "input_regex": ("input",), "output_regex": ("output",)}
    print("\n[사전 매칭] 정규식에 걸리는 D_ITEMS 라벨")
    for dn in DOCS:
        for it in PROFILE["documents"][dn]["checklist"]:
            for src in it.get("sources", []):
                if src["type"] in want:
                    hit = sorted({v[1] for v in p.item_map.values() if any(w in v[2] for w in want[src["type"]]) and re.search(src["regex"], "%s %s" % (v[1], v[3]), re.I)})
                    print("  %-7s %-16s %-13s %3d개 %s" % (dn, it["key"], src["type"], len(hit), ", ".join(hit[:5])))
    sd = pick_stay_days(p, None, a.day, a.n, a.seed)
    p.preload([s for s, _ in sd])
    stat = defaultdict(lambda: {"data": 0, "gap": 0, "absent": 0})
    for sid, day in sd:
        try:
            r = p.process(sid, day, None, write=False)
        except Exception as e:
            print("[warn] stay %s 실패: %r" % (sid, e))
            continue
        for dn in DOCS:
            for c in r["ctx"].checklist.get(dn, []):
                stat[(dn, c["item"]["key"])][c["status"]] += 1
    rows = []
    for dn in DOCS:
        for it in PROFILE["documents"][dn]["checklist"]:
            st = stat[(dn, it["key"])]
            n = sum(st.values()) or 1
            rows.append({"doc": dn, "key": it["key"], "label": it["label"], "section": it["section"], "required": bool(it.get("required")),
                         "sources": "+".join(s["type"] for s in it.get("sources", [])), "data_rate": round(st["data"] / float(n), 3)})
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "profile_check.csv"), index=False)
    print("\n[데이터 자동 채움 비율] stay-일 %d건 기준 (0이면 이 병원 데이터에 없음 → 수기 확인 항목)" % len(sd))
    print(df.to_string(index=False))


def read_note(path, doc):
    """외부 작성 문서(텍스트/마크다운)를 문장 목록으로. '## 제목'으로 섹션 추정."""
    secmap = SECTIONS if doc == "summary" else PARENT_SECTIONS
    txt = open(path, encoding="utf-8").read()
    sents, sec = [], list(secmap.keys())[0]
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    bul = re.compile(r"^([-*•]|\d+[.)])\s")
    bullets = [l for l in lines if bul.match(l)]
    for l in lines:
        if l.startswith("#"):
            h = l.lstrip("#").strip()
            for k, title in secmap.items():
                core = re.sub(r"^\d+\.\s*", "", title).split(" (")[0]
                if core and (core in h or h in title):
                    sec = k
            continue
        if bullets and not bul.match(l):
            continue
        body = re.sub(r"^([-*•]|\d+[.)])\s+", "", l)
        parts = [body] if bullets else [x.strip() for x in re.split(r"(?<=[.다])\s+", body) if x.strip()]
        for x in parts:
            sents.append({"section": sec, "text": x, "cites": cites_of(x)})
    for i, x in enumerate(sents, 1):
        x["sid"] = "S%03d" % i
    return sents


def cmd_evaluate_note(a):
    """사람이 쓴(또는 다른 시스템이 만든) 요약지·부모 설명지를 같은 Fact Sheet·체크리스트로 평가."""
    if not a.note or not a.stay:
        raise SystemExit("--note 파일과 --stay 가 필요합니다.")
    doc = "summary" if a.doc == "both" else a.doc
    p = mk(a)
    p.preload([a.stay])
    r = p.process(a.stay, a.day, None, write=False)
    llm = LLM()
    V = Verifier(r["ctx"], r["facts"], judge=llm if llm.ok else None, alerts=r["alerts"], doc=doc)
    sents = read_note(a.note, doc)
    vd = V.verify(sents)
    cov = V.coverage(sents, vd, only=required_fids(r["ctx"], doc))
    chk = checklist_in_note(V, sents, vd, checklist_report(r["ctx"], doc), doc=doc)
    m = metrics(vd, cov, None, chk)
    if doc == "parent":
        m.update(readability(sents))
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, "eval_%s_d%d_%s_%s" % (a.stay, r["ctx"].day_index, doc, os.path.splitext(os.path.basename(a.note))[0]))
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"metrics": m, "verdicts": vd, "coverage": cov, "checklist": chk, "sentences": sents}, f, ensure_ascii=False, indent=1, default=str)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(note_markdown(r["ctx"], sents, vd, cov, title="외부 문서 평가: %s" % os.path.basename(a.note), checklist=chk, doc=doc,
                              extra_metrics=m if doc == "parent" else None))
    print(json.dumps(m, ensure_ascii=False))
    miss = [c["label"] for c in chk if c["required"] and c["in_note"] == "missing"]
    print("필수 항목 중 문서에 없는 것(%d): %s" % (len(miss), ", ".join(miss) or "-"))
    print("[out] %s.md / .json" % base)


def kcd_main(argv=None):
    if isinstance(argv, str):
        argv = argv.split()
    ap = argparse.ArgumentParser(description="K-MIMIC NICU 1일 요약지·부모님 설명지 생성·검증·CDSS")
    sp = ap.add_subparsers(dest="cmd")
    for name in ("list", "run", "batch", "perturb", "arms", "init-profile", "profile-check", "evaluate-note"):
        s = sp.add_parser(name)
        s.add_argument("--data", required=(name != "init-profile"))
        s.add_argument("--profile", default=None, help="병원 프로파일 JSON(없으면 기본 양식)")
        s.add_argument("--cohort", default="nicu", choices=["nicu", "all"], help="all = 전체 ICU(코드 점검용)")
        s.add_argument("--out", default="out")
        s.add_argument("--stay", default=None)
        s.add_argument("--day", default=None, help="요약 일차(1부터) 또는 all. 생략 시 run 은 마지막 날, batch·perturb 는 stay당 무작위 하루")
        s.add_argument("--n", type=int, default=30)
        s.add_argument("--k", type=int, default=2, help="perturb: stay-일당 유형별 주입 횟수")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--mode", default="template", choices=["template", "grounded", "repair", "naive"])
        s.add_argument("--doc", default="both", choices=["both", "summary", "parent"])
        s.add_argument("--note", default=None, help="evaluate-note: 평가할 문서 파일")
        s.add_argument("--dur-csv", default=None, help="병용금기 CSV(성분A,성분B,사유)")
        s.add_argument("--no-write", action="store_true")
        s.add_argument("--time-unit", default=None, help="시간 컬럼이 숫자(상대시간)일 때 단위 D/h/min")
    a = ap.parse_args(argv)
    if not a.cmd:
        ap.print_help()
        return
    if a.time_unit:
        global TIME_UNIT
        TIME_UNIT = a.time_unit
    {"list": cmd_list, "run": cmd_run, "batch": cmd_batch, "perturb": cmd_perturb, "arms": cmd_arms, "init-profile": cmd_init_profile,
     "profile-check": cmd_profile_check, "evaluate-note": cmd_evaluate_note}[a.cmd](a)


if __name__ == "__main__" and not in_notebook_paste():
    kcd_main()
