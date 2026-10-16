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
  python kcd_nicu.py train-risk    --data "DIR/*" --out risk_models   # 트리 모델 2종 학습(악화 조기경보, 검사치 임계치)
  python kcd_nicu.py sample        --data "DIR/*" --n 30 --seed 0 --out sample [--stratify]   # 생후 28일 이하 평가 시점 무작위 추출(여러 폴더 가능)

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
            # NICU 간호 daily 인계 필수 항목(nicu_handoff_required_items_v2.md) 17개 영역 기준.
            # area = 영역 번호, required = Core, required_if = 해당 환아에게만 필수(preterm / invasive_vent / tube_feeding),
            # 조건부 영역(14~17)은 required=False: 기록이 있을 때만 포함. after_cdss = CDSS 이후 판정(중증도).
            "checklist": [
                _ck("a01_id", "환자 식별자", "D1", True, [_f("patient_info")], area=1),
                _ck("a01_ga", "재태주령(GA)", "D1", True, [{"type": "chart_regex", "regex": "재태|gestational|\\bGA\\b|주수"}, {"type": "manual"}],
                    area=1, mention_regex="재태|주수|\\bGA\\b"),
                _ck("a01_bw", "출생체중", "D1", True, [{"type": "chart_regex", "regex": "출생 ?체중|birth ?weight"}, {"type": "manual"}],
                    area=1, mention_regex="출생 ?체중|birth ?weight"),
                _ck("a01_dol", "생후일", "D1", True, [_f("patient_info")], area=1, mention_regex="생후|입원 \\d+일째"),
                _ck("a01_ca", "교정연령", "D1", True, [{"type": "chart_regex", "regex": "교정|corrected|\\bPMA\\b|\\bCGA\\b"}, {"type": "manual"}],
                    area=1, mention_regex="교정|PMA|CGA"),
                _ck("a01_wt", "현재 체중과 측정일", "D1", True, [_f("weight")], area=1),
                _ck("a01_multi", "다태아 순번", "D1", True, [{"type": "chart_regex", "regex": "다태|쌍둥이|쌍태|twin|multiple|birth order"}, {"type": "manual"}],
                    area=1, mention_regex="다태|쌍둥이|쌍태|twin"),
                _ck("a02_sev", "중증도(안정·관찰 필요·불안정)", "D2", True, [_f("severity")], area=2, after_cdss=True, mention_regex="중증도|불안정|관찰 필요|안정"),
                _ck("a02_prob", "활성 문제 목록", "D2", True, [_f("diagnosis_main", "diagnosis_other", "prematurity")], area=2),
                _ck("a03_support", "호흡 지지 방식·설정", "D3", True, [_f("resp_support", "resp_change")], area=3),
                _ck("a03_fio2", "FiO2", "D3", True, [{"type": "facts", "concepts": ["fio2"]}], area=3, gap_text="FiO2: 최근 기록 없음, 확인 필요.",
                    mention_regex="FiO2"),
                _ck("a03_target", "SpO2 목표 범위", "D3", True, [{"type": "chart_regex", "regex": "목표|target"}, {"type": "manual"}], area=3,
                    mention_regex="목표"),
                _ck("a03_gas", "최근 혈액가스", "D3", True, [_f("blood_gas")], area=3, gap_text="최근 혈액가스: 최근 72시간 기록 없음, 확인 필요.",
                    mention_regex="혈액가스|ABG|VBG|CBG|pCO2"),
                _ck("a04_ett", "ETT 크기·고정 깊이", "D3", True,
                    [{"type": "chart_regex", "regex": "ETT|E-tube|기관내관|endotracheal|tube size|고정 ?깊이|fixation|depth"}, {"type": "manual"}],
                    area=4, required_if="invasive_vent", mention_regex="ETT|기관내관|고정 ?깊이"),
                _ck("a04_ogng", "OG·NG 위치", "D4", True, [{"type": "chart_regex", "regex": "\\bOG\\b|\\bNG\\b|위관|orogastric|nasogastric|L-tube|Levin"}, {"type": "manual"}],
                    area=4, required_if="tube_feeding", mention_regex="OG|NG|위관 위치"),
                _ck("a05_events", "무호흡·서맥·탈포화 사건 횟수", "D3", True, [_f("neo_events")], area=5, required_if="preterm",
                    mention_regex="무호흡|서맥|탈포화|SpO2 \\d+% 미만"),
                _ck("a05_interv", "사건 중재 강도(자가 회복·자극·산소)", "D3", True,
                    [{"type": "chart_regex", "regex": "자극|stimulat|중재|intervention|자가 ?회복|bagging|BVM"}, {"type": "manual"}],
                    area=5, required_if="preterm", mention_regex="중재|자극|자가 ?회복"),
                _ck("a05_caffeine", "카페인 투여 여부", "D7", True, [{"type": "med_class", "classes": ["respiratory_stimulant"]}], area=5, required_if="preterm",
                    gap_text="카페인: 기준 시각 투여 기록 없음.", mention_regex="카페인|caffeine"),
                _ck("a06_hrbp", "HR·BP 범위", "D3", True, [{"type": "facts", "fact_types": ["vitals_24h"], "concepts": ["hr"]}], area=6),
                _ck("a06_perf", "관류(CRT 등)", "D3", True, [{"type": "chart_regex", "regex": "CRT|capillary|관류|모세혈관|perfusion"}, {"type": "manual"}],
                    area=6, mention_regex="관류|CRT|모세혈관"),
                _ck("a06_vaso", "혈관작용제(약물·용량·속도)", "D7", True, [{"type": "med_class", "classes": ["vasopressor"]}], area=6,
                    gap_text="혈관작용제: 기준 시각 투여 기록 없음.", mention_regex="혈관작용제|승압|강심|dopamine|dobutamine|epinephrine|milrinone"),
                _ck("a07_temp", "체온 범위", "D3", True, [{"type": "facts", "fact_types": ["vitals_24h"], "concepts": ["temp"]}], area=7),
                _ck("a07_env", "인큐베이터·보온기 모드, 습도", "D3", True,
                    [{"type": "chart_regex", "regex": "인큐베이터|incubator|보온기|warmer|습도|humid|servo|air mode|isolette"}, {"type": "manual"}],
                    area=7, mention_regex="인큐베이터|보온기|습도|incubator|warmer"),
                _ck("a08_line", "라인 종류·삽입일", "D3", True,
                    [{"type": "proc_regex", "regex": LINE_RX, "note": "제거 여부는 데이터로 확인되지 않음"},
                     {"type": "chart_regex", "regex": "insertion|삽입|catheter|카테터|UVC|UAC|PICC|PIV|말초정맥"}],
                    area=8, mention_regex="UVC|UAC|PICC|PIV|제대|중심정맥|카테터|catheter|정맥관"),
                _ck("a08_depth", "라인 삽입 깊이·부위 상태", "D3", True, [{"type": "chart_regex", "regex": "삽입 ?깊이|insertion depth|부위 ?상태|site"}, {"type": "manual"}],
                    area=8, mention_regex="삽입 ?깊이|부위"),
                _ck("a08_infusion", "주입 내용·속도", "D4", True,
                    [_f("infusion_episode", "fluids"), {"type": "med_class", "classes": ["nutrition"]}, _f("intake_24h")], area=8),
                _ck("a09_route", "수유 방식(NPO·위관·경구)", "D4", True, [_f("intake_24h")], area=9, mention_regex="수유|경관|경구|NPO|금식|위관"),
                _ck("a09_type", "수유 종류(모유·분유 등)", "D4", True,
                    [{"type": "input_regex", "regex": "모유|분유|breast|formula|EBM|HMF|강화제"}, {"type": "chart_regex", "regex": "모유|분유|breast|formula|EBM"},
                     {"type": "manual"}], area=9, mention_regex="모유|분유|EBM|formula"),
                _ck("a09_amount", "수유 양·간격", "D4", True, [_f("intake_24h")], area=9),
                _ck("a09_plan", "수유 증량 계획", "D4", True, [{"type": "chart_regex", "regex": "증량|advance"}, {"type": "manual"}], area=9, mention_regex="증량"),
                _ck("a09_intol", "수유 불내성(잔류량·복부팽만·구토·혈변)", "D4", True,
                    [{"type": "chart_regex", "regex": "잔류|residual|복부 ?팽만|abdominal girth|distension|구토|vomit|혈변|bloody"},
                     {"type": "output_regex", "regex": "잔류|residual|구토|vomit|Levin tube-output|OG.*output"}, {"type": "manual"}],
                    area=9, mention_regex="잔류|팽만|구토|혈변|불내성"),
                _ck("a10_urine", "소변량(mL/kg/h)", "D5", True, [_f("urine_24h")], area=10),
                _ck("a10_stool", "배변", "D5", True, [_f("stool")], area=10, gap_text="배변: 지난 24시간 기록 없음, 확인 필요.", mention_regex="배변|대변|stool"),
                _ck("a10_glucose", "혈당", "D6", True, [{"type": "facts", "concepts": ["glucose", "bst"]}], area=10,
                    gap_text="혈당: 최근 24시간 검사 기록 없음, 확인 필요.", mention_regex="혈당|glucose|BST"),
                _ck("a10_wtchange", "체중 변화", "D1", True, [_f("weight")], area=10),
                _ck("a10_bili", "빌리루빈", "D6", True, [{"type": "facts", "concepts": ["tbil"]}], area=10,
                    gap_text="빌리루빈: 최근 24시간 검사 기록 없음, 확인 필요.", mention_regex="빌리루빈|bilirubin|T\\.bil|황달"),
                _ck("a10_photo", "광선요법", "D6", True, [{"type": "proc_regex", "regex": "photo|광선"}, {"type": "chart_regex", "regex": "광선|photo"}, {"type": "manual"}],
                    area=10, mention_regex="광선|photo"),
                _ck("a11_meds", "투약과 마지막 투여 시각", "D7", True, [_f("med_active", "fluids", "infusion_episode", "med_stopped")], area=11,
                    gap_text="투약: 기준 시각 투여 기록 없음."),
                _ck("a11_sched", "정규 투약 시각", "D7", True, [{"type": "manual"}], area=11, mention_regex="정규 투약|투약 시각"),
                _ck("a11_status", "투여 상태(투여·미투여·보류)와 사유", "D7", True, [_f("med_status"), {"type": "manual"}], area=11,
                    mention_regex="미투여|보류|hold|not given"),
                _ck("a11_high", "고위험약", "D7", True, [{"type": "med_class", "classes": ["vasopressor", "sedation", "insulin", "anticoagulant"]}], area=11,
                    gap_text="고위험약: 기준 시각 투여 기록 없음.", mention_regex="고위험|승압|진정|인슐린|항응고"),
                _ck("a12_neuro", "활동성·경련", "D3", True, [{"type": "chart_regex", "regex": "활동|activity|경련|seizure|tone|긴장도|irritab"}, {"type": "manual"}],
                    area=12, mention_regex="활동성|경련|seizure"),
                _ck("a12_pain", "통증·진정 점수", "D3", True, [{"type": "chart_regex", "regex": "통증|pain|N-PASS|NIPS|PIPP|진정|sedation|COMFORT"}, {"type": "manual"}],
                    area=12, mention_regex="통증|진정 점수|N-PASS|NIPS|PIPP"),
                _ck("a13_pending", "결과 대기 검사", "D3", True, [_f("micro_pending")], area=13, gap_text="결과 대기 검사: 데이터상 미보고 배양 없음, 그 외 대기 검사 확인 필요.",
                    mention_regex="미보고|결과 대기|pending"),
                _ck("a13_planned", "예정 검사·시술, 시각이 정해진 할 일", "D3", True, [{"type": "manual"}], area=13, mention_regex="예정"),
                _ck("a14_plan", "비상 계획", "D8", False, [{"type": "chart_regex", "regex": "비상|contingency"}], area=14),
                _ck("a15_family", "가족(면회·설명·캥거루케어·모유 공급·교육)", "D1", False,
                    [{"type": "chart_regex", "regex": "보호자|면회|가족|캥거루|kangaroo|모유 ?공급|유축|교육"}], area=15),
                _ck("a16_skin", "피부(기구 압박손상·테이프 손상·침윤)", "D3", False,
                    [{"type": "chart_regex", "regex": "압박|pressure injury|테이프|tape|침윤|infiltrat|피부|skin|욕창"}], area=16),
                _ck("a17_iso", "감염·격리", "D1", False,
                    [{"type": "micro_org_regex", "regex": "MRSA|VRE|CRE|CRAB|CRPA|carbapenem|methicillin.resistant|vancomycin.resistant"},
                     {"type": "chart_regex", "regex": "격리|isolation"}], area=17),
            ],
        },
        "parent": {
            "title": "오늘 아이의 상태와 돌봄",
            "subtitle": "보호자 Daily Briefing · 간호 인계 기록을 바탕으로 쉽게 정리한 자료",
            "notice": "간호사가 내용을 확인한 뒤 설명과 함께 전달합니다. 보호자에게 자동으로 보내지 않습니다.",
            "sections": [{"id": "B1", "title": "오늘 꼭 알아둘 변화"}, {"id": "B2", "title": "현재 받고 있는 치료와 돌봄"},
                         {"id": "B3", "title": "아직 확인이 필요한 중요한 내용"}, {"id": "B4", "title": "기록된 다음 계획"},
                         {"id": "B5", "title": "보호자가 질문할 내용"}],
            "cdss_section": "B3",
            "closing_section": "B5",
            "section_of": {"resp_change": "B1", "neo_events": "B1", "procedure": "B1", "imaging": "B1", "micro_positive": "B1", "crrt_episode": "B1",
                           "lab_latest": "B1", "resp_support": "B2", "weight": "B2", "feeding_24h": "B2", "intake_24h": "B2", "med_active": "B2",
                           "infusion_episode": "B2", "fluids": "B2", "micro_pending": "B3"},
            "show_lab_values": False,
            "parent_visible_rules": ["NEO_HYPOGLYCEMIA", "NEO_HYPERGLYCEMIA", "NEO_BILI_HIGH", "NEO_BILI_RISE", "NEO_TEMP_INSTABILITY",
                                     "NEO_WEIGHT_LOSS", "NEO_SEPSIS_MARKERS", "NEO_OLIGURIA"],
            "alert_text": {
                "NEO_HYPOGLYCEMIA": "혈당이 낮게 나온 적이 있어서 간호사와 의사가 혈당을 자주 재고 있습니다.",
                "NEO_HYPERGLYCEMIA": "혈당이 높게 나온 적이 있어서 간호사와 의사가 혈당을 살펴보고 있습니다.",
                "NEO_BILI_HIGH": "황달 수치가 높게 나와서 치료가 필요한지 의사가 살펴보고 있습니다.",
                "NEO_BILI_RISE": "황달 수치가 빠르게 올라서 의사가 자주 확인하고 있습니다.",
                "NEO_TEMP_INSTABILITY": "체온이 정해진 범위를 벗어난 적이 있어서 체온을 자주 재고 있습니다.",
                "NEO_WEIGHT_LOSS": "입원했을 때보다 몸무게가 많이 줄어서 먹는 양을 함께 살펴보고 있습니다.",
                "NEO_SEPSIS_MARKERS": "감염이 있는지 알아보는 검사 결과를 살펴보고 있습니다.",
                "NEO_OLIGURIA": "소변 양이 적어서 몸에 들어가는 물의 양을 살펴보고 있습니다.",
            },
            "gap_text": "오늘 기록에서는 {label} 내용이 확인되지 않았습니다.",
            "questions": ["오늘 설명에서 더 알고 싶은 점이 있으신가요?",
                          "아이에게 지금 가장 중요한 치료가 무엇인지 궁금하시면 물어봐 주세요.",
                          "다음에 무엇을 확인하는지, 언제 다시 설명을 들을 수 있는지도 물어보실 수 있습니다."],
            "doctor_flag": {"dx_regex": "^P36|^A4[01]|^P77|^P52|^P91|^P29|^Q2[0-5]|^C[0-9]", "alert_severity": ["high"], "severity_levels": ["불안정"],
                            "fact_types": ["micro_positive"]},
            "checklist": [
                _ck("p_resp", "숨 쉬기", "B2", True, [_f("resp_support", "resp_change")], mention_regex="숨|호흡|인공호흡기|산소"),
                _ck("p_events", "산소 수치 저하·느린 맥박", "B1", False, [_f("neo_events", status="present")], mention_regex="산소 수치|심장"),
                _ck("p_weight", "몸무게", "B2", True, [_f("weight")], mention_regex="몸무게|체중"),
                _ck("p_feeding", "먹는 양", "B2", True, [_f("feeding_24h", "intake_24h")], mention_regex="먹|수유|영양|수분"),
                _ck("p_tests", "중요한 검사 변화", "B1", False, [_f("lab_latest", "micro_positive", "procedure", "imaging")], mention_regex="검사"),
                _ck("p_meds", "쓰고 있는 약", "B2", True, [_f("med_active", "fluids", "infusion_episode")], mention_regex="약|주사|수액"),
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
    "severity": {                # 중증도 3단계 판정 규칙(위에서부터 먼저 만족하는 단계). 근거는 기록지에 함께 표시
        "section": "D2",
        "unstable": {"alert_severity": ["high"], "resp_kind": ["invasive", "vent_unknown"], "active_classes": ["vasopressor"]},
        "watcher": {"alert_severity": ["medium"], "resp_kind": ["noninvasive", "o2"], "fact_status": ["neo_events:present"]},
    },
    "sampling": {},              # 임상 평가용 표본 추출 기준 덮어쓰기(기본값은 SAMPLING_DEFAULT 참고). 예: {"max_age_days": 28, "min_bp": 0}
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


SOURCE_TYPES = ("facts", "med_class", "chart_regex", "lab_regex", "proc_regex", "output_regex", "input_regex", "service", "micro_org_regex", "manual")


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



# =============================================================================
# 2-c. 인계 명세 JSON (예: nicu_daily_handoff_spec.json) → 요약지 섹션·필수 요소로 변환
#   - required_information 의 각 영역 = 요약지 섹션(순서·제목 그대로)
#   - 각 requirement 문장을 요소로 나눠 데이터 출처를 연결. 데이터 없으면 "확인 필요", 조건부 요소는 기록 있을 때만
#   - 영역에 "elements": [{"label", "sources", "conditional", "mention_regex"}] 를 쓰면 자동 분해 대신 그대로 사용
#   - 영역에 "fact_types": [...] 를 쓰면 그 영역에 배치할 fact 종류를 직접 지정
#   - report_context.required 의 시각 항목은 머리말에, hard_constraints·generation_instruction 은 LLM 지시문에 반영
# =============================================================================

SPEC_DEFAULT_FILE = "nicu_daily_handoff_spec.json"

# 영역 id → 이 영역에 배치할 fact 종류(명세에 fact_types 가 있으면 그것을 우선)
SPEC_FACT_MAP = {
    "patient_basics": ["patient_info", "weight"],
    "current_status_and_changes": ["severity", "diagnosis_main", "diagnosis_other", "prematurity", "procedure_icd", "resp_support", "resp_change",
                                   "vitals_24h", "neo_events", "procedure", "crrt_episode", "aki"],
    "feeding": ["feeding_24h"],
    "intake_output": ["intake_24h", "urine_24h", "stool", "io_balance"],
    "medications": ["med_active", "med_stopped", "med_status", "infusion_episode", "fluids"],
    "test_results": ["lab_latest", "lab_change", "lab_normal", "lab_none", "bst_24h", "blood_gas", "micro_positive", "micro_negative", "imaging"],
    "tubes_and_lines": ["drain"],
    "precautions_and_next_actions": ["micro_pending"],
}

_F = lambda *t, **kw: dict({"type": "facts", "fact_types": list(t)}, **kw)
# 영역 id → [(요소 문구 정규식, 출처 목록, 조건부 여부, 언급 정규식)]. 앞에서부터 처음 맞는 규칙을 씀
SPEC_VOCAB = {
    "patient_basics": [
        (r"\bID\b|식별", [_F("patient_info")], False, None),
        (r"나이|생후일|일령", [_F("patient_info")], False, r"생후|입원 \d+일째"),
        (r"성별", [_F("patient_info")], False, r"남아|여아|성별"),
        (r"체중|측정시각", [_F("weight")], False, r"체중|몸무게"),
        (r"혈액형|ABO", [{"type": "lab_regex", "regex": "ABO|혈액형|blood ?group|Rh"}, {"type": "chart_regex", "regex": "혈액형|ABO|blood ?type"},
                        {"type": "manual"}], False, r"혈액형|ABO"),
        (r"재태|\bGA\b", [{"type": "chart_regex", "regex": "재태|gestational|\\bGA\\b|주수"}, {"type": "manual"}], False, r"재태|주수|\bGA\b"),
        (r"PMA|교정", [{"type": "chart_regex", "regex": "\\bPMA\\b|교정|corrected|\\bCGA\\b"}, {"type": "manual"}], False, r"PMA|교정"),
        (r"출생 ?체중", [{"type": "chart_regex", "regex": "출생 ?체중|birth ?weight"}, {"type": "manual"}], False, r"출생 ?체중"),
    ],
    "current_status_and_changes": [
        (r"주요 문제|문제", [_F("severity", "diagnosis_main", "diagnosis_other", "prematurity")], False, r"중증도|진단"),
        (r"호흡지원|산소", [_F("resp_support", "resp_change")], False, r"호흡 보조|산소|FiO2|CPAP|NIPPV|인공호흡"),
        (r"활력", [_F("vitals_24h", "neo_events")], False, r"활력징후|HR|SpO2"),
        (r"사건|광선|벤틸레이터|ventil|이탈", [_F("resp_change", "procedure", "crrt_episode"), {"type": "proc_regex", "regex": "photo|광선"},
                                       {"type": "chart_regex", "regex": "광선|photo|발관|extubat|삽관|intubat"}], True, r"광선|발관|삽관|모드 변경"),
        (r"중재|반응", [{"type": "chart_regex", "regex": "중재|intervention|반응|response"}], True, r"중재|반응"),
        (r"보챔|BIS|상태 변화", [{"type": "chart_regex", "regex": "보챔|irritab|fuss|\\bBIS\\b|울음|crying"}], True, r"보챔|BIS"),
    ],
    "feeding": [
        (r"종류|경로", [{"type": "input_regex", "regex": "경구|oral|Levin|\\bOG\\b|\\bNG\\b|feeding|경관|수유|모유|분유"},
                     {"type": "chart_regex", "regex": "모유|분유|breast|formula|EBM|수유 ?종류"}, {"type": "manual"}], False, r"모유|분유|경관|경구|수유"),
        (r"회당량|간격", [_F("feeding_24h"), {"type": "chart_regex", "regex": "회당|수유량|간격"}], False, r"회당|간격"),
        (r"횟수|총량", [_F("feeding_24h")], False, r"수유 기록 \d+회|총 \d+"),
        (r"보류|NPO|금식", [{"type": "chart_regex", "regex": "보류|hold|NPO|금식"}], True, r"보류|NPO|금식"),
        (r"구토|팽만|특이", [{"type": "chart_regex", "regex": "잔류|residual|복부 ?팽만|distension|girth|구토|vomit|혈변|bloody"}], True, r"구토|팽만|잔류|혈변"),
    ],
    "intake_output": [
        (r"섭취|배출", [_F("intake_24h", "io_balance")], False, r"섭취|배설"),
        (r"소변|측정방법", [_F("urine_24h")], False, r"소변"),
        (r"배변", [_F("stool")], False, r"배변|대변"),
        (r"기저귀|순증량|혼합", [{"type": "output_regex", "regex": "기저귀|diaper|대소변|소변\\+대변"}, {"type": "chart_regex", "regex": "기저귀|diaper"}], True,
         r"기저귀|대소변"),
    ],
    "medications": [
        (r"약명|용량|단위|경로|투여시각|횟수", [_F("med_active", "med_stopped", "infusion_episode")], False, None),
        (r"PRN", [{"type": "chart_regex", "regex": "\\bPRN\\b|필요 ?시"}], True, r"PRN|필요 시"),
        (r"미투여|보류", [_F("med_status")], True, r"미투여|보류"),
        (r"지속주입", [_F("infusion_episode", "fluids"), {"type": "med_class", "classes": ["nutrition", "vasopressor", "sedation"]}], True, r"지속주입|주입속도"),
    ],
    "test_results": [
        (r"이상치|추세|정상범위|복귀|수치", [_F("lab_latest", "lab_change", "lab_normal", "lab_none", "bst_24h", "blood_gas")], False, None),
    ],
    "tubes_and_lines": [
        (r"종류|위치|고정|현재 상태", [{"type": "proc_regex", "regex": LINE_RX + "|tube|튜브|gavage|ETT", "note": "제거 여부는 데이터로 확인되지 않음"},
                                   {"type": "chart_regex", "regex": "insertion|삽입|catheter|카테터|UVC|UAC|PICC|PIV|ETT|tube|튜브|고정"}], False,
         r"UVC|UAC|PICC|PIV|카테터|ETT|tube|튜브|라인"),
        (r"연결 수액|주입속도", [_F("infusion_episode", "fluids"), {"type": "med_class", "classes": ["nutrition"]}], True, r"수액|주입"),
        (r"배액|성상|삽입부", [_F("drain"), {"type": "chart_regex", "regex": "삽입부|site|배액|drain"}], True, r"배액|삽입부"),
    ],
    "precautions_and_next_actions": [
        (r"알레르기", [{"type": "chart_regex", "regex": "알레르기|알러지|allergy|과민"}, {"type": "manual"}], False, r"알레르기|알러지|allergy"),
        (r"격리", [{"type": "micro_org_regex", "regex": "MRSA|VRE|CRE|CRAB|CRPA|carbapenem|methicillin.resistant|vancomycin.resistant"},
                 {"type": "chart_regex", "regex": "격리|isolation"}, {"type": "manual"}], False, r"격리|isolation"),
        (r"결과 대기", [_F("micro_pending")], False, r"미보고|결과 대기"),
        (r"예정 검사|예정시각|계획", [{"type": "chart_regex", "regex": "예정|계획|plan|schedule"}], True, r"예정|계획"),
    ],
}
SPEC_COND_CUE = r"해당 시|있으면|했다면|여부"


SPEC_RULE_NAME = {  # 요소 문구가 길 때 표에 쓸 짧은 이름
    r"사건|광선|벤틸레이터|ventil|이탈": "중요 사건(광선치료·벤틸레이터 적용·이탈 등)",
    r"중재|반응": "시행한 중재·반응",
    r"보챔|BIS|상태 변화": "보챔 등 상태 변화(BIS)",
    r"이상치|추세|정상범위|복귀|수치": "중요 이상치·변화 추세·정상범위 복귀값(수치·단위·시각)",
}
_JOIN_RX = re.compile(r"(?<!결)(?<=[가-힣)])(?:와|과) (?=\S)| 및 ")


def _split_top(text, seps):
    """괄호 밖의 구분자로만 나눔."""
    out, buf, depth = [], "", 0
    for ch in str(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if depth == 0 and ch in seps:
            if buf.strip(" "):
                out.append(buf.strip(" "))
            buf = ""
            continue
        buf += ch
    if buf.strip(" "):
        out.append(buf.strip(" "))
    return out


def split_requirement(text):
    """requirement 문장을 요소로 분해(괄호 밖 ',' '.' 기준). 끝맺음 표현은 제거."""
    t = re.sub(r"\s*(를|을) (포함|함께 표시)합니다\.?|\s*합니다\.?$", "", str(text))
    return [x.strip(" .") for x in _split_top(t, ",.") if x.strip(" .")]


def spec_elements(item):
    """명세 영역 → 요소 목록 [{label, sources, conditional, mention_regex}].
    요소 안의 '·' 조각마다 규칙을 찾고, 규칙이 없는 조각은 같은 요소의 첫 규칙에 붙임. 같은 규칙은 하나로 합침.
    조건 표현(해당 시·있으면·했다면·여부)은 조각 단위로 보고, 규칙의 모든 조각이 조건부일 때만 선택 요소로 둠."""
    if item.get("elements"):
        return [dict(e) for e in item["elements"]]
    vocab = SPEC_VOCAB.get(item.get("id"), [r for v in SPEC_VOCAB.values() for r in v])
    groups, unmatched = OrderedDict(), []
    for el in split_requirement(item.get("requirement", "")):
        pieces = []
        for pc in (_split_top(el, "·") or [el]):
            parts = _JOIN_RX.split(pc) if "(" not in pc else re.split(r"(?<=\))(?:와|과) ", pc)
            pieces += [x.strip() for x in parts if x.strip()]
        hit_rules, used = [], set()
        for pc in pieces:  # 조각마다 처음 맞는 규칙 하나에 배정
            rule = next((r for r in vocab if re.search(r[0], pc, re.I)), None)
            if rule is None:
                continue
            if rule not in hit_rules:
                hit_rules.append(rule)
            g = groups.setdefault(rule[0], {"pieces": [], "rule": rule})
            if pc not in g["pieces"]:
                g["pieces"].append(pc)
            used.add(pc)
        if not hit_rules:
            unmatched.append(el)
            continue
        rest = [pc for pc in pieces if pc not in used]
        if rest:
            groups[hit_rules[0][0]]["pieces"] += rest
    els = []
    for g in groups.values():
        rx, sources, cond, mrx = g["rule"]
        label = "·".join(g["pieces"])
        if len(label) > 40:
            label = SPEC_RULE_NAME.get(rx, label[:40] + "…")
        all_cue = all(re.search(SPEC_COND_CUE, pc) for pc in g["pieces"])
        e = {"label": label, "sources": sources, "conditional": bool(cond or all_cue)}
        if mrx:
            e["mention_regex"] = mrx
        els.append(e)
    for el in unmatched:  # 연결 규칙이 없는 요소: 데이터 출처 없음 → 조건 표현이 없으면 "확인 필요"
        els.append({"label": el, "sources": [{"type": "manual"}], "conditional": bool(re.search(SPEC_COND_CUE, el)), "unmatched": True})
    return els


def load_spec(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def find_default_spec():
    for d in (os.getcwd(), os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()):
        p = os.path.join(d, SPEC_DEFAULT_FILE)
        if os.path.exists(p):
            return p
    return None


def spec_to_profile(p, spec, path=None):
    """인계 명세를 프로파일의 요약지(documents.summary)로 변환. 기존 요약지 체크리스트는 대체됨."""
    items = spec.get("required_information", [])
    if not items:
        raise SystemExit("명세에 required_information 이 없습니다: %s" % path)
    secs, checklist, smap, unmatched = [], [], {}, []
    for i, it in enumerate(items, 1):
        sid = it.get("id") or "sec%d" % i
        secs.append({"id": sid, "title": "%d. %s" % (i, it.get("label", sid))})
        for ft in it.get("fact_types") or SPEC_FACT_MAP.get(sid, []):
            smap[ft] = sid
        for j, e in enumerate(spec_elements(it), 1):
            ck = {"key": "%s_%d" % (sid, j), "label": e["label"], "section": sid, "required": not e.get("conditional", False),
                  "sources": e.get("sources", [{"type": "manual"}]), "area": i}
            if e.get("mention_regex"):
                ck["mention_regex"] = e["mention_regex"]
            if e.get("unmatched"):
                unmatched.append("%s: %s" % (it.get("label", sid), e["label"]))
            checklist.append(ck)
    secs.append({"id": "cdss", "title": "참고: 규칙 기반 알림(CDSS, 인계문과 별도 확인용)"})
    s = p["documents"]["summary"]
    s.update({"title": spec.get("title", s["title"]) + " (자동 생성 초안, 의료진 검토 필요)", "sections": secs, "cdss_section": "cdss",
              "section_cap": {x["id"]: 24 for x in secs}, "checklist": checklist})
    p["section_of"] = dict(smap, **p.get("section_of", {}))
    p["default_section"] = secs[1]["id"] if len(secs) > 2 else secs[0]["id"]
    p.setdefault("severity", {})["section"] = smap.get("severity", p["default_section"])
    p["_spec"] = {"path": path, "spec_id": spec.get("spec_id"), "version": spec.get("version"), "report_context": spec.get("report_context", {}),
                  "hard_constraints": spec.get("hard_constraints", []), "generation_instruction": spec.get("generation_instruction", ""),
                  "required_information": items, "unmatched": unmatched}
    return p


def spec_system_prompt():
    S = PROFILE.get("_spec") or {}
    req = "\n".join("- %s: %s" % (x.get("label"), x.get("requirement")) for x in S.get("required_information", []))
    hc = "\n".join("- %s" % x for x in S.get("hard_constraints", []))
    return ("%s\n\n[필수정보]\n%s\n\n[반드시 지킬 규칙]\n%s\n- 주어진 facts만 사용하고, 문장마다 끝에 사용한 fact_id를 [F001] 형식으로 인용한다\n"
            "- 출력은 '- '로 시작하는 문장 목록만 쓴다." % (S.get("generation_instruction", ""), req, hc))


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
    if t == "lab_regex":
        L = ctx.lab
        if L.empty:
            return []
        L = L[(L["t"] <= ctx.t0) & L["label"].astype(str).str.contains(rx, case=False, regex=True, na=False)]
        if L.empty:
            return []
        r = L.sort_values("t").iloc[-1]
        return [Fact("checklist_item", sec, "%s: %s %s(%s 검사)." % (item["label"], r["label"], r.get("value"), fmt_t(r["t"])), 2, concepts={pre + key},
                     time=r["t"], prov=_prov("LABEVENTS", L.tail(1), "labevent_id"))]
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


def _required(ctx, facts, item):
    """Core 여부. required_if 가 있으면 해당 환아에게만 필수."""
    if not item.get("required"):
        return False
    cond = item.get("required_if")
    if not cond:
        return True
    if cond == "preterm":
        return any(f.ftype == "prematurity" for f in facts)
    if cond == "invasive_vent":
        return getattr(ctx, "resp_kind", None) in ("invasive", "vent_unknown")
    if cond == "tube_feeding":
        return any(f.ftype == "intake_24h" and f.extra.get("tube") for f in facts)
    if cond == "neonate":
        return bool(ctx.is_neonate)
    return True


def apply_checklist(ctx, facts, doc, post=False):
    """post=False: CDSS 이전 판정(after_cdss 항목 제외). post=True: after_cdss 항목만 판정(중증도 등)."""
    added = []
    if not post:
        ctx.checklist[doc] = []
    D = PROFILE["documents"][doc]
    pre = "chk:" if doc == "summary" else "pchk:"
    for item in D["checklist"]:
        if bool(item.get("after_cdss")) != post:
            continue
        got, used = [], None
        for src in item.get("sources", []):
            got = eval_source(ctx, facts + added, item, src, doc)
            if got:
                used = src["type"]
                break
        req = _required(ctx, facts + added, item)
        if got:
            for f in got:
                if f not in facts and f not in added:
                    added.append(f)
                if req and doc == "summary":
                    f.salience = max(f.salience, 2)
                f.concepts.add(pre + item["key"])
            ctx.checklist[doc].append({"item": item, "facts": got, "source": used, "status": "data", "req": req})
        elif req:
            if doc == "summary":
                txt = item.get("gap_text") or "%s: 데이터에 기록 없음, 확인 필요." % item["label"]
                g = Fact("checklist_gap", item["section"], txt, 2, concepts={pre + item["key"]}, status="gap", extra={"checklist": item["key"]})
            else:
                txt = item.get("gap_text") or D.get("gap_text", "{label}: 기록에서 확인되지 않았습니다.").format(label=item["label"])
                g = Fact("parent_gap", item["section"], txt, 1, concepts={pre + item["key"]}, status="gap", extra={"checklist": item["key"]})
            added.append(g)
            ctx.checklist[doc].append({"item": item, "facts": [g], "source": "gap", "status": "gap", "req": True})
        else:
            ctx.checklist[doc].append({"item": item, "facts": [], "source": None, "status": "absent" if not item.get("required") else "n/a", "req": False})
    if post:
        order = {it["key"]: i for i, it in enumerate(D["checklist"])}
        ctx.checklist[doc].sort(key=lambda c: order.get(c["item"]["key"], 999))
    return added


def checklist_report(ctx, doc):
    rows = []
    for c in ctx.checklist.get(doc, []):
        it = c["item"]
        rows.append({"key": it["key"], "area": it.get("area"), "label": it["label"], "section": it["section"],
                     "required": bool(c.get("req", it.get("required"))),
                     "status": c["status"], "source": c["source"], "fact_ids": [f.fid for f in c["facts"]]})
    return rows


def checklist_in_note(V, sents, verdicts, report, doc="summary"):
    """기록지(생성·외부 작성)가 체크리스트 항목을 다루는지. 인용 또는 항목 언급으로 판정."""
    vd = {v["sid"]: v for v in verdicts}
    ok_s = [s for s in sents if vd.get(s["sid"], {}).get("label") not in ("Contradicted", "NotAddressed") and not s.get("fixed")]
    pre = "chk:" if doc == "summary" else "pchk:"
    out = []
    for r in report:
        if r["status"] in ("absent", "n/a") and not r["required"]:
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
    dflt = PROFILE.get("default_section")
    if dflt:
        for f in facts:
            if f.section not in SECTIONS:
                f.section = dflt
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
    parts = ["환자 ID %s" % ctx.subject_id, ("%s세 %s" % (fmt_num(ctx.age), sex_ko)) if (not neo and ctx.age is not None) else sex_ko]
    nums = []
    if pd.notna(admit):
        ctx.admit = pd.Timestamp(admit)
        ctx.hosp_day = (ctx.t0.normalize() - ctx.admit.normalize()).days + 1
        parts.append("입원 %d일째(입원 %s)" % (ctx.hosp_day, fmt_t(admit)))
        nums.append(ctx.hosp_day)
        if neo:
            ctx.age_days = int((ctx.t0 - ctx.admit).total_seconds() // 86400)
            parts.append("생후 약 %d일(입원 시각을 출생 시각으로 근사)" % ctx.age_days)
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
        facts.append(Fact("resp_support", "D3", txt, 3, concepts={"resp"} | ({"ev_vent"} if kind != "noninvasive" else set()) | ({"fio2"} if fio2 is not None else set()),
                          status="active",
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
            orgs = [o for o in orgs if not re.search(NEG_ORG_RX, o, re.I)]  # 'No MRSA isolated' 같은 음성 문구는 균 동정이 아님
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
    fd = d[d["concept"].isin(["in_oral", "in_tube"])].sort_values("t")
    if len(fd):
        per = fd["v"].median()
        gaps = fd["t"].diff().dropna().dt.total_seconds() / 3600.0
        parts = []
        for c, nm in (("in_oral", "경구"), ("in_tube", "경관")):
            x = fd[fd["concept"] == c]
            if len(x):
                parts.append("%s %d회 %s mL" % (nm, len(x), fmt_num(round(x["v"].sum()))))
        ftxt = "지난 %s시간 수유 기록 %d회, 총 %s mL(%s), 회당 중앙값 %s mL%s." % (
            HH(ctx), len(fd), fmt_num(round(fd["v"].sum())), ", ".join(parts), fmt_num(round(per, 1)),
            (", 간격 중앙값 %s시간" % fmt_num(round(gaps.median(), 1))) if len(gaps) else "")
        facts.append(Fact("feeding_24h", "D4", ftxt, 3, concepts={"feeding", "intake"}, key_nums=[len(fd), round(fd["v"].sum())],
                          prov=_prov("INPUTEVENTS", fd.head(3), "inputevent_id"), extra={"n": len(fd), "total": round(fd["v"].sum())}))
    facts.append(Fact("intake_24h", "D4", txt + ".", 3, concepts={"intake", "io"}, key_nums=nums, prov=_prov("INPUTEVENTS", d.head(3), "inputevent_id"),
                      extra={"total": round(total), "iv": round(sums["in_iv"]), "enteral": round(ent), "tube": round(sums["in_tube"]),
                             "oral": round(sums["in_oral"]), "mlkgday": ctx.intake_ml_kg_day}))
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
        bg = [c for c in ("ph", "pco2", "po2", "hco3") if c in LAB_DISPLAY]
        G = ctx.lab[ctx.lab["concept"].isin(bg) & (ctx.lab["t"] <= ctx.t0) & (ctx.lab["t"] > ctx.t0 - pd.Timedelta(hours=72))].dropna(subset=["v"])
        if len(G):
            tl = G["t"].max()
            last = G[G["t"] >= tl - pd.Timedelta(minutes=30)]
            parts, vals, cs = [], [], set()
            for c in bg:
                x = last[last["concept"] == c].sort_values("t")
                if len(x):
                    parts.append("%s %s" % (LAB_DISPLAY[c], fmt_num(x["v"].iloc[-1])))
                    vals.append(float(x["v"].iloc[-1]))
                    cs.add(c)
            facts.append(Fact("blood_gas", "D6", "최근 혈액가스(%s): %s." % (fmt_t(tl), ", ".join(parts)), 2, concepts={"blood_gas"} | cs, key_nums=vals, time=tl,
                              prov=_prov("LABEVENTS", last, "labevent_id")))
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


NOT_GIVEN_RX = r"not given|held|hold|보류|미투여|refus|거부|missed|omitted|delay|지연|투여 ?안 ?함"

NEG_ORG_RX = r"^\s*no\b|not isolated|no growth|negative|음성|자라지 않|무균|없음"

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
            n_win = int(((gg["t"] > ws) & (gg["t"] <= t0)).sum())
            rows.append(dict(generic=g, cls=gg["cls"].iloc[-1], name=gg["medication"].iloc[-1], first=gg["t"].min(), last=gg["t"].max(), source="EMAR",
                             prov=_prov("EMAR", gg.tail(2), "emar_id"), rate=rate, n_win=n_win))
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
    rxinfo = {}
    for r in rows:
        if r["source"] == "PRESCRIPTIONS" and (r.get("dose") or r.get("route")):
            rxinfo[r["generic"].lower()] = r
    fluids = []
    for k, r in by.items():
        if r["source"] == "EMAR" and k in rxinfo:
            r = dict(r, dose=rxinfo[k].get("dose"), route=rxinfo[k].get("route"))
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
                cnt = (", 지난 %s시간 투여 기록 %d회" % (HH(ctx), r["n_win"])) if r.get("n_win") else ""
                txt = "%s%s(%s) 투여 중%s%s%s, 마지막 기록 %s." % ((ck + " ") if ck else "", r["generic"], r["name"], extra_txt, cnt, newtxt, fmt_t(r["last"]))
            facts.append(Fact("med_active", "D7", txt, 3 if r["cls"] in HIGH_ALERT else 2, concepts={"med:" + r["generic"].lower(), "cls:" + r["cls"]},
                              status="active", time=r["last"], prov=r["prov"], extra=ex))
        else:
            txt = "%s%s(%s) 오늘 중단, 마지막 기록 %s." % ((ck + " ") if ck else "", r["generic"], r["name"], fmt_t(r["last"]))
            facts.append(Fact("med_stopped", "D7", txt, 2, concepts={"med:" + r["generic"].lower(), "cls:" + r["cls"]}, status="stopped", time=r["last"],
                              prov=r["prov"], extra=ex))
    if not ctx.emar.empty and "event_txt" in ctx.emar.columns:
        e = ctx.win(ctx.emar)
        ne = e[e["event_txt"].astype(str).str.contains(NOT_GIVEN_RX, case=False, regex=True, na=False)]
        if len(ne):
            parts = []
            for (g, ev), gg in ne.groupby(["generic", "event_txt"]):
                why = ""
                if "note_txt" in gg.columns:
                    nt = [str(x) for x in gg["note_txt"].dropna().unique() if str(x) not in ("nan", "")]
                    why = (", 사유 기록: %s" % "; ".join(nt[:2])) if nt else ""
                parts.append("%s %s %d건(마지막 %s%s)" % (g, ev, len(gg), fmt_t(gg["t"].max()), why))
            facts.append(Fact("med_status", "D7", "지난 %s시간 투여 상태 기록: %s." % (HH(ctx), "; ".join(parts[:6])), 2, concepts={"med_status"},
                              prov=_prov("EMAR", ne.head(3), "emar_id")))
    if fluids:
        facts.append(Fact("fluids", "D7", "수액 투여 중: %s." % ", ".join(sorted(set(f["name"] for f in fluids))), 1,
                          concepts={"cls:fluid"} | {"med:" + f["generic"].lower() for f in fluids}, status="active"))
    return facts


# =============================================================================
# 4. CDSS — NICU 규칙(결정론). 기준값은 프로파일 neonatal/critical_values/tdm 에서 읽음.
#    알림 문장은 근거 fact 값만 사용하고, 계산값은 출처가 남는 파생 fact로 등록해 검증기 대조 대상에 포함.
# =============================================================================


def add_severity(ctx, facts, alerts):
    """중증도 3단계(불안정 > 관찰 필요 > 안정)를 프로파일 severity 규칙으로 판정하고 근거와 함께 fact로 등록."""
    R = PROFILE.get("severity", {})
    rk_ko = {"invasive": "침습적 인공호흡", "vent_unknown": "인공호흡기 사용", "noninvasive": "비침습 호흡 보조", "o2": "산소 투여"}
    sev_ko = {"high": "고위험", "medium": "중간", "info": "참고"}

    def hit(rule):
        reasons, trig = [], []
        for sv in rule.get("alert_severity", []):
            al = [a for a in alerts if a["severity"] == sv]
            if al:
                reasons.append("%s 알림: %s" % (sev_ko.get(sv, sv), ", ".join(dict.fromkeys(a["title"] for a in al))))
                trig += [t for a in al for t in a["trigger_fact_ids"]]
        rk = getattr(ctx, "resp_kind", None)
        if rk in rule.get("resp_kind", []):
            reasons.append(rk_ko.get(rk, rk))
            trig += [f.fid for f in facts if f.ftype == "resp_support"]
        cls = set(rule.get("active_classes", []))
        act = [f for f in facts if f.ftype in ("infusion_episode", "med_active") and f.status == "active" and f.extra.get("cls") in cls]
        if act:
            reasons.append("%s 투여 중" % ", ".join(dict.fromkeys("%s %s" % (CLS_KO.get(f.extra["cls"], ""), f.extra["generic"]) for f in act)))
            trig += [f.fid for f in act]
        for fs in rule.get("fact_status", []):
            ft, st = fs.split(":")
            fl = [f for f in facts if f.ftype == ft and f.status == st]
            if fl:
                reasons.append({"neo_events": "무호흡·서맥·탈포화 기록"}.get(ft, ft))
                trig += [f.fid for f in fl]
        return reasons, trig

    level, reasons, trig = "안정", ["고위험·중간 알림, 침습적 호흡 보조, 혈관작용제 투여 없음"], []
    for key, name in (("unstable", "불안정"), ("watcher", "관찰 필요")):
        r, t = hit(R.get(key, {}))
        if r:
            level, reasons, trig = name, r, t
            break
    sec = R.get("section", "D2")
    f = Fact("severity", sec, "중증도: %s(규칙 기반 판정). 근거: %s." % (level, "; ".join(reasons)), 3, concepts={"severity"}, status=level,
             prov=[{"derived_from": list(dict.fromkeys(trig))}], extra={"level": level})
    f.fid = "F%03d" % (len(facts) + 1)
    idx = next((i for i, x in enumerate(facts) if x.section == sec), len(facts))
    facts.insert(idx, f)
    ctx.severity = level
    return f


def finalize_facts(ctx, facts, alerts):
    """CDSS 이후 단계: 중증도 판정, after_cdss 체크리스트 항목 판정, 새 fact 에 ID 부여."""
    add_severity(ctx, facts, alerts)
    for doc in ("summary", "parent"):
        new = apply_checklist(ctx, facts, doc, post=True)
        for g in new:
            if g not in facts:
                g.fid = "F%03d" % (len(facts) + 1)
                facts.append(g)


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
                       "trigger_fact_ids": [t for t in trig if t], "reference": ref, "parent_visible": rule in vis, "action": cdss_action(rule)})

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


# ---- 보호자 Daily Briefing(부모님 요약지): 간호 인계 fact(SSOT)에서 보호자에게 필요한 사실만 쉬운 말로. 숫자는 fact 값 그대로.
PB = {
    "opening": "%s 기준으로 확인된 내용을 말씀드리겠습니다.",
    "r_inv": "아이는 숨 쉬는 것을 돕는 기계(인공호흡기)를 쓰고 있습니다.",
    "r_niv": "아이는 코로 공기를 넣어 숨 쉬기를 도와주는 장치를 쓰고 있습니다.",
    "r_o2": "아이는 코에 연결된 줄로 산소를 받고 있습니다.",
    "r_none": "최근 기록에서는 숨 쉬기를 돕는 장치를 쓴 기록이 확인되지 않았습니다.",
    "r_change": "오늘 숨 쉬기를 돕는 방법이 바뀌었습니다.",
    "ev_desat": "오늘 산소 수치가 잠깐 낮아진 적이 %d번 있었습니다.",
    "ev_brady": "심장이 잠깐 느리게 뛴 적은 %d번 있었습니다.",
    "w_cur": "아이 몸무게는 %s %s입니다.",
    "w_prev": "어제 잰 몸무게는 %s %s이었습니다.",
    "fd_tube": "아이는 위로 연결된 관으로 먹고 있습니다.",
    "fd_oral": "아이는 입으로 먹고 있습니다.",
    "fd_both": "아이는 입과 위로 연결된 관으로 먹고 있습니다.",
    "fd_amount": "지난 하루 동안 %d번, 모두 %s mL를 먹었습니다.",
    "fd_none": "오늘 기록에서는 입이나 관으로 먹은 기록이 확인되지 않았습니다.",
    "fd_iv": "수분과 영양은 주사로 받고 있습니다.",
    "meds": "지금 %s 쓰고 있습니다.",
    "meds_more": "그 밖에 필요한 약과 영양제도 쓰고 있습니다.",
    "fluid": "수액도 맞고 있습니다.",
    "photo": "빛을 쬐는 치료(광선치료)를 받고 있습니다.",
    "lab_high": "피검사에서 %s 기준보다 높게 나왔습니다.",
    "lab_low": "피검사에서 %s 기준보다 낮게 나왔습니다.",
    "proc": "오늘 %s 했습니다.",
    "mpos": "균 검사 결과가 나왔습니다. 자세한 내용은 담당 의사에게 들으실 수 있습니다.",
    "mpend": "감염이 있는지 알아보는 검사 결과를 아직 기다리고 있습니다.",
    "crrt": "콩팥 기능을 돕는 투석 치료를 받았습니다.",
    "plan": "다음 계획으로 '%s' 내용이 기록되어 있습니다.",
}
PB_LAB = {"tbil": "황달 수치", "glucose": "혈당", "bst": "혈당", "crp": "염증 수치", "na": "나트륨(몸속 소금 성분)", "k": "칼륨", "hb": "빈혈 수치",
          "plt": "혈소판", "wbc": "백혈구", "anc": "백혈구", "cr": "콩팥 기능 수치", "bun": "콩팥 기능 수치", "ph": "혈액 산도", "pco2": "혈액 이산화탄소",
          "lactate": "젖산", "ica": "칼슘", "mg": "마그네슘", "phos": "인", "albumin": "알부민", "ast": "간 수치", "alt": "간 수치"}
PB_CLASS = {"vasopressor": "혈압과 심장을 돕는 약", "sedation": "편안하게 해 주는 약", "insulin": "혈당을 조절하는 약", "anticoagulant": "피가 굳지 않게 하는 약",
            "antibiotic": "감염을 막거나 치료하는 항생제", "antifungal": "곰팡이 감염을 치료하는 약", "diuretic": "소변이 잘 나오게 하는 약",
            "respiratory_stimulant": "숨 멈춤을 줄여 주는 약", "surfactant": "폐가 잘 펴지게 돕는 약", "nutrition": "주사로 주는 영양",
            "anticonvulsant": "경련을 막는 약", "antiplatelet": "피가 덜 뭉치게 하는 약"}


def _josa_list(names, pair="을를"):
    names = list(dict.fromkeys(names))
    if not names:
        return ""
    body = ", ".join(names[:-1]) + (", " if len(names) > 1 else "") + names[-1]
    return josa(body, pair)


def ssot_id(ctx, facts):
    base = {"stay": ctx.stay_id, "t0": str(ctx.t0), "facts": [f.text for f in facts if f.ftype not in ("parent_gap",)],
            "spec": (PROFILE.get("_spec") or {}).get("version"), "profile": PROFILE.get("profile_name"), "code": VERSION}
    return sha(base)


def doctor_flags(ctx, facts, alerts):
    """보호자에게 직접 설명하지 않고 진료팀 설명이 필요하다고 간호사에게 표시할 항목."""
    R = PROFILE["documents"]["parent"].get("doctor_flag", {})
    out = []
    rx = R.get("dx_regex")
    for f in facts:
        if f.ftype in ("diagnosis_main", "diagnosis_other") and rx:
            codes = [c for c in re.findall(r"\b[A-Z]\d{2,4}\b", f.text) if re.search(rx, c)]
            if codes:
                out.append(("중대한 진단 코드(%s)" % ", ".join(codes), [f.fid]))
        if f.ftype in R.get("fact_types", []):
            out.append(("%s" % f.text[:60], [f.fid]))
        if f.ftype == "severity" and f.status in R.get("severity_levels", []):
            out.append(("중증도 %s" % f.status, [f.fid]))
        if f.ftype == "lab_latest" and f.extra.get("critical"):
            out.append(("검사 위험값: %s" % f.text[:50], [f.fid]))
    for a in alerts:
        if a["severity"] in R.get("alert_severity", []):
            out.append(("고위험 알림: %s" % a["title"], a["trigger_fact_ids"]))
    seen, res = set(), []
    for t, ids in out:
        if t not in seen:
            seen.add(t)
            res.append((t, ids))
    return res


def render_parent(ctx, facts, alerts):
    """보호자 브리핑 = 문장 리스트(section, text, cites). 근거 없는 묶음은 비움."""
    P = PROFILE["documents"]["parent"]
    smap = P.get("section_of", {})
    secs = list(PARENT_SECTIONS.keys())
    psec, csec = P.get("cdss_section", "B3"), P.get("closing_section", "B5")
    by = OrderedDict((s, []) for s in secs)

    def put(sec, text, fids):
        if sec in by and text:
            by[sec].append({"section": sec, "text": "%s [%s]" % (text, ", ".join(dict.fromkeys(fids))), "cites": list(dict.fromkeys(fids))})
    F = {}
    for f in facts:
        F.setdefault(f.ftype, []).append(f)
    # 오늘 꼭 알아둘 변화
    for f in F.get("resp_change", []):
        put(smap.get("resp_change"), PB["r_change"], [f.fid])
    for f in F.get("neo_events", []):
        if f.status == "present":
            t = PB["ev_desat"] % f.extra["n_desat"] if f.extra.get("n_desat") else ""
            if f.extra.get("n_brady"):
                t = (t + " " if t else "") + PB["ev_brady"] % f.extra["n_brady"]
            put(smap.get("neo_events"), t, [f.fid])
    labs = [f for f in F.get("lab_latest", [])]
    for hi, key in ((True, "lab_high"), (False, "lab_low")):
        g = [f for f in labs if bool(f.extra.get("high")) == hi]
        names = [PB_LAB.get(f.extra.get("concept"), LAB_DISPLAY.get(f.extra.get("concept"), f.extra.get("concept"))) for f in g]
        if g:
            put(smap.get("lab_latest"), PB[key] % _josa_list(names, "이가"), [f.fid for f in g])
    for f in F.get("procedure", []) + F.get("imaging", []):
        names = [lay_proc(n) for n in (f.extra.get("names") or [f.extra.get("label", "")])]
        put(smap.get(f.ftype), PB["proc"] % _josa_list(names, "을를"), [f.fid])
    for f in F.get("micro_positive", []):
        if f.salience >= 3:
            put(smap.get("micro_positive"), PB["mpos"], [f.fid])
    for f in F.get("crrt_episode", []):
        put(smap.get("crrt_episode"), PB["crrt"], [f.fid])
    # 현재 받고 있는 치료와 돌봄
    for f in F.get("resp_support", []):
        k = f.extra.get("kind")
        t = PB["r_inv"] if k in ("invasive", "vent_unknown") else PB["r_niv"] if k == "noninvasive" else PB["r_o2"] if k == "o2" else PB["r_none"]
        put(smap.get("resp_support"), t, [f.fid])
    photo = [f for f in facts if f.ftype == "checklist_item" and re.search(r"광선|photo", f.text, re.I)]
    if photo:
        put(smap.get("resp_support", "B2"), PB["photo"], [f.fid for f in photo[:2]])
    fd, it = (F.get("feeding_24h") or [None])[0], (F.get("intake_24h") or [None])[0]
    if fd:
        route = PB["fd_both"] if ("경구" in fd.text and "경관" in fd.text) else PB["fd_tube"] if "경관" in fd.text else PB["fd_oral"]
        put(smap.get("feeding_24h"), route + " " + PB["fd_amount"] % (fd.extra["n"], fmt_num(fd.extra["total"])), [fd.fid] + ([it.fid] if it else []))
    elif it:
        put(smap.get("intake_24h"), PB["fd_none"] + (" " + PB["fd_iv"] if it.extra.get("iv") else ""), [it.fid])
    for f in F.get("weight", []):
        e = f.extra
        t = PB["w_cur"] % (fmt_num(e["cur"]), e.get("unit", "g"))
        if "prev" in e:
            t += " " + PB["w_prev"] % (fmt_num(e["prev"]), e.get("unit", "g"))
        put(smap.get("weight"), t, [f.fid])
    act = [f for f in F.get("med_active", []) + F.get("infusion_episode", []) if f.status == "active"]
    if act:
        names = [PB_CLASS[f.extra.get("cls")] for f in act if f.extra.get("cls") in PB_CLASS]
        other = [f for f in act if f.extra.get("cls") not in PB_CLASS]
        t = (PB["meds"] % _josa_list(names, "을를")) if names else ""
        if other:
            t = (t + " " if t else "") + PB["meds_more"]
        if F.get("fluids"):
            t += " " + PB["fluid"]
        put(smap.get("med_active"), t.strip(), [f.fid for f in act] + [f.fid for f in F.get("fluids", [])])
    elif F.get("fluids"):
        put(smap.get("fluids"), PB["fluid"], [f.fid for f in F["fluids"]])
    # 아직 확인이 필요한 중요한 내용
    if F.get("micro_pending"):
        put(smap.get("micro_pending"), PB["mpend"], [f.fid for f in F["micro_pending"]])
    lay = P.get("alert_text", {})
    seen = set()
    for a in alerts:
        r = a["rule_id"]
        if a.get("parent_visible") and r in lay and r not in seen:
            seen.add(r)
            put(psec, lay[r], a["trigger_fact_ids"])
    for f in F.get("parent_gap", []):
        put(psec, f.text, [f.fid])
    # 기록된 다음 계획: 기록에 계획·예정 항목이 있을 때만
    for f in facts:
        if f.ftype == "checklist_item" and re.search(r"예정|계획|plan", f.text, re.I) and not re.search(r"광선|photo", f.text, re.I):
            body = f.text.split(":", 1)[-1].strip().rstrip(".")
            put("B4" if "B4" in by else psec, PB["plan"] % body[:60], [f.fid])
    sents = []
    for s in secs:
        if s == csec:
            sents += [{"section": s, "text": q, "cites": [], "fixed": True} for q in P.get("questions", [])]
        else:
            sents += by[s]
    for i, x in enumerate(sents, 1):
        x["sid"] = "S%03d" % i
    return sents


def _kor_time(t):
    t = pd.Timestamp(t)
    h = t.hour
    return "%d월 %d일 %s %d시%s" % (t.month, t.day, "오전" if h < 12 else "오후", h if h <= 12 else h - 12, (" %d분" % t.minute) if t.minute else "")


def parent_markdown(ctx, sents, facts=None, alerts=None, verdicts=None, coverage=None, nurse=False, sid_=None, metrics_=None):
    """보호자용(nurse=False): 불렛 없이 묶음별 문단. 간호사 확인용(nurse=True): 문장별 근거·확인할 부분·진료팀 설명 필요 항목."""
    P = PROFILE["documents"]["parent"]
    hrs = (ctx.t0 - ctx.win_start).total_seconds() / 3600.0
    rng = "지난 24시간" if hrs >= 23.5 else "입실 후 %s시간" % fmt_num(round(hrs))
    clean = [x for x in sents if x["section"] in PARENT_SECTIONS]
    if not nurse:
        L = ["# " + P["title"], "", "%s 기준 · 범위: %s" % (_kor_time(ctx.t0), rng), "", PB["opening"] % _kor_time(ctx.t0).split(" ", 2)[-1], ""]
        for s, title in PARENT_SECTIONS.items():
            body = " ".join(_strip_cites(x["text"]) for x in clean if x["section"] == s)
            if body:
                L += ["## " + title, "", body, ""]
        L += ["---", "", "%s 간호사 확인 전 초안입니다. 기준 자료 ID %s" % (P.get("notice", ""), sid_ or "-")]
        return "\n".join(L) + "\n"
    vd = {v["sid"]: v for v in (verdicts or [])}
    FF = {f.fid: f for f in (facts or [])}
    flags = doctor_flags(ctx, facts or [], alerts or [])
    flag_ids = {i for _, ids in flags for i in ids}
    L = ["# 보호자 브리핑 간호사 확인용", "", "- 기준 자료 ID: %s (인계 자료가 바뀌면 다시 생성해야 하며, 이전 브리핑은 사용 중지로 바뀜)" % (sid_ or "-"),
         "- 기준 시각 %s, 범위 %s, stay %s" % (fmt_t(ctx.t0), rng, ctx.stay_id), "- 상태: 간호사 확인 전 초안. 보호자 자동 발송 안 함", ""]
    L += ["## 진료팀 설명이 필요한 항목", ""]
    L += (["- %s [%s]" % (t, ", ".join(ids)) for t, ids in flags] or ["- 해당 없음"])
    L += ["", "## 문장별 근거와 확인할 부분", "", "| 번호 | 묶음 | 보호자용 문장 | 근거 원문 | 자동 대조 | 확인할 부분 |", "|---|---|---|---|---|---|"]
    for x in clean:
        v = vd.get(x["sid"], {})
        src = " / ".join("%s: %s" % (c, FF[c].text[:90]) for c in x.get("cites", []) if c in FF) or ("고정 문구" if x.get("fixed") else "-")
        chk = []
        if v.get("label") not in (None, "Supported"):
            chk.append("%s: %s" % (v["label"], v.get("reason", "")))
        if extract_nums(x["text"]):
            chk.append("수치 확인")
        if set(x.get("cites", [])) & flag_ids:
            chk.append("진료팀 설명 필요")
        L.append("| %s | %s | %s | %s | %s | %s |" % (x["sid"], PARENT_SECTIONS.get(x["section"], ""), _strip_cites(x["text"]).replace("|", "/"),
                                                   src.replace("|", "/"), "근거 일치" if v.get("label") == "Supported" else (v.get("label") or "-"),
                                                   "; ".join(chk) or "-"))
    used = {c for x in clean for c in x.get("cites", [])}
    left = [f for f in (facts or []) if f.salience >= 3 and f.fid not in used and f.ftype not in ("checklist_gap", "parent_gap")]
    L += ["", "## 보호자 문서에서 뺀 중요 정보(전달 여부 판단)", ""] + (["- %s: %s" % (f.fid, f.text[:100]) for f in left] or ["- 없음"])
    if coverage is not None:
        miss = [c for c in coverage if c["status"] != "Covered"]
        L += ["", "## 자동 대조 요약", "", "- 문장 %d개 중 근거 일치 %d개" % (len(vd), sum(1 for v in vd.values() if v["label"] == "Supported")),
              "- 필수 항목 누락·부분 전달 %d건" % len(miss)]
        if metrics_:
            L.append("- 문장 평균 %s자, 최장 %s자" % (metrics_.get("read_avg_chars"), metrics_.get("read_max_chars")))
    return "\n".join(L) + "\n"


def supersede(path, new_id):
    """같은 경로에 다른 기준 자료 ID로 만든 브리핑이 있으면 '사용 중지' 이름으로 바꿔 재사용을 막음."""
    if not os.path.exists(path):
        return None
    old = open(path, encoding="utf-8").read()
    m = re.search(r"기준 자료 ID:? ([0-9a-f]{6,})", old)
    if m and m.group(1) != new_id:
        dst = path.replace(".md", ".SUPERSEDED_%s.md" % m.group(1))
        os.replace(path, dst)
        return dst
    return None


def note_markdown(ctx, sents, verdicts=None, coverage=None, title=None, checklist=None, doc="summary", show_cites=True, extra_metrics=None,
                  facts=None, alerts=None, sid_=None):
    if doc == "parent" and title is None:
        return parent_markdown(ctx, sents, facts, alerts, verdicts, coverage, nurse=show_cites, sid_=sid_, metrics_=extra_metrics)
    D = PROFILE["documents"][doc]
    title = title or D["title"]
    secmap = SECTIONS if doc == "summary" else PARENT_SECTIONS
    vd = {v["sid"]: v for v in (verdicts or [])}
    if doc == "summary":
        lines = ["# " + title, ""]
        sp = PROFILE.get("_spec")
        if sp:
            hrs = (ctx.t0 - ctx.win_start).total_seconds() / 3600.0
            val = {"요약 기준시각": fmt_t(ctx.t0), "집계 시작시각": fmt_t(ctx.win_start), "집계 종료시각": fmt_t(ctx.t0)}
            for k in sp.get("report_context", {}).get("required", []):
                lines.append("- %s: %s" % (k, val.get(k, "확인 필요")))
            lines.append("- 집계 구분: %s(%s시간)" % ("24시간 요약" if hrs >= 23.5 else "부분 기간 요약(입실 후 24시간 미만)", fmt_num(round(hrs, 1))))
            lines.append("- 명세: %s v%s" % (sp.get("spec_id"), sp.get("version")))
        lines += ["- stay_id %s / hadm_id %s / 기준 시각 %s (%d/%d일째 요약)" % (ctx.stay_id, ctx.hadm_id, fmt_t(ctx.t0), ctx.day_index, ctx.n_days),
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
        lines += ["", "## 필수 항목 체크리스트 (%s)" % (("명세 %s v%s" % (PROFILE["_spec"]["spec_id"], PROFILE["_spec"]["version"])) if (PROFILE.get("_spec") and doc == "summary") else PROFILE.get("profile_name", "")), "", "| 영역 | 항목 | 필수 | 데이터 | 문서 반영 | 근거 |", "|---|---|---|---|---|---|"]
        lab = {"data": "있음", "gap": "없음(확인 필요)", "absent": "없음", "n/a": "해당 없음"}
        inl = {"covered": "반영", "missing": "누락", "n/a": "-"}
        for c in checklist:
            lines.append("| %s | %s | %s | %s | %s | %s |" % (c.get("area") or "", c["label"], "필수" if c["required"] else "선택", lab.get(c["status"], c["status"]),
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
        max_tokens = int(os.environ.get("KCD_LLM_MAX_TOKENS", 0)) or max_tokens   # 추론(thinking) 모델은 크게(예: 8000)
        body = json.dumps({"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                           "temperature": temperature, "max_tokens": max_tokens}).encode("utf-8")
        req = urllib.request.Request(self.base.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key})
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read().decode("utf-8"))
        txt = out["choices"][0]["message"].get("content") or ""
        # 추론 모델의 <think>…</think> 는 버림(안의 '- ' 줄이 인계 문장으로 섞이지 않도록). 닫는 태그가 없으면(잘림) 빈 응답 취급
        if "<think>" in txt or "</think>" in txt:
            txt = txt.split("</think>", 1)[1] if "</think>" in txt else ""
        return txt.strip()


SYS_GROUNDED = ("당신은 NICU 환아의 1일 요약지를 작성하는 보조자다. 주어진 facts만 사용한다.\n"
                "규칙: 1) facts에 없는 정보·추정·권고를 쓰지 않는다 2) 문장마다 끝에 사용한 fact_id를 [F001] 형식으로 인용한다\n"
                "3) 수치·단위·시각(MM-DD HH:MM)은 fact 표기를 그대로 복사한다 4) 개조식(~함/~임)으로 쓴다\n"
                "5) 출력은 '- '로 시작하는 문장 목록만 쓴다. 제목·설명은 쓰지 않는다.")
SYS_PARENT = ("당신은 간호 인계 기록(facts)을 바탕으로 간호사가 보호자에게 오늘 아이 상태를 설명할 때 쓸 자료를 쓰는 보조자다.\n"
              "규칙: 1) 초등학교 6학년도 이해할 쉬운 한국어, 존댓말(~습니다), 한 문장은 짧게 2) 1~2분 안에 말할 분량\n"
              "3) 전문용어·불필요한 수치는 줄이되 중요한 악화나 불확실성은 빼지 않는다\n"
              "4) facts에 없는 진단·원인·예후·안심 표현을 만들지 않는다. 수치·단위·시간과 실제 투여 여부를 바꾸지 않는다\n"
              "5) '기록 없음'을 '정상'·'미투여'·'결과 대기'로 바꾸지 않는다 6) 문장마다 끝에 fact_id를 [F001] 형식으로 인용한다\n"
              "7) '- '로 시작하는 문장 목록만 쓴다.")


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
    sysp = (spec_system_prompt() if PROFILE.get("_spec") else SYS_GROUNDED) if doc == "summary" else SYS_PARENT
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
    texts += list(PB.values()) + list(PB_LAB.values()) + list(PB_CLASS.values()) + list(P.get("questions", [])) + [P.get("subtitle", "")]
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
            for ph in list(LAY_CLASS.values()) + list(PB_CLASS.values()):
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
            if self.doc == "parent" and re.match(r"[가-힣]", t):
                core = re.sub(r"(을|를|이|가|은|는|도|와|과|로|으로|에|에서|의)$", "", t)
                if len(core) < 2 or core in self.lex:
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
                hide = (f.ftype in ("lab_latest", "lab_change", "lab_normal", "bst_24h") and not PROFILE["documents"]["parent"].get("show_lab_values")) \
                    or f.ftype == "intake_24h"
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
# 7-b. 환자별 통합 보고서 (1. 1일 요약지 + 검사 추이  2. CDSS  3. 검증 정보  4. 부모님 요약지) → md, html
#   검사 추이 항목은 명세 JSON 의 "trend_items"(개념 키 목록)로 바꿀 수 있음. 없으면 TREND_DEFAULT.
# =============================================================================

TREND_DEFAULT = ["tbil", "crp", "glucose", "bst", "na", "k", "cr", "hb", "plt", "wbc", "ph", "pco2", "weight"]
SPARK = "▁▂▃▄▅▆▇█"


def trend_items():
    S = PROFILE.get("_spec") or {}
    return S.get("trend_items") or PROFILE.get("trend_items") or TREND_DEFAULT


# ---------------------------------------------------------------------------
# 문헌 기준값: 신생아 검사 참고범위·치료 기준선(neonatal_ref_literature.json)
#   파일의 entries 를 그대로 읽어 재태주수(GA)·생후시간(PNA)에 맞는 항목을 고른다. JSON 을 고치면 바로 반영.
#   같은 조건에 여러 문헌이 맞으면 모두 보여 주고, 띠는 파일에서 먼저 나온 문헌(=우선순위)으로 그린다.
#   GA 를 알 수 없으면 GA 제한이 있는 문헌으로 띠를 그리지 않고 "재태주수 확인 필요"로 표시(추정하지 않음).
# ---------------------------------------------------------------------------
REF_LIT_FILE = "neonatal_ref_literature.json"
REF_LIT = {"entries": [], "path": None}
REF_BAND_KINDS = ("reference_interval", "prediction_limit")
REF_LINE_KINDS = ("treatment_threshold", "target_range", "lower_limit")
# (개념, 원 단위, 표시 단위) → 곱할 값. 단위는 _unorm 으로 정규화한 문자열
UNIT_CONV = {("cr", "umol/l", "mg/dl"): 1 / 88.4, ("tbil", "umol/l", "mg/dl"): 1 / 17.1, ("dbil", "umol/l", "mg/dl"): 1 / 17.1,
             ("glucose", "mmol/l", "mg/dl"): 18.0, ("hb", "g/l", "g/dl"): 0.1, ("hb", "mmol/l", "g/dl"): 1.611, ("crp", "mg/l", "mg/dl"): 0.1,
             ("bun", "mmol/l", "mg/dl"): 2.8, ("ca", "mmol/l", "mg/dl"): 4.008, ("ica", "mmol/l", "mg/dl"): 4.008, ("mg", "mmol/l", "mg/dl"): 2.43,
             ("phos", "mmol/l", "mg/dl"): 3.097, ("alb", "g/l", "g/dl"): 0.1, ("hct", "l/l", "%"): 100.0}


def _unorm(u):
    u = str(u or "").strip().lower().replace("µ", "u").replace("μ", "u").replace("㎕", "ul").replace("㎗", "dl").replace("³", "^3").replace(" ", "")
    u = u.replace("×", "x").replace("meq/l", "mmol/l")
    if re.fullmatch(r"x?10\^?9/l|x?10\^?3/ul|k/ul|x?10\^?3/mm\^?3|10\*3/ul", u):
        return "10^3/ul"
    return u


def unit_factor(concept, frm, to):
    a, b = _unorm(frm), _unorm(to)
    if a == b or not a:
        return 1.0
    if (concept, a, b) in UNIT_CONV:
        return UNIT_CONV[(concept, a, b)]
    if (concept, b, a) in UNIT_CONV:
        return 1.0 / UNIT_CONV[(concept, b, a)]
    if a in ("cells/ul", "/ul", "cells/mm^3", "/mm^3") and b == "10^3/ul":   # 혈구 수: 개/µL → ×10³/µL
        return 0.001
    if a == "10^3/ul" and b in ("cells/ul", "/ul"):
        return 1000.0
    return None


def load_ref_lit(path=None):
    if path in ("none", "None"):
        REF_LIT.update({"entries": [], "path": None})
        return
    if path is None:
        for d in (os.getcwd(), os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()):
            p = os.path.join(d, REF_LIT_FILE)
            if os.path.exists(p):
                path = p
                break
    if not path:
        return
    with open(path, encoding="utf-8") as fh:
        J = json.load(fh)
    es = [e for e in J.get("entries", []) if e.get("concept") and not e.get("disabled")]
    srcs = J.get("sources", {})
    for i, e in enumerate(es):
        e.setdefault("id", "L%04d" % (i + 1))
        if isinstance(e.get("source"), str):   # 공통 문헌 정보는 sources 에 한 번만 적고 키로 참조
            e["source"] = dict(srcs.get(e["source"], {"citation": e["source"]}), **e.get("source_extra", {}))
    REF_LIT.update({"entries": es, "path": path, "meta": {k: J.get(k) for k in ("version", "note", "updated")}})


def ga_info(ctx):
    """재태주수: 1) 차트 기록 숫자(재태·GA·gestational) 2) 진단 코드(P07.2x/P07.3x 주수 코드, P07.2=<28주, P07.3=28~36주). 없으면 None."""
    if hasattr(ctx, "_ga"):
        return ctx._ga
    out = {"wk": None, "lo": None, "hi": None, "src": "재태주수 기록 없음"}
    for df, table in ((ctx.chart, "CHARTEVENTS"), (getattr(ctx, "dte", None), "DATETIMEEVENTS")):
        if df is None or df.empty or "itemid" not in df.columns:
            continue
        _labelize(ctx, df)
        m = df[df["full_label"].str.contains(r"재태|gestational|\bGA\b", case=False, regex=True, na=False)]
        for _, r in m.iterrows():
            s = str(r.get("value"))
            g = re.search(r"(\d{2})\s*(?:\+|주\s*)(\d)?", s) or re.fullmatch(r"\s*(\d{2}(?:\.\d+)?)\s*", s)
            if g:
                wk = float(g.group(1)) + (float(g.group(2)) / 7.0 if g.lastindex and g.lastindex >= 2 and g.group(2) else 0.0)
                if 20 <= wk <= 45:
                    out = {"wk": wk, "lo": wk, "hi": wk, "src": "%s %s=%s" % (table, r["full_label"], s)}
                    break
        if out["wk"] is not None:
            break
    if out["wk"] is None and not ctx.dx.empty and "icd_code" in ctx.dx.columns:
        codes = ctx.dx["icd_code"].astype(str).str.replace(".", "", regex=False).str.upper().tolist()
        wk_code = {"P0721": (None, 22.9), "P0722": (23, 23.9), "P0723": (24, 24.9), "P0724": (25, 25.9), "P0725": (26, 26.9), "P0726": (27, 27.9),
                   "P0731": (28, 28.9), "P0732": (29, 29.9), "P0733": (30, 30.9), "P0734": (31, 31.9), "P0735": (32, 32.9), "P0736": (33, 33.9),
                   "P0737": (34, 34.9), "P0738": (35, 35.9), "P0739": (36, 36.9)}
        for c in codes:
            if c[:5] in wk_code:
                lo, hi = wk_code[c[:5]]
                out = {"wk": None, "lo": lo, "hi": hi, "src": "진단 코드 %s" % c}
                break
        else:
            for c in codes:
                if c.startswith("P072"):
                    out = {"wk": None, "lo": None, "hi": 27.9, "src": "진단 코드 %s(28주 미만)" % c}
                    break
                if c.startswith("P073"):
                    out = {"wk": None, "lo": 28, "hi": 36.9, "src": "진단 코드 %s(28~36주)" % c}
                    break
    ctx._ga = out
    return out


def ctx_birth(ctx):
    b = getattr(ctx, "birth", None)
    if b is not None:
        return pd.Timestamp(b), getattr(ctx, "birth_src", "")
    if ctx.is_neonate and getattr(ctx, "admit", None) is not None:
        return pd.Timestamp(ctx.admit), "입원 시각 근사"
    return None, "출생 시각 확인 불가"


def _ga_fit(e, ga):
    """문헌 GA 구간과 환아 GA 의 관계: 'ok'(포함) / 'any'(문헌에 GA 제한 없음) / 'unknown'(환아 GA 모름) / None(불일치)"""
    lo, hi = e.get("ga_min_wk"), e.get("ga_max_wk")
    if lo is None and hi is None:
        return "any"
    if ga["lo"] is None and ga["hi"] is None:
        return "unknown"
    glo = ga["lo"] if ga["lo"] is not None else 0
    ghi = ga["hi"] if ga["hi"] is not None else 99
    elo = lo if lo is not None else 0
    ehi = (hi + 0.99) if hi is not None else 99   # "36주"는 36+6 까지 포함
    if elo <= glo and ghi <= ehi:
        return "ok"
    if ghi < elo or glo > ehi:
        return None
    return "unknown"


def lit_match(concept, ga, pna_h, unit):
    """해당 개념·GA·생후시간에 맞는 문헌 항목. 반환 [(entry, ga_fit, low, high)] (값은 unit 으로 환산, 환산 불가면 제외)"""
    out = []
    for e in REF_LIT["entries"]:
        if e.get("concept") != concept or e.get("kind") in ("equation", "formula"):
            continue
        if pna_h is not None:
            if e.get("pna_min_h") is not None and pna_h < e["pna_min_h"]:
                continue
            if e.get("pna_max_h") is not None and pna_h >= e["pna_max_h"]:
                continue
        elif e.get("pna_min_h") is not None or e.get("pna_max_h") is not None:
            continue
        g = ga
        if e.get("ga_basis") == "pma" and pna_h is not None:   # 교정(월경후)연령 기준 표: GA + 생후 주수
            w = pna_h / 168.0
            g = {"wk": None, "lo": ga["lo"] + w if ga["lo"] is not None else None, "hi": ga["hi"] + w if ga["hi"] is not None else None}
        fit = _ga_fit(e, g)
        if fit is None:
            continue
        f = unit_factor(concept, e.get("unit"), unit) if unit else 1.0
        if f is None:
            continue
        lo = e["low"] * f if e.get("low") is not None else None
        hi = e["high"] * f if e.get("high") is not None else None
        out.append((e, fit, lo, hi))
    return out


def lit_has(concept):
    return any(e.get("concept") == concept for e in REF_LIT["entries"])


def lit_cite(e):
    s = e.get("source", {})
    return s.get("short") or s.get("citation") or "출처 미기재"


def lit_cond(e):
    g = (e.get("ga_min_wk"), e.get("ga_max_wk"))
    p = (e.get("pna_min_h"), e.get("pna_max_h"))
    gs = "재태 전체" if g == (None, None) else "재태 %s~%s주" % (fmt_num(g[0]) if g[0] is not None else "", fmt_num(g[1]) if g[1] is not None else "")
    if p[1] is None and (p[0] is None or p[0] == 0):
        ps = "생후 전체"
    elif p[1] is None:
        ps = "생후 %s시간 이후" % fmt_num(p[0])
    else:
        ps = "생후 %s~%s시간 미만" % (fmt_num(p[0] or 0), fmt_num(p[1]))
    return "%s · %s" % (gs, ps)


def apply_lit(ctx, x):
    """추이 항목 x 에 점별 문헌 띠(lit)·기준선(lines)·최근 값의 전체 비교(lit_last)를 붙임."""
    x["lit"], x["lines"], x["lit_last"] = [], [], []
    if not REF_LIT["entries"]:
        return x
    ga = ga_info(ctx)
    b, bsrc = ctx_birth(ctx)
    x["ga"], x["birth_src"] = ga, bsrc
    for i, t in enumerate(x["t"]):
        pna = (pd.Timestamp(t) - b).total_seconds() / 3600.0 if b is not None else None
        ms = lit_match(x["key"], ga, pna, x["unit"])
        band = next(((lo, hi, e) for e, fit, lo, hi in ms if e.get("kind", "reference_interval") in REF_BAND_KINDS and fit in ("ok", "any")), None)
        x["lit"].append(band)
        x["lines"].append([(lo, hi, e) for e, fit, lo, hi in ms if e.get("kind") in REF_LINE_KINDS and fit in ("ok", "any") and e.get("chart", True)])
        if i == len(x["t"]) - 1:
            x["lit_last"] = [dict(e=e, fit=fit, lo=lo, hi=hi, pna=pna) for e, fit, lo, hi in ms]
    return x


def rng_txt(lo, hi):
    f = lambda z: fmt_num(round(z, 2))
    if lo is not None and hi is not None:
        return "%s ~ %s" % (f(lo), f(hi))
    if hi is not None:
        return "상한 %s" % f(hi)
    return "하한 %s" % f(lo) if lo is not None else "-"


def lit_judge(v, lo, hi):
    if lo is not None and v < lo:
        return "아래"
    if hi is not None and v > hi:
        return "위"
    return "안"


def lit_rows(x):
    """최근 값과 모든 문헌 기준 비교 행: [(문헌, 종류, 조건, 기준, 판정, GA 적합)]"""
    v = x["v"][-1]
    rows = []
    for m in x.get("lit_last", []):
        e = m["e"]
        rng = rng_txt(m["lo"], m["hi"])
        lab = "%s–%s" % (e.get("low_label") or "", e.get("high_label") or "")
        kind = {"reference_interval": "참고범위", "prediction_limit": "예측한계", "treatment_threshold": "치료 기준선", "target_range": "목표범위",
                "lower_limit": "하한"}.get(e.get("kind"), e.get("kind", ""))
        below = e.get("act_when") == "below"
        one = m["hi"] if m["lo"] is None else (m["lo"] if m["hi"] is None else None)
        if e.get("kind") in REF_LINE_KINDS and one is not None:
            if below:
                jd = "기준 미만(해당)" if v < one else "기준 이상"
            else:
                jd = "기준 초과(해당)" if v > one else "기준 이하"
        elif e.get("kind") in REF_LINE_KINDS:
            jd = {"아래": "기준 구간 아래", "위": "기준 구간 위", "안": "기준 구간 안"}[lit_judge(v, m["lo"], m["hi"])]
        else:
            jd = {"아래": "범위 아래", "위": "범위 위", "안": "범위 안"}[lit_judge(v, m["lo"], m["hi"])]
        if m["fit"] == "unknown":
            jd += "(재태주수 확인 필요)"
        if e.get("risk_group"):
            kind += " · " + e["risk_group"]
        rows.append((e, kind, lit_cond(e), "%s %s (%s)" % (rng, x["unit"], lab.strip("–")), jd))
    return rows


def trend_series(ctx):
    """ICU 입실 ~ 기준 시각의 항목별 시계열(원자료). 반환: [{key, label, unit, t[], v[], ref}]"""
    out = []
    for c in trend_items():
        t, v, unit, ref = [], [], "", (None, None)
        if c == "weight":
            ws = [w for w in weight_series(ctx) if w[0] >= ctx.intime]
            t, v, unit = [w[0] for w in ws], [w_disp(w[1]) for w in ws], ("g" if ws and ws[-1][1] < 10 else "kg")
            label = "체중"
        elif c == "bst":
            if ctx.chart.empty:
                continue
            s = ctx.chart[(ctx.chart["concept"] == "bst") & (ctx.chart["t"] >= ctx.intime) & (ctx.chart["t"] <= ctx.t0)].dropna(subset=["v"]).sort_values("t")
            t, v, unit, label = list(s["t"]), [float(x) for x in s["v"]], "mg/dL", "BST(현장 혈당)"
            ref = tuple(PROFILE["ref_ranges"].get("glucose", (None, None)))
        else:
            if ctx.lab.empty:
                continue
            s = ctx.lab[(ctx.lab["concept"] == c) & (ctx.lab["t"] >= ctx.intime) & (ctx.lab["t"] <= ctx.t0)].dropna(subset=["v"]).sort_values("t")
            if len(s):
                t, v = list(s["t"]), [float(x) for x in s["v"]]
                u = str(s["valueuom"].dropna().iloc[-1]) if ("valueuom" in s.columns and s["valueuom"].notna().any()) else ""
                unit = "" if u in ("nan", "None") else u
                ref = ref_range(s)
            label = LAB_DISPLAY.get(c, c)
        if len(t) >= 2:  # 기록이 1개뿐인 항목은 추이가 없으므로 제외
            out.append(apply_lit(ctx, {"key": c, "label": label, "unit": unit, "t": t, "v": v, "ref": ref}))
    return out


def spark(vals, n=20):
    if not vals:
        return ""
    if len(vals) > n:
        step = len(vals) / float(n)
        vals = [vals[int(i * step)] for i in range(n - 1)] + [vals[-1]]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return SPARK[3] * len(vals)
    return "".join(SPARK[int((x - lo) / (hi - lo) * (len(SPARK) - 1))] for x in vals)


def trend_markdown(ctx, series, max_days=14):
    if not series:
        return ["- ICU 입실 이후 기록이 2개 이상인 추이 항목 없음."]
    L = ["- 기간: ICU 입실 %s ~ 기준 시각 %s. 기록이 2개 이상인 항목만 표시. 원자료 집계값이며 검증기 대조 대상이 아님." % (fmt_t(ctx.intime), fmt_t(ctx.t0)), "",
         "| 항목 | 단위 | 첫 값(시각) | 최저 | 최고 | 최근 값(시각) | 변화 | 추이 | 기록 수 |", "|---|---|---|---|---|---|---|---|---|"]
    for x in series:
        v, t = x["v"], x["t"]
        d = v[-1] - v[0]
        L.append("| %s | %s | %s(%s) | %s | %s | %s(%s) | %s%s | %s | %d |" % (
            x["label"], x["unit"], fmt_num(v[0]), fmt_t(t[0]), fmt_num(min(v)), fmt_num(max(v)), fmt_num(v[-1]), fmt_t(t[-1]),
            "+" if d > 0 else "", fmt_num(round(d, 2)), spark(v), len(v)))
    days = pd.date_range(ctx.intime.normalize(), ctx.t0.normalize(), freq="D")
    days = list(days[-max_days:])
    L += ["", "일별 마지막 값(최근 %d일)" % len(days), "", "| 항목 | " + " | ".join(d.strftime("%m-%d") for d in days) + " |",
          "|---|" + "---|" * len(days)]
    for x in series:
        s = pd.Series(x["v"], index=pd.DatetimeIndex(x["t"]))
        last = s.groupby(s.index.normalize()).last()
        L.append("| %s | " % x["label"] + " | ".join(fmt_num(last[d]) if d in last.index else "" for d in days) + " |")
    L += [""] + lit_markdown(series)
    return L


def lit_markdown(series):
    if not REF_LIT["entries"]:
        return ["- 문헌 기준값 파일(%s) 없음: 그래프의 띠는 프로파일 고정 범위." % REF_LIT_FILE]
    L = ["문헌 기준과 최근 값 비교(%s, 같은 조건에 맞는 문헌은 모두 표시)" % os.path.basename(REF_LIT["path"]), "",
         "| 항목 | 최근 값 | 문헌 | 종류 | 적용 조건 | 기준 | 판정 |", "|---|---|---|---|---|---|---|"]
    n = 0
    for x in series:
        for e, kind, cond, rng, jd in lit_rows(x):
            L.append("| %s | %s %s | %s | %s | %s | %s | %s |" % (x["label"], fmt_num(x["v"][-1]), x["unit"], lit_cite(e), kind, cond, rng, jd))
            n += 1
        if not x.get("lit_last") and lit_has(x["key"]):
            L.append("| %s | %s %s | - | - | - | 맞는 문헌 없음 | - |" % (x["label"], fmt_num(x["v"][-1]), x["unit"]))
    for x in series[:1]:
        ga = x.get("ga") or {}
        L += ["", "- 재태주수: %s. 생후시간 기준: %s. 재태주수를 알 수 없으면 주수별 문헌은 띠로 그리지 않고 '재태주수 확인 필요'로 표시." % (
            (fmt_num(round(ga["wk"], 1)) + "주 (" + ga.get("src", "") + ")") if ga.get("wk") else ga.get("src", "확인 불가"), x.get("birth_src") or "확인 불가")]
    return L


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def point_band(x, i):
    """i 번째 점의 (low, high, 출처문구). 문헌 띠가 있으면 문헌, 없으면 프로파일 고정 범위."""
    lit = x.get("lit") or []
    if any(lit):
        b = lit[i] if i < len(lit) else None
        return (b[0], b[1], lit_cite(b[2])) if b else (None, None, None)
    rl, rh = x["ref"] if x["ref"] else (None, None)
    return (rl, rh, "프로파일 고정 범위") if (rl is not None or rh is not None) else (None, None, None)


def trend_svg(x, w=560, h=150):
    t = [pd.Timestamp(a).value for a in x["t"]]
    v = x["v"]
    n = len(t)
    bands = [point_band(x, i) for i in range(n)]
    lines = x.get("lines") or [[] for _ in range(n)]
    lo, hi = min(v), max(v)
    for b in bands:
        for r in b[:2]:
            if r is not None:
                lo, hi = min(lo, r), max(hi, r)
    for ls in lines:
        for a, b, e in ls:
            for r in (a, b):
                if r is not None:
                    lo, hi = min(lo, r), max(hi, r)
    pad = (hi - lo) * 0.1 or 1.0
    lo, hi = lo - pad, hi + pad
    t0, t1 = min(t), max(t)
    span = (t1 - t0) or 1
    L, R, T, B = 48, 12, 18, 26

    def X(a):
        return L + (a - t0) / span * (w - L - R)

    def Y(b):
        return T + (hi - b) / (hi - lo) * (h - T - B)

    def seg(i):
        xa = X(t[i])
        xb = X(t[i + 1]) if i + 1 < n else w - R
        if i == 0:
            xa = L
        return xa, max(xb, xa + 2)
    parts = ['<svg viewBox="0 0 %d %d" width="100%%" style="max-width:%dpx" role="img" aria-label="%s 추이">' % (w, h, w, _esc(x["label"]))]
    for i, (rl, rh, src) in enumerate(bands):   # 점마다 생후시간이 달라 기준이 바뀌면 계단형 띠
        if rl is None and rh is None:
            continue
        xa, xb = seg(i)
        ya, yb = Y(rh) if rh is not None else T, Y(rl) if rl is not None else h - B
        parts.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#2e7d32" opacity="0.12"><title>%s ~ %s (%s)</title></rect>' % (
            xa, ya, xb - xa, max(0, yb - ya), fmt_num(round(rl, 2)) if rl is not None else "", fmt_num(round(rh, 2)) if rh is not None else "", _esc(src)))
    for i, ls in enumerate(lines):
        xa, xb = seg(i)
        for a, b, e in ls:
            for r in (a, b):
                if r is not None:
                    parts.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#c62828" stroke-width="1.4" stroke-dasharray="4 3"><title>%s %s (%s)</title></line>' % (
                        xa, Y(r), xb, Y(r), _esc(e.get("high_label") or e.get("low_label") or "기준선"), fmt_num(round(r, 2)), _esc(lit_cite(e))))
    parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#999" stroke-width="1"/>' % (L, h - B, w - R, h - B))
    pts = " ".join("%.1f,%.1f" % (X(a), Y(b)) for a, b in zip(t, v))
    parts.append('<polyline points="%s" fill="none" stroke="#1565c0" stroke-width="1.8"/>' % pts)
    for i, (a, b) in enumerate(zip(t, v)):
        rl, rh, src = bands[i]
        out = (rl is not None and b < rl) or (rh is not None and b > rh)
        ref_t = (" · 기준 %s~%s (%s)" % (fmt_num(round(rl, 2)) if rl is not None else "", fmt_num(round(rh, 2)) if rh is not None else "", src)) if src else ""
        parts.append('<circle cx="%.1f" cy="%.1f" r="2.6" fill="%s"><title>%s %s%s</title></circle>' % (X(a), Y(b), "#c62828" if out else "#1565c0",
                                                                                                      fmt_t(pd.Timestamp(a)), fmt_num(b), _esc(ref_t)))
    for b in (max(v), min(v)):
        parts.append('<text x="%d" y="%.1f" font-size="10" text-anchor="end" fill="#555">%s</text>' % (L - 4, Y(b) + 3, fmt_num(b)))
    parts.append('<text x="%d" y="%d" font-size="10" fill="#555">%s</text>' % (L, h - 8, fmt_t(pd.Timestamp(t0))))
    parts.append('<text x="%d" y="%d" font-size="10" text-anchor="end" fill="#555">%s</text>' % (w - R, h - 8, fmt_t(pd.Timestamp(t1))))
    parts.append('<text x="%d" y="12" font-size="11" fill="#222">%s %s</text>' % (L, _esc(x["label"]), _esc(("(" + x["unit"] + ")") if x["unit"] else "")))
    parts.append("</svg>")
    return "".join(parts)


def ref_caption(x):
    """그래프 아래 기준 설명 한 줄(무엇을 초록 띠·빨간 점선으로 그렸는지)."""
    lit = [b for b in (x.get("lit") or []) if b]
    ga = x.get("ga") or {}
    if lit:
        es = []
        for b in lit:
            if b[2] not in es:
                es.append(b[2])
        s = "초록 띠: " + "; ".join("%s (%s–%s, %s)" % (lit_cite(e), e.get("low_label") or "", e.get("high_label") or "", lit_cond(e)) for e in es)
    elif x.get("ref") and any(r is not None for r in x["ref"]):
        s = "초록 띠: 프로파일 고정 범위(문헌·재태주수·생후시간 미반영)"
        if REF_LIT["entries"]:
            pend = [m for m in x.get("lit_last", []) if m["fit"] == "unknown"]
            s += " — 맞는 문헌 기준 없음" + (", 재태주수 확인 필요(후보 %d개)" % len(pend) if pend else "")
    else:
        s = "기준 범위 없음"
    ln = []
    for ls in x.get("lines") or []:
        for a, b, e in ls:
            if e not in ln:
                ln.append(e)
    if ln:
        s += " · 빨간 점선: " + "; ".join("%s %s (%s)" % (e.get("high_label") or e.get("low_label") or "기준선", lit_cite(e), lit_cond(e)) for e in ln)
    if REF_LIT["entries"]:
        s += " · 재태주수: %s · 생후시간: %s" % ((fmt_num(round(ga["wk"], 1)) + "주") if ga.get("wk") else ga.get("src", "확인 불가"), x.get("birth_src") or "확인 불가")
    return s


def _strip_cites(t):
    return re.sub(r"\s*\[F\d{3}(?:\s*,\s*F\d{3})*\]", "", re.sub(r"\s*\(규칙 [^)]*\)", "", t))


def report_markdown(ctx, res, series):
    s, q = res["summary"], res["parent"]
    sm, pm = s["metrics"], q["metrics"]
    title = PROFILE["documents"]["summary"]["title"]
    pre = note_markdown(ctx, [x for x in s["sents"] if x["section"] != CDSS_SEC()], doc="summary").splitlines()
    head = []
    for l in pre[1:]:
        if l.startswith("## "):
            break
        if l.startswith("- "):
            head.append(l)
    L = ["# 환자별 NICU 보고서 — 환자 ID %s, %s 기준" % (ctx.subject_id, fmt_t(ctx.t0)), ""] + head + ["",
         "> 자동 생성 초안이며 의료진 검토가 필요합니다. 문장 끝 [F###]은 근거 fact 번호이고, 3장 검증 정보와 연결됩니다.", ""]
    L += ["## 1. 1일 요약지 — %s" % title.replace(" (자동 생성 초안, 의료진 검토 필요)", "")]
    vd = {v["sid"]: v for v in s["verdicts"]}
    cur = None
    for x in s["sents"]:
        if x["section"] == CDSS_SEC():
            continue
        if x["section"] != cur:
            cur = x["section"]
            L += ["", "### " + SECTIONS.get(cur, cur)]
        mark = "  ⚠️검증: %s" % vd[x["sid"]]["label"] if vd.get(x["sid"], {}).get("label") not in (None, "Supported") else ""
        L.append("- " + x["text"] + mark)
    L += ["", "### 검사 추이(ICU 입실 이후)", ""] + trend_markdown(ctx, series)
    L += ["", "## 2. CDSS 알림(규칙 기반 + 트리 모델, 인계문과 별도)"]
    if res["alerts"]:
        L += ["", "| 심각도 | 알림 | 내용 | 권고 행동(검토 제안) | 근거 fact | 규칙·모델 | 참고 |", "|---|---|---|---|---|---|---|"]
        order = {"high": 0, "medium": 1, "info": 2}
        for a in sorted(res["alerts"], key=lambda x: order.get(x["severity"], 3)):
            L.append("| %s | %s | %s | %s | %s | %s | %s |" % (a["severity"].upper(), a["title"], a["text"], a.get("action", "-"), ", ".join(a["trigger_fact_ids"]),
                                                             a["rule_id"], a["reference"]))
        L.append("- 권고 행동은 병원 프로토콜 확인을 돕는 검토 제안이며 처방이 아님. 문구는 프로파일 cdss.actions 에서 수정.")
    else:
        L.append("- 규칙 기반 알림 없음.")
    L += ["", "## 3. 검증 정보", "", "| 지표 | 1일 요약지 | 부모님 요약지 |", "|---|---|---|"]
    for k, nm in (("n_sentences", "문장 수"), ("supported", "근거 일치"), ("contradicted", "근거와 불일치"), ("not_addressed", "기록에 근거 없음"),
                  ("unverifiable", "판정 불가"), ("hallucination_rate", "할루시네이션율"), ("n_must", "필수 정보 수"), ("missing", "필수 정보 누락"),
                  ("partial", "부분 전달"), ("omission_rate", "누락율"), ("checklist_required", "명세 필수 요소"), ("checklist_from_data", "데이터로 채움"),
                  ("checklist_gap", "확인 필요로 표시"), ("checklist_required_missing_in_note", "문서에서 빠진 필수 요소")):
        L.append("| %s | %s | %s |" % (nm, sm.get(k, "-"), pm.get(k, "-")))
    L.append("| 부모님 요약지 문장 평균 길이(자) | - | %s |" % pm.get("read_avg_chars", "-"))
    bad = [(x, vd[x["sid"]]) for x in s["sents"] if vd.get(x["sid"], {}).get("label") not in (None, "Supported")]
    qvd = {v["sid"]: v for v in q["verdicts"]}
    bad += [(x, qvd[x["sid"]]) for x in q["sents"] if qvd.get(x["sid"], {}).get("label") not in (None, "Supported")]
    L += ["", "- 판정 기준: 문장의 수치·시각·약물·사건이 인용한 근거 fact와 일치하면 '근거 일치'. 필수 정보는 명세 요소와 중요도 2 이상 fact 기준."]
    L.append("- 근거와 맞지 않는 문장: " + ("없음" if not bad else "%d건" % len(bad)))
    for x, v in bad[:20]:
        L.append("  - %s → %s(%s)" % (_strip_cites(x["text"])[:80], v["label"], v["reason"]))
    miss = [c for c in s["coverage"] + q["coverage"] if c["status"] != "Covered"]
    L.append("- 누락·부분 전달 필수 정보: " + ("없음" if not miss else "%d건" % len(miss)))
    for c in miss[:20]:
        L.append("  - %s %s: %s" % (c["fact_id"], c["status"], c["text"][:80]))
    L += ["", "명세 필수 요소 반영 현황", "", "| 영역 | 요소 | 필수 | 데이터 | 문서 반영 | 근거 |", "|---|---|---|---|---|---|"]
    lab = {"data": "있음", "gap": "없음(확인 필요)", "absent": "없음", "n/a": "해당 없음"}
    inl = {"covered": "반영", "missing": "누락", "n/a": "-"}
    for c in s["checklist"]:
        L.append("| %s | %s | %s | %s | %s | %s |" % (c.get("area") or "", c["label"], "필수" if c["required"] else "선택", lab.get(c["status"], c["status"]),
                                                   inl.get(c.get("in_note"), c.get("in_note")), ", ".join(c["fact_ids"][:4])))
    L += ["", "근거 fact 목록(출처 테이블)", "", "| fact | 유형 | 내용 | 출처 |", "|---|---|---|---|"]
    for f in res["facts"]:
        src = sorted({p.get("table", "파생") for p in f.prov if isinstance(p, dict)}) or ["-"]
        L.append("| %s | %s | %s | %s |" % (f.fid, f.ftype, f.text.replace("|", "/")[:120], ", ".join(src)))
    L += ["", "- 한계: 검증기는 문장이 추출된 fact와 맞는지를 확인합니다. 원자료에서 fact를 뽑는 단계의 오류(측정 오류값, 라벨 오인식)는 이 표로 확인할 수 없어 의료진 검토가 필요합니다."]
    L += ["", "## 4. 부모님 요약지(보호자 Daily Briefing, 간호사 확인 전)", ""] + [
        l.replace("## ", "### ", 1) if l.startswith("## ") else l
        for l in parent_markdown(ctx, q["sents"], res["facts"], res["alerts"], sid_=res.get("ssot_id")).splitlines()[1:]]
    return "\n".join(L) + "\n"


def md_to_html(md, title, svgs):
    """보고서용 최소 마크다운 → HTML 변환(제목·목록·표·인용). '### 검사 추이' 아래에 SVG 그래프 삽입."""
    out, in_list, in_table = [], False, False

    def close():
        nonlocal in_list, in_table
        if in_list:
            out.append("</ul>")
            in_list = False
        if in_table:
            out.append("</tbody></table>")
            in_table = False
    lines = md.splitlines()
    for i, l in enumerate(lines):
        if l.startswith("|"):
            cells = [c.strip() for c in l.strip().strip("|").split("|")]
            if re.match(r"^\|(\s*-+\s*\|)+$", l.strip()):
                continue
            if not in_table:
                close()
                out.append("<table><thead><tr>%s</tr></thead><tbody>" % "".join("<th>%s</th>" % _esc(c) for c in cells))
                in_table = True
            else:
                out.append("<tr>%s</tr>" % "".join("<td>%s</td>" % _esc(c) for c in cells))
            continue
        if l.startswith("- ") or l.startswith("  - "):
            if in_table:
                close()
            if not in_list:
                out.append("<ul>")
                in_list = True
            cls = ' class="sub"' if l.startswith("  - ") else ""
            out.append("<li%s>%s</li>" % (cls, _esc(l.strip()[2:])))
            continue
        close()
        if l.startswith("#"):
            n = len(l) - len(l.lstrip("#"))
            out.append("<h%d>%s</h%d>" % (n, _esc(l[n:].strip()), n))
            if l.startswith("### 검사 추이") and svgs:
                out.append('<div class="charts">%s</div>' % "".join('<div class="chart">%s</div>' % s for s in svgs))
        elif l.startswith("> "):
            out.append("<blockquote>%s</blockquote>" % _esc(l[2:]))
        elif l.strip():
            out.append("<p>%s</p>" % _esc(l))
    close()
    css = ("body{font-family:-apple-system,'Malgun Gothic','Apple SD Gothic Neo',sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222;line-height:1.5}"
           "h1{font-size:22px;border-bottom:2px solid #333;padding-bottom:6px}h2{font-size:18px;margin-top:28px;border-bottom:1px solid #ccc}h3{font-size:15px}"
           "table{border-collapse:collapse;margin:8px 0;font-size:13px}th,td{border:1px solid #ccc;padding:3px 6px;vertical-align:top}th{background:#f2f2f2}"
           "li.sub{margin-left:18px;color:#555}blockquote{color:#555;border-left:3px solid #ccc;margin:8px 0;padding-left:10px}"
           ".charts{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:8px}.chart{border:1px solid #eee;padding:4px}")
    return "<!doctype html><html lang=\"ko\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>%s</title><style>%s</style></head><body>%s</body></html>" % (
        _esc(title), css, "\n".join(out))



# ---- 대화형 HTML 보고서: 값 호버 → 검증 판정·근거 fact·원자료 행. 맨 위에 필수 정보 점검 패널.
RAW_SRC = {  # 출처 테이블 → (StayContext 속성, ID 열, 표시 열)
    "CHARTEVENTS": ("chart", "chartevent_id", ["charttime", "itemid", "value", "valueuom"]),
    "LABEVENTS": ("lab", "labevent_id", ["charttime", "itemid", "label", "value", "valueuom", "ref_range_lower", "ref_range_upper"]),
    "INPUTEVENTS": ("inp", "inputevent_id", ["starttime", "endtime", "itemid", "amount", "amountuom", "patientweight"]),
    "OUTPUTEVENTS": ("out", "outputevent_id", ["charttime", "itemid", "value", "valueuom"]),
    "EMAR": ("emar", "emar_id", ["charttime", "medication", "event_txt", "infusion_rate_hr", "infusion_rate_min"]),
    "MICROBIOLOGYEVENTS": ("micro", "microevent_id", ["charttime", "spec_type_desc", "test_name", "org_name", "ab_name", "interpretation", "storetime"]),
    "PROCEDUREEVENTS": ("proc", "procedureevent_id", ["starttime", "endtime", "itemid", "value"]),
    "TESTEVENTS": ("test", "testevent_id", ["charttime", "itemid"]),
    "DATETIMEEVENTS": ("dte", "datetimeevent_id", ["charttime", "itemid", "value"]),
    "DIAGNOSES_ICD": ("dx", "icd_code", ["seq_num", "icd_code", "icd_version"]),
    "PROCEDURES_ICD": ("px", "icd_code", ["chartdate", "icd_code", "icd_version"]),
    "PRESCRIPTIONS": ("rx", "pharmacy_id", ["starttime", "stoptime", "drug", "dose_val_rx", "dose_unit_rx", "route"]),
    "SERVICES": ("services", "hadm_id", ["transfertime", "curr_service"]),
    "ICUSTAYS": (None, "stay_id", ["first_careunit", "intime", "outtime", "los"]),
    "PATIENTS": ("pat", "subject_id", ["subject_id", "sex", "anchor_age", "anchor_year"]),
    "ADMISSIONS": ("adm", "hadm_id", ["admittime", "dischtime", "admission_type"]),
}
_VAL_COLS = {"chart": "v", "lab": "v", "inp": "v", "out": "v"}


def _raw_label(ctx, r):
    i = r.get("itemid")
    if i is not None and i in ctx.item_map:
        return ctx.item_map[i][1]
    if i is not None and i in ctx.lab_map:
        return ctx.lab_map[i][1]
    return r.get("label") or r.get("medication") or ""


def _raw_row(ctx, table, row, idcol):
    keep = RAW_SRC.get(table, (None, None, []))[2]
    d = {"table": table, "id": str(row.get(idcol, "")), "label": _raw_label(ctx, row)}
    for c in keep:
        if c in row and pd.notna(row[c]) and str(row[c]) not in ("nan", ""):
            d[c] = str(row[c])
    return d


def raw_rows_of(ctx, f, limit=12):
    """fact 출처(prov)의 원자료 행. 파생 fact 는 입력 fact 의 출처를 따라감."""
    out = []
    for p in f.prov:
        if not isinstance(p, dict):
            continue
        t = p.get("table")
        if t == "ICUSTAYS":
            out.append({"table": t, "id": str(ctx.stay_id), "label": "ICU 입실",
                        **{c: str(ctx.row.get(c)) for c in RAW_SRC[t][2] if c in ctx.row and pd.notna(ctx.row.get(c))}})
            continue
        if t not in RAW_SRC:
            continue
        attr, idc, _ = RAW_SRC[t]
        df = getattr(ctx, attr, None)
        if df is None or df.empty or idc not in df.columns:
            continue
        m = df[df[idc].astype(str) == str(p.get("row_id"))]
        for _, r in m.head(3).iterrows():
            out.append(_raw_row(ctx, t, r, idc))
        if len(out) >= limit:
            break
    return out[:limit]


def raw_match(ctx, facts, x, limit=3):
    """문장 속 숫자 x 를 만든 원자료 행 찾기: 인용 fact 개념의 원자료 중 값이 같은 행(ICU 입실~기준 시각)."""
    hits = []
    for f in facts:
        cons = {c for c in f.concepts if not c.startswith(("chk:", "pchk:", "cls:", "med:"))}
        for attr, table, idc in (("chart", "CHARTEVENTS", "chartevent_id"), ("lab", "LABEVENTS", "labevent_id"),
                                 ("inp", "INPUTEVENTS", "inputevent_id"), ("out", "OUTPUTEVENTS", "outputevent_id")):
            df = getattr(ctx, attr, None)
            if df is None or df.empty or "concept" not in df.columns:
                continue
            d = df[df["concept"].isin(cons) & (df["t"] >= ctx.intime) & (df["t"] <= ctx.t0)]
            if d.empty:
                continue
            v = to_num(d["v"])
            ok = (v - x).abs() <= max(0.051, 0.01 * abs(x))
            if "weight" in cons:
                ok = ok | ((v / 1000.0 - x).abs() <= 0.051) | ((v * 1000.0 - x).abs() <= 1.0)
            for _, r in d[ok].sort_values("t").tail(limit).iterrows():
                hits.append(_raw_row(ctx, table, r, idc))
        if len(hits) >= limit:
            break
    return hits[:limit]


def _rows_html(rows):
    if not rows:
        return "<div class='muted'>연결된 원자료 행 없음(고정 문구·계산값·데이터 없음 표시)</div>"
    cols = ["table", "id", "label"] + [c for c in ("charttime", "starttime", "endtime", "storetime", "value", "amount", "valueuom", "amountuom",
                                                   "ref_range_lower", "ref_range_upper", "medication", "event_txt", "infusion_rate_hr", "infusion_rate_min",
                                                   "spec_type_desc", "org_name", "ab_name", "interpretation", "drug", "dose_val_rx", "dose_unit_rx", "route",
                                                   "icd_code", "seq_num", "patientweight", "first_careunit", "intime", "outtime", "curr_service", "sex",
                                                   "admittime") if any(c in r for r in rows)]
    h = "<table class='raw'><tr>%s</tr>" % "".join("<th>%s</th>" % _esc(c) for c in cols)
    for r in rows:
        h += "<tr>%s</tr>" % "".join("<td>%s</td>" % _esc(r.get(c, "")) for c in cols)
    return h + "</table>"


VERDICT_KO = {"Supported": ("근거 일치", "ok"), "Contradicted": ("근거와 불일치", "bad"), "NotAddressed": ("기록에 근거 없음", "bad"),
              "Unverifiable": ("판정 불가", "warn")}


def sentence_html(ctx, x, v, F, sid_anchor=True):
    """문장 → 숫자마다 호버(원자료 행), 끝에 판정 배지 호버(판정·사유·근거 fact·출처 행)."""
    text = x["text"]
    cf = [F[c] for c in x.get("cites", []) if c in F]
    body = re.sub(r"\s*\[F\d{3}(?:\s*,\s*F\d{3})*\]", "", re.sub(r"\s*\(규칙 [^)]*\)", "", text))
    masked = strip_keep(body)
    parts, pos = [], 0
    for m in NUM_RX.finditer(masked):
        parts.append(_esc(body[pos:m.start()]))
        val = float(m.group(0))
        hits = raw_match(ctx, cf, val) if cf else []
        inner = ("<b>값 %s의 원자료</b>" % _esc(m.group(0))) + (_rows_html(hits) if hits else
                                                         "<div class='muted'>한 행과 바로 일치하지 않음 → 여러 행의 합계·비율 같은 계산값. 아래 근거 fact의 출처 행 참고.</div>"
                                                         + "".join(_rows_html(raw_rows_of(ctx, f, 6)) for f in cf[:2]))
        parts.append("<span class='tip num %s' tabindex='0'>%s<span class='pop'>%s</span></span>" % ("hit" if hits else "calc", _esc(body[m.start():m.end()]), inner))
        pos = m.end()
    parts.append(_esc(body[pos:]))
    lab, cls = VERDICT_KO.get((v or {}).get("label"), ("고정 문구", "fix")) if not x.get("fixed") else ("고정 문구", "fix")
    pop = "<b>검증: %s</b>" % lab
    if v and v.get("reason"):
        pop += "<div>사유: %s</div>" % _esc(v["reason"])
    for f in cf:
        pop += "<div class='fact'><b>%s</b> %s <span class='muted'>(%s, 중요도 %d)</span></div>%s" % (f.fid, _esc(f.text), f.ftype, f.salience, _rows_html(raw_rows_of(ctx, f)))
        der = [p for p in f.prov if isinstance(p, dict) and "derived_from" in p]
        if der:
            pop += "<div class='muted'>파생 fact: 입력 %s</div>" % _esc(der[0]["derived_from"])
    badge = "<span class='tip badge %s' tabindex='0'>%s<span class='pop wide'>%s</span></span>" % (cls, "✔" if cls == "ok" else ("•" if cls == "fix" else "⚠"), pop)
    anchor = " id='%s'" % x["sid"] if sid_anchor else ""
    return "<li%s>%s %s</li>" % (anchor, "".join(parts), badge)


def coverage_panel_html(res, doc_label=("1일 요약지", "부모님 요약지")):
    """필수 정보 점검: 명세 요소별 상태 칸 + 중요 fact 미인용 점검."""
    s = res["summary"]
    cited = {c for x in s["sents"] for c in x.get("cites", [])}
    st_cls = {"covered": "ok", "missing": "bad"}
    req = [c for c in s["checklist"] if c["required"]]
    n_cov = sum(1 for c in req if c["status"] == "data" and c.get("in_note") == "covered")
    n_gap = sum(1 for c in req if c["status"] == "gap")
    n_miss = sum(1 for c in req if c.get("in_note") == "missing")
    must = [f for f in res["facts"] if f.must and f.section in SECTIONS and f.section != CDSS_SEC() and f.ftype not in ("checklist_gap",)]
    unc = [f for f in must if f.fid not in cited]
    covv = [c for c in s["coverage"] if c["status"] != "Covered"]
    h = ["<section class='panel'><h2>0. 필수 정보 점검</h2>",
         "<div class='kpis'>"
         "<div class='kpi ok'><b>%d</b><span>필수 요소 데이터로 반영</span></div>" % n_cov +
         "<div class='kpi warn'><b>%d</b><span>데이터 없음 → '확인 필요' 표시</span></div>" % n_gap +
         "<div class='kpi %s'><b>%d</b><span>문서에서 빠진 필수 요소</span></div>" % ("bad" if n_miss else "ok", n_miss) +
         "<div class='kpi %s'><b>%d</b><span>중요 fact 중 문서에 안 쓰인 것(%d개 중)</span></div>" % ("bad" if unc else "ok", len(unc), len(must)) +
         "<div class='kpi %s'><b>%d</b><span>필수 fact 누락·부분 전달</span></div>" % ("bad" if covv else "ok", len(covv)) + "</div>",
         "<p class='muted'>판정 방법: (1) 명세 JSON의 영역별 필수 요소마다 원자료 출처를 찾아 데이터가 있으면 문서 반영 여부를, 없으면 '확인 필요' 표시 여부를 확인. "
         "(2) 원자료에서 중요도 2 이상으로 뽑힌 fact가 문서 어느 문장에 인용됐는지 거꾸로 확인. 칸을 누르면 해당 문장으로 이동.</p>"]
    areas = OrderedDict()
    for c in s["checklist"]:
        areas.setdefault(c["section"], []).append(c)
    h.append("<div class='grid'>")
    for sec, items in areas.items():
        h.append("<div class='area'><div class='aname'>%s</div>" % _esc(SECTIONS.get(sec, sec)))
        for c in items:
            if c["status"] == "data":
                cls, lab = st_cls.get(c.get("in_note"), "ok"), ("반영" if c.get("in_note") == "covered" else "누락")
            elif c["status"] == "gap":
                cls, lab = "warn", "확인 필요"
            else:
                cls, lab = "na", "해당 없음" if not c["required"] else "-"
            fid = c["fact_ids"][0] if c["fact_ids"] else None
            target = next((x["sid"] for x in s["sents"] if fid and fid in x.get("cites", [])), None)
            tag = "a href='#%s'" % target if target else "span"
            h.append("<%s class='cell %s' title='%s · %s · 근거 %s'>%s<small>%s%s</small></%s>" % (
                tag, cls, _esc(c["label"]), lab, ", ".join(c["fact_ids"][:4]) or "-", _esc(c["label"]), lab, "" if c["required"] else " · 선택",
                "a" if target else "span"))
        h.append("</div>")
    h.append("</div>")
    if unc:
        h.append("<details open><summary>문서에 안 쓰인 중요 fact %d개</summary><ul>%s</ul></details>" % (
            len(unc), "".join("<li>%s %s</li>" % (f.fid, _esc(f.text)) for f in unc)))
    h.append("<div class='legend'><span class='cell ok'>반영</span><span class='cell warn'>데이터 없음·확인 필요</span>"
             "<span class='cell bad'>문서에서 누락</span><span class='cell na'>조건부·해당 없음</span></div></section>")
    return "\n".join(h)


REPORT_CSS = """
:root{--ok:#2e7d32;--okb:#e8f5e9;--warn:#b26a00;--warnb:#fff4e0;--bad:#c62828;--badb:#fdecea;--na:#757575;--nab:#f2f2f2;--ink:#222;--line:#ddd}
body{font-family:-apple-system,'Malgun Gothic','Apple SD Gothic Neo',sans-serif;max-width:1180px;margin:20px auto;padding:0 16px;color:var(--ink);line-height:1.55;background:#fff}
h1{font-size:22px;border-bottom:2px solid #333;padding-bottom:6px}h2{font-size:18px;margin-top:30px;border-bottom:1px solid #ccc;padding-bottom:4px}h3{font-size:15px;margin:16px 0 4px}
.meta{color:#555;font-size:13px}.muted{color:#777;font-size:12px}
ul.sents{list-style:none;padding-left:0}ul.sents li{padding:3px 0;border-bottom:1px dashed #eee}
.tip{position:relative;cursor:help}.tip:focus{outline:2px solid #90caf9}
.num{border-bottom:2px dotted #1565c0;padding:0 1px}.num.calc{border-bottom-color:#b26a00}
.tip .pop{display:none;position:absolute;z-index:20;left:0;top:1.5em;min-width:min(340px,calc(100vw - 24px));max-width:min(720px,calc(100vw - 24px));box-sizing:border-box;overflow-x:auto;background:#fff;border:1px solid #999;
 box-shadow:0 4px 14px rgba(0,0,0,.18);padding:8px 10px;font-size:12px;font-weight:normal;color:#222;white-space:normal}
.chart .cap{font-size:11px;color:#555;margin:2px 0 8px 0}
.tip .pop.wide{min-width:min(520px,calc(100vw - 24px))}
.tip .pop.fixed{position:fixed;top:0;left:0}
.tip:hover>.pop,.tip:focus>.pop,.tip:focus-within>.pop{display:block}
.badge{display:inline-block;width:18px;height:18px;line-height:18px;text-align:center;border-radius:9px;font-size:11px;color:#fff;margin-left:4px}
.badge.ok{background:var(--ok)}.badge.bad{background:var(--bad)}.badge.warn{background:var(--warn)}.badge.fix{background:#9e9e9e}
table{border-collapse:collapse;margin:6px 0;font-size:13px}th,td{border:1px solid var(--line);padding:3px 6px;vertical-align:top}th{background:#f4f4f4}
table.raw{font-size:11px;margin:4px 0}table.raw td,table.raw th{padding:2px 4px}
.fact{margin-top:6px;border-top:1px solid #eee;padding-top:4px}
.panel{border:1px solid #cfd8dc;background:#fafcfd;padding:4px 14px 12px;border-radius:6px}
.kpis{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}.kpi{flex:1 1 160px;border-radius:6px;padding:8px 10px}.kpi b{font-size:22px;display:block}
.kpi span{font-size:12px}.kpi.ok{background:var(--okb);color:var(--ok)}.kpi.warn{background:var(--warnb);color:var(--warn)}.kpi.bad{background:var(--badb);color:var(--bad)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:8px}.area{border:1px solid var(--line);border-radius:6px;padding:6px;background:#fff}
.aname{font-weight:bold;font-size:13px;margin-bottom:4px}
.cell{display:inline-block;margin:2px;padding:2px 6px;border-radius:4px;font-size:12px;text-decoration:none;color:inherit}
.cell small{display:block;font-size:10px;opacity:.8}.cell.ok{background:var(--okb);color:var(--ok)}.cell.warn{background:var(--warnb);color:var(--warn)}
.cell.bad{background:var(--badb);color:var(--bad);font-weight:bold}.cell.na{background:var(--nab);color:var(--na)}
.legend{margin-top:8px}.legend .cell{font-size:11px}
.charts{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:8px}.chart{border:1px solid #eee;padding:4px}
.sev-high{color:var(--bad);font-weight:bold}.sev-medium{color:var(--warn);font-weight:bold}.sev-info{color:#555}
.brief p{font-size:15px;line-height:1.8}.note{background:#f6f8fa;border-left:3px solid #90a4ae;padding:6px 10px;font-size:13px;color:#444}
"""


def lit_src_pop(e):
    s = e.get("source", {})
    rows = [("인용", s.get("citation")), ("위치", s.get("location")), ("대상", e.get("population")), ("n", e.get("n")),
            ("원문", e.get("quote")), ("원 단위", e.get("unit")), ("DOI", s.get("doi")), ("PMID", s.get("pmid"))]
    body = "".join("<div><b>%s</b> %s</div>" % (k, _esc(v)) for k, v in rows if v not in (None, ""))
    if s.get("url"):
        body += "<div><b>URL</b> <a href='%s' target='_blank' rel='noopener'>%s</a></div>" % (_esc(s["url"]), _esc(s["url"]))
    return "<span class='tip num hit' tabindex='0'>%s<span class='pop wide'>%s</span></span>" % (_esc(lit_cite(e)), body)


def lit_html(series):
    if not REF_LIT["entries"]:
        return "<p class='muted'>문헌 기준값 파일(%s) 없음: 그래프의 띠는 프로파일 고정 범위입니다.</p>" % REF_LIT_FILE
    H = ["<h4>문헌 기준과 최근 값 비교</h4><p class='muted'>같은 재태주수·생후시간 조건에 맞는 문헌을 모두 보여줍니다. 문헌 이름에 마우스를 올리면 원문 위치·대상·인용 문구가 보입니다. "
         "그래프의 띠는 표에서 위에 있는 문헌(파일 순서)으로 그립니다.</p>",
         "<table><tr><th>항목</th><th>최근 값</th><th>문헌</th><th>종류</th><th>적용 조건</th><th>기준</th><th>판정</th></tr>"]
    for x in series:
        rows = lit_rows(x)
        if not rows and lit_has(x["key"]):
            H.append("<tr><td>%s</td><td>%s %s</td><td colspan='5' class='muted'>맞는 문헌 없음</td></tr>" % (_esc(x["label"]), fmt_num(x["v"][-1]), _esc(x["unit"])))
        for e, kind, cond, rng, jd in rows:
            cls = "warn" if ("해당" in jd or "범위 위" in jd or "범위 아래" in jd) else ""
            H.append("<tr><td>%s</td><td>%s %s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td class='%s'>%s</td></tr>" % (
                _esc(x["label"]), fmt_num(x["v"][-1]), _esc(x["unit"]), lit_src_pop(e), _esc(kind), _esc(cond), _esc(rng), cls, _esc(jd)))
    H.append("</table>")
    return "".join(H)


def report_html(ctx, res, series, md):
    s, q = res["summary"], res["parent"]
    F = {f.fid: f for f in res["facts"]}
    vd = {v["sid"]: v for v in s["verdicts"]}
    sp = PROFILE.get("_spec") or {}
    hrs = (ctx.t0 - ctx.win_start).total_seconds() / 3600.0
    H = ["<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
         "<title>NICU 보고서 %s %s</title><style>%s</style></head><body>" % (_esc(ctx.subject_id), _esc(fmt_t(ctx.t0)), REPORT_CSS),
         "<h1>환자별 NICU 보고서 — 환자 ID %s</h1>" % _esc(ctx.subject_id),
         "<div class='meta'>요약 기준시각 %s · 집계 %s ~ %s(%s시간) · 명세 %s v%s · stay %s · 기준 자료 ID %s · 생성 %s</div>" % (
             fmt_t(ctx.t0), fmt_t(ctx.win_start), fmt_t(ctx.t0), fmt_num(round(hrs, 1)), _esc(sp.get("spec_id", "-")), _esc(sp.get("version", "-")),
             _esc(ctx.stay_id), _esc(res.get("ssot_id", "-")), time.strftime("%Y-%m-%d %H:%M")),
         "<p class='note'>자동 생성 초안이며 의료진 검토가 필요합니다. <b>밑줄 친 숫자</b>에 마우스를 올리면 그 값을 만든 원자료 행이, "
         "문장 끝 <b>✔·⚠</b>에 올리면 검증 판정·근거 fact·출처 행이 보입니다. 파란 점선은 원자료 한 행과 일치, 주황 점선은 여러 행으로 계산한 값입니다.</p>",
         coverage_panel_html(res)]
    H.append("<h2>1. 1일 요약지 — %s</h2>" % _esc(PROFILE["documents"]["summary"]["title"].replace(" (자동 생성 초안, 의료진 검토 필요)", "")))
    cur = None
    for x in s["sents"]:
        if x["section"] == CDSS_SEC():
            continue
        if x["section"] != cur:
            if cur is not None:
                H.append("</ul>")
            cur = x["section"]
            H.append("<h3>%s</h3><ul class='sents'>" % _esc(SECTIONS.get(cur, cur)))
        H.append(sentence_html(ctx, x, vd.get(x["sid"]), F))
    if cur is not None:
        H.append("</ul>")
    H.append("<h3>검사 추이(ICU 입실 이후, 기록 2개 이상)</h3>")
    if series:
        H.append("<div class='charts'>%s</div>" % "".join("<div class='chart'>%s<div class='cap'>%s</div></div>" % (trend_svg(x), _esc(ref_caption(x))) for x in series))
        H.append(lit_html(series))
        tm = trend_markdown(ctx, series)
        tm = tm[:next((i for i, l in enumerate(tm) if l.startswith("문헌 기준") or l.startswith("- 문헌 기준값")), len(tm))]
        groups, g = [], []
        for l in tm + [""]:
            if l.startswith("|"):
                g.append(l)
            elif g:
                groups.append(g)
                g = []
        for g in groups:   # 표마다 따로 변환(붙이면 두 표가 한 표로 합쳐짐)
            H.append(md_to_html("\n".join(g), "", []).split("<body>", 1)[1].rsplit("</body>", 1)[0])
    else:
        H.append("<p class='muted'>기록이 2개 이상인 추이 항목 없음.</p>")
    H.append("<h2>2. CDSS 알림(규칙 기반 + 트리 모델, 인계문과 별도)</h2>")
    if res["alerts"]:
        order = {"high": 0, "medium": 1, "info": 2}
        H.append("<table><tr><th>심각도</th><th>알림</th><th>내용</th><th>권고 행동(검토 제안)</th><th>근거</th><th>규칙·모델</th><th>참고</th></tr>")
        for a in sorted(res["alerts"], key=lambda z: order.get(z["severity"], 3)):
            ev = "".join("<span class='tip num hit' tabindex='0'>%s<span class='pop wide'><b>%s</b> %s%s</span></span> " % (
                i, i, _esc(F[i].text), _rows_html(raw_rows_of(ctx, F[i]))) for i in a["trigger_fact_ids"] if i in F)
            H.append("<tr><td class='sev-%s'>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
                a["severity"], a["severity"].upper(), _esc(a["title"]), _esc(a["text"]), _esc(a.get("action", "-")), ev, _esc(a["rule_id"]), _esc(a["reference"])))
        H.append("</table><p class='muted'>권고 행동은 병원 프로토콜 확인을 돕는 검토 제안이며 처방이 아님(프로파일 cdss.actions 에서 수정).</p>")
    else:
        H.append("<p>규칙 기반 알림 없음.</p>")
    sec3 = md.split("## 3. 검증 정보", 1)[1].split("## 4.", 1)[0] if "## 3. 검증 정보" in md else ""
    H.append("<h2>3. 검증 정보</h2>" + md_to_html(sec3, "", []).split("<body>", 1)[1].rsplit("</body>", 1)[0])
    H.append("<h2>4. 부모님 요약지(보호자 Daily Briefing, 간호사 확인 전)</h2><div class='brief'>")
    P = PROFILE["documents"]["parent"]
    qvd = {v["sid"]: v for v in q["verdicts"]}
    H.append("<p class='muted'>%s 기준 · 범위 %s</p>" % (_esc(_kor_time(ctx.t0)), "지난 24시간" if hrs >= 23.5 else "입실 후 %s시간" % fmt_num(round(hrs))))
    for sec, title in PARENT_SECTIONS.items():
        xs = [x for x in q["sents"] if x["section"] == sec]
        if not xs:
            continue
        H.append("<h3>%s</h3><p>" % _esc(title))
        for x in xs:
            v = qvd.get(x["sid"])
            lab, cls = ("고정 문구", "fix") if x.get("fixed") else VERDICT_KO.get((v or {}).get("label"), ("-", "warn"))
            cf = [F[c] for c in x.get("cites", []) if c in F]
            pop = "<b>검증: %s</b>%s" % (lab, "".join("<div class='fact'><b>%s</b> %s</div>%s" % (f.fid, _esc(f.text), _rows_html(raw_rows_of(ctx, f, 6))) for f in cf))
            H.append("<span class='tip' tabindex='0'>%s<span class='pop wide'>%s</span></span> " % (_esc(_strip_cites(x["text"])), pop))
        H.append("</p>")
    H.append("<p class='muted'>%s 기준 자료 ID %s</p></div>" % (_esc(P.get("notice", "")), _esc(res.get("ssot_id", "-"))))
    H.append("""<script>
(function(){function place(t){var p=t.querySelector(':scope>.pop');if(!p)return;p.classList.add('fixed');p.style.display='block';
var r=t.getBoundingClientRect(),w=p.offsetWidth,h=p.offsetHeight,vw=window.innerWidth,vh=window.innerHeight;
var x=Math.min(Math.max(8,r.left),vw-w-8),y=r.bottom+4;if(y+h>vh-8)y=Math.max(8,r.top-h-4);p.style.left=x+'px';p.style.top=y+'px';}
function hide(t){var p=t.querySelector(':scope>.pop');if(p){p.style.display='';}}
document.querySelectorAll('.tip').forEach(function(t){t.addEventListener('mouseenter',function(){place(t)});t.addEventListener('focus',function(){place(t)});
t.addEventListener('mouseleave',function(){if(document.activeElement!==t)hide(t)});t.addEventListener('blur',function(){hide(t)});});
window.addEventListener('scroll',function(){document.querySelectorAll('.pop.fixed').forEach(function(p){p.style.display='';});},{passive:true});})();
</script>""")
    H.append("</body></html>")
    return "\n".join(H)


def write_report(ctx, res, out_dir):
    """환자별 통합 보고서를 stay 폴더와 reports/ 폴더에 md·html로 저장. 반환: md 경로"""
    series = trend_series(ctx)
    md = report_markdown(ctx, res, series)
    html = report_html(ctx, res, series, md)
    rd = os.path.join(out_dir, "reports")
    os.makedirs(rd, exist_ok=True)
    base = "report_%s_%s" % (ctx.subject_id, pd.Timestamp(ctx.t0).strftime("%m%d_%H%M"))
    paths = []
    for d in (rd, os.path.join(out_dir, "stay_%s_d%d" % (ctx.stay_id, ctx.day_index))):
        os.makedirs(d, exist_ok=True)
        for ext, body in (("md", md), ("html", html)):
            p = os.path.join(d, "%s.%s" % (base if d == rd else "report", ext))
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(body)
            paths.append(p)
    return paths[0]


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
    def __init__(self, data_dir, dur_csv=None, verbose=True, profile=None, cohort="nicu", spec=None):
        prof = load_profile(profile)
        spec = find_default_spec() if spec is None else (None if spec in ("none", "") else spec)
        if spec:
            prof = spec_to_profile(prof, load_spec(spec), spec)
            validate_profile(prof)
        apply_profile(prof)
        load_ref_lit(os.environ.get("KCD_REF_LIT") or PROFILE.get("ref_literature"))   # 생략 시 같은 폴더의 neonatal_ref_literature.json
        if verbose:
            print("[ref] 문헌 기준값 %s" % ("%d개 (%s)" % (len(REF_LIT["entries"]), os.path.basename(REF_LIT["path"])) if REF_LIT["entries"] else "없음 → 프로파일 고정 범위 사용"))
        if verbose and PROFILE.get("_spec"):
            S = PROFILE["_spec"]
            print("[spec] %s v%s (%s): 영역 %d개, 필수 요소 %d개" % (S["spec_id"], S["version"], os.path.basename(S["path"] or ""), len(S["required_information"]),
                                                       sum(1 for c in PROFILE["documents"]["summary"]["checklist"] if c["required"])))
            if S["unmatched"]:
                print("[spec] 데이터 출처를 연결하지 못한 요소(항상 '확인 필요'로 표시): " + "; ".join(S["unmatched"]))
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
        if not getattr(self, "_births", None):
            S = dict(SAMPLING_DEFAULT)
            S.update(PROFILE.get("sampling", {}))
            self._births = birth_times(self, S)
        ctx.birth, ctx.birth_src = self._births.get(str(stay_id), (None, "출생 시각 확인 불가"))
        risk_alerts(ctx, facts, alerts, ctx.birth)
        finalize_facts(ctx, facts, alerts)
        audit.log("cdss", {"facts": [f.fid for f in facts]}, {"alerts": [a["rule_id"] for a in alerts]})
        judge = llm if (llm is not None and llm.ok) else None
        res = {"ctx": ctx, "facts": facts, "alerts": alerts, "ssot_id": ssot_id(ctx, facts)}
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
                    sid_ = res["ssot_id"]
                    pp, pv = os.path.join(sd, "parent_%s.md" % mode), os.path.join(sd, "parent_%s_nurse_check.md" % mode)
                    old = [supersede(pp, sid_), supersede(pv, sid_)]
                    with open(pp, "w", encoding="utf-8") as f:
                        f.write(parent_markdown(ctx, sents, facts, alerts, sid_=sid_))
                    with open(pv, "w", encoding="utf-8") as f:
                        f.write(parent_markdown(ctx, sents, facts, alerts, vd, cov, nurse=True, sid_=sid_, metrics_=m))
                    with open(os.path.join(sd, "parent_%s_meta.json" % mode), "w", encoding="utf-8") as f:
                        json.dump({"ssot_id": sid_, "status": "간호사 확인 전 초안", "auto_send": False, "superseded": [x for x in old if x],
                                   "doctor_flags": [{"item": t, "facts": ids} for t, ids in doctor_flags(ctx, facts, alerts)]}, f, ensure_ascii=False, indent=1)
                with open(os.path.join(sd, "verify_%s_%s.json" % (doc, mode)), "w", encoding="utf-8") as f:
                    json.dump({"metrics": m, "verdicts": vd, "coverage": cov, "checklist": chk, "sentences": sents}, f, ensure_ascii=False, indent=1, default=str)
        allm["secs"] = round(time.time() - t1, 2)
        allm["ssot_id"] = res["ssot_id"]
        if sd and write and all(d in res for d in DOCS):
            res["report"] = write_report(ctx, res, out_dir)
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
    return Pipeline(a.data, a.dur_csv, profile=a.profile, cohort=a.cohort, spec=a.spec)


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
        print("\n" + parent_markdown(r["ctx"], q["sents"], r["facts"], r["alerts"], sid_=r["ssot_id"]))
        if r.get("report"):
            print("[report] %s (html 같은 이름)" % r["report"])
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


# =============================================================================
# 9. 임상 평가용 표본 추출 (sample): 생후 28일 이하 NICU 환아의 (person_id, 평가 시점) 무작위 추출
#    평가 시점 = 매일 cutoff 시각(기본 07:00). 생성은 이 시각 이전 기록만, 특히 직전 24시간 기록으로 한다.
#    직전 24시간 데이터가 부족해 Core 인계 항목을 만들 수 없는 시점은 제외한다. 기준값은 프로파일 sampling 에서 수정.
# =============================================================================

SAMPLING_DEFAULT = {
    "max_age_days": 28,                 # 평가 시점 생후 일수 상한(출생일 = 0일)
    "birth_columns": ["dob", "birth_datetime", "birthdate", "birth_time", "date_of_birth"],
    "birth_proxy_max_hours": 48,        # 생년월일 컬럼이 없으면: 신생아(나이 0세, 또는 NICU 병동이면서 나이 결측)이고 입원→NICU 입실 48시간 이내일 때만 입원 시각을 출생 시각으로 근사
    "missing_age_as_neonate": True,     # K-MIMIC은 NICU 환아의 anchor_age 를 비워 둠 → NICU 병동 + 나이 결측이면 신생아로 간주
    "min_icu_hours_before": 24,         # 평가 시점 직전 24시간 전체가 같은 NICU 재실 기간 안
    "min_hr": 6, "min_spo2": 6, "min_temp": 2, "min_bp": 1,   # 직전 24시간 기록 건수 하한(호흡·순환·체온 항목)
    "weight_within_days": 7,            # 평가 시점 이전 7일 안에 체중 기록(현재 체중·mL/kg 계산)
    "min_intake_records": 1,            # 직전 24시간 섭취(수액·수유) 기록(영양·수액 항목)
    "min_urine_records": 1,             # 직전 24시간 소변 기록(배설 항목)
    "exclude_after_death": True,        # 사망 시각 이후 시점 제외
    "one_per_person": True,             # 한 환아당 한 시점(같은 환아의 여러 날이 표본을 차지하지 않게)
    "dol_strata": [[0, 3], [4, 7], [8, 14], [15, 28]],   # --stratify 시 생후일 구간별 균등 추출
}


def _np():
    import numpy as np
    return np


def _times_by(df, keycol, tcol, mask=None):
    """hadm별 정렬된 시각 배열 dict."""
    np = _np()
    if df is None or df.empty or tcol not in df.columns:
        return {}
    d = df if mask is None else df[mask]
    t = to_dt(d[tcol])
    d = d.assign(_t=t).dropna(subset=["_t"])
    return {k: np.sort(g["_t"].values) for k, g in d.groupby(keycol)}


def _count(arr, a, b):
    """(a, b] 구간 건수."""
    np = _np()
    if arr is None or len(arr) == 0:
        return 0
    return int(np.searchsorted(arr, np.datetime64(b), side="right") - np.searchsorted(arr, np.datetime64(a), side="right"))


def _last_before(arr, b):
    np = _np()
    if arr is None or len(arr) == 0:
        return None
    i = np.searchsorted(arr, np.datetime64(b), side="right")
    return pd.Timestamp(arr[i - 1]) if i > 0 else None


def birth_times(p, S):
    """stay별 (출생 시각, 출처). 생년월일 컬럼 우선, 없으면 입원 시각 근사(조건부)."""
    pats = p.kd.get("PATIENTS")
    adm = p.kd.get("ADMISSIONS")
    col = next((c for c in S["birth_columns"] if c in pats.columns), None) if not pats.empty else None
    bmap = {}
    if col:
        bt = to_dt(pats[col])
        bmap = dict(zip(pats["subject_id"], bt))
    amap = dict(zip(adm["hadm_id"], to_dt(adm["admittime"]))) if (not adm.empty and "admittime" in adm.columns) else {}
    out = {}
    for _, r in p.cands.iterrows():
        b = bmap.get(r["subject_id"]) if col else None
        if b is not None and pd.notna(b):
            out[r["stay_id"]] = (pd.Timestamp(b), "PATIENTS.%s" % col)
            continue
        age = to_num(pd.Series([r.get("anchor_age")])).iloc[0]
        a = amap.get(r["hadm_id"])
        neo = (pd.notna(age) and age < 1) or (pd.isna(age) and S.get("missing_age_as_neonate", True) and bool(r.get("_unit_match", False)))
        if neo and a is not None and pd.notna(a) and \
                0 <= (pd.Timestamp(r["intime_dt"]) - pd.Timestamp(a)).total_seconds() <= S["birth_proxy_max_hours"] * 3600:
            out[r["stay_id"]] = (pd.Timestamp(a), "입원 시각 근사" + ("(나이 결측 NICU)" if pd.isna(age) else ""))
        else:
            out[r["stay_id"]] = (None, "출생 시각 확인 불가")
    return out


def expand_dirs(spec):
    """--data 값을 폴더 목록으로. 쉼표 구분과 와일드카드(*) 허용. ICUSTAYS 가 있는 폴더만."""
    import glob as _g
    out = []
    for part in str(spec).split(","):
        part = os.path.expanduser(part.strip())
        cands = sorted(_g.glob(part)) if any(ch in part for ch in "*?[") else [part]
        for d in cands:
            if os.path.isdir(d) and any(os.path.exists(os.path.join(d, f)) for f in ("ICUSTAYS.csv", "icustays.csv", "ICUSTAYS.csv.gz", "ICUSTAYS.parquet")):
                out.append(d)
            elif os.path.isdir(d):
                out += sorted(x for x in _g.glob(os.path.join(d, "*")) if os.path.exists(os.path.join(x, "ICUSTAYS.csv")))
    return list(OrderedDict.fromkeys(out))


def _sample_one_dir(a, data_dir):
    """한 데이터 폴더에서 적격 평가 시점과 단계별 흐름 계산. 반환: (rows, steps, flow)"""
    S = dict(SAMPLING_DEFAULT)
    p = Pipeline(data_dir, a.dur_csv, verbose=False, profile=a.profile, cohort=a.cohort, spec=a.spec)
    S.update(PROFILE.get("sampling", {}))
    C = p.cands.copy()
    hadms = C["hadm_id"].tolist()
    print("[sample] %s: 코호트 %d stay" % (os.path.basename(os.path.normpath(data_dir)), len(C)))
    chart = p.kd.get("CHARTEVENTS", hadms)
    if not chart.empty:
        chart = chart.assign(concept=chart["itemid"].map(lambda i: p.item_map.get(i, (None,))[0]))
    want = {"hr": ("hr",), "spo2": ("spo2",), "temp": ("temp",), "bp": ("sbp", "map", "dbp"), "weight": ("weight",),
            "resp": tuple(VENT_CONCEPTS | {"fio2", "o2_flow"})}
    T = {k: _times_by(chart, "hadm_id", "charttime", chart["concept"].isin(v)) if not chart.empty else {} for k, v in want.items()}
    inp = p.kd.get("INPUTEVENTS", hadms)
    T["intake"] = _times_by(inp, "hadm_id", "starttime")
    out = p.kd.get("OUTPUTEVENTS", hadms)
    if not out.empty:
        out = out.assign(concept=out["itemid"].map(lambda i: p.item_map.get(i, (None,))[0]))
        T["urine"] = _times_by(out, "hadm_id", "charttime", out["concept"] == "urine")
    else:
        T["urine"] = {}
    T["lab"] = _times_by(p.kd.get("LABEVENTS", hadms), "hadm_id", "charttime")
    T["emar"] = _times_by(p.kd.get("EMAR", hadms), "hadm_id", "charttime")
    adm = p.kd.get("ADMISSIONS")
    death = dict(zip(adm["hadm_id"], to_dt(adm["deathtime"]))) if (not adm.empty and "deathtime" in adm.columns) else {}
    births = birth_times(p, S)
    steps = ["1. NICU 코호트 후보 시점(매일 %02d:00)" % int(PROFILE["daily"]["cutoff_hour"]), "2. 출생 시각 확인 가능",
             "3. 직전 %d시간 전체가 NICU 재실 중" % S["min_icu_hours_before"], "4. 생후 %d일 이하" % S["max_age_days"], "5. 사망 이전 시점",
             "6. 활력징후 충분(HR≥%d, SpO2≥%d, 체온≥%d, 혈압≥%d)" % (S["min_hr"], S["min_spo2"], S["min_temp"], S["min_bp"]),
             "7. %d일 내 체중 기록" % S["weight_within_days"], "8. 섭취 기록≥%d" % S["min_intake_records"], "9. 소변 기록≥%d" % S["min_urine_records"]]
    flow = [[0, set()] for _ in steps]
    rows = []
    for _, r in C.iterrows():
        sid, h, person = r["stay_id"], r["hadm_id"], r["subject_id"]
        intime, outtime = pd.Timestamp(r["intime_dt"]), pd.Timestamp(r["outtime_dt"])
        t0s = [t for t in day_t0s(r) if t.hour == int(PROFILE["daily"]["cutoff_hour"]) and t.minute == 0]
        b, bsrc = births.get(sid, (None, None))
        for k, t0 in enumerate(day_t0s(r), 1):
            if t0 not in t0s:
                continue  # 퇴실 시각 같은 비정규 시점은 평가 시점에서 제외
            ws = t0 - pd.Timedelta(hours=24)
            ok = [True]
            ok.append(b is not None)
            ok.append(ok[-1] and (t0 - intime).total_seconds() >= S["min_icu_hours_before"] * 3600 and t0 <= outtime)
            age_days = int((t0 - b).total_seconds() // 86400) if b is not None else None
            ok.append(ok[-1] and age_days is not None and 0 <= age_days <= S["max_age_days"])
            dt_ = death.get(h)
            ok.append(ok[-1] and not (S["exclude_after_death"] and dt_ is not None and pd.notna(dt_) and pd.Timestamp(dt_) <= t0))
            n = {k2: _count(T[k2].get(h), ws, t0) for k2 in ("hr", "spo2", "temp", "bp", "intake", "urine", "lab", "emar", "resp")}
            ok.append(ok[-1] and n["hr"] >= S["min_hr"] and n["spo2"] >= S["min_spo2"] and n["temp"] >= S["min_temp"] and n["bp"] >= S["min_bp"])
            lw = _last_before(T["weight"].get(h), t0)
            wdays = round((t0 - lw).total_seconds() / 86400.0, 1) if lw is not None else None
            ok.append(ok[-1] and wdays is not None and wdays <= S["weight_within_days"])
            ok.append(ok[-1] and n["intake"] >= S["min_intake_records"])
            ok.append(ok[-1] and n["urine"] >= S["min_urine_records"])
            for i, v in enumerate(ok):
                if v:
                    flow[i][0] += 1
                    flow[i][1].add(person)
            if all(ok):
                rows.append({"person_id": person, "index_date": t0.strftime("%Y-%m-%d"), "index_datetime": str(t0), "data_dir": data_dir, "stay_id": sid, "hadm_id": h,
                             "kcd_day": k, "age_days": age_days, "birth_source": bsrc, "icu_day": (t0.normalize() - intime.normalize()).days + 1,
                             "n_hr_24h": n["hr"], "n_spo2_24h": n["spo2"], "n_temp_24h": n["temp"], "n_bp_24h": n["bp"], "n_resp_24h": n["resp"],
                             "n_intake_24h": n["intake"], "n_urine_24h": n["urine"], "n_lab_24h": n["lab"], "n_emar_24h": n["emar"], "weight_days_ago": wdays})
    return rows, steps, flow


def cmd_sample(a):
    dirs = expand_dirs(a.data)
    if not dirs:
        raise SystemExit("데이터 폴더를 찾지 못했습니다: %s" % a.data)
    print("[sample] 데이터 폴더 %d개" % len(dirs))
    rows, steps, flow, skipped = [], None, None, []
    for d in dirs:
        try:
            r, st, fl_ = _sample_one_dir(a, d)
        except SystemExit as e:
            skipped.append((d, str(e)[:80]))
            continue
        rows += r
        if flow is None:
            steps, flow = st, fl_
        else:
            for i, f in enumerate(fl_):
                flow[i][0] += f[0]
                flow[i][1] |= f[1]
    if flow is None:
        raise SystemExit("모든 폴더에서 NICU 코호트가 0건입니다: %s" % skipped[:3])
    S = dict(SAMPLING_DEFAULT)
    S.update(PROFILE.get("sampling", {}))
    elig = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    fl = pd.DataFrame([{"step": s, "candidate_days": f[0], "persons": len(f[1])} for s, f in zip(steps, flow)])
    if skipped:
        print("[sample] 코호트 0건으로 건너뛴 폴더 %d개" % len(skipped))
    if not elig.empty:
        dup = elig.groupby("person_id")["data_dir"].nunique()
        if (dup > 1).any():
            print("[sample] 주의: 여러 폴더에 걸친 person_id %d명(한 환아당 1시점 규칙은 person_id 기준으로 적용)" % int((dup > 1).sum()))
    fl.to_csv(os.path.join(a.out, "sample_flow.csv"), index=False)
    print("\n[선정 흐름]\n" + fl.to_string(index=False))
    if elig.empty:
        print("\n조건을 만족하는 평가 시점이 없습니다. sample_flow.csv 에서 탈락 단계를 확인하고 프로파일 sampling 기준을 조정하세요.")
        return
    rnd = random.Random(a.seed)
    if S["one_per_person"]:
        elig = elig.sample(frac=1.0, random_state=rnd.randrange(10 ** 9)).drop_duplicates("person_id").reset_index(drop=True)
    strata = S["dol_strata"]
    elig["dol_stratum"] = elig["age_days"].map(lambda d: next(("%d-%d일" % (lo, hi) for lo, hi in strata if lo <= d <= hi), "기타"))
    n = a.n
    if a.stratify:
        per = max(1, n // len(strata))
        parts = []
        for lo, hi in strata:
            g = elig[elig["dol_stratum"] == "%d-%d일" % (lo, hi)]
            parts.append(g.sample(n=min(per, len(g)), random_state=a.seed) if len(g) else g)
        smp = pd.concat(parts)
    else:
        smp = elig.sample(n=min(n, len(elig)), random_state=a.seed)
    smp = smp.sort_values(["index_date", "person_id"]).reset_index(drop=True)
    smp.insert(0, "sample_no", range(1, len(smp) + 1))
    smp.to_csv(os.path.join(a.out, "nicu_sample.csv"), index=False)
    elig.to_csv(os.path.join(a.out, "nicu_eligible_one_per_person.csv"), index=False)
    with open(os.path.join(a.out, "nicu_sample_commands.sh"), "w", encoding="utf-8") as f:
        f.write("# 표본별 1일 요약지·부모님 설명지 생성 명령\n")
        for _, x in smp.iterrows():
            f.write("python kcd_nicu.py run --data %s --stay %s --day %d --out %s%s\n" % (x["data_dir"], x["stay_id"], x["kcd_day"], os.path.join(a.out, "generated"),
                                                                                   (" --profile %s" % a.profile) if a.profile else ""))
    print("\n[표본] 적격 환아 %d명 중 %d명 추출 (seed=%d%s)" % (len(elig), len(smp), a.seed, ", 생후일 층화" if a.stratify else ""))
    print(smp["dol_stratum"].value_counts().sort_index().to_string())
    print("\n[out] %s/nicu_sample.csv, sample_flow.csv, nicu_eligible_one_per_person.csv, nicu_sample_commands.sh" % a.out)
    print("주의: nicu_sample.csv 에는 person_id 와 날짜가 있으므로 분석환경 밖으로 반출하지 마세요. 외부 공유는 sample_flow.csv 집계만.")


# =============================================================================
# 10. 트리 모델 CDSS: (1) 환자 악화 조기경보 (2) 검사치 임계치 알림
#   학습: python kcd_nicu.py train-risk --data "DIR/*" --out risk_models   (분석환경 안에서만)
#   추론: run 시 risk_models/ 에 모델이 있으면 CDSS 장에 예측 위험·주요 요인·권고 행동을 추가
#   시점 = 매일 07:00(1일 요약 기준 시각), 입력 = 직전 24시간(검사는 48시간) 기록, 예측 = 다음 24시간
# =============================================================================

RISK_MODEL_DIR = "risk_models"
RISK_TASKS = OrderedDict([
    ("deterioration", {"title": "환자 악화 조기경보", "label": "다음 24시간 안에 사망·새 침습적 인공호흡·승압제 신규 시작 중 하나"}),
    ("critical_lab", {"title": "검사치 임계치 알림", "label": "다음 24시간 안에 검사 위험값(K·Na·혈당·빌리루빈·Hb·혈소판·pH 등) 발생"}),
])
RISK_VITALS = [("hr", "심박수"), ("rr", "호흡수"), ("spo2", "SpO2"), ("temp", "체온"), ("sbp", "수축기혈압"), ("map", "평균혈압")]
RISK_LABS = [("cr", "Cr"), ("k", "K"), ("na", "Na"), ("glucose", "혈당"), ("tbil", "빌리루빈"), ("crp", "CRP"), ("hb", "Hb"), ("plt", "혈소판"),
             ("wbc", "백혈구"), ("ph", "pH"), ("pco2", "pCO2"), ("lactate", "Lactate")]
RISK_MEDS = [("vasopressor", "승압제"), ("sedation", "진정제"), ("antibiotic", "항생제"), ("insulin", "인슐린"),
             ("respiratory_stimulant", "카페인"), ("nutrition", "정맥영양")]


def risk_feature_names():
    n = []
    for k, _ in RISK_VITALS:
        n += ["%s_min" % k, "%s_max" % k, "%s_mean" % k, "%s_last" % k, "%s_n" % k]
    n += ["desat_n", "brady_n", "tachy_n", "temp_low_n", "temp_high_n", "resp_level", "fio2_last"]
    for k, _ in RISK_LABS:
        n += ["%s_last" % k, "%s_delta" % k]
    n += ["bst_min", "bst_max", "hours_since_lab", "weight_kg", "weight_pct", "intake_ml_kg_day", "urine_ml_kg_h"]
    n += ["med_%s" % k for k, _ in RISK_MEDS]
    n += ["age_days", "icu_day", "preterm"]
    return n


RISK_FEATURE_KO = {"desat_n": "SpO2 85% 미만 횟수", "brady_n": "HR 100 미만 횟수", "tachy_n": "HR 180 초과 횟수", "temp_low_n": "저체온 횟수",
                   "temp_high_n": "고체온 횟수", "resp_level": "호흡 보조 단계", "fio2_last": "최근 FiO2", "bst_min": "BST 최저", "bst_max": "BST 최고",
                   "hours_since_lab": "마지막 검사 후 경과 시간", "weight_kg": "체중", "weight_pct": "입실 후 체중 변화율", "intake_ml_kg_day": "섭취 mL/kg/일",
                   "urine_ml_kg_h": "소변 mL/kg/h", "age_days": "생후일", "icu_day": "NICU 재실일", "preterm": "미숙아 진단"}
for _k, _ko in RISK_VITALS:
    for _s, _sko in (("min", "최저"), ("max", "최고"), ("mean", "평균"), ("last", "최근"), ("n", "기록 수")):
        RISK_FEATURE_KO["%s_%s" % (_k, _s)] = "%s %s" % (_ko, _sko)
for _k, _ko in RISK_LABS:
    RISK_FEATURE_KO["%s_last" % _k] = "%s 최근값" % _ko
    RISK_FEATURE_KO["%s_delta" % _k] = "%s 변화량" % _ko
for _k, _ko in RISK_MEDS:
    RISK_FEATURE_KO["med_%s" % _k] = "%s 투여" % _ko


def _arr(df, by_concept=True):
    import numpy as np
    out = {}
    if df is None or df.empty or "t" not in df.columns:
        return out
    d = df.dropna(subset=["t"])
    if by_concept:
        d = d.dropna(subset=["concept"]) if "concept" in d.columns else d
        for c, g in d.groupby("concept"):
            g = g.sort_values("t")
            out[c] = (g["t"].values.astype("datetime64[ns]"), to_num(g["v"]).values.astype(float) if "v" in g.columns else np.zeros(len(g)))
    return out


def ctx_arrays(ctx):
    """StayContext → 개념별 (시각, 값) 배열. 학습·추론이 같은 함수를 씀."""
    import numpy as np
    A = {"chart": _arr(ctx.chart), "lab": _arr(ctx.lab), "out": _arr(ctx.out)}
    if not ctx.inp.empty:
        d = ctx.inp.dropna(subset=["t"]).sort_values("t")
        A["inp"] = (d["t"].values.astype("datetime64[ns]"), to_num(d["v"]).fillna(0).values.astype(float))
    else:
        A["inp"] = (np.array([], dtype="datetime64[ns]"), np.array([]))
    modes = ctx.chart[(ctx.chart["concept"] == "vent_mode")].dropna(subset=["t"]).sort_values("t") if not ctx.chart.empty else ctx.chart
    A["mode"] = (modes["t"].values.astype("datetime64[ns]"), modes["value"].astype(str).values) if len(modes) else (np.array([], dtype="datetime64[ns]"), np.array([]))
    vent = ctx.chart[ctx.chart["concept"].isin(VENT_CONCEPTS)].dropna(subset=["t"]).sort_values("t") if not ctx.chart.empty else ctx.chart
    A["vent_t"] = vent["t"].values.astype("datetime64[ns]") if len(vent) else np.array([], dtype="datetime64[ns]")
    A["emar"] = {}
    if not ctx.emar.empty:
        for c, g in ctx.emar.dropna(subset=["t"]).groupby("cls"):
            A["emar"][c] = np.sort(g["t"].values.astype("datetime64[ns]"))
    if not ctx.inp.empty and (ctx.inp["concept"] == "in_vasopressor").any():
        extra = ctx.inp[ctx.inp["concept"] == "in_vasopressor"]["t"].dropna().values.astype("datetime64[ns]")
        A["emar"]["vasopressor"] = np.sort(np.concatenate([A["emar"].get("vasopressor", np.array([], dtype="datetime64[ns]")), extra]))
    intub = ctx.proc[ctx.proc["label"].astype(str).str.contains(r"intubat|삽관", case=False, regex=True, na=False)] if not ctx.proc.empty else ctx.proc
    A["intub_t"] = np.sort(intub["t"].dropna().values.astype("datetime64[ns]")) if len(intub) else np.array([], dtype="datetime64[ns]")
    w = []
    if not ctx.chart.empty:
        x = ctx.chart[ctx.chart["concept"] == "weight"].dropna(subset=["v", "t"]).sort_values("t")
        w = [(t, kg(v)) for t, v in zip(x["t"], x["v"])]
    if not w and not ctx.inp.empty and "patientweight" in ctx.inp.columns:
        x = ctx.inp.assign(_w=to_num(ctx.inp["patientweight"])).dropna(subset=["_w", "t"]).sort_values("t")
        w = [(t, kg(v)) for t, v in zip(x["t"], x["_w"])]
    w = [(t, v) for t, v in w if 0.2 <= v <= 300]
    A["weight"] = (np.array([pd.Timestamp(t).to_datetime64() for t, _ in w], dtype="datetime64[ns]"), np.array([v for _, v in w], dtype=float))
    adm = ctx.adm.iloc[0] if len(ctx.adm) else {}
    dt_ = to_dt(pd.Series([adm.get("deathtime") if hasattr(adm, "get") else None])).iloc[0]
    A["death"] = pd.Timestamp(dt_) if pd.notna(dt_) else None
    codes = ctx.dx["icd_code"].astype(str).str.replace(".", "", regex=False).str.upper() if not ctx.dx.empty and "icd_code" in ctx.dx.columns else pd.Series(dtype=str)
    A["preterm"] = float(codes.str.startswith("P07").any())
    return A


def _win(tv, a, b):
    import numpy as np
    t, v = tv
    i = np.searchsorted(t, np.datetime64(a), side="right")
    j = np.searchsorted(t, np.datetime64(b), side="right")
    return t[i:j], v[i:j]


def _cnt(t, a, b):
    import numpy as np
    return int(np.searchsorted(t, np.datetime64(b), side="right") - np.searchsorted(t, np.datetime64(a), side="right"))


def risk_features(A, t0, intime, birth):
    import numpy as np
    nan = float("nan")
    a24, a48 = t0 - pd.Timedelta(hours=24), t0 - pd.Timedelta(hours=48)
    f = []
    ch = A["chart"]
    vals = {}
    for k, _ in RISK_VITALS:
        _, v = _win(ch[k], a24, t0) if k in ch else (None, np.array([]))
        v = v[~np.isnan(v)] if len(v) else v
        vals[k] = v
        f += ([float(v.min()), float(v.max()), float(v.mean()), float(v[-1]), float(len(v))] if len(v) else [nan, nan, nan, nan, 0.0])
    N = PROFILE["neonatal"]
    sp, hr, tp = vals["spo2"], vals["hr"], vals["temp"]
    f += [float((sp < N["desat_spo2"]).sum()) if len(sp) else 0.0, float((hr < N["brady_hr"]).sum()) if len(hr) else 0.0,
          float((hr > N["vitals_normal"]["hr"][1]).sum()) if len(hr) else 0.0,
          float((tp < N["vitals_normal"]["temp"][0]).sum()) if len(tp) else 0.0, float((tp > N["vitals_normal"]["temp"][1]).sum()) if len(tp) else 0.0]
    f.append(float(resp_level_at(A, t0)))
    _, fi = _win(ch["fio2"], a24, t0) if "fio2" in ch else (None, np.array([]))
    f.append(float(fi[-1] if fi[-1] > 1 else fi[-1] * 100) if len(fi) and not np.isnan(fi[-1]) else nan)
    lb = A["lab"]
    last_lab = None
    for k, _ in RISK_LABS:
        if k in lb:
            t, v = _win(lb[k], a48, t0)
            t2, v2 = _win(lb[k], t0 - pd.Timedelta(hours=120), t0)
            f += [float(v[-1]) if len(v) else nan, float(v2[-1] - v2[-2]) if len(v2) >= 2 and len(v) else nan]
            if len(t):
                last_lab = t[-1] if last_lab is None or t[-1] > last_lab else last_lab
        else:
            f += [nan, nan]
    _, b = _win(ch["bst"], a24, t0) if "bst" in ch else (None, np.array([]))
    f += [float(b.min()) if len(b) else nan, float(b.max()) if len(b) else nan]
    f.append(float((np.datetime64(t0) - last_lab) / np.timedelta64(1, "h")) if last_lab is not None else nan)
    wt, wv = A["weight"]
    i = np.searchsorted(wt, np.datetime64(t0), side="right")
    wk = float(wv[i - 1]) if i > 0 else nan
    w0 = wv[np.searchsorted(wt, np.datetime64(intime), side="left")] if len(wt) and np.searchsorted(wt, np.datetime64(intime)) < i else nan
    f += [wk, float((wk - w0) / w0 * 100) if (i > 0 and not np.isnan(w0) and w0) else nan]
    _, iv = _win(A["inp"], a24, t0)
    _, uv = _win(A["out"]["urine"], a24, t0) if "urine" in A["out"] else (None, np.array([]))
    f += [float(np.nansum(iv) / wk) if (len(iv) and wk == wk and wk) else nan, float(np.nansum(uv) / wk / 24.0) if (len(uv) and wk == wk and wk) else nan]
    for k, _ in RISK_MEDS:
        t = A["emar"].get(k)
        f.append(float(_cnt(t, a24, t0) > 0) if t is not None else 0.0)
    f += [float((t0 - birth).total_seconds() / 86400.0) if birth is not None else nan, float((t0 - intime).total_seconds() / 86400.0), A["preterm"]]
    return f


def resp_level_at(A, t0, hours=12):
    """0 없음, 1 산소, 2 비침습, 3 침습(모드 미상 포함)."""
    import numpy as np
    a = t0 - pd.Timedelta(hours=hours)
    if _cnt(A["vent_t"], a, t0):
        mt, mv = A["mode"]
        i = np.searchsorted(mt, np.datetime64(t0), side="right")
        j = np.searchsorted(mt, np.datetime64(a), side="right")
        mode = str(mv[i - 1]) if i > j else None
        return 2 if resp_kind(mode, []) == "noninvasive" else 3
    ch = A["chart"]
    if "o2_flow" in ch:
        _, v = _win(ch["o2_flow"], a, t0)
        if len(v) and np.nanmax(v) > 0:
            return 1
    return 0


def risk_labels(A, t0):
    """다음 24시간 라벨. 반환 dict(deterioration, critical_lab, 세부)."""
    import numpy as np
    b = t0 + pd.Timedelta(hours=24)
    death = A["death"] is not None and t0 < A["death"] <= b
    lvl = resp_level_at(A, t0)
    mt, mv = A["mode"]
    _, fm = _win((mt, mv), t0, b) if len(mt) else (None, np.array([]))
    inv_future = any(resp_kind(str(m), []) == "invasive" for m in fm) or _cnt(A["intub_t"], t0, b) > 0
    new_inv = bool(inv_future and lvl < 3)
    vt = A["emar"].get("vasopressor")
    new_vaso = bool(vt is not None and _cnt(vt, t0, b) > 0 and _cnt(vt, t0 - pd.Timedelta(hours=24), t0) == 0)
    crit = {}
    lb = A["lab"]
    for c, (lo, hi) in PROFILE["critical_values"].items():
        if c in lb:
            _, v = _win(lb[c], t0, b)
            v = v[~np.isnan(v)]
            if len(v) and ((lo is not None and (v <= lo).any()) or (hi is not None and (v >= hi).any())):
                crit[c] = True
    gl = PROFILE["neonatal"]["glucose_low"]
    for src in (lb.get("glucose"), A["chart"].get("bst")):
        if src is not None:
            _, v = _win(src, t0, b)
            if len(v) and (v[~np.isnan(v)] < gl).any():
                crit["glucose_low"] = True
    if "tbil" in lb:
        _, v = _win(lb["tbil"], t0, b)
        if len(v) and (v >= PROFILE["neonatal"]["tsb_alert_mg_dl"]).any():
            crit["tbil_high"] = True
    measured = any(_cnt(lb[c][0], t0, b) for c in lb) if lb else False
    return {"deterioration": int(death or new_inv or new_vaso), "critical_lab": int(bool(crit)), "death": int(death), "new_invasive": int(new_inv),
            "new_vasopressor": int(new_vaso), "crit_types": ",".join(sorted(crit)), "lab_measured_next24": int(measured)}


def build_risk_dataset(a):
    """폴더별로 NICU stay 를 읽어 (stay-일) 특징·라벨 표 생성. 직전 24시간이 같은 재실 안이고 다음 24시간 관찰 가능한 시점만."""
    S = dict(SAMPLING_DEFAULT)
    rows, names = [], risk_feature_names()
    dirs = expand_dirs(a.data)
    for di, d in enumerate(dirs, 1):
        try:
            p = Pipeline(d, a.dur_csv, verbose=False, profile=a.profile, cohort=a.cohort, spec=a.spec)
        except SystemExit:
            continue
        S.update(PROFILE.get("sampling", {}))
        births = birth_times(p, S)
        p.preload(p.cands["stay_id"].tolist())
        for _, r in p.cands.iterrows():
            ctx = StayContext(p.kd, r, p.item_map, p.lab_map, p.d_test)
            A = ctx_arrays(ctx)
            b, _src = births.get(r["stay_id"], (None, None))
            intime, outtime = pd.Timestamp(r["intime_dt"]), pd.Timestamp(r["outtime_dt"])
            for t0 in day_t0s(r):
                if t0.hour != int(PROFILE["daily"]["cutoff_hour"]) or t0.minute:
                    continue
                if (t0 - intime).total_seconds() < 24 * 3600:
                    continue
                if A["death"] is not None and A["death"] <= t0:
                    continue
                lab = risk_labels(A, t0)
                if t0 + pd.Timedelta(hours=24) > outtime and not lab["death"]:
                    continue  # 다음 24시간을 관찰할 수 없음(퇴실)
                if b is not None and (t0 - b).days > int(S.get("max_age_days", 28)) and a.cohort == "nicu":
                    continue
                feats = risk_features(A, t0, intime, b)
                rows.append(dict(zip(names, feats), person_id=r["subject_id"], stay_id=r["stay_id"], data_dir=d, t0=str(t0), **lab))
        print("[train-risk] 폴더 %d/%d %s: 누적 %d 시점" % (di, len(dirs), os.path.basename(os.path.normpath(d)), len(rows)))
    return pd.DataFrame(rows), names


def _metrics(y, p):
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    m = {"n": int(len(y)), "positives": int(np.sum(y)), "prevalence": round(float(np.mean(y)), 4)}
    if len(set(y)) > 1:
        m.update(auroc=round(float(roc_auc_score(y, p)), 4), auprc=round(float(average_precision_score(y, p)), 4))
    m["brier"] = round(float(brier_score_loss(y, p)), 4)
    return m


def cmd_train_risk(a):
    try:
        import numpy as np
        import joblib
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.inspection import permutation_importance
    except ImportError:
        raise SystemExit("scikit-learn 이 없습니다. offline_wheels_ml 폴더로 설치하세요:\n"
                         "  python -m pip install --no-index --find-links offline_wheels_ml scikit-learn")
    df, names = build_risk_dataset(a)
    os.makedirs(a.out, exist_ok=True)
    if df.empty:
        raise SystemExit("학습 가능한 시점이 없습니다.")
    df.drop(columns=["person_id", "stay_id", "data_dir"]).to_csv(os.path.join(a.out, "risk_dataset_features_only.csv"), index=False)
    report = ["# 트리 모델 CDSS 학습 결과", "", "- 생성: %s, kcd_nicu.py v%s" % (time.strftime("%Y-%m-%d %H:%M"), VERSION),
              "- 데이터: 폴더 %d개, stay-일 %d개, 환아 %d명" % (df["data_dir"].nunique(), len(df), df["person_id"].nunique()),
              "- 시점: 매일 %02d:00, 입력 직전 24시간(검사 48시간), 예측 다음 24시간. 학습·평가는 환자 단위 8:2 분할" % int(PROFILE["daily"]["cutoff_hour"]), ""]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=a.seed)
    tr, te = next(gss.split(df, groups=df["person_id"]))
    X = df[names].astype(float).values
    for task, info in RISK_TASKS.items():
        y = df[task].values.astype(int)
        report += ["## %s" % info["title"], "", "- 정의: %s" % info["label"], "- 양성: 전체 %d / %d (%.1f%%)" % (y.sum(), len(y), 100 * y.mean())]
        if y[tr].sum() < 5 or len(set(y[te])) < 2:
            report += ["- 양성 사례가 너무 적어 학습하지 않음(학습 양성 %d, 평가 양성 %d)" % (y[tr].sum(), y[te].sum()), ""]
            continue
        clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=400, l2_regularization=1.0, min_samples_leaf=40,
                                             early_stopping=True, validation_fraction=0.15, class_weight="balanced", random_state=a.seed)
        clf.fit(X[tr], y[tr])
        pte, ptr = clf.predict_proba(X[te])[:, 1], clf.predict_proba(X[tr])[:, 1]
        cut = float(np.quantile(ptr, 1 - a.alert_top))
        flag = pte >= cut
        sens = float(flag[y[te] == 1].mean()) if (y[te] == 1).any() else float("nan")
        ppv = float(y[te][flag].mean()) if flag.any() else float("nan")
        m = _metrics(y[te], pte)
        m.update(threshold=round(cut, 4), alert_top=a.alert_top, sensitivity=round(sens, 4), ppv=round(ppv, 4), alert_rate=round(float(flag.mean()), 4))
        pi = permutation_importance(clf, X[te], y[te], scoring="roc_auc", n_repeats=3, random_state=a.seed)
        imp = sorted(zip(names, pi.importances_mean), key=lambda x: -x[1])
        med = {n: (float(np.nanmedian(X[tr][:, i])) if np.isfinite(X[tr][:, i]).any() else None) for i, n in enumerate(names)}
        bins = pd.qcut(pd.Series(pte), 5, duplicates="drop")
        cal = pd.DataFrame({"p": pte, "y": y[te], "bin": bins}).groupby("bin", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
        joblib.dump(clf, os.path.join(a.out, "%s.joblib" % task))
        meta = {"task": task, "title": info["title"], "label": info["label"], "features": names, "medians": med, "threshold": cut, "metrics": m,
                "top_features": [n for n, v in imp[:15]], "train_quantiles": [float(x) for x in np.quantile(ptr, np.linspace(0, 1, 101))],
                "created": time.strftime("%Y-%m-%d %H:%M"), "code_version": VERSION, "sklearn": __import__("sklearn").__version__}
        with open(os.path.join(a.out, "%s_meta.json" % task), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        report += ["- 평가(환자 %d명, 시점 %d개): AUROC %s, AUPRC %s, Brier %s" % (df.iloc[te]["person_id"].nunique(), len(te), m.get("auroc"), m.get("auprc"), m["brier"]),
                   "- 경보 기준: 학습 예측 상위 %d%%(위험 %.1f%% 이상) → 평가 경보율 %.1f%%, 민감도 %.1f%%, 양성예측도 %.1f%%" % (
                       round(a.alert_top * 100), cut * 100, m["alert_rate"] * 100, sens * 100, ppv * 100), "",
                   "보정(예측 5분위)", "", "| 구간 | 평균 예측 | 실제 비율 | 시점 수 |", "|---|---|---|---|"]
        report += ["| %s | %.3f | %.3f | %d |" % (i + 1, r_.pred, r_.obs, r_.n) for i, r_ in enumerate(cal.itertuples())]
        report += ["", "주요 입력(순열 중요도, AUROC 감소량)", "", "| 순위 | 입력 | 중요도 |", "|---|---|---|"]
        report += ["| %d | %s | %.4f |" % (i + 1, RISK_FEATURE_KO.get(n, n), v) for i, (n, v) in enumerate(imp[:12])]
        report.append("")
        print("[train-risk] %s: AUROC %s, AUPRC %s, 양성 %d" % (info["title"], m.get("auroc"), m.get("auprc"), y.sum()))
    sub = df[["death", "new_invasive", "new_vasopressor", "lab_measured_next24"]].sum().to_dict()
    report += ["## 라벨 세부(전체 시점 수)", "", "- 사망 %d, 새 침습적 인공호흡 %d, 승압제 신규 %d, 다음 24시간 검사 시행 %d" % (
        sub["death"], sub["new_invasive"], sub["new_vasopressor"], sub["lab_measured_next24"]), "",
               "- 한계: 라벨은 구조화 기록으로 정의한 대리 결과이며, 외부 검증 전에는 임상 의사결정에 단독으로 쓰지 않음."]
    with open(os.path.join(a.out, "train_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")
    print("[train-risk] 저장: %s/(deterioration|critical_lab).joblib, *_meta.json, train_report.md" % a.out)


_RISK_CACHE = {}


def load_risk_models():
    if "models" in _RISK_CACHE:
        return _RISK_CACHE["models"]
    out = {}
    d = next((x for x in (os.environ.get("KCD_RISK_DIR"), os.path.join(os.getcwd(), RISK_MODEL_DIR),
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), RISK_MODEL_DIR)) if x and os.path.isdir(x)), None)
    if d:
        try:
            import joblib
            for task in RISK_TASKS:
                mp, jp = os.path.join(d, task + ".joblib"), os.path.join(d, task + "_meta.json")
                if os.path.exists(mp) and os.path.exists(jp):
                    out[task] = (joblib.load(mp), json.load(open(jp, encoding="utf-8")))
        except ImportError:
            print("[risk] scikit-learn 이 없어 트리 모델 경보를 건너뜀")
    _RISK_CACHE["models"] = out
    return out


def risk_alerts(ctx, facts, alerts, birth=None):
    """트리 모델 예측을 CDSS 알림으로 추가. 예측값·기준값은 파생 fact로 등록해 검증기 대조 대상에 포함."""
    import numpy as np
    models = load_risk_models()
    if not models:
        return
    A = ctx_arrays(ctx)
    x = np.array([risk_features(A, ctx.t0, ctx.intime, birth)], dtype=float)
    for task, (clf, meta) in models.items():
        names = meta["features"]
        if names != risk_feature_names():
            continue
        p = float(clf.predict_proba(x)[0, 1])
        contrib = []
        for n in meta["top_features"][:10]:
            i = names.index(n)
            if not np.isfinite(x[0, i]) or meta["medians"].get(n) is None:
                continue
            x2 = x.copy()
            x2[0, i] = meta["medians"][n]
            d = p - float(clf.predict_proba(x2)[0, 1])
            if d > 0.005:
                contrib.append((d, n, x[0, i]))
        contrib.sort(reverse=True)
        why = ", ".join("%s %s" % (RISK_FEATURE_KO.get(n, n), fmt_num(round(v, 2))) for d, n, v in contrib[:3]) or "뚜렷한 단일 요인 없음"
        pct = int(np.searchsorted(meta["train_quantiles"], p) - 1)
        txt = "%s 예측 위험 %s%%(경보 기준 %s%%, 학습 분포 상위 %d%%). 주요 요인: %s." % (meta["title"], fmt_num(round(p * 100, 1)), fmt_num(round(meta["threshold"] * 100, 1)),
                                                                     max(1, 100 - pct), why)
        f = Fact("model_" + task, CDSS_SEC(), txt, 1, concepts={"model_" + task}, key_nums=[round(p * 100, 1)], prov=[{"derived_from": "tree_model", "model": task}])
        f.fid = "F%03d" % (len(facts) + 1)
        facts.append(f)
        hi = p >= meta["threshold"]
        alerts.append({"alert_id": "AL%02d" % (len(alerts) + 1), "rule_id": "MODEL_" + task.upper(), "severity": "high" if hi else "info",
                       "title": meta["title"] + (" 경보" if hi else "(경보 기준 미만)"), "text": txt, "trigger_fact_ids": [f.fid],
                       "reference": "트리 모델(GBM), 평가 AUROC %s" % meta["metrics"].get("auroc"), "parent_visible": False,
                       "action": cdss_action("MODEL_" + task.upper()) if hi else "현재 모델 경보 기준 미만. 규칙 알림과 임상 판단을 우선함."})


CDSS_ACTIONS_DEFAULT = {
    "NEO_HYPOGLYCEMIA": "30분 안에 혈당 재측정, 당 주입 속도(GIR)와 수유 확인, 기준 미만이 지속되면 담당의 보고.",
    "NEO_HYPERGLYCEMIA": "당 주입 속도와 수액 농도 확인, 재측정 후 지속 시 담당의 보고.",
    "NEO_TEMP_INSTABILITY": "인큐베이터·보온기 설정 확인 후 30~60분 뒤 체온 재측정, 발열이면 감염 징후 함께 평가.",
    "NEO_TACHYCARDIA": "통증·발열·저혈량 여부 확인, 지속 시 담당의 보고.",
    "NEO_TACHYPNEA": "호흡 양상·SpO2 재평가, 호흡 보조 조정 필요성 담당의와 상의.",
    "NEO_DESAT_BRADY": "사건 시각·중재 강도 기록, 카페인 투여·호흡 보조 설정 검토를 담당의와 상의.",
    "NEO_WEIGHT_LOSS": "수유량·수액량 재계산, 영양 계획을 담당의·영양팀과 검토.",
    "NEO_FLUID_RANGE": "실제 섭취량과 목표 수분량 비교, 수액·수유 계획 재확인.",
    "NEO_OLIGURIA": "기저귀 무게·도뇨 확인으로 소변량 재측정, 지속 시 수분 상태와 신기능 평가 요청.",
    "NEO_BILI_HIGH": "재태주수·생후시간별 광선치료 기준표와 비교, 광선치료 시작·강화 여부 담당의 확인.",
    "NEO_BILI_RISE": "빌리루빈 추적 간격 단축, 용혈 검사 필요성 담당의와 상의.",
    "NEO_SEPSIS_MARKERS": "감염 징후 재평가, 배양 결과와 항생제 계획을 담당의와 확인.",
    "CRITICAL_LAB": "검체 오류 여부 확인 후 재검, 결과 즉시 담당의 보고.",
    "AKI_KDIGO": "소변량·수분 균형 확인, 신독성 약물 검토와 용량 조절 필요성 상의.",
    "PRESSOR_ACTIVE": "혈압·관류 지표 확인 간격 유지, 주입 라인·속도 재확인.",
    "TDM_MISSING": "다음 투여 전 혈중농도 채혈 필요성 담당의·약제부와 확인.",
    "LINE_DAYS": "중심정맥관 유지 필요성 검토, 삽입부 관찰과 관리 지침 준수 확인.",
    "MICRO_PENDING": "결과 확인 시각 정하고 인계에 포함.",
    "ABX_MISMATCH": "감수성 결과와 현재 항생제 비교 후 변경 필요성 담당의 즉시 상의.",
    "DUR_PAIR": "병용금기 여부 약제부 확인 후 처방 재검토.",
    "MODEL_DETERIORATION": "활력징후·호흡 상태 관찰 간격 단축, 담당의에게 모델 경보와 주요 요인 공유 후 재평가.",
    "MODEL_CRITICAL_LAB": "다음 검사 결과 확인 시각을 정하고, 주요 요인 항목(전해질·혈당 등) 재측정 필요성 확인.",
}


def cdss_action(rule):
    acts = dict(CDSS_ACTIONS_DEFAULT, **(PROFILE.get("cdss", {}).get("actions", {}) if PROFILE else {}))
    for k, v in acts.items():
        if rule == k or rule.startswith(k):
            return v
    return "담당의와 상의."


def kcd_main(argv=None):
    if isinstance(argv, str):
        argv = argv.split()
    ap = argparse.ArgumentParser(description="K-MIMIC NICU 1일 요약지·부모님 설명지 생성·검증·CDSS")
    sp = ap.add_subparsers(dest="cmd")
    for name in ("list", "run", "batch", "perturb", "arms", "init-profile", "profile-check", "evaluate-note", "sample", "train-risk"):
        s = sp.add_parser(name)
        s.add_argument("--data", required=(name != "init-profile"))
        s.add_argument("--profile", default=None, help="병원 프로파일 JSON(없으면 기본 양식)")
        s.add_argument("--spec", default=None, help="인계 명세 JSON. 생략 시 같은 폴더의 nicu_daily_handoff_spec.json 자동 사용, none 이면 사용 안 함")
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
        s.add_argument("--stratify", action="store_true", help="sample: 생후일 구간별 균등 추출")
        s.add_argument("--alert-top", type=float, default=0.10, help="train-risk: 학습 예측 상위 비율을 경보 기준으로(기본 0.10)")
        s.add_argument("--time-unit", default=None, help="시간 컬럼이 숫자(상대시간)일 때 단위 D/h/min")
    a = ap.parse_args(argv)
    if not a.cmd:
        ap.print_help()
        return
    if a.time_unit:
        global TIME_UNIT
        TIME_UNIT = a.time_unit
    {"list": cmd_list, "run": cmd_run, "batch": cmd_batch, "perturb": cmd_perturb, "arms": cmd_arms, "init-profile": cmd_init_profile,
     "profile-check": cmd_profile_check, "evaluate-note": cmd_evaluate_note, "sample": cmd_sample,
     "train-risk": cmd_train_risk}[a.cmd](a)


if __name__ == "__main__" and not in_notebook_paste():
    kcd_main()
