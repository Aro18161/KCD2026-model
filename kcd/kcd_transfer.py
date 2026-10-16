#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kcd_transfer.py — K-MIMIC 기반 ICU→병동 전동기록지 생성(A) / 누락·할루시네이션 검증(B) / CDSS(C)

단일 파일, 의존성 pandas만 필요(LLM은 선택: OpenAI 호환 엔드포인트, 표준 라이브러리 urllib 사용).

사용 (터미널 또는 Jupyter의 %run):
  python kcd_transfer.py list    --data DIR                       # 전동 대상 ICU stay 후보 수와 ID 일부
  python kcd_transfer.py run     --data DIR --stay auto --out out # 1건 생성·검증·CDSS
  python kcd_transfer.py run     --data DIR --stay 12345 --out out
  python kcd_transfer.py batch   --data DIR --n 50 --out out      # N건 + 집계표(summary.csv/md)
  python kcd_transfer.py perturb --data DIR --n 50 --out out      # B3 검증기 자체 평가(합성 오류 주입)
  python kcd_transfer.py arms    --data DIR --n 10 --out out      # A0~A3 비교(LLM 필요)

Jupyter 셀에 통째로 붙여넣은 경우:
  kcd_main("run --data datasets/kmimic-emr-001/268 --stay auto --out out")

LLM(선택) 환경변수:
  KCD_LLM_BASE_URL=http://localhost:8000/v1   KCD_LLM_MODEL=모델명   KCD_LLM_API_KEY=(선택)

실제 환자 데이터 출력물은 분석환경 밖으로 반출하지 말 것. 외부 공유는 summary.csv의 집계 지표만.
"""
from __future__ import print_function

import argparse
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

VERSION = "0.2.0"
PROMPT_VERSION = "p1"

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


def candidate_stays(kd):
    """ICU에서 생존 퇴실 후 같은 입원 중 병동으로 간 stay(=전동기록지 대상)."""
    icu = kd.get("ICUSTAYS").copy()
    adm = kd.get("ADMISSIONS").copy()
    if icu.empty:
        raise SystemExit("ICUSTAYS가 없습니다.")
    icu["intime_dt"] = to_dt(icu["intime"])
    icu["outtime_dt"] = to_dt(icu["outtime"])
    if not adm.empty:
        adm = adm[["hadm_id"] + [c for c in ["dischtime", "deathtime", "hospital_expire_flag", "icu_expire_flag"] if c in adm.columns]]
        icu = icu.merge(adm, on="hadm_id", how="left")
        icu["dischtime_dt"] = to_dt(icu["dischtime"]) if "dischtime" in icu.columns else pd.NaT
        died_icu = to_num(icu.get("icu_expire_flag", pd.Series(0, index=icu.index))).fillna(0) == 1
        ok = (~died_icu) & icu["outtime_dt"].notna() & (icu["dischtime_dt"].isna() | (icu["dischtime_dt"] > icu["outtime_dt"] + pd.Timedelta(hours=1)))
        icu = icu[ok]
    icu = icu[icu["outtime_dt"].notna() & icu["intime_dt"].notna()]
    return icu.sort_values("stay_id").reset_index(drop=True)


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
    ("weight", "체중", r"weight|체중", None),
]
VENT_CONCEPTS = {"vent_mode", "peep", "pip", "paw_mean", "vt", "ve", "compliance"}

OUTPUT_CONCEPTS = [
    ("crrt", r"CVVHDF|CRRT|투석|dialysis|ultrafiltrat"),
    ("urine_stool", r"소변\+대변|대소변"),
    ("urine", r"도뇨|배뇨|foley|urine|소변|void"),
    ("drain", r"tube|drain|흉관|PCD|J-P|Jackson|C-tube|배액|hemovac|EVD"),
]
INPUT_CONCEPTS = [("in_iv", r"정맥|IV|intravenous"), ("in_oral", r"경구|oral|\bPO\b"), ("in_tube", r"Levin|L-tube|feeding|경관")]

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
    ("antibiotic", r"vancomycin|teicoplanin|piperacillin|tazobactam|meropenem|imipenem|ertapenem|ceftriaxone|cefazolin|cefepime|ceftazidime|cefotaxime|cefuroxime|ampicillin|amoxicillin|ciprofloxacin|levofloxacin|moxifloxacin|metronidazole|linezolid|colistin|amikacin|gentamicin|tigecycline|clindamycin|azithromycin|clarithromycin|trimethoprim|sulfamethoxazole|doxycycline|daptomycin", ["J01"]),
    ("antifungal", r"fluconazole|voriconazole|caspofungin|micafungin|amphotericin|itraconazole", ["J02"]),
    ("diuretic", r"furosemide|lasix|torsemide|spironolactone|bumetanide", ["C03"]),
    ("fluid", r"saline|hartmann|dextrose|plasma solution|ringer|D5W|albumin 5|NaCl", ["B05"]),
]
HIGH_ALERT = {"vasopressor", "sedation", "insulin", "anticoagulant"}
QT_DRUGS = r"ondansetron|haloperidol|amiodarone|levofloxacin|ciprofloxacin|moxifloxacin|azithromycin|clarithromycin|quetiapine|methadone|sotalol|domperidone|erythromycin|fluconazole"
DUR_PAIRS_DEFAULT = [  # 예시. 실제 사용 시 --dur-csv 로 식약처 DUR 병용금기 목록(ingredient_a, ingredient_b, reason)으로 교체
    ("ciprofloxacin", "tizanidine", "티자니딘 혈중농도 상승(병용금기 예시)"),
    ("clarithromycin", "simvastatin", "횡문근융해 위험(병용금기 예시)"),
    ("itraconazole", "simvastatin", "횡문근융해 위험(병용금기 예시)"),
    ("sildenafil", "nitroglycerin", "중증 저혈압(병용금기 예시)"),
]
RENAL_DRUGS = [  # (ingredient regex, eGFR 기준 미만, 메시지)
    (r"enoxaparin|clexane|crexane", 30, "eGFR<30에서 에녹사파린 용량 조절 필요"),
    (r"metformin", 30, "eGFR<30에서 메트포르민 금기"),
    (r"dabigatran", 30, "eGFR<30에서 다비가트란 사용 주의/금기"),
    (r"rivaroxaban", 15, "eGFR<15에서 리바록사반 사용 금기"),
    (r"gabapentin|pregabalin", 30, "eGFR<30에서 용량 감량 필요"),
]

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
# 3. Fact Sheet 생성 (기준 시각 t0 = ICU 퇴실)
# =============================================================================

SECTIONS = OrderedDict([
    ("S1", "1. 환자·전동 정보"),
    ("S2", "2. 입원·ICU 입실 사유 (진단·수술)"),
    ("S3", "3. ICU 경과 요약"),
    ("S4", "4. 전동 시점 상태"),
    ("S5", "5. 전동 시 투약"),
    ("S6", "6. 병동 주의사항 (CDSS 규칙 기반)"),
])
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
        self.allowed_times = extract_times(text)
        self.codes = extract_codes(text)

    @property
    def must(self):
        return self.salience >= 2

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

    def __init__(self, kd, icu_row, item_map, lab_map, d_test=None):
        self.kd = kd
        self.row = icu_row
        self.subject_id = icu_row["subject_id"]
        self.hadm_id = icu_row["hadm_id"]
        self.stay_id = icu_row["stay_id"]
        self.intime = pd.Timestamp(icu_row["intime_dt"])
        self.outtime = pd.Timestamp(icu_row["outtime_dt"])
        self.t0 = self.outtime
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


def build_facts(ctx):
    facts = []
    t0 = ctx.t0
    pat = ctx.pat.iloc[0] if len(ctx.pat) else {}
    adm = ctx.adm.iloc[0] if len(ctx.adm) else {}
    age = to_num(pd.Series([pat.get("anchor_age") if hasattr(pat, "get") else None])).iloc[0]
    sex = pat.get("sex") if hasattr(pat, "get") else None
    sex_ko = {"M": "남성", "F": "여성"}.get(str(sex), "성별 미상")
    age_txt = ("%s세" % fmt_num(age)) if pd.notna(age) else "나이 미상"
    ctx.age, ctx.sex = (float(age) if pd.notna(age) else None), sex
    # ---------------- S1
    facts.append(Fact("demographic", "S1", "%s %s." % (age_txt, sex_ko), 3, concepts={"demo"},
                      key_nums=[age] if pd.notna(age) else [], prov=_prov("PATIENTS", ctx.pat, "subject_id")))
    unit = ctx.row.get("first_careunit") or ctx.row.get("last_careunit") or "ICU"
    los = (ctx.outtime - ctx.intime).total_seconds() / 86400.0
    facts.append(Fact("icu_stay", "S1", "%s 입실 %s, 퇴실(전동) %s, ICU 재실 %s일." % (unit, fmt_t(ctx.intime), fmt_t(ctx.outtime), fmt_num(round(los, 1))),
                      3, concepts={"icu_stay"}, key_nums=[round(los, 1)], time=ctx.outtime, prov=[{"table": "ICUSTAYS", "row_id": ctx.stay_id}]))
    if hasattr(adm, "get") and adm.get("admission_type"):
        facts.append(Fact("admission", "S1", "입원 경로: %s, 입원 %s." % (adm.get("admission_type"), fmt_t(to_dt(pd.Series([adm.get("admittime")])).iloc[0])),
                          2, concepts={"admission"}, prov=[{"table": "ADMISSIONS", "row_id": ctx.hadm_id}]))
    opf = str(ctx.row.get("op_flag", ""))
    if opf not in ("", "nan", "None", "0"):
        facts.append(Fact("op_flag", "S1", "수술 관련 ICU 입실(op_flag=%s)." % opf, 2, concepts={"ev_surgery"}, prov=[{"table": "ICUSTAYS", "row_id": ctx.stay_id}]))
    nxt = None
    if not ctx.transfers.empty:
        tr = ctx.transfers.copy()
        tr["t"] = to_dt(tr["intime"])
        after = tr[(tr["t"] >= ctx.outtime - pd.Timedelta(hours=2))].sort_values("t")
        after = after[after["careunit"].astype(str) != str(unit)]
        if len(after):
            nxt = after.iloc[0]
    if nxt is not None:
        facts.append(Fact("transfer_dest", "S1", "전동 병동: %s (%s)." % (nxt["careunit"], fmt_t(nxt["t"])), 3, concepts={"transfer_dest"},
                          prov=_prov("TRANSFERS", after.head(1), "transfer_id")))
    else:
        facts.append(Fact("transfer_dest", "S1", "전동 병동: 이동 기록 없음.", 1, concepts={"transfer_dest"}))
    # ---------------- S2 진단·수술
    if not ctx.dx.empty:
        dx = ctx.dx.copy()
        dx["seq"] = to_num(dx.get("seq_num", pd.Series(range(len(dx)), index=dx.index)))
        dx = dx[~dx["icd_code"].astype(str).isin(["Others", "nan", "None", ""])].sort_values("seq")
        dx = dx.drop_duplicates("icd_code")
        items = ["%s %s" % (c, KCD_NAMES.get(str(c).replace(".", ""), "")) for c in dx["icd_code"].astype(str)]
        items = [i.strip() for i in items]
        if items:
            facts.append(Fact("diagnosis_main", "S2", "주진단: %s." % items[0], 3, concepts={"dx"}, prov=_prov("DIAGNOSES_ICD", dx.head(1), "icd_code")))
        if len(items) > 1:
            facts.append(Fact("diagnosis_other", "S2", "기타 진단: %s." % ", ".join(items[1:8]), 2, concepts={"dx"}, prov=_prov("DIAGNOSES_ICD", dx.iloc[1:8], "icd_code")))
    if not ctx.px.empty:
        px = ctx.px[~ctx.px["icd_code"].astype(str).isin(["Others", "nan", "None", ""])].copy()
        if len(px):
            px["code"] = px["icd_code"].map(norm_icd9p)
            px["d"] = to_dt(px["chartdate"]) if "chartdate" in px.columns else pd.NaT
            px = px.sort_values("d").drop_duplicates("code")
            items = ["ICD-9 %s %s(%s)" % (r["code"], ICD9P_NAMES.get(r["code"], ""), pd.Timestamp(r["d"]).strftime("%m-%d") if pd.notna(r["d"]) else "날짜 미상")
                     for _, r in px.iterrows()]
            facts.append(Fact("procedure_icd", "S2", "수술·시술 코드: %s." % ", ".join(i.replace(" (", "(") for i in items[:6]), 2,
                              concepts={"px", "ev_surgery"}, prov=_prov("PROCEDURES_ICD", px, "icd_code")))
    # ---------------- S3 ICU 경과
    facts += course_facts(ctx)
    # ---------------- S4 전동 시점 상태
    facts += status_facts(ctx)
    # ---------------- S5 투약
    facts += med_facts(ctx)
    order = list(SECTIONS.keys())
    facts.sort(key=lambda f: (order.index(f.section), 0 if f.time is None or pd.isna(f.time) else 1, f.time if f.time is not None and not pd.isna(f.time) else pd.Timestamp.min))
    for i, f in enumerate(facts, 1):
        f.fid = "F%03d" % i
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


def course_facts(ctx):
    facts = []
    t0 = ctx.t0
    # 인공호흡
    if not ctx.chart.empty:
        v = ctx.icu_slice(ctx.chart)
        v = v[v["concept"].isin(VENT_CONCEPTS)]
        eps = episodes(v["t"].tolist(), 12)
        for a, b in eps[:3]:
            dur = (b - a).total_seconds() / 86400.0
            st = "active" if (t0 - b).total_seconds() <= 12 * 3600 else "stopped"
            txt = "인공호흡 관련 기록 %s ~ %s(약 %s일)%s." % (fmt_t(a), fmt_t(b), fmt_num(round(dur, 1)), ", 전동 직전까지 기록 지속" if st == "active" else ", 이후 기록 없음")
            facts.append(Fact("vent_episode", "S3", txt, 3, concepts={"ev_vent"}, status=st, time=a, prov=_prov("CHARTEVENTS", v[(v["t"] >= a) & (v["t"] <= b)].head(3), "chartevent_id")))
    # 승압제·진정제·인슐린 지속주입 (EMAR)
    if not ctx.emar.empty:
        e = ctx.icu_slice(ctx.emar)
        for cls in ("vasopressor", "sedation"):
            ec = e[e["cls"] == cls]
            for g, gg in ec.groupby("generic"):
                for a, b in episodes(gg["t"].tolist(), 6)[:2]:
                    seg = gg[(gg["t"] >= a) & (gg["t"] <= b)]
                    rates = []
                    for c in ("infusion_rate_min", "infusion_rate_hr"):
                        if c in seg.columns:
                            r = to_num(seg[c]).dropna()
                            if len(r):
                                rates.append((c, r.max()))
                    rtxt = ""
                    if rates:
                        c, mx = rates[0]
                        rtxt = ", 최대 주입속도 기록값 %s(%s)" % (fmt_num(mx), "/min" if c.endswith("min") else "/hr")
                    st = "active" if (t0 - b).total_seconds() <= 4 * 3600 else "stopped"
                    stxt = "전동 직전까지 투여 중" if st == "active" else "%s 마지막 투여 후 중단" % fmt_t(b)
                    kind = "승압제" if cls == "vasopressor" else "진정·진통제"
                    txt = "%s %s 사용 %s ~ %s%s, %s." % (kind, g, fmt_t(a), fmt_t(b), rtxt, stxt)
                    facts.append(Fact("infusion_episode", "S3", txt, 3, concepts={"med:" + g.lower(), "cls:" + cls}, status=st, time=a,
                                      key_nums=[], prov=_prov("EMAR", seg.head(3), "emar_id"), extra={"cls": cls, "last": str(b), "generic": g}))
    # 입력 테이블의 승압제(MIMIC 형식 대비)
    if not ctx.inp.empty and (ctx.inp["concept"] == "in_vasopressor").any():
        iv = ctx.icu_slice(ctx.inp)
        iv = iv[iv["concept"] == "in_vasopressor"]
        for g, gg in iv.groupby("label"):
            for a, b in episodes(gg["t"].tolist(), 6)[:2]:
                seg = gg[(gg["t"] >= a) & (gg["t"] <= b)]
                if "t_end" in seg.columns and seg["t_end"].notna().any():
                    b = max(b, min(seg["t_end"].max(), t0))
                rt = to_num(seg["rate"]).dropna() if "rate" in seg.columns else pd.Series(dtype=float)
                ru = str(seg["rateuom"].dropna().iloc[0]) if ("rateuom" in seg.columns and seg["rateuom"].notna().any()) else ""
                rtxt = (", 최대 %s %s" % (fmt_num(rt.max()), ru)) if len(rt) else ""
                st = "active" if (t0 - b).total_seconds() <= 4 * 3600 else "stopped"
                txt = "승압제 %s 사용 %s ~ %s%s, %s." % (g, fmt_t(a), fmt_t(b), rtxt, "전동 직전까지 투여 중" if st == "active" else "이후 중단")
                facts.append(Fact("infusion_episode", "S3", txt, 3, concepts={"med:" + med_generic(g).lower(), "cls:vasopressor"}, status=st, time=a,
                                  prov=_prov("INPUTEVENTS", gg.head(3), "inputevent_id"), extra={"cls": "vasopressor", "last": str(b), "generic": g}))
    # CRRT
    if not ctx.out.empty:
        o = ctx.icu_slice(ctx.out)
        cr = o[o["concept"] == "crrt"]
        for a, b in episodes(cr["t"].tolist(), 12)[:2]:
            seg = cr[(cr["t"] >= a) & (cr["t"] <= b)]
            st = "active" if (t0 - b).total_seconds() <= 12 * 3600 else "stopped"
            txt = "CRRT(%s) 기록 %s ~ %s, 총 제거량 %s cc%s." % (seg["label"].iloc[0], fmt_t(a), fmt_t(b), fmt_num(seg["v"].sum()), ", 전동 직전까지 시행" if st == "active" else "")
            facts.append(Fact("crrt_episode", "S3", txt, 3, concepts={"ev_crrt"}, status=st, time=a, key_nums=[round(seg["v"].sum())],
                              prov=_prov("OUTPUTEVENTS", seg.head(3), "outputevent_id")))
    # 시술·검사(PROCEDUREEVENTS)
    if not ctx.proc.empty:
        p = ctx.icu_slice(ctx.proc)
        if "value" in p.columns:
            vv = to_num(p["value"])
            p = p[vv.isna() | (vv != 0)]
        for lab, g in p.groupby("label"):
            txt = "%s 시행 기록(첫 기록 %s, %d회)." % (lab, fmt_t(g["t"].min()), len(g))
            sal = 2 if re.search("수술|surgery|Intubation|삽관|central|중심정맥", str(g["cat"].iloc[0]) + lab, re.I) else 1
            ev = {"procedure"}
            for evn, rx in EVENT_VOCAB:
                if re.search(rx, lab, re.I):
                    ev.add(evn)
            facts.append(Fact("procedure", "S3", txt, sal, concepts=ev, time=g["t"].min(), key_nums=[], prov=_prov("PROCEDUREEVENTS", g.head(3), "procedureevent_id")))
    # 검사 추세 (ICU 기간 최고/최저)
    if not ctx.lab.empty:
        L = ctx.lab
        icu = ctx.icu_slice(L)
        for c in KEY_TREND_LABS:
            s = icu[icu["concept"] == c].dropna(subset=["v"]).sort_values("t")
            if len(s) < 2:
                continue
            lo_ref, hi_ref = ref_range(s)
            disp = LAB_DISPLAY.get(c, c)
            if c in ("hb", "plt"):
                r = s.loc[s["v"].idxmin()]
                if lo_ref is not None and r["v"] < lo_ref:
                    txt = "ICU 기간 %s 최저 %s(%s), 입실 시 %s." % (disp, fmt_num(r["v"]), fmt_t(r["t"]), fmt_num(s["v"].iloc[0]))
                    facts.append(Fact("lab_trend", "S3", txt, 2, concepts={c}, key_nums=[r["v"]], time=r["t"], prov=_prov("LABEVENTS", s.loc[[r.name]], "labevent_id")))
            else:
                r = s.loc[s["v"].idxmax()]
                if hi_ref is not None and r["v"] > hi_ref:
                    txt = "ICU 기간 %s 최고 %s(%s), 입실 시 %s." % (disp, fmt_num(r["v"]), fmt_t(r["t"]), fmt_num(s["v"].iloc[0]))
                    facts.append(Fact("lab_trend", "S3", txt, 2, concepts={c}, key_nums=[r["v"]], time=r["t"], prov=_prov("LABEVENTS", s.loc[[r.name]], "labevent_id")))
        # AKI (KDIGO Cr 기준): 입실 7일 전부터
        ctx.aki = aki_check(ctx)
        if ctx.aki:
            a = ctx.aki
            txt = "Cr %s(%s) → %s(%s) 상승." % (fmt_num(a["base"]), fmt_t(a["base_t"]), fmt_num(a["peak"]), fmt_t(a["peak_t"]))
            facts.append(Fact("aki", "S3", txt, 3, concepts={"cr", "aki"}, key_nums=[a["base"], a["peak"]], time=a["peak_t"], prov=a["prov"], extra={"criterion": a["criterion"]}))
    # 미생물
    if not ctx.micro.empty:
        m = ctx.micro[(ctx.micro["t"] >= ctx.intime - pd.Timedelta(days=3)) & (ctx.micro["t"] <= ctx.t0)]
        key = "micro_specimen_id" if "micro_specimen_id" in m.columns else "t"
        for sid_, g in m.groupby(key):
            spec = str(g["spec_type_desc"].iloc[0]) if "spec_type_desc" in g.columns else "검체"
            orgs = [o for o in g.get("org_name", pd.Series(dtype=str)).dropna().astype(str).unique() if o not in ("nan", "")]
            if orgs:
                res = g[g.get("interpretation", pd.Series(dtype=str)).astype(str).isin(["R"])]["ab_name"].dropna().astype(str).unique().tolist() if "interpretation" in g.columns else []
                txt = "%s(%s) 배양 %s 동정%s." % (spec, fmt_t(g["t"].min()), ", ".join(orgs), ("; 내성: " + ", ".join(res)) if res else "")
                facts.append(Fact("micro_positive", "S3", txt, 3, concepts={"ev_bloodcx", "micro"}, time=g["t"].min(), prov=_prov("MICROBIOLOGYEVENTS", g.head(3), "microevent_id"),
                                  extra={"orgs": orgs, "resistant": res}))
            elif g["t_store"].isna().all():
                txt = "%s(%s) 배양 결과 미보고." % (spec, fmt_t(g["t"].min()))
                facts.append(Fact("micro_pending", "S3", txt, 2, concepts={"ev_bloodcx", "micro"}, status="pending", time=g["t"].min(), prov=_prov("MICROBIOLOGYEVENTS", g.head(2), "microevent_id")))
            else:
                txt = "%s(%s) 배양 동정균 없음." % (spec, fmt_t(g["t"].min()))
                facts.append(Fact("micro_negative", "S3", txt, 1, concepts={"micro"}, time=g["t"].min(), prov=_prov("MICROBIOLOGYEVENTS", g.head(1), "microevent_id")))
    # 영상(TESTEVENTS, 판독문 없음)
    if not ctx.test.empty:
        te = ctx.test.copy()
        te["t"] = to_dt(te["charttime"])
        te = te[(te["t"] >= ctx.intime) & (te["t"] <= ctx.t0)]
        if len(te):
            lab = te["itemid"].map(lambda i: i)
            if ctx.d_test is not None and not ctx.d_test.empty:
                dm = dict(zip(ctx.d_test["itemid"], ctx.d_test["label"]))
                lab = te["itemid"].map(lambda i: dm.get(i, i))
            names = ", ".join(sorted(set(lab.astype(str))))[:120]
            facts.append(Fact("imaging", "S3", "영상검사 시행: %s(판독문은 데이터에 없음)." % names, 1, concepts={"imaging"}, prov=_prov("TESTEVENTS", te.head(3), "testevent_id")))
    return facts


def ref_range(s):
    lo = s["ref_range_lower"].dropna() if "ref_range_lower" in s.columns else pd.Series(dtype=float)
    hi = s["ref_range_upper"].dropna() if "ref_range_upper" in s.columns else pd.Series(dtype=float)
    c = s["concept"].iloc[0]
    fb = LAB_FALLBACK_REF.get(c, (None, None))
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
        rows = s[s["t"].isin([best["base_t"], best["peak_t"]])]
        best["prov"] = _prov("LABEVENTS", rows, "labevent_id")
    return best


def status_facts(ctx):
    facts = []
    t0 = ctx.t0
    ctx.last_vitals = {}
    if not ctx.chart.empty:
        c = ctx.icu_slice(ctx.chart)
        lastv = {}
        rows = []
        for k in ["hr", "sbp", "dbp", "map", "rr", "spo2", "temp"]:
            s = c[c["concept"] == k].dropna(subset=["v"]).sort_values("t")
            if len(s):
                r = s.iloc[-1]
                lastv[k] = (float(r["v"]), r["t"])
                rows.append(s.tail(1))
        ctx.last_vitals = lastv
        if lastv:
            parts = []
            if "hr" in lastv:
                parts.append("HR %s회/분" % fmt_num(lastv["hr"][0]))
            if "sbp" in lastv and "dbp" in lastv:
                parts.append("BP %s/%s mmHg" % (fmt_num(lastv["sbp"][0]), fmt_num(lastv["dbp"][0])))
            elif "sbp" in lastv:
                parts.append("SBP %s mmHg" % fmt_num(lastv["sbp"][0]))
            if "map" in lastv:
                parts.append("MAP %s mmHg" % fmt_num(lastv["map"][0]))
            if "rr" in lastv:
                parts.append("RR %s회/분" % fmt_num(lastv["rr"][0]))
            if "spo2" in lastv:
                parts.append("SpO2 %s%%" % fmt_num(lastv["spo2"][0]))
            if "temp" in lastv:
                parts.append("BT %s℃" % fmt_num(lastv["temp"][0]))
            tmax = max(v[1] for v in lastv.values())
            gap_h = (ctx.t0 - tmax).total_seconds() / 3600.0
            when = "%s 기준" % fmt_t(tmax) + (", 전동 %s시간 전 기록" % fmt_num(round(gap_h)) if gap_h > 6 else "")
            txt = "최종 활력징후(%s): %s." % (when, ", ".join(parts))
            facts.append(Fact("vitals_last", "S4", txt, 3, concepts=set(lastv.keys()) | {"vitals"}, key_nums=[v[0] for v in lastv.values()],
                              time=tmax, prov=sum([_prov("CHARTEVENTS", r, "chartevent_id") for r in rows], [])))
        # 최근 24시간 범위
        rec = ctx.last_window(c, 24)
        rng = []
        for k, disp in (("map", "MAP"), ("hr", "HR"), ("spo2", "SpO2")):
            s = rec[rec["concept"] == k]["v"].dropna()
            if len(s) >= 2 and fmt_num(s.min()) != fmt_num(s.max()):
                rng.append("%s %s~%s" % (disp, fmt_num(s.min()), fmt_num(s.max())))
        if rng:
            facts.append(Fact("vitals_range", "S4", "최근 24시간 범위: %s." % ", ".join(rng), 1, concepts={"vitals", "map", "hr", "spo2"}, prov=[]))
        # GCS
        g = {}
        for k in ("gcs_e", "gcs_v", "gcs_m", "gcs"):
            s = c[c["concept"] == k].dropna(subset=["v"]).sort_values("t")
            if len(s):
                g[k] = (float(s["v"].iloc[-1]), s["t"].iloc[-1])
        tot = g.get("gcs", (None,))[0]
        if tot is None and all(k in g for k in ("gcs_e", "gcs_v", "gcs_m")):
            tot = g["gcs_e"][0] + g["gcs_v"][0] + g["gcs_m"][0]
        ctx.gcs = tot
        if tot is not None:
            sub = ""
            if all(k in g for k in ("gcs_e", "gcs_v", "gcs_m")):
                sub = "(E%s V%s M%s)" % (fmt_num(g["gcs_e"][0]), fmt_num(g["gcs_v"][0]), fmt_num(g["gcs_m"][0]))
            tt = max(v[1] for v in g.values())
            facts.append(Fact("gcs", "S4", "의식: GCS %s%s, %s 기록." % (fmt_num(tot), sub, fmt_t(tt)), 3 if tot < 15 else 2, concepts={"gcs"},
                              key_nums=[tot], time=tt))
        # 호흡 보조
        ctx.on_vent = False
        ctx.on_o2 = False
        ctx.fio2 = None
        rec12 = ctx.last_window(c, 12)
        vv = rec12[rec12["concept"].isin(VENT_CONCEPTS)]
        fi = c[c["concept"] == "fio2"].dropna(subset=["v"]).sort_values("t")
        if len(fi):
            f = float(fi["v"].iloc[-1])
            ctx.fio2 = f / 100.0 if f > 1 else f
        if len(vv):
            ctx.on_vent = True
            mode = rec12[rec12["concept"] == "vent_mode"]["value"].dropna().astype(str)
            peep = rec12[rec12["concept"] == "peep"]["v"].dropna()
            parts = []
            if len(mode):
                parts.append("모드 %s" % mode.iloc[-1])
            if ctx.fio2 is not None and len(fi) and fi["t"].iloc[-1] > t0 - pd.Timedelta(hours=12):
                parts.append("FiO2 %s%%" % fmt_num(ctx.fio2 * 100))
            if len(peep):
                parts.append("PEEP %s cmH2O" % fmt_num(peep.iloc[-1]))
            txt = "호흡: 전동 전 12시간 내 인공호흡기 기록 있음%s." % ((" (" + ", ".join(parts) + ")") if parts else "")
            facts.append(Fact("resp_support", "S4", txt, 3, concepts={"ev_vent", "resp"}, status="active", time=vv["t"].max(), prov=_prov("CHARTEVENTS", vv.tail(2), "chartevent_id")))
        else:
            o2 = rec12[rec12["concept"] == "o2_flow"].dropna(subset=["v"]).sort_values("t")
            if len(o2) and o2["v"].iloc[-1] > 0:
                ctx.on_o2 = True
                facts.append(Fact("resp_support", "S4", "호흡: 산소 %s L/min(%s).", 2, concepts={"resp", "o2_flow"}, key_nums=[o2["v"].iloc[-1]], time=o2["t"].iloc[-1]))
                facts[-1].text = "호흡: 산소 %s L/min(%s)." % (fmt_num(o2["v"].iloc[-1]), fmt_t(o2["t"].iloc[-1]))
                facts[-1].allowed_nums = extract_nums(facts[-1].text)
                facts[-1].allowed_times = extract_times(facts[-1].text)
            else:
                facts.append(Fact("resp_support", "S4", "호흡: 전동 전 12시간 내 인공호흡기·산소유량 기록 없음.", 1, concepts={"resp"}))
    # 최신 검사 (48시간 내)
    ctx.latest_labs = {}
    if not ctx.lab.empty:
        L = ctx.last_window(ctx.lab, 48)
        normal = []
        for c, disp, _, lo_c, hi_c in LAB_CONCEPTS:
            s = L[L["concept"] == c].dropna(subset=["v"]).sort_values("t")
            if not len(s):
                continue
            r = s.iloc[-1]
            ctx.latest_labs[c] = (float(r["v"]), r["t"])
            lo, hi = ref_range(s)
            unit = str(r.get("valueuom", "")) if pd.notna(r.get("valueuom", None)) else ""
            unit = "" if unit in ("nan", "None") else unit
            crit = (lo_c is not None and r["v"] <= lo_c) or (hi_c is not None and r["v"] >= hi_c)
            abn = (lo is not None and r["v"] < lo) or (hi is not None and r["v"] > hi)
            if c == "egfr":
                abn = r["v"] < 60
            if crit or abn:
                direction = "높음" if (hi is not None and r["v"] > hi) else "낮음"
                unit_txt = (" " + unit) if unit and unit.lower() != disp.lower() else ""
                txt = "%s %s%s(%s, %s)." % (disp, fmt_num(r["v"]), unit_txt, fmt_t(r["t"]), "위험값" if crit else "참고범위보다 " + direction)
                facts.append(Fact("lab_latest", "S4", txt, 3 if crit else 2, concepts={c}, key_nums=[r["v"]], time=r["t"], prov=_prov("LABEVENTS", s.tail(1), "labevent_id"),
                                  extra={"critical": bool(crit), "ref": [lo, hi]}))
            else:
                normal.append((disp, r["v"]))
        if normal:
            txt = "최근 48시간 참고범위 내 검사: %s." % ", ".join("%s %s" % (d, fmt_num(v)) for d, v in normal[:14])
            facts.append(Fact("lab_normal", "S4", txt, 1, concepts={"labs_normal"}))
        if not ctx.latest_labs:
            lt = ctx.lab["t"].max()
            facts.append(Fact("lab_none", "S4", "전동 전 48시간 내 검사 기록 없음(마지막 검사 %s)." % fmt_t(lt), 2, concepts={"labs_none"}, time=lt))
    # 섭취·배설 (24시간)
    w = ctx.weight()
    ctx.wt = w
    io_in = ctx.last_window(ctx.inp, 24) if not ctx.inp.empty else pd.DataFrame()
    io_out = ctx.last_window(ctx.out, 24) if not ctx.out.empty else pd.DataFrame()
    if len(io_in) or len(io_out):
        tin = float(io_in["v"].sum()) if len(io_in) else 0.0
        tout = float(io_out["v"].sum()) if len(io_out) else 0.0
        ur = io_out[io_out["concept"] == "urine"]["v"].sum() if len(io_out) else 0.0
        hrs = 24.0
        if len(io_out):
            span = (ctx.t0 - io_out["t"].min()).total_seconds() / 3600.0
            hrs = max(1.0, min(24.0, span + 1.0))
        parts = ["섭취 %s cc" % fmt_num(round(tin)) if len(io_in) else "섭취 기록 없음",
                 "배설 %s cc" % fmt_num(round(tout)) if len(io_out) else "배설 기록 없음"]
        nums = ([round(tin)] if len(io_in) else []) + ([round(tout)] if len(io_out) else [])
        ctx.urine_rate = None
        if ur > 0:
            parts.append("소변 %s cc" % fmt_num(round(ur)))
            nums.append(round(ur))
            if w:
                rate = ur / w / hrs
                ctx.urine_rate = rate
                parts.append("%s mL/kg/h" % fmt_num(round(rate, 2)))
                nums.append(round(rate, 2))
        if len(io_in) and len(io_out):
            bal = round(tin - tout)
            parts.append("균형 %s%s cc" % ("+" if bal >= 0 else "", fmt_num(bal)))
        txt = "최근 24시간 섭취·배설: %s." % ", ".join(parts)
        sal = 3 if (ctx.urine_rate is not None and ctx.urine_rate < 0.5) else 2
        facts.append(Fact("io_24h", "S4", txt, sal, concepts={"io", "urine"}, key_nums=nums, prov=_prov("OUTPUTEVENTS", io_out.head(3), "outputevent_id") + _prov("INPUTEVENTS", io_in.head(2), "inputevent_id")))
    # 배액관
    if len(io_out):
        d = io_out[io_out["concept"] == "drain"]
        for lab, g in d.groupby("label"):
            txt = "배액관 %s 유지, 최근 24시간 배액 %s cc." % (lab, fmt_num(round(g["v"].sum())))
            facts.append(Fact("drain", "S4", txt, 2, concepts={"drain", "drain:" + lab.lower()}, key_nums=[round(g["v"].sum())], status="active",
                              prov=_prov("OUTPUTEVENTS", g.head(2), "outputevent_id")))
    return facts


def med_facts(ctx):
    facts = []
    t0 = ctx.t0
    ctx.active_meds = []  # (generic, cls, name, last_time, route)
    rows = []
    if not ctx.emar.empty:
        e = ctx.emar[(ctx.emar["t"] <= t0) & (ctx.emar["t"] >= ctx.intime)]
        for g, gg in e.groupby("generic"):
            last = gg["t"].max()
            name = gg["medication"].iloc[-1]
            cls = gg["cls"].iloc[-1]
            rate = None
            for c in ("infusion_rate_min", "infusion_rate_hr"):
                if c in gg.columns:
                    r = to_num(gg.sort_values("t")[c]).dropna()
                    if len(r):
                        rate = (float(r.iloc[-1]), "/min" if c.endswith("min") else "/hr")
                        break
            rows.append(dict(generic=g, cls=cls, name=name, last=last, source="EMAR", prov=_prov("EMAR", gg.tail(2), "emar_id"), rate=rate, route=None))
    if not ctx.rx.empty:
        r = ctx.rx[(ctx.rx["t"] <= t0)]
        for _, x in r.iterrows():
            active = pd.isna(x["t_stop"]) or x["t_stop"] > t0
            last = t0 if active else x["t_stop"]
            last = x["t"] if active else x["t_stop"]
            rows.append(dict(generic=x["generic"], cls=x["cls"], name=x["drug"], last=last, source="PRESCRIPTIONS", prov=[{"table": "PRESCRIPTIONS", "row_id": x.get("pharmacy_id")}],
                             open_end=pd.isna(x["t_stop"]),
                             rate=None, route=x.get("route"), dose="%s %s" % (fmt_num(to_num(pd.Series([x.get("dose_val_rx")])).iloc[0]), x.get("dose_unit_rx", "")) if pd.notna(to_num(pd.Series([x.get("dose_val_rx")])).iloc[0]) else "",
                             rx_active=active))
    # 성분 단위로 합침
    by = OrderedDict()
    for r in rows:  # EMAR(실제 투여 기록)을 처방보다 우선
        k = r["generic"].lower()
        if k not in by or (r["source"] == "EMAR" and by[k]["source"] != "EMAR") or (r["source"] == by[k]["source"] and r["last"] > by[k]["last"]):
            by[k] = r
    # 같은 약이 ICU 경과(S3)에서 승압제 지속주입(INPUTEVENTS) 에피소드로 기술되면 그 상태를 따름
    inp_last = {}
    if not ctx.inp.empty and (ctx.inp["concept"] == "in_vasopressor").any():
        iv = ctx.inp[ctx.inp["concept"] == "in_vasopressor"]
        for g, gg in iv.groupby("label"):
            end = gg["t_end"].max() if "t_end" in gg.columns else gg["t"].max()
            inp_last[med_generic(g).lower()] = min(end, t0)
    fluids = []
    for k, r in by.items():
        if k in inp_last and r["source"] == "PRESCRIPTIONS":
            r = dict(r, last=inp_last[k], source="INPUTEVENTS")
            active = (t0 - r["last"]).total_seconds() <= 4 * 3600
        elif r["source"] == "PRESCRIPTIONS":
            active = r.get("rx_active", False)
        else:
            win = 4 if r["cls"] in ("vasopressor", "sedation") else 24
            active = (t0 - r["last"]).total_seconds() <= win * 3600
        if r["cls"] == "fluid":
            if active:
                fluids.append(r)
            continue
        cls_ko = {"vasopressor": "승압제", "sedation": "진정·진통제", "insulin": "인슐린", "anticoagulant": "항응고제", "antiplatelet": "항혈소판제",
                  "antibiotic": "항생제", "antifungal": "항진균제", "diuretic": "이뇨제"}.get(r["cls"], "")
        extra_txt = ""
        if r.get("rate"):
            extra_txt += ", 마지막 주입속도 기록값 %s(%s)" % (fmt_num(r["rate"][0]), r["rate"][1])
        if r.get("dose"):
            extra_txt += ", %s" % r["dose"].strip()
        if r.get("route") and str(r["route"]) not in ("nan", "None"):
            extra_txt += " %s" % r["route"]
        if active:
            ctx.active_meds.append((r["generic"], r["cls"], r["name"], r["last"], r.get("route")))
            if r["source"] == "PRESCRIPTIONS":
                tail = "처방 시작 %s%s" % (fmt_t(r["last"]), ", 종료 기록 없음" if r.get("open_end") else "")
                txt = "%s%s(%s) 처방 유지 중%s, %s." % ((cls_ko + " ") if cls_ko else "", r["generic"], r["name"], extra_txt, tail)
            else:
                txt = "%s%s(%s) 투여 중%s, 마지막 기록 %s." % ((cls_ko + " ") if cls_ko else "", r["generic"], r["name"], extra_txt, fmt_t(r["last"]))
            sal = 3 if r["cls"] in HIGH_ALERT else 2
            facts.append(Fact("med_active", "S5", txt, sal, concepts={"med:" + r["generic"].lower(), "cls:" + r["cls"]}, status="active", time=r["last"], prov=r["prov"],
                              extra={"cls": r["cls"], "generic": r["generic"]}))
        else:
            if r["cls"] in ("vasopressor", "sedation"):
                continue  # 경과(S3)에 이미 기술
            txt = "%s%s(%s) ICU 중 사용 후 중단, 마지막 기록 %s." % ((cls_ko + " ") if cls_ko else "", r["generic"], r["name"], fmt_t(r["last"]))
            sal = 2 if r["cls"] in HIGH_ALERT | {"antibiotic"} else 1
            facts.append(Fact("med_stopped", "S5", txt, sal, concepts={"med:" + r["generic"].lower(), "cls:" + r["cls"]}, status="stopped", time=r["last"], prov=r["prov"],
                              extra={"cls": r["cls"], "generic": r["generic"]}))
    if fluids:
        txt = "수액 투여 중: %s." % ", ".join(sorted(set(f["name"] for f in fluids)))
        facts.append(Fact("fluids", "S5", txt, 1, concepts={"cls:fluid"} | {"med:" + f["generic"].lower() for f in fluids}, status="active"))
    return facts


# =============================================================================
# 4. CDSS (결정론 규칙; 알림 문장은 trigger fact 값만 사용)
# =============================================================================


def news2(ctx):
    v = {k: x[0] for k, x in getattr(ctx, "last_vitals", {}).items()}
    if not v:
        return None, {}
    pts = {}
    rr = v.get("rr")
    if rr is not None:
        pts["RR"] = 3 if rr <= 8 else 1 if rr <= 11 else 0 if rr <= 20 else 2 if rr <= 24 else 3
    s = v.get("spo2")
    if s is not None:
        pts["SpO2"] = 3 if s <= 91 else 2 if s <= 93 else 1 if s <= 95 else 0
    pts["O2"] = 2 if (getattr(ctx, "on_vent", False) or getattr(ctx, "on_o2", False)) else 0
    sbp = v.get("sbp")
    if sbp is not None:
        pts["SBP"] = 3 if sbp <= 90 else 2 if sbp <= 100 else 1 if sbp <= 110 else 0 if sbp <= 219 else 3
    hr = v.get("hr")
    if hr is not None:
        pts["HR"] = 3 if hr <= 40 else 1 if hr <= 50 else 0 if hr <= 90 else 1 if hr <= 110 else 2 if hr <= 130 else 3
    g = getattr(ctx, "gcs", None)
    pts["의식"] = 3 if (g is not None and g < 15) else 0
    t = v.get("temp")
    if t is not None:
        pts["BT"] = 3 if t <= 35.0 else 1 if t <= 36.0 else 0 if t <= 38.0 else 1 if t <= 39.0 else 2
    return sum(pts.values()), pts


def swift(ctx):
    """SWIFT (Gajic 2008) ICU 퇴실 후 재입실 위험. 입실 경로는 TRANSFERS로 근사."""
    pts = {}
    src = 8
    if not ctx.transfers.empty:
        tr = ctx.transfers.copy()
        tr["t"] = to_dt(tr["intime"])
        prev = tr[tr["t"] < ctx.intime - pd.Timedelta(minutes=1)].sort_values("t")
        if len(prev) == 0 or str(prev["careunit"].iloc[-1]).upper() in ("ED", "EMERGENCY DEPARTMENT", "응급실"):
            src = 0
    pts["입실경로"] = src
    los = (ctx.outtime - ctx.intime).total_seconds() / 86400.0
    pts["재실기간"] = 0 if los < 2 else 1 if los <= 10 else 14
    po2 = getattr(ctx, "latest_labs", {}).get("po2")
    fio2 = ctx.fio2 if (getattr(ctx, "fio2", None) and getattr(ctx, "on_vent", False)) else 0.21
    if po2:
        pf = po2[0] / fio2
        pts["P/F"] = 0 if pf >= 400 else 5 if pf >= 150 else 10 if pf >= 100 else 13
    g = getattr(ctx, "gcs", None)
    if g is not None:
        pts["GCS"] = 0 if g >= 15 else 6 if g >= 11 else 14 if g >= 8 else 24
    pco2 = getattr(ctx, "latest_labs", {}).get("pco2")
    if pco2:
        pts["PaCO2"] = 5 if pco2[0] > 45 else 0
    return sum(pts.values()), pts


def ckd_epi_2021(cr, age, sex):
    if cr is None or age is None or sex not in ("M", "F"):
        return None
    k, a = (0.7, -0.241) if sex == "F" else (0.9, -0.302)
    e = 142 * (min(cr / k, 1) ** a) * (max(cr / k, 1) ** -1.200) * (0.9938 ** age)
    return e * 1.012 if sex == "F" else e


def run_cdss(ctx, facts, dur_pairs=None):
    alerts = []
    F = {f.fid: f for f in facts}

    def fid_of(pred):
        return [f.fid for f in facts if pred(f)]

    def derived(ftype, text, inputs, nums=()):
        """CDSS가 계산한 값(점수·eGFR)을 출처(입력 fact)가 남는 파생 fact로 등록."""
        f = Fact(ftype, "S6", text, 1, concepts={ftype}, key_nums=nums, prov=[{"derived_from": inputs}])
        f.fid = "F%03d" % (len(facts) + 1)
        facts.append(f)
        return f.fid

    def add(rule, sev, title, text, trig, ref):
        alerts.append({"alert_id": "AL%02d" % (len(alerts) + 1), "rule_id": rule, "severity": sev, "title": title, "text": text,
                       "trigger_fact_ids": trig, "reference": ref})

    vit = fid_of(lambda f: f.ftype == "vitals_last")
    score, pts = news2(ctx)
    if score is not None:
        sev = "high" if score >= 7 else "medium" if score >= 5 or 3 in pts.values() else "low"
        if sev != "low":
            inputs = vit + fid_of(lambda f: f.ftype in ("gcs", "resp_support"))
            dtxt = "NEWS2 %s점(%s)" % (score, ", ".join("%s %s" % (k, v) for k, v in pts.items() if v))
            did = derived("score_news2", dtxt + ".", inputs, [score])
            add("NEWS2_AT_TRANSFER", sev, "전동 시점 NEWS2 %s점" % score,
                "전동 시점 %s, 병동 활력징후 측정 간격 단축 검토." % dtxt, [did] + inputs, "RCP NEWS2 (2017)")
    sw, sp = swift(ctx)
    if sw >= 15:
        inputs = fid_of(lambda f: f.ftype in ("icu_stay", "gcs") or (f.ftype == "lab_latest" and f.concepts & {"po2", "pco2"}))
        dtxt = "SWIFT %s점(%s)" % (sw, ", ".join("%s %s" % (k, v) for k, v in sp.items() if v))
        did = derived("score_swift", dtxt + ".", inputs, [sw])
        add("SWIFT_READMIT", "high" if sw > 25 else "medium", "SWIFT %s점: ICU 재입실 위험 증가" % sw,
            "%s, 재입실 위험 증가 범위로 전동 후 집중 관찰 필요." % dtxt, [did] + inputs, "Gajic O. et al., Crit Care Med 2008")
    for f in facts:
        if f.ftype == "infusion_episode" and f.extra.get("cls") == "vasopressor":
            last = pd.Timestamp(f.extra["last"])
            hrs = (ctx.t0 - last).total_seconds() / 3600.0
            if f.status == "active":
                add("PRESSOR_AT_TRANSFER", "high", "승압제 투여 중 전동", "승압제 %s 전동 직전까지 투여 중, 전동 적합성 재확인 필요." % f.extra["generic"], [f.fid], "ICU 퇴실 기준 일반 원칙")
            elif hrs < 24:
                add("PRESSOR_RECENT_OFF", "medium", "승압제 중단 후 하루 미만", "승압제 %s %s 마지막 투여, 중단 후 하루가 지나지 않아 혈압 재저하 관찰 필요." % (f.extra["generic"], fmt_t(last)), [f.fid], "ICU 퇴실 기준 일반 원칙")
        if f.ftype == "infusion_episode" and f.extra.get("cls") == "sedation" and f.status == "active":
            add("SEDATION_AT_TRANSFER", "high", "진정제 지속주입 중 전동", "%s 전동 직전까지 투여 중, 병동 지속주입 가능 여부 확인 필요." % f.extra["generic"], [f.fid], "고위험 약물 관리 원칙")
    rs = fid_of(lambda f: f.ftype == "resp_support" and f.status == "active")
    if rs:
        add("VENT_AT_TRANSFER", "high", "전동 직전 인공호흡기 기록", "전동 전 12시간 내 인공호흡기 기록 있음, 이탈 여부 확인 필요.", rs, "ICU 퇴실 기준 일반 원칙")
    cr_ep = fid_of(lambda f: f.ftype == "crrt_episode")
    for fid in cr_ep:
        add("CRRT_HANDOFF", "high" if F[fid].status == "active" else "medium", "CRRT 시행 이력", "ICU 중 CRRT 시행, 병동에서 신대체요법 계획·투석 접근로 확인 필요.", [fid], "KDIGO AKI 2012")
    aki = fid_of(lambda f: f.ftype == "aki")
    if aki:
        a = ctx.aki
        add("AKI_KDIGO", "high", "급성신손상(KDIGO Cr 기준)", "Cr %s → %s, 신독성 약물 점검과 Cr 추적 필요." % (fmt_num(a["base"]), fmt_num(a["peak"])), aki, "KDIGO AKI 2012")
    if getattr(ctx, "urine_rate", None) is not None and ctx.urine_rate < 0.5 and fid_of(lambda f: f.ftype == "io_24h"):
        io = fid_of(lambda f: f.ftype == "io_24h")
        add("OLIGURIA_24H", "medium", "소변량 감소", "최근 24시간 소변 %s mL/kg/h, 병동 소변량 측정 필요." % fmt_num(round(ctx.urine_rate, 2)), io, "KDIGO AKI 2012")
    for f in facts:
        if f.ftype == "lab_latest" and f.extra.get("critical"):
            c = list(f.concepts)[0]
            add("CRITICAL_LAB_%s" % c.upper(), "high", "%s 위험값" % LAB_DISPLAY.get(c, c), "%s %s 위험값, 전동 전 교정·재검 확인 필요." % (LAB_DISPLAY.get(c, c), fmt_num(f.key_nums[0])), [f.fid], "검사 위험값 보고 원칙")
    lac = getattr(ctx, "latest_labs", {}).get("lactate")
    if lac and 2.0 < lac[0] < 4.0:
        add("LACTATE_NOT_CLEARED", "medium", "Lactate 미정상화", "Lactate %s, 정상화 전 전동으로 재검 필요." % fmt_num(lac[0]),
            fid_of(lambda f: f.ftype == "lab_latest" and "lactate" in f.concepts), "Surviving Sepsis Campaign 2021")
    # 약물
    act = getattr(ctx, "active_meds", [])
    gens = [(g.lower(), cls, name) for g, cls, name, last, route in act]
    med_fid = {list(c for c in f.concepts if c.startswith("med:"))[0][4:]: f.fid for f in facts if f.ftype == "med_active" and any(c.startswith("med:") for c in f.concepts)}
    if any(cls == "insulin" for _, cls, _ in gens):
        ins_drip = (not ctx.emar.empty) and ((ctx.emar["cls"] == "insulin") & (to_num(ctx.emar.get("infusion_rate_hr", pd.Series(dtype=float))).notna() |
                                                                                 to_num(ctx.emar.get("infusion_rate_min", pd.Series(dtype=float))).notna())).any()
        trig = [med_fid[g] for g, cls, _ in gens if cls == "insulin" and g in med_fid]
        add("INSULIN_TRANSITION", "medium", "인슐린 투여 중 전동", "인슐린 %s투여 중, 병동 혈당 측정 계획과 피하 인슐린 전환 확인 필요." % ("지속주입 기록 있음, " if ins_drip else ""), trig, "고위험 약물 관리 원칙")
    cr = getattr(ctx, "latest_labs", {}).get("cr")
    eg = getattr(ctx, "latest_labs", {}).get("egfr")
    egfr = eg[0] if eg else (ckd_epi_2021(cr[0], ctx.age, ctx.sex) if cr else None)
    ctx.egfr_used = egfr
    if egfr is not None:
        for g, cls, name in gens:
            for rx, thr, msg in RENAL_DRUGS:
                if re.search(rx, g + " " + name, re.I) and egfr < thr:
                    trig = [x for x in [med_fid.get(g)] + fid_of(lambda f: f.ftype == "lab_latest" and f.concepts & {"cr", "egfr"}) if x]
                    if not eg:
                        trig.append(derived("calc_egfr", "eGFR %s(CKD-EPI 2021 계산)." % fmt_num(round(egfr, 1)), trig[:], [round(egfr, 1)]))
                    add("RENAL_DOSE", "high", "신기능 기반 용량 경고", "%s 투여 중, eGFR %s(%s). %s." % (g, fmt_num(round(egfr, 1)), "검사값" if eg else "CKD-EPI 2021 계산", msg), trig, "CKD-EPI 2021, 약품 허가사항")
    pairs = dur_pairs or DUR_PAIRS_DEFAULT
    names = " ".join(g + " " + n.lower() for g, _, n in gens)
    for a, b, why in pairs:
        if re.search(a, names, re.I) and re.search(b, names, re.I):
            trig = [fid for g, fid in med_fid.items() if re.search(a, g, re.I) or re.search(b, g, re.I)]
            add("DUR_PAIR", "high", "병용금기 조합", "%s + %s 병용 중: %s." % (a, b, why), trig, "식약처 DUR 병용금기(예시 목록)")
    k = getattr(ctx, "latest_labs", {}).get("k")
    mg = getattr(ctx, "latest_labs", {}).get("mg")
    qt = [g for g, _, n in gens if re.search(QT_DRUGS, g + " " + n, re.I)]
    if qt and ((k and k[0] < 3.5) or (mg and mg[0] < 1.6)):
        trig = [med_fid[g] for g in qt if g in med_fid] + fid_of(lambda f: f.ftype == "lab_latest" and f.concepts & {"k", "mg"})
        add("QT_RISK", "medium", "전해질 저하 + QT 연장 약물", "%s 투여 중 저칼륨·저마그네슘, 전해질 교정·심전도 확인 필요." % ", ".join(qt), trig, "CredibleMeds 목록 기반")
    for f in facts:
        if f.ftype == "micro_pending":
            add("MICRO_PENDING", "medium", "배양 결과 미보고", "배양 결과 미보고 상태로 전동, 병동에서 결과 확인 필요.", [f.fid], "인수인계 원칙(I-PASS)")
        if f.ftype == "micro_positive" and f.extra.get("resistant"):
            res = [x.lower() for x in f.extra["resistant"]]
            abx = [g for g, cls, n in gens if cls == "antibiotic" and any(r.split("/")[0][:6] in (g + n).lower() for r in res)]
            if abx:
                add("ABX_MISMATCH", "high", "항생제-감수성 불일치", "%s 투여 중이나 분리균 내성 보고, 항생제 재검토 필요." % ", ".join(abx), [f.fid] + [med_fid[g] for g in abx if g in med_fid], "IDSA 원칙")
    drains = fid_of(lambda f: f.ftype == "drain")
    if drains:
        add("DRAIN_HANDOFF", "info", "배액관 유지", "배액관 유지 상태로 전동, 병동 배액량·위치 관리 필요.", drains, "인수인계 원칙(I-PASS)")
    return alerts


# =============================================================================
# 5. 생성 모델 A — 플래너 + 템플릿 렌더러(+ 선택적 LLM)
# =============================================================================

SECTION_CAP = {"S1": 6, "S2": 4, "S3": 12, "S4": 14, "S5": 14}


def plan_sections(facts):
    plan = OrderedDict((s, []) for s in SECTIONS if s != "S6")
    for s in plan:
        fs = [f for f in facts if f.section == s]
        must = [f for f in fs if f.must]
        rest = sorted([f for f in fs if not f.must], key=lambda f: -f.salience)
        plan[s] = must + rest[: max(0, SECTION_CAP[s] - len(must))]
    return plan


def illness_severity(ctx, facts, alerts):
    hi = [a for a in alerts if a["severity"] == "high"]
    return ("주의(watcher)" if hi else "안정(stable)"), [x for a in hi for x in a["trigger_fact_ids"]][:6]


def render_template(ctx, facts, alerts):
    """기록지 = 문장 리스트. 문장마다 section, text, cites."""
    plan = plan_sections(facts)
    sents = []
    for s, fs in plan.items():
        if not fs:
            sents.append({"section": s, "text": "해당 사항 없음.", "cites": []})
        for f in fs:
            sents.append({"section": s, "text": "%s [%s]" % (f.text, f.fid), "cites": [f.fid]})
    if alerts:
        for a in alerts:
            sents.append({"section": "S6", "text": "[%s] %s [%s] (규칙 %s)" % (a["severity"].upper(), a["text"], ", ".join(a["trigger_fact_ids"]), a["rule_id"]) if a["trigger_fact_ids"]
                          else "[%s] %s (규칙 %s)" % (a["severity"].upper(), a["text"], a["rule_id"]), "cites": list(a["trigger_fact_ids"]), "alert": a["alert_id"]})
    else:
        sents.append({"section": "S6", "text": "규칙 기반 알림 없음.", "cites": []})
    for i, x in enumerate(sents, 1):
        x["sid"] = "S%03d" % i
    return sents


def note_markdown(ctx, sents, verdicts=None, coverage=None, title="ICU → 병동 전동기록지 (자동 생성 초안, 의료진 검토 필요)"):
    vd = {v["sid"]: v for v in (verdicts or [])}
    lines = ["# " + title, "",
             "- stay_id %s / hadm_id %s / 기준 시각(ICU 퇴실) %s" % (ctx.stay_id, ctx.hadm_id, fmt_t(ctx.t0)),
             "- 생성: kcd_transfer.py v%s, %s" % (VERSION, time.strftime("%Y-%m-%d %H:%M")), ""]
    cur = None
    for x in sents:
        if x["section"] != cur:
            cur = x["section"]
            lines += ["", "## " + SECTIONS.get(cur, cur)]
        mark = ""
        if x["sid"] in vd and vd[x["sid"]]["label"] != "Supported":
            mark = "  ⚠️검증: %s" % vd[x["sid"]]["label"]
        lines.append("- " + x["text"] + mark)
    if coverage is not None:
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


SYS_GROUNDED = ("당신은 ICU에서 병동으로 전동하는 환자의 전동기록지를 작성하는 보조자다. 주어진 facts만 사용한다.\n"
                "규칙: 1) facts에 없는 정보·추정·권고를 쓰지 않는다 2) 문장마다 끝에 사용한 fact_id를 [F001] 형식으로 인용한다\n"
                "3) 수치·단위·시각(MM-DD HH:MM)은 fact 표기를 그대로 복사한다 4) 개조식(~함/~임)으로 쓴다\n"
                "5) 출력은 '- '로 시작하는 문장 목록만 쓴다. 제목·설명은 쓰지 않는다.")


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


def render_llm(ctx, facts, alerts, llm, audit, repair=True, max_rounds=2):
    plan = plan_sections(facts)
    F = {f.fid: f for f in facts}
    sents = []
    for s, fs in plan.items():
        if not fs:
            sents.append({"section": s, "text": "해당 사항 없음.", "cites": []})
            continue
        user = "섹션: %s\nfacts:\n%s" % (SECTIONS[s], "\n".join("%s: %s" % (f.fid, f.text) for f in fs))
        t1 = time.time()
        try:
            out = llm.chat(SYS_GROUNDED, user)
        except Exception as e:
            out = ""
            audit.log("llm_error", {"section": s}, {"error": repr(e)})
        audit.log("llm_render", {"section": s, "facts": [f.fid for f in fs]}, {"n_chars": len(out)}, time.time() - t1, llm.model)
        bl = parse_bullets(out) or [("%s [%s]" % (f.text, f.fid)) for f in fs]
        for b in bl:
            sents.append({"section": s, "text": b, "cites": cites_of(b)})
    for a in alerts:
        sents.append({"section": "S6", "text": "[%s] %s [%s] (규칙 %s)" % (a["severity"].upper(), a["text"], ", ".join(a["trigger_fact_ids"]), a["rule_id"]), "cites": a["trigger_fact_ids"]})
    for i, x in enumerate(sents, 1):
        x["sid"] = "S%03d" % i
    stats = {"repaired": 0, "replaced": 0, "appended": 0}
    if not repair:
        return sents, stats
    V = Verifier(ctx, facts, alerts=alerts)
    for rnd in range(max_rounds):
        bad = [(x, V.check(x)) for x in sents if x["section"] != "S6"]
        bad = [(x, v) for x, v in bad if v["label"] in ("Contradicted", "NotAddressed")]
        if not bad:
            break
        for x, v in bad:
            ev = [F[c] for c in x["cites"] if c in F]
            user = "다음 문장이 근거 facts와 일치하지 않는다(사유: %s). facts만 사용해 한 문장으로 다시 쓰고 인용을 붙여라.\n문장: %s\nfacts:\n%s" % (
                v["reason"], x["text"], "\n".join("%s: %s" % (f.fid, f.text) for f in (ev or plan_sections(facts)[x["section"]])))
            try:
                out = llm.chat(SYS_GROUNDED, user)
                b = parse_bullets(out)
                if b:
                    x["text"] = b[0]
                    x["cites"] = cites_of(b[0])
                    stats["repaired"] += 1
            except Exception as e:
                audit.log("llm_error", {"sid": x["sid"]}, {"error": repr(e)})
    for x in sents:
        if x["section"] == "S6":
            continue
        v = V.check(x)
        if v["label"] in ("Contradicted", "NotAddressed"):
            ev = [F[c] for c in x["cites"] if c in F]
            if ev:
                x["text"] = " ".join("%s [%s]" % (f.text, f.fid) for f in ev)
                x["cites"] = [f.fid for f in ev]
            else:
                x["text"] = "(삭제됨: 근거 없는 문장)"
                x["cites"] = []
            stats["replaced"] += 1
    cov = V.coverage(sents)
    for c in cov:
        if c["status"] == "Missing":
            f = F[c["fact_id"]]
            idx = max([i for i, x in enumerate(sents) if x["section"] == f.section] or [len(sents) - 1])
            sents.insert(idx + 1, {"section": f.section, "text": "%s [%s] (검증기 보완)" % (f.text, f.fid), "cites": [f.fid]})
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
    lines.append("환자: %s세 %s, ICU 입실 %s, 퇴실 %s" % (getattr(ctx, "age", "?"), getattr(ctx, "sex", "?"), ctx.intime, ctx.outtime))
    dump("DIAGNOSES_ICD", ctx.dx, ["icd_code", "seq_num"], 10)
    dump("CHARTEVENTS(최근)", ctx.last_window(ctx.chart, 12) if not ctx.chart.empty else ctx.chart, ["charttime", "itemid", "value", "valueuom"], 80)
    dump("LABEVENTS(최근 48h)", ctx.last_window(ctx.lab, 48) if not ctx.lab.empty else ctx.lab, ["charttime", "label", "value", "valueuom"], 80)
    dump("EMAR", ctx.icu_slice(ctx.emar) if not ctx.emar.empty else ctx.emar, ["charttime", "medication", "infusion_rate_hr", "infusion_rate_min"], 60)
    dump("PRESCRIPTIONS", ctx.rx, ["starttime", "stoptime", "drug", "dose_val_rx", "dose_unit_rx", "route"], 40)
    dump("OUTPUTEVENTS(24h)", ctx.last_window(ctx.out, 24) if not ctx.out.empty else ctx.out, ["charttime", "label", "value"], 40)
    dump("INPUTEVENTS(24h)", ctx.last_window(ctx.inp, 24) if not ctx.inp.empty else ctx.inp, ["starttime", "label", "amount"], 30)
    dump("PROCEDUREEVENTS", ctx.icu_slice(ctx.proc) if not ctx.proc.empty else ctx.proc, ["starttime", "label"], 20)
    return "\n".join(lines[:max_lines])


def render_naive(ctx, llm, audit):
    sys_ = "당신은 ICU 의사다. 아래 환자 데이터로 ICU에서 병동으로 전동하는 환자의 전동기록지를 작성하라. '- '로 시작하는 개조식 문장 목록으로, 다음 섹션 제목을 '## '로 붙여 쓴다: " + ", ".join(v for k, v in SECTIONS.items() if k != "S6")
    t1 = time.time()
    out = llm.chat(sys_, naive_prompt(ctx), max_tokens=2000)
    audit.log("llm_naive", {"stay": ctx.stay_id}, {"n_chars": len(out)}, time.time() - t1, llm.model)
    sents, sec = [], "S1"
    names = {v: k for k, v in SECTIONS.items()}
    for line in out.splitlines():
        l = line.strip()
        if l.startswith("#"):
            h = l.lstrip("#").strip()
            for v, k in names.items():
                if v[3:7] in h or h in v:
                    sec = k
        elif l.startswith(("- ", "* ")):
            sents.append({"section": sec, "text": l[2:], "cites": []})
    for i, x in enumerate(sents, 1):
        x["sid"] = "S%03d" % i
    return sents


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
KNOWN_MEDS = sorted(set(re.findall(r"[a-z]{4,}", "|".join(rx for _, rx, _ in MED_CLASSES))) - {"insulin", "saline", "dextrose", "solution", "plasma", "ringer"})
TEMPLATE_LEX = set("""최종 활력징후 기준 최근 시간 범위 의식 기록 호흡 전동 인공호흡기 산소유량 없음 있음 참고범위 검사 위험값 높음 낮음 섭취 배설 소변 균형 배액관 유지 배액
투여 마지막 중단 사용 승압제 진정 진통제 인슐린 항응고제 항혈소판제 항생제 항진균제 이뇨제 수액 입실 퇴실 재실 입원 경로 병동 이동 주진단 기타 진단 수술 시술 코드 관련 경과
최고 최저 상승 배양 동정 미보고 결과 내성 동정균 영상검사 판독문 데이터 해당 사항 규칙 알림 필요 확인 주입속도 기록값 후 직전 까지 중 이후 지속 시행 회 총 제거량 남성 여성 나이 미상
모드 날짜 명세 상태 관찰 주의 계획 측정 간격 단축 검토 재확인 적합성 가능 여부 교정 재검 추적 점검 신독성 약물 전환 혈당 피하 저칼륨 저마그네슘 전해질 심전도 정상화 재저하
혈압 이상 미만 증가 집중 위험 재입실 신대체요법 투석 접근로 범위로 신기능 용량 경고 계산 감수성 불일치 재검토 분리균 보고 위치 관리 개 검증기 보완 삭제됨 문장 근거""".split())


class Verifier(object):
    def __init__(self, ctx, facts, judge=None, alerts=None):
        self.alert_lex = set()
        self.alert_texts = [x.get("text", "") for x in (alerts or [])]
        for a in (alerts or []):
            self.alert_lex |= set(re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}|[가-힣]{2,}", a.get("text", "") + " " + a.get("title", "")))
        self.ctx = ctx
        self.facts = facts
        self.F = {f.fid: f for f in facts}
        self.judge = judge
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
    def mentions(self, text):
        t = CITE_RX.sub(" ", text)
        t = re.sub(r"\(규칙 [^)]*\)", " ", t)
        found = set()
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
        if re.match(r"^(해당 사항 없음|규칙 기반 알림 없음)", text) or text.startswith("(삭제됨"):
            return {"sid": s["sid"], "label": "Supported", "method": "empty", "reason": "", "citation_error": False}
        cites = [c for c in s.get("cites", []) if c in self.F]
        bad_cite = [c for c in s.get("cites", []) if c not in self.F]
        cf = [self.F[c] for c in cites]
        nums = extract_nums(text)
        times = extract_times(text)
        ment = self.mentions(text)
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
            if s.get("section") == "S6" and c in ALIAS and any(re.search(ALIAS[c], t, re.I) for t in self.alert_texts):
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
        st_active = bool(re.search(r"투여 중|유지 중|적용 중|지속 중|투여중|사용 중|처방 유지", text))
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
            if s.get("section") == "S6":
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
            if t in src_toks or t in TEMPLATE_LEX or t.lower() in self.med_names:
                continue
            if re.match(r"[가-힣]", t) and (t[:2] in src_stems or t[:-1] in TEMPLATE_LEX or t[:-2] in TEMPLATE_LEX or t.rstrip("함임됨음") in TEMPLATE_LEX):
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

    def coverage(self, sents, verdicts=None):
        vd = {v["sid"]: v for v in (verdicts or self.verify(sents))}
        res = []
        for f in self.facts:
            if not f.must:
                continue
            cand = []
            mention_only = False
            for s in sents:
                v = vd.get(s["sid"])
                if v and v["label"] in ("Contradicted", "NotAddressed"):
                    continue
                if f.fid in s.get("cites", []):
                    cand.append(s)
                    continue
                m = self.mentions(s["text"])
                key_c = {c for c in f.concepts if not c.startswith("cls:")}
                if m & key_c:
                    if f.key_nums:
                        ok = any(self._num_match(x, f.key_nums) for x in extract_nums(s["text"]))
                    elif f.allowed_times:  # 수치 없는 사건 fact: 같은 시각 언급이 있어야 전달로 인정(부정문 오인 방지)
                        ok = bool(set(extract_times(s["text"])) & set(f.allowed_times))
                    else:
                        ok = not re.search(r"없음|없다|기록 없|no record", s["text"])
                    if ok:
                        cand.append(s)
                    else:
                        mention_only = True
            if cand:
                if f.key_nums:
                    got = [n for n in f.key_nums if any(self._num_match(x, [n]) for s in cand for x in extract_nums(s["text"]))]
                    frac = len(got) / float(len(f.key_nums))
                else:
                    frac = 1.0
                    if f.status:
                        txt = " ".join(s["text"] for s in cand)
                        if f.status == "active" and re.search(r"중단", txt) and not re.search(r"투여 중|유지|지속|직전까지", txt):
                            frac = 0.5
                st = "Covered" if frac >= 0.999 else "Partial"
            else:
                st, frac = ("Partial", 0.0) if mention_only else ("Missing", 0.0)
            res.append({"fact_id": f.fid, "fact_type": f.ftype, "salience": f.salience, "status": st, "frac": round(frac, 2), "text": f.text})
        return res


def metrics(verdicts, coverage, alerts=None):
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
    if alerts is not None:
        out["alerts_high"] = sum(1 for a in alerts if a["severity"] == "high")
        out["alerts_total"] = len(alerts)
    return out


# =============================================================================
# 7. B3 — 합성 오류 주입으로 검증기 자체 평가
# =============================================================================

SWAP_MEDS = ["Dopamine", "Vancomycin", "Meropenem", "Heparin", "Midazolam", "Amiodarone", "Dobutamine", "Ceftriaxone"]
FABRICATIONS = ["기관절개술 시행함.", "ECMO 적용 중임.", "혈액배양에서 MRSA 동정됨.", "농축적혈구 2단위 수혈함.", "심폐소생술 시행 이력 있음."]


def perturb(sents, facts, rng, kind):
    """반환: 새 문장 리스트, 정답 {'kind', 'sid' 또는 'fact_id'}"""
    S = [dict(x) for x in sents]
    body = [i for i, x in enumerate(S) if x["section"] != "S6" and x["cites"]]
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
        cand = [i for i in body if re.search(r"투여 중|중단", S[i]["text"]) and any(F[c].status in ("active", "stopped") for c in S[i]["cites"] if c in F)]
        if not cand:
            return None
        i = rng.choice(cand)
        t = S[i]["text"]
        if "투여 중" in t:
            t = re.sub(r"전동 직전까지 투여 중|투여 중", "중단", t, count=1)
        else:
            t = re.sub(r"[^,]*마지막 투여 후 중단|사용 후 중단|중단", "투여 중", t, count=1)
        S[i]["text"] = t
        return S, {"kind": kind, "sid": S[i]["sid"]}
    if kind == "fabricate":
        rec_ev = set(c for f in facts for c in f.concepts)
        fab = [x for x in FABRICATIONS if not any(re.search(rx, x, re.I) and ev in rec_ev for ev, rx in EVENT_VOCAB)]
        if not fab:
            return None
        i = rng.choice(body)
        new = {"section": S[i]["section"], "text": "%s [%s]" % (rng.choice(fab), S[i]["cites"][0]), "cites": [S[i]["cites"][0]], "sid": "SFAB"}
        S.insert(i + 1, new)
        return S, {"kind": kind, "sid": "SFAB"}
    if kind == "delete":
        must = [f for f in facts if f.must and any(f.fid in x["cites"] for x in S)]
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
                                     "secs": round(secs, 3), "model": model, "prompt_version": PROMPT_VERSION, "code_version": VERSION},
                                    ensure_ascii=False, default=str) + "\n")

    def close(self):
        if self.f:
            self.f.close()


class Pipeline(object):
    def __init__(self, data_dir, dur_csv=None, verbose=True):
        self.kd = KData(data_dir, verbose=verbose)
        self.item_map = map_items(self.kd.get("D_ITEMS"))
        self.lab_map = map_labs(self.kd.get("D_LABITEMS"))
        self.d_test = self.kd.get("D_TESTITEMS")
        self.cands = candidate_stays(self.kd)
        self.dur = None
        if dur_csv:
            d = pd.read_csv(dur_csv, dtype=str)
            cols = [c.lower() for c in d.columns]
            d.columns = cols
            self.dur = list(zip(d.iloc[:, 0].str.lower(), d.iloc[:, 1].str.lower(), d.iloc[:, 2] if d.shape[1] > 2 else ["병용금기"] * len(d)))
        if verbose:
            um = sum(1 for v in self.item_map.values() if v[0] is None)
            ul = sum(1 for v in self.lab_map.values() if v[0] is None)
            print("[map] D_ITEMS %d개 중 개념 미매핑 %d, D_LABITEMS %d개 중 미매핑 %d" % (len(self.item_map), um, len(self.lab_map), ul))
            print("[cand] 전동기록지 대상 ICU stay %d건" % len(self.cands))

    def preload(self, stays):
        h = self.cands[self.cands["stay_id"].isin(stays)]["hadm_id"].tolist()
        for t in ["CHARTEVENTS", "LABEVENTS", "INPUTEVENTS", "OUTPUTEVENTS", "PROCEDUREEVENTS", "EMAR", "EMAR_DETAIL", "PRESCRIPTIONS",
                  "MICROBIOLOGYEVENTS", "TESTEVENTS", "DATETIMEEVENTS"]:
            if self.kd.has(t):
                df = self.kd.get(t, h)
                # stay 단위 sub() 가 같은 캐시를 쓰도록 단일 hadm 키에도 등록
                for hid, g in (df.groupby("hadm_id") if (not df.empty and "hadm_id" in df.columns) else []):
                    self.kd.cache[(t, (hid,))] = g
                for hid in h:
                    self.kd.cache.setdefault((t, (hid,)), df.iloc[0:0] if not df.empty else df)

    def context(self, stay_id):
        r = self.cands[self.cands["stay_id"] == str(stay_id)]
        if r.empty:
            raise SystemExit("stay_id %s 가 전동 대상 후보에 없습니다(사망 퇴실이거나 ICUSTAYS에 없음)." % stay_id)
        return StayContext(self.kd, r.iloc[0], self.item_map, self.lab_map, self.d_test)

    def process(self, stay_id, out_dir=None, mode="template", llm=None, write=True):
        t1 = time.time()
        ctx = self.context(stay_id)
        sd = os.path.join(out_dir, "stay_%s" % stay_id) if out_dir else None
        if sd and write:
            os.makedirs(sd, exist_ok=True)
        audit = Audit(os.path.join(sd, "audit.jsonl") if (sd and write) else None)
        facts = build_facts(ctx)
        audit.log("build_facts", {"stay_id": stay_id, "t0": str(ctx.t0)}, {"n_facts": len(facts), "n_must": sum(f.must for f in facts)}, time.time() - t1)
        alerts = run_cdss(ctx, facts, self.dur)
        audit.log("cdss", {"facts": [f.fid for f in facts]}, {"alerts": [a["rule_id"] for a in alerts]})
        stats = {}
        if mode == "template":
            sents = render_template(ctx, facts, alerts)
        elif mode == "grounded":
            sents, stats = render_llm(ctx, facts, alerts, llm, audit, repair=False)
        elif mode == "repair":
            sents, stats = render_llm(ctx, facts, alerts, llm, audit, repair=True)
        elif mode == "naive":
            sents = render_naive(ctx, llm, audit)
        else:
            raise ValueError(mode)
        judge = llm if (llm is not None and llm.ok) else None
        V = Verifier(ctx, facts, judge=judge, alerts=alerts)
        verdicts = V.verify(sents)
        cov = V.coverage(sents, verdicts)
        m = metrics(verdicts, cov, alerts)
        m.update({"stay_id": stay_id, "mode": mode, "n_facts": len(facts), "secs": round(time.time() - t1, 2)})
        m.update({"repair_" + k: v for k, v in stats.items()})
        audit.log("verify", {"n_sent": len(sents)}, m)
        if sd and write:
            with open(os.path.join(sd, "note_%s.md" % mode), "w", encoding="utf-8") as f:
                f.write(note_markdown(ctx, sents, verdicts, cov))
            with open(os.path.join(sd, "factsheet.json"), "w", encoding="utf-8") as f:
                json.dump({"stay_id": stay_id, "hadm_id": ctx.hadm_id, "t0": str(ctx.t0), "facts": [x.to_dict() for x in facts]}, f, ensure_ascii=False, indent=1, default=str)
            with open(os.path.join(sd, "verify_%s.json" % mode), "w", encoding="utf-8") as f:
                json.dump({"metrics": m, "verdicts": verdicts, "coverage": cov, "sentences": sents}, f, ensure_ascii=False, indent=1, default=str)
            with open(os.path.join(sd, "alerts.json"), "w", encoding="utf-8") as f:
                json.dump(alerts, f, ensure_ascii=False, indent=1, default=str)
        audit.close()
        return {"ctx": ctx, "facts": facts, "alerts": alerts, "sents": sents, "verdicts": verdicts, "coverage": cov, "metrics": m}


def pick_stays(pipe, stay, n, seed):
    if stay and stay != "auto":
        return [str(stay)]
    ids = pipe.cands["stay_id"].tolist()
    if stay == "auto":
        return ids[:1]
    rnd = random.Random(seed)
    return ids if n >= len(ids) else sorted(rnd.sample(ids, n))


def write_summary(rows, out_dir, name):
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, name + ".csv"), index=False)
    num = df.select_dtypes("number")
    agg = num.mean().round(4)
    with open(os.path.join(out_dir, name + ".md"), "w", encoding="utf-8") as f:
        f.write("# %s (n=%d)\n\n| 지표 | 평균 |\n|---|---|\n" % (name, len(df)))
        for k, v in agg.items():
            f.write("| %s | %s |\n" % (k, v))
    return df, agg


def cmd_list(a):
    p = Pipeline(a.data, verbose=True)
    print("후보 stay 수: %d" % len(p.cands))
    print("앞쪽 stay_id 20개:", ", ".join(p.cands["stay_id"].head(20).tolist()))
    los = (p.cands["outtime_dt"] - p.cands["intime_dt"]).dt.total_seconds() / 86400
    print("ICU 재실일 분포: 중앙 %.1f, IQR %.1f~%.1f" % (los.median(), los.quantile(.25), los.quantile(.75)))


def cmd_run(a):
    p = Pipeline(a.data, a.dur_csv)
    os.makedirs(a.out, exist_ok=True)
    llm = LLM()
    mode = a.mode
    if mode != "template" and not llm.ok:
        raise SystemExit("LLM 모드는 KCD_LLM_BASE_URL, KCD_LLM_MODEL 환경변수가 필요합니다.")
    for sid in pick_stays(p, a.stay or "auto", 1, a.seed):
        p.preload([sid])
        r = p.process(sid, a.out, mode=mode, llm=llm if llm.ok else None)
        print("\n" + note_markdown(r["ctx"], r["sents"], r["verdicts"], r["coverage"]))
        print("[metrics]", json.dumps(r["metrics"], ensure_ascii=False))
        print("[out] %s" % os.path.join(a.out, "stay_%s" % sid))


def cmd_batch(a):
    p = Pipeline(a.data, a.dur_csv)
    os.makedirs(a.out, exist_ok=True)
    stays = pick_stays(p, None, a.n, a.seed)
    p.preload(stays)
    rows, rule_counts = [], defaultdict(int)
    for i, sid in enumerate(stays, 1):
        try:
            r = p.process(sid, a.out, mode="template", write=not a.no_write)
            rows.append(r["metrics"])
            for al in r["alerts"]:
                rule_counts[al["rule_id"].split("_LAB_")[0] if "CRITICAL_LAB" in al["rule_id"] else al["rule_id"]] += 1
        except Exception as e:
            print("[warn] stay %s 실패: %r" % (sid, e))
        if i % 10 == 0:
            print("  %d/%d" % (i, len(stays)))
    df, agg = write_summary(rows, a.out, "summary_template")
    with open(os.path.join(a.out, "cdss_rule_counts.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rule_counts), f, ensure_ascii=False, indent=1)
    print(agg.to_string())
    print("CDSS 규칙별 발화 stay 수:", dict(rule_counts))


def cmd_perturb(a):
    p = Pipeline(a.data, a.dur_csv)
    os.makedirs(a.out, exist_ok=True)
    stays = pick_stays(p, None, a.n, a.seed)
    p.preload(stays)
    rng = random.Random(a.seed)
    kinds = ["num_change", "drug_swap", "time_shift", "status_flip", "fabricate", "delete"]
    res = defaultdict(lambda: {"n": 0, "detected": 0})
    fp_n = fp_bad = 0
    clean_n = clean_bad = 0
    for sid in stays:
        try:
            r = p.process(sid, None, mode="template", write=False)
        except Exception as e:
            print("[warn] stay %s 실패: %r" % (sid, e))
            continue
        V = Verifier(r["ctx"], r["facts"], alerts=r["alerts"])
        clean_v = V.verify(r["sents"])
        clean_n += len(clean_v)
        clean_bad += sum(1 for v in clean_v if v["label"] != "Supported")
        for k in kinds:
            for _ in range(a.k):
                out = perturb(r["sents"], r["facts"], rng, k)
                if out is None:
                    continue
                S, truth = out
                vd = V.verify(S)
                res[k]["n"] += 1
                if k == "delete":
                    cov = V.coverage(S, vd)
                    st = [c for c in cov if c["fact_id"] == truth["fact_id"]]
                    if st and st[0]["status"] in ("Missing", "Partial"):
                        res[k]["detected"] += 1
                else:
                    tv = [v for v in vd if v["sid"] == truth["sid"]]
                    if tv and tv[0]["label"] != "Supported":
                        res[k]["detected"] += 1
                    others = [v for v in vd if v["sid"] != truth["sid"]]
                    fp_n += len(others)
                    fp_bad += sum(1 for v in others if v["label"] != "Supported")
    rows = [{"perturbation": k, "n": v["n"], "detected": v["detected"], "recall": round(v["detected"] / float(v["n"]), 4) if v["n"] else None} for k, v in res.items()]
    rows.append({"perturbation": "clean_sentences_false_flag", "n": clean_n, "detected": clean_bad, "recall": round(clean_bad / float(clean_n), 4) if clean_n else None})
    rows.append({"perturbation": "unperturbed_in_perturbed_false_flag", "n": fp_n, "detected": fp_bad, "recall": round(fp_bad / float(fp_n), 4) if fp_n else None})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "perturb_eval.csv"), index=False)
    print(df.to_string(index=False))


def cmd_arms(a):
    llm = LLM()
    if not llm.ok:
        raise SystemExit("arms 비교에는 LLM이 필요합니다(KCD_LLM_BASE_URL, KCD_LLM_MODEL).")
    p = Pipeline(a.data, a.dur_csv)
    os.makedirs(a.out, exist_ok=True)
    stays = pick_stays(p, None, a.n, a.seed)
    p.preload(stays)
    rows = []
    for sid in stays:
        for mode in ("naive", "grounded", "repair", "template"):
            try:
                r = p.process(sid, a.out, mode=mode, llm=llm)
                rows.append(r["metrics"])
            except Exception as e:
                print("[warn] stay %s %s 실패: %r" % (sid, mode, e))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "arms.csv"), index=False)
    cols = ["hallucination_rate", "contradiction_rate", "not_addressed", "unverifiable", "omission_rate", "weighted_omission", "critical_missing", "n_sentences", "secs"]
    print(df.groupby("mode")[[c for c in cols if c in df.columns]].mean().round(4).to_string())


def kcd_main(argv=None):
    if isinstance(argv, str):
        argv = argv.split()
    ap = argparse.ArgumentParser(description="K-MIMIC ICU→병동 전동기록지 생성·검증·CDSS")
    sp = ap.add_subparsers(dest="cmd")
    for name in ("list", "run", "batch", "perturb", "arms"):
        s = sp.add_parser(name)
        s.add_argument("--data", required=True)
        s.add_argument("--out", default="out")
        s.add_argument("--stay", default=None)
        s.add_argument("--n", type=int, default=30)
        s.add_argument("--k", type=int, default=2, help="perturb: stay당 유형별 주입 횟수")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--mode", default="template", choices=["template", "grounded", "repair", "naive"])
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
    {"list": cmd_list, "run": cmd_run, "batch": cmd_batch, "perturb": cmd_perturb, "arms": cmd_arms}[a.cmd](a)


if __name__ == "__main__" and not in_notebook_paste():
    kcd_main()
