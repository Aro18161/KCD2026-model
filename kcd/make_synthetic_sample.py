#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_synthetic_sample.py — K-MIMIC 데이터톤 공개판과 같은 컬럼 구조의 아주 작은 합성 CSV 생성.
실제 환자 데이터가 아니다. 로컬에서 파이프라인을 끝까지 실행·테스트하기 위한 용도다.

시나리오
  stay 1 (subject 1, 67M): 패혈성 쇼크 + AKI. 승압제 증량, 삽관, 혈액배양 E. coli(pip/tazo 내성), 반코마이신 농도 미측정.
  stay 2 (subject 2, 82F): 수술 후 안정. eGFR<30 + 에녹사파린, 저칼륨 + 온단세트론(QT), DUR 병용금기 예시 쌍, 인슐린 주입 중 혈당 간격 공백.
  stay 3 (subject 3, 45M): 시술 후 관찰. 거의 정상 → 비어 있는 섹션 테스트.

사용: python make_synthetic_sample.py [--out data/synthetic]
의존성: 표준 라이브러리만.
"""
import argparse
import csv
import os
from datetime import datetime, timedelta

FMT = "%Y-%m-%d %H:%M:%S"


def ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M")


def f(dt):
    return dt.strftime(FMT) if dt else ""


COLUMNS = {
    "ADMISSIONS": ["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "DEATHTIME", "ADMISSION_TYPE", "ADMISSION_LOCATION", "DISCHARGE_LOCATION", "INSURANCE", "MARITAL_STATUS", "EDREGTIME", "EDOUTTIME", "HOSPITAL_EXPIRE_FLAG", "ICU_EXPIRE_FLAG", "NATIONALITY", "HOSPITAL_ID"],
    "PATIENTS": ["SUBJECT_ID", "SEX", "ANCHOR_AGE", "ANCHOR_YEAR", "ANCHOR_YEAR_GROUP", "DOD"],
    "TRANSFERS": ["SUBJECT_ID", "HADM_ID", "TRANSFER_SEQ", "TRANSFER_ID", "EVENTTYPE", "CAREUNIT", "INTIME", "OUTTIME"],
    "DIAGNOSES_ICD": ["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD_CODE", "ICD_VERSION", "ITEMID", "IS_ICU"],
    "D_LABITEMS": ["ITEMID", "LABEL", "FLUID", "CATEGORY", "EDI_CODE"],
    "D_TESTITEMS": ["ITEMID", "LABEL", "CATEGORY", "EDI_CODE"],
    "EMAR": ["SUBJECT_ID", "HADM_ID", "EMAR_TYPE", "EMAR_ID", "EMAR_SEQ", "POE_ID", "PHARMACY_ID", "CHARTTIME", "MEDICATION", "EVENT_TXT", "STORETIME", "ITEMID", "STAY_ID"],
    "EMAR_DETAIL": ["SUBJECT_ID", "HADM_ID", "EMAR_ID", "EMAR_SEQ", "PARENT_FIELD_ORDINAL", "PHARMACY_ID", "NOTE_TXT", "ON_OFF", "INFUSION_RATE_DAY", "INFUSION_RATE_HR", "INFUSION_RATE_MIN", "ITEMID", "STAY_ID"],
    "LABEVENTS": ["LABEVENT_ID", "SUBJECT_ID", "HADM_ID", "SPECIMEN_ID", "ITEMID", "CHARTTIME", "STORETIME", "VALUE", "VALUENUM", "VALUEUOM", "REF_RANGE_LOWER", "REF_RANGE_UPPER", "FLAG", "COMMENTS", "STAY_ID"],
    "MICROBIOLOGYEVENTS": ["MICROEVENT_ID", "SUBJECT_ID", "HADM_ID", "MICRO_SPECIMEN_ID", "CHARTDATE", "CHARTTIME", "SPEC_ITEMID", "SPEC_TYPE_DESC", "TEST_SEQ", "STOREDATE", "STORETIME", "TEST_ITEMID", "TEST_NAME", "ORG_ITEMID", "ORG_NAME", "ISOLATE_NUM", "QUANTITY", "AB_ITEMID", "AB_NAME", "DILUTION_TEXT", "DILUTION_COMPARISON", "DILUTION_VALUE", "INTERPRETATION", "COMMENTS", "ORDERDATE", "ORDER_COMMENTS", "STAY_ID"],
    "PRESCRIPTIONS": ["SUBJECT_ID", "HADM_ID", "PHARMACY_ID", "STARTTIME", "STOPTIME", "DRUG_TYPE", "DRUG", "GSN", "NDC", "PROD_STRENGTH", "FORM_RX", "DOSE_VAL_RX", "DOSE_UNIT_RX", "FORM_VAL_DISP", "FORM_UNIT_DISP", "DOSES_PER_24_HRS", "ROUTE", "ITEMID", "STAY_ID"],
    "PROCEDURES_ICD": ["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "CHARTDATE", "ICD_CODE", "ICD_VERSION", "ITEMID", "STAY_ID"],
    "SERVICES": ["SUBJECT_ID", "HADM_ID", "TRANSFERTIME", "PREV_SERVICE", "CURR_SERVICE"],
    "TESTEVENTS": ["TESTEVENT_ID", "SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "STORETIME", "FILE_PATH", "STAY_ID"],
    "CHARTEVENTS": ["CHARTEVENT_ID", "SUBJECT_ID", "HADM_ID", "STAY_ID", "CHARTTIME", "STORETIME", "ITEMID", "VALUE", "VALUENUM", "VALUEUOM", "WARNING"],
    "DATETIMEEVENTS": ["DATETIMEEVENT_ID", "SUBJECT_ID", "HADM_ID", "STAY_ID", "CHARTTIME", "STORETIME", "ITEMID", "VALUE", "VALUEUOM", "WARNING"],
    "D_ITEMS": ["ITEMID", "LABEL", "ABBREVIATION", "LINKSTO", "CATEGORY", "UNITNAME", "PARAM_TYPE", "LOWNORMALVALUE", "HIGHNORMALVALUE"],
    "ICUSTAYS": ["SUBJECT_ID", "HADM_ID", "STAY_ID", "FIRST_CAREUNIT", "LAST_CAREUNIT", "INTIME", "OUTTIME", "LOS", "OP_FLAG"],
    "INPUTEVENTS": ["INPUTEVENT_ID", "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTTIME", "ENDTIME", "ITEMID", "AMOUNT", "AMOUNTUOM", "RATE", "RATEUOM", "PATIENTWEIGHT", "STORETIME", "CGID", "ORDERID", "LINKORDERID"],
    "OUTPUTEVENTS": ["OUTPUTEVENT_ID", "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "CHARTTIME", "STORETIME", "ITEMID", "VALUE", "VALUEUOM", "WARNING"],
    "PROCEDUREEVENTS": ["PROCEDUREEVENT_ID", "SUBJECT_ID", "HADM_ID", "STAY_ID", "STARTTIME", "ENDTIME", "STORETIME", "ITEMID", "VALUE", "VALUEUOM", "LOCATION", "LOCATIONCATEGORY", "ORDERCATEGORYNAME", "SECONDARYORDERCATEGORYNAME", "ORDERCATEGORYDESCRIPTION", "PATIENTWEIGHT", "ORIGINALAMOUNT", "ORIGINALRATE"],
    "EDSTAY": ["SUBJECT_ID", "HADM_ID", "STAY_ID", "INTIME", "OUTTIME", "ARRIVAL_TRANSPORT", "DISPOSITION"],
    "TRIAGE": ["SUBJECT_ID", "STAY_ID", "TEMPERATURE", "HEARTRATE", "RESPRATE", "O2SAT", "SBP", "DBP", "PAIN", "ACUITY", "CHIEFCOMPLAINT"],
    "MEDRECON": ["SUBJECT_ID", "STAY_ID", "CHARTTIME", "NAME", "GSN", "NDC", "ETC_RN", "ETCCODE", "ETCDESCRIPTION"],
    "HOSPITAL": ["HOSPITAL_ID", "HOSPITAL_NAME", "HOSPITAL_TYPE", "REGION"],
}

# ---- 사전(label) : 한글·영문 혼재로 정규화 사전을 테스트 ----
D_ITEMS = [
    # itemid, label, abbr, linksto, category, unit, param_type, low, high
    (220045, "Heart Rate", "HR", "chartevents", "Routine Vital Signs", "bpm", "Numeric", 60, 100),
    (220050, "Arterial Blood Pressure systolic", "ABPs", "chartevents", "Routine Vital Signs", "mmHg", "Numeric", 90, 140),
    (220051, "Arterial Blood Pressure diastolic", "ABPd", "chartevents", "Routine Vital Signs", "mmHg", "Numeric", 60, 90),
    (220052, "Arterial Blood Pressure mean", "ABPm", "chartevents", "Routine Vital Signs", "mmHg", "Numeric", 65, 110),
    (220210, "호흡수", "RR", "chartevents", "Routine Vital Signs", "insp/min", "Numeric", 12, 20),
    (220277, "SpO2", "SpO2", "chartevents", "Routine Vital Signs", "%", "Numeric", 95, 100),
    (223762, "체온(℃)", "Temp C", "chartevents", "Routine Vital Signs", "°C", "Numeric", 36.0, 37.5),
    (220739, "GCS - Eye Opening", "GCS E", "chartevents", "Neurological", "", "Numeric", 4, 4),
    (223900, "GCS - Verbal Response", "GCS V", "chartevents", "Neurological", "", "Numeric", 5, 5),
    (223901, "GCS - Motor Response", "GCS M", "chartevents", "Neurological", "", "Numeric", 6, 6),
    (227011, "GCS Total", "GCS", "chartevents", "Neurological", "", "Numeric", 15, 15),
    (223835, "Inspired O2 Fraction", "FiO2", "chartevents", "Respiratory", "", "Numeric", 21, 100),
    (220339, "PEEP set", "PEEP", "chartevents", "Respiratory", "cmH2O", "Numeric", 5, 10),
    (223849, "Ventilator Mode", "Vent Mode", "chartevents", "Respiratory", "", "Text", "", ""),
    (224639, "Daily Weight", "Weight", "chartevents", "General", "kg", "Numeric", "", ""),
    (221906, "Norepinephrine", "NE", "inputevents", "Medications", "mg", "Numeric", "", ""),
    (222168, "Propofol", "Propofol", "inputevents", "Medications", "mg", "Numeric", "", ""),
    (223258, "Insulin - Regular", "Insulin", "inputevents", "Medications", "units", "Numeric", "", ""),
    (225158, "NaCl 0.9%", "NS", "inputevents", "Fluids - Other (Not In Use)", "mL", "Numeric", "", ""),
    (226559, "Foley (소변량)", "Foley", "outputevents", "Output", "mL", "Numeric", "", ""),
    (224385, "Intubation", "Intubation", "procedureevents", "Intubation/Extubation", "", "Text", "", ""),
    (225792, "Invasive Ventilation", "Invasive Vent", "procedureevents", "Ventilation", "", "Text", "", ""),
    (225315, "중심정맥관 삽입 (Central Venous Line)", "CVL", "procedureevents", "Access Lines - Invasive", "", "Text", "", ""),
    (229351, "Foley Catheter Insertion Date", "Foley Ins", "datetimeevents", "Access Lines - Invasive", "", "Date", "", ""),
    (225752, "Arterial Line", "A-line", "procedureevents", "Access Lines - Invasive", "", "Text", "", ""),
]
D_LABITEMS = [
    (50912, "Creatinine", "Blood", "Chemistry", "B2520"),
    (50813, "Lactate", "Blood", "Blood Gas", "B2670"),
    (50971, "Potassium", "Blood", "Chemistry", "B2600"),
    (50983, "나트륨 (Sodium)", "Blood", "Chemistry", "B2590"),
    (51301, "White Blood Cells", "Blood", "Hematology", "B1010"),
    (50889, "C-Reactive Protein", "Blood", "Chemistry", "C2610"),
    (51222, "Hemoglobin", "Blood", "Hematology", "B1010"),
    (51265, "Platelet Count", "Blood", "Hematology", "B1010"),
    (50885, "Bilirubin, Total", "Blood", "Chemistry", "B2500"),
    (50821, "pO2", "Blood", "Blood Gas", "B2650"),
    (50931, "Glucose", "Blood", "Chemistry", "B2540"),
    (51237, "INR(PT)", "Blood", "Hematology", "B1430"),
    (51275, "PTT", "Blood", "Hematology", "B1440"),
    (50960, "Magnesium", "Blood", "Chemistry", "B2610"),
    (51009, "Vancomycin", "Blood", "Chemistry", "B4010"),
    (50882, "Bicarbonate", "Blood", "Chemistry", "B2510"),
    (51006, "Urea Nitrogen", "Blood", "Chemistry", "B2530"),
]
D_TESTITEMS = [(90001, "Chest X-ray (PA)", "Radiology", "G1101"), (90002, "CT Abdomen", "Radiology", "HA441")]
HOSPITAL = [(268, "Synthetic General Hospital", "Tertiary", "Seoul")]

LAB_REF = {50912: (0.5, 1.2, "mg/dL"), 50813: (0.5, 2.0, "mmol/L"), 50971: (3.5, 5.1, "mmol/L"), 50983: (135, 145, "mmol/L"),
           51301: (4.0, 10.0, "K/uL"), 50889: (0, 0.5, "mg/dL"), 51222: (12.0, 16.0, "g/dL"), 51265: (150, 400, "K/uL"),
           50885: (0.2, 1.2, "mg/dL"), 50821: (80, 100, "mmHg"), 50931: (70, 140, "mg/dL"), 51237: (0.9, 1.2, ""),
           51275: (25, 36, "sec"), 50960: (1.7, 2.4, "mg/dL"), 51009: (10, 20, "ug/mL"), 50882: (22, 29, "mmol/L"), 51006: (8, 20, "mg/dL")}


class DB:
    def __init__(self):
        self.rows = {t: [] for t in COLUMNS}
        self.ids = {}

    def nid(self, key):
        self.ids[key] = self.ids.get(key, 0) + 1
        return self.ids[key]

    def add(self, table, **kw):
        row = {c: "" for c in COLUMNS[table]}
        for k, v in kw.items():
            assert k in row, (table, k)
            row[k] = v
        self.rows[table].append(row)
        return row

    def lab(self, sid, hadm, stay, itemid, when, value, comments=""):
        lo, hi, uom = LAB_REF[itemid]
        flag = "abnormal" if (value < lo or value > hi) else ""
        self.add("LABEVENTS", LABEVENT_ID=self.nid("lab"), SUBJECT_ID=sid, HADM_ID=hadm, SPECIMEN_ID=self.nid("spec"), ITEMID=itemid,
                 CHARTTIME=f(when), STORETIME=f(when + timedelta(minutes=50)), VALUE=str(value), VALUENUM=value, VALUEUOM=uom,
                 REF_RANGE_LOWER=lo, REF_RANGE_UPPER=hi, FLAG=flag, COMMENTS=comments, STAY_ID=stay)

    def chart(self, sid, hadm, stay, itemid, when, value, uom="", num=True):
        self.add("CHARTEVENTS", CHARTEVENT_ID=self.nid("chart"), SUBJECT_ID=sid, HADM_ID=hadm, STAY_ID=stay, CHARTTIME=f(when),
                 STORETIME=f(when + timedelta(minutes=5)), ITEMID=itemid, VALUE=str(value), VALUENUM=(value if num else ""), VALUEUOM=uom, WARNING=0)

    def vitals(self, sid, hadm, stay, when, hr, sbp, dbp, mapv, rr, spo2, temp, gcs=(4, 5, 6), fio2=None, peep=None, mode=None):
        self.chart(sid, hadm, stay, 220045, when, hr, "bpm")
        self.chart(sid, hadm, stay, 220050, when, sbp, "mmHg")
        self.chart(sid, hadm, stay, 220051, when, dbp, "mmHg")
        self.chart(sid, hadm, stay, 220052, when, mapv, "mmHg")
        self.chart(sid, hadm, stay, 220210, when, rr, "insp/min")
        self.chart(sid, hadm, stay, 220277, when, spo2, "%")
        self.chart(sid, hadm, stay, 223762, when, temp, "°C")
        self.chart(sid, hadm, stay, 220739, when, gcs[0])
        self.chart(sid, hadm, stay, 223900, when, gcs[1])
        self.chart(sid, hadm, stay, 223901, when, gcs[2])
        self.chart(sid, hadm, stay, 227011, when, sum(gcs))
        if fio2 is not None:
            self.chart(sid, hadm, stay, 223835, when, fio2, "")
        if peep is not None:
            self.chart(sid, hadm, stay, 220339, when, peep, "cmH2O")
        if mode:
            self.chart(sid, hadm, stay, 223849, when, mode, "", num=False)

    def infusion(self, sid, hadm, stay, itemid, start, end, rate, rateuom, amount, amountuom, weight):
        oid = self.nid("order")
        self.add("INPUTEVENTS", INPUTEVENT_ID=self.nid("input"), SUBJECT_ID=sid, HADM_ID=hadm, ICUSTAY_ID=stay, STARTTIME=f(start), ENDTIME=f(end),
                 ITEMID=itemid, AMOUNT=amount, AMOUNTUOM=amountuom, RATE=rate, RATEUOM=rateuom, PATIENTWEIGHT=weight, STORETIME=f(start),
                 CGID=1, ORDERID=oid, LINKORDERID=oid)

    def output(self, sid, hadm, stay, itemid, when, value, uom="mL"):
        self.add("OUTPUTEVENTS", OUTPUTEVENT_ID=self.nid("out"), SUBJECT_ID=sid, HADM_ID=hadm, ICUSTAY_ID=stay, CHARTTIME=f(when), STORETIME=f(when),
                 ITEMID=itemid, VALUE=value, VALUEUOM=uom, WARNING=0)

    def rx(self, sid, hadm, stay, drug, atc, edi, start, stop, dose, unit, route, strength="", per24=""):
        self.add("PRESCRIPTIONS", SUBJECT_ID=sid, HADM_ID=hadm, PHARMACY_ID=self.nid("pharm"), STARTTIME=f(start), STOPTIME=f(stop) if stop else "",
                 DRUG_TYPE="MAIN", DRUG=drug, GSN=atc, NDC=edi, PROD_STRENGTH=strength, FORM_RX="", DOSE_VAL_RX=dose, DOSE_UNIT_RX=unit,
                 FORM_VAL_DISP="", FORM_UNIT_DISP="", DOSES_PER_24_HRS=per24, ROUTE=route, ITEMID="", STAY_ID=stay)

    def proc(self, sid, hadm, stay, itemid, start, end, cat, weight):
        self.add("PROCEDUREEVENTS", PROCEDUREEVENT_ID=self.nid("proc"), SUBJECT_ID=sid, HADM_ID=hadm, STAY_ID=stay, STARTTIME=f(start), ENDTIME=f(end) if end else "",
                 STORETIME=f(start), ITEMID=itemid, VALUE=1, VALUEUOM="", LOCATION="", LOCATIONCATEGORY="", ORDERCATEGORYNAME=cat,
                 SECONDARYORDERCATEGORYNAME="", ORDERCATEGORYDESCRIPTION="", PATIENTWEIGHT=weight, ORIGINALAMOUNT="", ORIGINALRATE="")

    def micro(self, sid, hadm, stay, spec_desc, chartdate, test_name, org, ab_rows, storetime):
        msid = self.nid("mspec")
        if not ab_rows:
            self.add("MICROBIOLOGYEVENTS", MICROEVENT_ID=self.nid("micro"), SUBJECT_ID=sid, HADM_ID=hadm, MICRO_SPECIMEN_ID=msid, CHARTDATE=chartdate.strftime("%Y-%m-%d"),
                     CHARTTIME=f(chartdate), SPEC_ITEMID=70011, SPEC_TYPE_DESC=spec_desc, TEST_SEQ=1, STOREDATE=storetime.strftime("%Y-%m-%d") if storetime else "",
                     STORETIME=f(storetime) if storetime else "", TEST_ITEMID=90033, TEST_NAME=test_name, ORG_ITEMID="", ORG_NAME=org or "", ISOLATE_NUM="", QUANTITY="",
                     AB_ITEMID="", AB_NAME="", DILUTION_TEXT="", DILUTION_COMPARISON="", DILUTION_VALUE="", INTERPRETATION="", COMMENTS="", ORDERDATE=f(chartdate), ORDER_COMMENTS="", STAY_ID=stay)
        for ab, interp in ab_rows:
            self.add("MICROBIOLOGYEVENTS", MICROEVENT_ID=self.nid("micro"), SUBJECT_ID=sid, HADM_ID=hadm, MICRO_SPECIMEN_ID=msid, CHARTDATE=chartdate.strftime("%Y-%m-%d"),
                     CHARTTIME=f(chartdate), SPEC_ITEMID=70011, SPEC_TYPE_DESC=spec_desc, TEST_SEQ=1, STOREDATE=storetime.strftime("%Y-%m-%d"),
                     STORETIME=f(storetime), TEST_ITEMID=90033, TEST_NAME=test_name, ORG_ITEMID=80002, ORG_NAME=org, ISOLATE_NUM=1, QUANTITY="",
                     AB_ITEMID="", AB_NAME=ab, DILUTION_TEXT="", DILUTION_COMPARISON="", DILUTION_VALUE="", INTERPRETATION=interp, COMMENTS="", ORDERDATE=f(chartdate), ORDER_COMMENTS="", STAY_ID=stay)


def build():
    db = DB()
    for r in D_ITEMS:
        db.add("D_ITEMS", **dict(zip(COLUMNS["D_ITEMS"], r)))
    for r in D_LABITEMS:
        db.add("D_LABITEMS", **dict(zip(COLUMNS["D_LABITEMS"], r)))
    for r in D_TESTITEMS:
        db.add("D_TESTITEMS", **dict(zip(COLUMNS["D_TESTITEMS"], r)))
    for r in HOSPITAL:
        db.add("HOSPITAL", **dict(zip(COLUMNS["HOSPITAL"], r)))

    # ================= stay 1: septic shock + AKI =================
    sid, hadm, stay, w = 1, 1001, 1, 65
    adm, icu_in, icu_out, disch = ts("2023-02-28 10:00"), ts("2023-02-28 14:00"), ts("2023-03-06 09:00"), ts("2023-03-10 11:00")
    db.add("PATIENTS", SUBJECT_ID=sid, SEX="M", ANCHOR_AGE=67, ANCHOR_YEAR=2023, ANCHOR_YEAR_GROUP="2020 - 2023", DOD="")
    db.add("ADMISSIONS", SUBJECT_ID=sid, HADM_ID=hadm, ADMITTIME=f(adm), DISCHTIME=f(disch), DEATHTIME="", ADMISSION_TYPE="EMERGENCY", ADMISSION_LOCATION="EMERGENCY ROOM",
           DISCHARGE_LOCATION="HOME", INSURANCE="NHI", MARITAL_STATUS="MARRIED", EDREGTIME=f(adm), EDOUTTIME=f(icu_in), HOSPITAL_EXPIRE_FLAG=0, ICU_EXPIRE_FLAG=0, NATIONALITY="KR", HOSPITAL_ID=268)
    db.add("ICUSTAYS", SUBJECT_ID=sid, HADM_ID=hadm, STAY_ID=stay, FIRST_CAREUNIT="MICU", LAST_CAREUNIT="MICU", INTIME=f(icu_in), OUTTIME=f(icu_out), LOS=round((icu_out - icu_in).total_seconds() / 86400, 2), OP_FLAG=0)
    db.add("TRANSFERS", SUBJECT_ID=sid, HADM_ID=hadm, TRANSFER_SEQ=1, TRANSFER_ID=db.nid("tr"), EVENTTYPE="ED", CAREUNIT="Emergency Department", INTIME=f(adm), OUTTIME=f(icu_in))
    db.add("TRANSFERS", SUBJECT_ID=sid, HADM_ID=hadm, TRANSFER_SEQ=2, TRANSFER_ID=db.nid("tr"), EVENTTYPE="admit", CAREUNIT="MICU", INTIME=f(icu_in), OUTTIME=f(icu_out))
    db.add("SERVICES", SUBJECT_ID=sid, HADM_ID=hadm, TRANSFERTIME=f(icu_in), PREV_SERVICE="", CURR_SERVICE="MED")
    db.add("EDSTAY", SUBJECT_ID=sid, HADM_ID=hadm, STAY_ID=5001, INTIME=f(adm), OUTTIME=f(icu_in), ARRIVAL_TRANSPORT="AMBULANCE", DISPOSITION="ADMITTED")
    db.add("TRIAGE", SUBJECT_ID=sid, STAY_ID=5001, TEMPERATURE=38.9, HEARTRATE=124, RESPRATE=28, O2SAT=92, SBP=86, DBP=50, PAIN=3, ACUITY=1, CHIEFCOMPLAINT="발열, 저혈압")
    db.add("MEDRECON", SUBJECT_ID=sid, STAY_ID=5001, CHARTTIME=f(adm), NAME="아스피린프로텍트정 100mg", GSN="B01AC06", NDC="643500680", ETC_RN=1, ETCCODE="", ETCDESCRIPTION="antiplatelet")
    for i, (code, ver) in enumerate([("A41.9", 10), ("N17.9", 10), ("J96.00", 10), ("I10", 10)], 1):
        db.add("DIAGNOSES_ICD", SUBJECT_ID=sid, HADM_ID=hadm, SEQ_NUM=i, ICD_CODE=code, ICD_VERSION=ver, ITEMID="", IS_ICU=1)
    # labs
    for when, cr in [("2023-02-28 15:00", 0.9), ("2023-03-01 06:00", 0.9), ("2023-03-02 06:00", 1.1), ("2023-03-02 18:00", 1.5), ("2023-03-03 01:00", 1.9)]:
        db.lab(sid, hadm, stay, 50912, ts(when), cr)
    for when, v in [("2023-02-28 15:00", 3.1), ("2023-03-02 15:00", 4.5), ("2023-03-02 21:00", 3.2), ("2023-03-03 01:00", 2.8)]:
        db.lab(sid, hadm, stay, 50813, ts(when), v)
    for when, v in [("2023-02-28 15:00", 4.4), ("2023-03-02 16:00", 6.2), ("2023-03-02 22:00", 5.4), ("2023-03-03 01:00", 5.2)]:
        db.lab(sid, hadm, stay, 50971, ts(when), v)
    db.lab(sid, hadm, stay, 50983, ts("2023-03-03 01:00"), 134)
    db.lab(sid, hadm, stay, 51301, ts("2023-02-28 15:00"), 15.2)
    db.lab(sid, hadm, stay, 51301, ts("2023-03-02 18:00"), 18.5)
    db.lab(sid, hadm, stay, 50889, ts("2023-03-02 18:00"), 25.3)
    db.lab(sid, hadm, stay, 51222, ts("2023-02-28 15:00"), 11.8)
    db.lab(sid, hadm, stay, 51222, ts("2023-03-02 18:00"), 9.5)
    db.lab(sid, hadm, stay, 51265, ts("2023-02-28 15:00"), 160)
    db.lab(sid, hadm, stay, 51265, ts("2023-03-02 18:00"), 88)
    db.lab(sid, hadm, stay, 50885, ts("2023-03-02 18:00"), 2.1)
    db.lab(sid, hadm, stay, 50821, ts("2023-03-02 23:30"), 72)
    db.lab(sid, hadm, stay, 50931, ts("2023-03-02 18:00"), 210)
    db.lab(sid, hadm, stay, 50931, ts("2023-03-03 00:00"), 185)
    db.lab(sid, hadm, stay, 51237, ts("2023-03-02 18:00"), 1.6)
    db.lab(sid, hadm, stay, 51006, ts("2023-03-03 01:00"), 48)
    db.lab(sid, hadm, stay, 50882, ts("2023-03-03 01:00"), 17)
    # vitals q2h in window (03-02 14:00 ~ 03-03 02:00) plus earlier
    db.vitals(sid, hadm, stay, ts("2023-03-02 08:00"), 112, 98, 52, 66, 24, 94, 38.2, (3, 4, 6))
    series = [("2023-03-02 14:00", 118, 92, 48, 62, 26, 93, 38.6), ("2023-03-02 16:00", 124, 88, 46, 59, 28, 92, 38.7),
              ("2023-03-02 18:00", 132, 86, 44, 58, 30, 91, 38.8), ("2023-03-02 20:00", 128, 94, 50, 64, 30, 92, 38.5),
              ("2023-03-02 22:00", 121, 100, 54, 68, 28, 93, 38.3), ("2023-03-03 00:00", 116, 104, 56, 70, 22, 95, 38.0),
              ("2023-03-03 02:00", 110, 106, 58, 72, 20, 96, 37.8)]
    for i, (when, hr, sbp, dbp, mp, rr, spo2, temp) in enumerate(series):
        vent = ts(when) >= ts("2023-03-02 23:00")
        db.vitals(sid, hadm, stay, ts(when), hr, sbp, dbp, mp, rr, spo2, temp, (3, 2, 5) if vent else (3, 4, 6),
                  fio2=(50 if vent else None), peep=(8 if vent else None), mode=("PCV" if vent else None))
    db.chart(sid, hadm, stay, 224639, ts("2023-03-02 06:00"), w, "kg")
    # infusions
    db.infusion(sid, hadm, stay, 221906, ts("2023-03-02 10:00"), ts("2023-03-02 14:00"), 0.1, "mcg/kg/min", 1.56, "mg", w)
    db.infusion(sid, hadm, stay, 221906, ts("2023-03-02 14:00"), ts("2023-03-02 20:00"), 0.2, "mcg/kg/min", 4.68, "mg", w)
    db.infusion(sid, hadm, stay, 221906, ts("2023-03-02 20:00"), ts("2023-03-03 06:00"), 0.3, "mcg/kg/min", 11.7, "mg", w)
    db.infusion(sid, hadm, stay, 222168, ts("2023-03-02 23:00"), ts("2023-03-03 06:00"), 30, "mg/hour", 210, "mg", w)
    db.infusion(sid, hadm, stay, 225158, ts("2023-03-02 15:00"), ts("2023-03-02 16:00"), 1000, "mL/hour", 1000, "mL", w)
    # urine output hourly: low
    t = ts("2023-03-02 14:00")
    for k in range(13):
        db.output(sid, hadm, stay, 226559, t + timedelta(hours=k), [25, 20, 15, 20, 15, 10, 20, 15, 20, 25, 20, 15, 20][k])
    # prescriptions (한글 상품명 혼재)
    db.rx(sid, hadm, stay, "타조페란주 4.5g (Piperacillin/Tazobactam)", "J01CR05", "653101011", ts("2023-02-28 16:00"), None, 4.5, "g", "IV", "4.5g/vial", 3)
    db.rx(sid, hadm, stay, "반코마이신염산염주 1g", "J01XA01", "648300021", ts("2023-02-28 16:00"), None, 1, "g", "IV", "1g/vial", 2)
    db.rx(sid, hadm, stay, "노르핀주 (Norepinephrine 4mg/4mL)", "C01CA03", "645900220", ts("2023-03-02 10:00"), None, 4, "mg", "IV", "4mg/4mL", "")
    db.rx(sid, hadm, stay, "프로포폴 1% 주사", "N01AX10", "643200380", ts("2023-03-02 23:00"), None, 200, "mg", "IV", "200mg/20mL", "")
    db.rx(sid, hadm, stay, "헤파린나트륨주 5000IU", "B01AB01", "641400130", ts("2023-02-28 18:00"), None, 5000, "IU", "SC", "5000IU/mL", 2)
    db.rx(sid, hadm, stay, "판토라인주 40mg (Pantoprazole)", "A02BC02", "643500940", ts("2023-02-28 18:00"), None, 40, "mg", "IV", "40mg/vial", 1)
    db.rx(sid, hadm, stay, "휴물린알주 (Insulin Regular)", "A10AB01", "644000010", ts("2023-03-02 18:00"), None, 4, "units", "SC", "100IU/mL", 4)
    db.rx(sid, hadm, stay, "라식스주 20mg (Furosemide)", "C03CA01", "641900460", ts("2023-03-01 08:00"), ts("2023-03-02 16:00"), 20, "mg", "IV", "20mg/2mL", 2)
    # EMAR vancomycin administrations
    for k, when in enumerate(["2023-02-28 16:30", "2023-03-01 04:30", "2023-03-01 16:30", "2023-03-02 04:30", "2023-03-02 16:30"], 1):
        eid = "E%03d" % db.nid("emar")
        db.add("EMAR", SUBJECT_ID=sid, HADM_ID=hadm, EMAR_TYPE="", EMAR_ID=eid, EMAR_SEQ=k, POE_ID="", PHARMACY_ID=2, CHARTTIME=f(ts(when)), MEDICATION="반코마이신염산염주 1g", EVENT_TXT="Administered", STORETIME=f(ts(when)), ITEMID="", STAY_ID=stay)
        db.add("EMAR_DETAIL", SUBJECT_ID=sid, HADM_ID=hadm, EMAR_ID=eid, EMAR_SEQ=k, PARENT_FIELD_ORDINAL="", PHARMACY_ID=2, NOTE_TXT="", ON_OFF="", INFUSION_RATE_DAY="", INFUSION_RATE_HR="", INFUSION_RATE_MIN="", ITEMID="", STAY_ID=stay)
    # micro
    db.micro(sid, hadm, stay, "BLOOD CULTURE", ts("2023-02-28 16:00"), "Blood Culture, Routine", "ESCHERICHIA COLI",
             [("PIPERACILLIN/TAZO", "R"), ("CEFTRIAXONE", "S"), ("MEROPENEM", "S"), ("CIPROFLOXACIN", "R"), ("GENTAMICIN", "S")], ts("2023-03-02 20:00"))
    db.micro(sid, hadm, stay, "URINE", ts("2023-03-02 16:00"), "Urine Culture", None, [], None)
    # procedures
    db.proc(sid, hadm, stay, 225315, ts("2023-02-28 16:00"), ts("2023-02-28 16:30"), "Invasive Lines", w)
    db.proc(sid, hadm, stay, 224385, ts("2023-03-02 23:00"), ts("2023-03-02 23:15"), "Intubation/Extubation", w)
    db.proc(sid, hadm, stay, 225792, ts("2023-03-02 23:15"), None, "Ventilation", w)
    db.add("PROCEDURES_ICD", SUBJECT_ID=sid, HADM_ID=hadm, SEQ_NUM=1, CHARTDATE="2023-03-02", ICD_CODE="96.04", ICD_VERSION=9, ITEMID="", STAY_ID=stay)
    db.add("TESTEVENTS", TESTEVENT_ID=db.nid("test"), SUBJECT_ID=sid, HADM_ID=hadm, ITEMID=90001, CHARTTIME=f(ts("2023-03-02 23:30")), STORETIME=f(ts("2023-03-02 23:40")), FILE_PATH="dicom/synthetic/1.dcm", STAY_ID=stay)
    db.add("DATETIMEEVENTS", DATETIMEEVENT_ID=db.nid("dte"), SUBJECT_ID=sid, HADM_ID=hadm, STAY_ID=stay, CHARTTIME=f(ts("2023-03-02 14:00")), STORETIME=f(ts("2023-03-02 14:00")), ITEMID=229351, VALUE=f(ts("2023-02-28 15:00")), VALUEUOM="Date", WARNING=0)

    # ================= stay 2: post-op, renal dosing / QT / DUR / insulin TDM gap =================
    sid, hadm, stay, w = 2, 1002, 2, 55
    adm, icu_in, icu_out, disch = ts("2023-04-10 07:00"), ts("2023-04-10 12:00"), ts("2023-04-12 10:00"), ts("2023-04-18 11:00")
    db.add("PATIENTS", SUBJECT_ID=sid, SEX="F", ANCHOR_AGE=82, ANCHOR_YEAR=2023, ANCHOR_YEAR_GROUP="2020 - 2023", DOD="")
    db.add("ADMISSIONS", SUBJECT_ID=sid, HADM_ID=hadm, ADMITTIME=f(adm), DISCHTIME=f(disch), DEATHTIME="", ADMISSION_TYPE="ELECTIVE", ADMISSION_LOCATION="PHYSICIAN REFERRAL",
           DISCHARGE_LOCATION="HOME", INSURANCE="NHI", MARITAL_STATUS="WIDOWED", EDREGTIME="", EDOUTTIME="", HOSPITAL_EXPIRE_FLAG=0, ICU_EXPIRE_FLAG=0, NATIONALITY="KR", HOSPITAL_ID=268)
    db.add("ICUSTAYS", SUBJECT_ID=sid, HADM_ID=hadm, STAY_ID=stay, FIRST_CAREUNIT="SICU", LAST_CAREUNIT="SICU", INTIME=f(icu_in), OUTTIME=f(icu_out), LOS=round((icu_out - icu_in).total_seconds() / 86400, 2), OP_FLAG=1)
    db.add("TRANSFERS", SUBJECT_ID=sid, HADM_ID=hadm, TRANSFER_SEQ=1, TRANSFER_ID=db.nid("tr"), EVENTTYPE="admit", CAREUNIT="Surgery Ward", INTIME=f(adm), OUTTIME=f(icu_in))
    db.add("TRANSFERS", SUBJECT_ID=sid, HADM_ID=hadm, TRANSFER_SEQ=2, TRANSFER_ID=db.nid("tr"), EVENTTYPE="transfer", CAREUNIT="SICU", INTIME=f(icu_in), OUTTIME=f(icu_out))
    db.add("SERVICES", SUBJECT_ID=sid, HADM_ID=hadm, TRANSFERTIME=f(adm), PREV_SERVICE="", CURR_SERVICE="SURG")
    for i, (code, ver) in enumerate([("K80.1", 10), ("N18.4", 10), ("E11.9", 10)], 1):
        db.add("DIAGNOSES_ICD", SUBJECT_ID=sid, HADM_ID=hadm, SEQ_NUM=i, ICD_CODE=code, ICD_VERSION=ver, ITEMID="", IS_ICU=1)
    db.add("PROCEDURES_ICD", SUBJECT_ID=sid, HADM_ID=hadm, SEQ_NUM=1, CHARTDATE="2023-04-10", ICD_CODE="51.23", ICD_VERSION=9, ITEMID="", STAY_ID=stay)
    db.lab(sid, hadm, stay, 50912, ts("2023-04-10 08:00"), 1.8)
    db.lab(sid, hadm, stay, 50912, ts("2023-04-11 05:00"), 1.7)
    db.lab(sid, hadm, stay, 50971, ts("2023-04-10 20:00"), 3.4)
    db.lab(sid, hadm, stay, 50971, ts("2023-04-11 05:00"), 3.1)
    db.lab(sid, hadm, stay, 50960, ts("2023-04-11 05:00"), 1.6)
    db.lab(sid, hadm, stay, 50983, ts("2023-04-11 05:00"), 139)
    db.lab(sid, hadm, stay, 51222, ts("2023-04-10 20:00"), 10.8)
    db.lab(sid, hadm, stay, 51222, ts("2023-04-11 05:00"), 10.5)
    db.lab(sid, hadm, stay, 51265, ts("2023-04-11 05:00"), 210)
    db.lab(sid, hadm, stay, 51301, ts("2023-04-11 05:00"), 11.2)
    for when, g in [("2023-04-10 18:00", 240), ("2023-04-10 23:00", 190), ("2023-04-11 05:00", 165)]:
        db.lab(sid, hadm, stay, 50931, ts(when), g)
    db.lab(sid, hadm, stay, 51006, ts("2023-04-11 05:00"), 34)
    for when in ["2023-04-10 14:00", "2023-04-10 18:00", "2023-04-10 22:00", "2023-04-11 02:00", "2023-04-11 06:00"]:
        db.vitals(sid, hadm, stay, ts(when), 78, 128, 70, 88, 16, 97, 36.8, (4, 5, 6))
    db.chart(sid, hadm, stay, 224639, ts("2023-04-10 13:00"), w, "kg")
    db.infusion(sid, hadm, stay, 223258, ts("2023-04-10 18:00"), ts("2023-04-11 08:00"), 2, "units/hour", 28, "units", w)
    t = ts("2023-04-10 18:00")
    for k in range(13):
        db.output(sid, hadm, stay, 226559, t + timedelta(hours=k), 60)
    db.rx(sid, hadm, stay, "크렉산주 40mg (Enoxaparin)", "B01AB05", "653400140", ts("2023-04-10 20:00"), None, 40, "mg", "SC", "40mg/0.4mL", 1)
    db.rx(sid, hadm, stay, "온단세트론주 4mg", "A04AA01", "641300470", ts("2023-04-10 14:00"), None, 4, "mg", "IV", "4mg/2mL", 3)
    db.rx(sid, hadm, stay, "시프로바이정 500mg (Ciprofloxacin)", "J01MA02", "640900180", ts("2023-04-10 20:00"), None, 500, "mg", "PO", "500mg", 2)
    db.rx(sid, hadm, stay, "실다루드정 1mg (Tizanidine)", "M03BX02", "648200130", ts("2023-04-10 20:00"), None, 1, "mg", "PO", "1mg", 3)
    db.rx(sid, hadm, stay, "휴물린알주 (Insulin Regular)", "A10AB01", "644000010", ts("2023-04-10 18:00"), None, 2, "units", "IV", "100IU/mL", "")
    db.rx(sid, hadm, stay, "아세트아미노펜정 650mg", "N02BE01", "641200080", ts("2023-04-10 13:00"), None, 650, "mg", "PO", "650mg", 3)
    db.proc(sid, hadm, stay, 225752, ts("2023-04-10 12:30"), ts("2023-04-10 12:45"), "Invasive Lines", w)

    # ================= stay 3: nearly normal =================
    sid, hadm, stay, w = 3, 1003, 3, 72
    adm, icu_in, icu_out, disch = ts("2023-05-20 07:00"), ts("2023-05-20 08:00"), ts("2023-05-21 08:00"), ts("2023-05-22 10:00")
    db.add("PATIENTS", SUBJECT_ID=sid, SEX="M", ANCHOR_AGE=45, ANCHOR_YEAR=2023, ANCHOR_YEAR_GROUP="2020 - 2023", DOD="")
    db.add("ADMISSIONS", SUBJECT_ID=sid, HADM_ID=hadm, ADMITTIME=f(adm), DISCHTIME=f(disch), DEATHTIME="", ADMISSION_TYPE="ELECTIVE", ADMISSION_LOCATION="PHYSICIAN REFERRAL",
           DISCHARGE_LOCATION="HOME", INSURANCE="NHI", MARITAL_STATUS="SINGLE", EDREGTIME="", EDOUTTIME="", HOSPITAL_EXPIRE_FLAG=0, ICU_EXPIRE_FLAG=0, NATIONALITY="KR", HOSPITAL_ID=268)
    db.add("ICUSTAYS", SUBJECT_ID=sid, HADM_ID=hadm, STAY_ID=stay, FIRST_CAREUNIT="CCU", LAST_CAREUNIT="CCU", INTIME=f(icu_in), OUTTIME=f(icu_out), LOS=1.0, OP_FLAG=1)
    db.add("TRANSFERS", SUBJECT_ID=sid, HADM_ID=hadm, TRANSFER_SEQ=1, TRANSFER_ID=db.nid("tr"), EVENTTYPE="admit", CAREUNIT="CCU", INTIME=f(icu_in), OUTTIME=f(icu_out))
    db.add("SERVICES", SUBJECT_ID=sid, HADM_ID=hadm, TRANSFERTIME=f(adm), PREV_SERVICE="", CURR_SERVICE="CMED")
    db.add("DIAGNOSES_ICD", SUBJECT_ID=sid, HADM_ID=hadm, SEQ_NUM=1, ICD_CODE="I25.10", ICD_VERSION=10, ITEMID="", IS_ICU=1)
    db.lab(sid, hadm, stay, 50912, ts("2023-05-20 09:00"), 0.8)
    db.lab(sid, hadm, stay, 50971, ts("2023-05-20 09:00"), 4.1)
    db.lab(sid, hadm, stay, 50983, ts("2023-05-20 09:00"), 140)
    db.lab(sid, hadm, stay, 51222, ts("2023-05-20 09:00"), 14.1)
    db.lab(sid, hadm, stay, 51003, ts("2023-05-20 09:00"), 0.01) if 51003 in LAB_REF else None
    for when in ["2023-05-20 08:00", "2023-05-20 12:00", "2023-05-20 16:00", "2023-05-20 20:00"]:
        db.vitals(sid, hadm, stay, ts(when), 72, 122, 74, 90, 14, 98, 36.6, (4, 5, 6))
    db.rx(sid, hadm, stay, "판토라인주 40mg (Pantoprazole)", "A02BC02", "643500940", ts("2023-05-20 09:00"), None, 40, "mg", "IV", "40mg/vial", 1)
    db.rx(sid, hadm, stay, "아스피린프로텍트정 100mg", "B01AC06", "643500680", ts("2023-05-20 09:00"), None, 100, "mg", "PO", "100mg", 1)
    t = ts("2023-05-20 08:00")
    for k in range(13):
        db.output(sid, hadm, stay, 226559, t + timedelta(hours=k), 70)
    return db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    db = build()
    for t, rows in db.rows.items():
        p = os.path.join(args.out, t + ".csv")
        with open(p, "w", newline="", encoding="utf-8") as fh:
            wtr = csv.DictWriter(fh, fieldnames=COLUMNS[t])
            wtr.writeheader()
            for r in rows:
                wtr.writerow(r)
        with open(p + ".CHECKSUM", "w") as fh:
            fh.write("synthetic\n")
    print("wrote %d tables to %s" % (len(db.rows), args.out))
    for t, rows in db.rows.items():
        print("  %-20s %d rows" % (t, len(rows)))


if __name__ == "__main__":
    main()
