#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_synthetic_nicu.py — K-MIMIC(SNUH) 형식의 소형 NICU 합성 데이터(25개 테이블 CSV) 생성. 실제 환자 데이터가 아니다.
ITEMID·라벨은 Synthetic K-MIMIC(SYN-ICU)에서 확인한 SNUH 형식(예: 001C_1021_25105 'HR > 심박동수')을 따르고, 신생아 항목은 같은 형식으로 추가.

시나리오 (요약 기준 시각은 매일 07:00)
  stay 101 (여아, 재태 29주 미숙아): 4일째 요약에서 SIMV→CPAP 변경, SpO2 저하·서맥, 저혈당(BST 38), 저나트륨, 빌리루빈 상승,
           카페인·ampicillin 투여 중, gentamicin 오늘 중단, TPN + 경관수유(수분 목표 미달), UVC 유지, 뇌초음파.
  stay 102 (남아, 만삭 패혈증 의심): 3일째 요약에서 SIMV 인공호흡, dopamine 지속주입, ampicillin+gentamicin(혈중농도 없음),
           CRP 상승, 혈액배양 미보고, 소변 감소, 발열·빈맥, Cr 상승(AKI), 칼륨 위험값, 고혈당.
  stay 103 (여아, 재태 35주): 5일째 요약에서 실내 공기, 경구 수유, 체중 11% 감소, 빌리루빈 16.5(고빌리루빈혈증), 투여 약 없음.

사용: python make_synthetic_nicu.py [--out data/synthetic_nicu]
"""
import argparse
import csv
import os
import random
import zlib
from datetime import datetime, timedelta

COLUMNS = {
    "ADMISSIONS": ["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "DEATHTIME", "ADMISSION_TYPE", "ADMISSION_LOCATION", "DISCHARGE_LOCATION", "INSURANCE", "MARITAL_STATUS", "EDREGTIME", "EDOUTTIME", "HOSPITAL_EXPIRE_FLAG", "ICU_EXPIRE_FLAG", "NATIONALITY", "HOSPITAL_ID"],
    "PATIENTS": ["SUBJECT_ID", "SEX", "ANCHOR_AGE", "ANCHOR_YEAR", "ANCHOR_YEAR_GROUP", "DOD"],
    "TRANSFERS": ["SUBJECT_ID", "HADM_ID", "TRANSFER_ID", "EVENTTYPE", "CAREUNIT", "INTIME", "OUTTIME"],
    "DIAGNOSES_ICD": ["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD_CODE", "ICD_VERSION", "ITEMID", "IS_ICU"],
    "D_LABITEMS": ["ITEMID", "LABEL", "FLUID", "CATEGORY", "EDI_CODE"],
    "D_TESTITEMS": ["ITEMID", "LABEL", "CATEGORY", "EDI_CODE"],
    "EMAR": ["SUBJECT_ID", "HADM_ID", "EMAR_ID", "EMAR_SEQ", "POE_ID", "PHARMACY_ID", "CHARTTIME", "MEDICATION", "EVENT_TXT", "STORETIME", "EMAR_TYPE", "ITEMID", "STAY_ID"],
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
    "PROCEDUREEVENTS": ["PROCEDUREEVENT_ID", "SUBJECT_ID", "HADM_ID", "STAY_ID", "STARTTIME", "ENDTIME", "STORETIME", "ITEMID", "VALUE", "VALUEUOM", "LOCATION", "LOCATIONCATEGORY", "ORDERCATEGORYNAME", "SECONDARYORDERCATEGORYNAME", "ORDERCATEGORYDESCRIPTION", "PATIENTWEIGHT"],
    "EDSTAY": ["SUBJECT_ID", "HADM_ID", "STAY_ID", "INTIME", "OUTTIME", "ARRIVAL_TRANSPORT", "DISPOSITION"],
    "TRIAGE": ["SUBJECT_ID", "STAY_ID", "TEMPERATURE", "HEARTRATE", "RESPRATE", "O2SAT", "SBP", "DBP", "PAIN", "ACUITY", "CHIEFCOMPLAINT"],
    "MEDRECON": ["SUBJECT_ID", "STAY_ID", "CHARTTIME", "NAME", "GSN", "NDC", "ETC_RN", "ETCCODE", "ETCDESCRIPTION"],
    "HOSPITAL": ["HOSPITAL_ID", "HOSPITAL_NAME", "HOSPITAL_TYPE", "REGION"],
}
HR, RR, SPO2, BT, SBP, DBP, MBP = "001C_1021_25105", "001C_1023_27310", "001C_1003_24480", "001C_1026_26520", "001C_1012_24795", "001C_1013_25640", "001C_1014_27095"
MODE, FIO2, PEEP, O2F, WT, BST = "001C_1810_22385", "001C_1894_20720", "001C_1567_21330", "001C_1838_24465", "001C_1101_20001", "001C_1301_20002"
IV, OG, ORAL = "001I_1315_26175", "001I_1305_20003", "001I_1301_23155"
URINE, STOOL = "001O_1479_25880", "001O_1500_20004"
UVC, BRAIN, INTUB = "001P_UVC01", "001P_RS1001", "001P_INT01"
D_ITEMS = [
    (HR, "HR > 심박동수", "HR(회/분)", "chartevents", "활력징후", "회/min", "Real Number"), (RR, "RR > 호흡수", "RR(회/분)", "chartevents", "활력징후", "회/min", "Real Number"),
    (SPO2, "SpO2 > 산소포화도", "%", "chartevents", "활력징후", "%", "Real Number"), (BT, "BT > 체온", "체온(℃)", "chartevents", "활력징후", "℃", "Real Number"),
    (SBP, "SBP > 수축기혈압", "SBP(mmHg)", "chartevents", "활력징후", "mmHg", "Real Number"), (DBP, "DBP > 이완기혈압", "DBP(mmHg)", "chartevents", "활력징후", "mmHg", "Real Number"),
    (MBP, "mean BP(연동) > 평균혈압(연동)", "", "chartevents", "활력징후", "mmHg", "Real Number"),
    (MODE, "vent 모드 > ventilator 모드 종류", "모드", "chartevents", "인공호흡기1", "", "Coded Text"), (FIO2, "FIO2 > FiO2", "Fio2(%)", "chartevents", "인공호흡기1", "%", "Real Number"),
    (PEEP, "set PEEP > PEEP level", "PEEP(CmH2O)", "chartevents", "인공호흡기1", "cmH2O", "Real Number"), (O2F, "flow rate > 산소량", "O2(L/min)", "chartevents", "인공호흡기1", "L/min", "Real Number"),
    (WT, "체중 > 체중(g)", "BW(g)", "chartevents", "신체계측", "g", "Real Number"), (BST, "BST > 혈당(BST)", "BST", "chartevents", "혈당", "mg/dL", "Real Number"),
    (IV, "정맥주입 > 정맥주입량(cc)", "", "inputevents", "섭취량", "cc", "Real Number"), (OG, "경관영양 > OG tube 수유량(cc)", "", "inputevents", "섭취량", "cc", "Real Number"),
    (ORAL, "경구 > 경구", "", "inputevents", "섭취량", "cc", "Real Number"),
    (URINE, "자연배뇨 > 자연배뇨(cc)", "", "outputevents", "배설량", "cc", "Real Number"), (STOOL, "대변 > 대변량(cc)", "", "outputevents", "배설량", "cc", "Real Number"),
    (UVC, "UVC insertion(제대정맥관 삽입)", "", "procedureevents", "시술", "", ""), (BRAIN, "Brain sono (portable)", "", "procedureevents", "영상의학과 > 초음파 > Brain", "", ""),
    (INTUB, "Intubation(기관내삽관)", "", "procedureevents", "시술", "", ""),
]
LAB = {"tbil": ("001L3011", "Bilirubin, total(검사24시간가능)", "mg/dL", 0.2, 1.2), "glucose": ("001L3005", "Glucose(검사24시간가능)", "mg/dL", 70, 110),
       "na": ("001L3043", "Sodium (serum)(검사24시간가능)", "mmol/L", 136, 145), "k": ("001L3044", "Potassium (serum)(검사24시간가능)", "mmol/L", 3.5, 5.1),
       "crp": ("001L3092", "hs-CRP quantitation(검사24시간가능)", "mg/dL", 0, 0.5), "hb": ("001L2003", "Hb(검사24시간가능)", "g/dL", 12, 16),
       "plt": ("001L2009", "PLT(검사24시간가능)", "×10³/㎕", 150, 400), "cr": ("001L3041", "Creatinine(검사24시간가능)", "mg/dL", 0.5, 1.2),
       "wbc": ("001L2001", "WBC(검사24시간가능)", "×10³/㎕", 4, 10), "genta": ("001L4501", "Gentamicin (trough)", "ug/mL", 0, 2)}
FMT = "%Y-%m-%d %H:%M:%S"


def ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M")


def f(d):
    return d.strftime(FMT) if d else ""


class DB(object):
    def __init__(self):
        self.rows = {t: [] for t in COLUMNS}
        self.n = {}

    def nid(self, k):
        self.n[k] = self.n.get(k, 0) + 1
        return self.n[k]

    def add(self, t, **kw):
        r = {c: "" for c in COLUMNS[t]}
        for k, v in kw.items():
            assert k in r, (t, k)
            r[k] = v
        self.rows[t].append(r)

    def chart(self, p, item, t, v, uom="", num=True):
        sid, hadm, stay = p
        self.add("CHARTEVENTS", CHARTEVENT_ID=self.nid("c"), SUBJECT_ID=sid, HADM_ID=hadm, STAY_ID=stay, CHARTTIME=f(t), STORETIME=f(t + timedelta(minutes=3)),
                 ITEMID=item, VALUE=v, VALUENUM=v if num else "", VALUEUOM=uom, WARNING=0)

    def lab(self, p, key, t, v):
        sid, hadm, stay = p
        item, label, uom, lo, hi = LAB[key]
        self.add("LABEVENTS", LABEVENT_ID=self.nid("l"), SUBJECT_ID=sid, HADM_ID=hadm, SPECIMEN_ID=self.nid("sp"), ITEMID=item, CHARTTIME=f(t),
                 STORETIME=f(t + timedelta(minutes=40)), VALUE=v, VALUENUM=v, VALUEUOM=uom, REF_RANGE_LOWER=lo, REF_RANGE_UPPER=hi, STAY_ID=stay)

    def inp(self, p, item, t, ml, w):
        sid, hadm, stay = p
        self.add("INPUTEVENTS", INPUTEVENT_ID=self.nid("i"), SUBJECT_ID=sid, HADM_ID=hadm, ICUSTAY_ID=stay, STARTTIME=f(t), ENDTIME=f(t + timedelta(hours=1)),
                 ITEMID=item, AMOUNT=ml, AMOUNTUOM="cc", PATIENTWEIGHT=w, STORETIME=f(t))

    def out(self, p, item, t, ml):
        sid, hadm, stay = p
        self.add("OUTPUTEVENTS", OUTPUTEVENT_ID=self.nid("o"), SUBJECT_ID=sid, HADM_ID=hadm, ICUSTAY_ID=stay, CHARTTIME=f(t), STORETIME=f(t), ITEMID=item, VALUE=ml, VALUEUOM="cc", WARNING=0)

    def emar(self, p, med, t, rate_hr="", rate_min=""):
        sid, hadm, stay = p
        eid = self.nid("e")
        self.add("EMAR", SUBJECT_ID=sid, HADM_ID=hadm, EMAR_ID=eid, EMAR_SEQ=1, PHARMACY_ID=eid, CHARTTIME=f(t), MEDICATION=med, EVENT_TXT="Administered",
                 STORETIME=f(t), EMAR_TYPE="Note", ITEMID=zlib.crc32(med.encode("utf-8")) % 10 ** 9, STAY_ID=stay)
        self.add("EMAR_DETAIL", SUBJECT_ID=sid, HADM_ID=hadm, EMAR_ID=eid, EMAR_SEQ=1, PHARMACY_ID=eid, INFUSION_RATE_HR=rate_hr, INFUSION_RATE_MIN=rate_min,
                 ITEMID=zlib.crc32(med.encode("utf-8")) % 10 ** 9, STAY_ID=stay)

    def proc(self, p, item, t, w):
        sid, hadm, stay = p
        self.add("PROCEDUREEVENTS", PROCEDUREEVENT_ID=self.nid("p"), SUBJECT_ID=sid, HADM_ID=hadm, STAY_ID=stay, STARTTIME=f(t), ENDTIME=f(t + timedelta(minutes=30)),
                 STORETIME=f(t), ITEMID=item, VALUE=1, PATIENTWEIGHT=w)

    def patient(self, p, sex, admit, icu_in, icu_out, disch, adm_type, dx):
        sid, hadm, stay = p
        self.add("PATIENTS", SUBJECT_ID=sid, SEX=sex, ANCHOR_AGE=0, ANCHOR_YEAR=2023, ANCHOR_YEAR_GROUP="2022-2024")
        self.add("ADMISSIONS", SUBJECT_ID=sid, HADM_ID=hadm, ADMITTIME=f(admit), DISCHTIME=f(disch), ADMISSION_TYPE=adm_type, DISCHARGE_LOCATION="Home",
                 INSURANCE="National Healthcare Insurance Service", MARITAL_STATUS="single", HOSPITAL_EXPIRE_FLAG=0, ICU_EXPIRE_FLAG=0, NATIONALITY="Korean", HOSPITAL_ID=1)
        self.add("ICUSTAYS", SUBJECT_ID=sid, HADM_ID=hadm, STAY_ID=stay, FIRST_CAREUNIT="NICU", LAST_CAREUNIT="NICU", INTIME=f(icu_in), OUTTIME=f(icu_out),
                 LOS=round((icu_out - icu_in).total_seconds() / 86400, 2), OP_FLAG=0)
        self.add("TRANSFERS", SUBJECT_ID=sid, HADM_ID=hadm, TRANSFER_ID="T%d" % stay, EVENTTYPE="admit", CAREUNIT="NICU", INTIME=f(icu_in), OUTTIME=f(icu_out))
        self.add("SERVICES", SUBJECT_ID=sid, HADM_ID=hadm, TRANSFERTIME=f(icu_in), CURR_SERVICE="NEO")
        for i, code in enumerate(dx):
            self.add("DIAGNOSES_ICD", SUBJECT_ID=sid, HADM_ID=hadm, SEQ_NUM=i, ICD_CODE=code, ICD_VERSION="KCD8", IS_ICU=1)


def vitals(db, p, start, hours, gen):
    for k in range(hours):
        t = start + timedelta(hours=k)
        v = gen(k, t)
        for item, key, uom in ((HR, "hr", "회/min"), (RR, "rr", "회/min"), (SPO2, "spo2", "%"), (BT, "bt", "℃"), (SBP, "sbp", "mmHg"), (DBP, "dbp", "mmHg"), (MBP, "map", "mmHg")):
            if key in v:
                db.chart(p, item, t, v[key], uom)


def build():
    db = DB()
    rnd = random.Random(7)
    for r in D_ITEMS:
        db.add("D_ITEMS", ITEMID=r[0], LABEL=r[1], ABBREVIATION=r[2], LINKSTO=r[3], CATEGORY=r[4], UNITNAME=r[5], PARAM_TYPE=r[6])
    for key, (item, label, uom, lo, hi) in LAB.items():
        db.add("D_LABITEMS", ITEMID=item, LABEL=label, FLUID="Serum", CATEGORY="진단검사의학과및 기타 검체검사 > 일반화학(1)", EDI_CODE="KMM90000")
    db.add("HOSPITAL", HOSPITAL_ID=1, HOSPITAL_NAME="Synthetic Hospital", HOSPITAL_TYPE="Tertiary", REGION="Seoul")

    # ---------- stay 101: 29주 미숙아
    p = (101, "H101", 101)
    birth = ts("2023-03-01 02:00")
    db.patient(p, "F", birth, ts("2023-03-01 02:30"), ts("2023-03-20 10:00"), ts("2023-04-10 10:00"), "Admission", ["P073", "P220", "P590"])
    for t, g in (("2023-03-01 03:00", 1300), ("2023-03-02 06:00", 1270), ("2023-03-03 06:00", 1250), ("2023-03-04 06:00", 1235)):
        db.chart(p, WT, ts(t), g, "g")
    vitals(db, p, ts("2023-03-01 03:00"), 76, lambda k, t: {"hr": 150 + rnd.randint(-8, 12), "rr": 50 + rnd.randint(-8, 8), "spo2": 93 + rnd.randint(-2, 3),
                                                             "bt": round(36.8 + rnd.uniform(-0.2, 0.3), 1), "sbp": 52, "dbp": 30, "map": 38})
    db.chart(p, SPO2, ts("2023-03-03 19:20"), 82, "%")
    db.chart(p, SPO2, ts("2023-03-04 02:10"), 80, "%")
    db.chart(p, HR, ts("2023-03-04 02:11"), 88, "회/min")
    for k in range(0, 60):
        t = ts("2023-03-01 03:00") + timedelta(hours=k)
        db.chart(p, MODE, t, "SIMV", "", num=False)
        db.chart(p, FIO2, t, 30, "%")
        db.chart(p, PEEP, t, 5, "cmH2O")
    for k in range(0, 16):
        t = ts("2023-03-03 16:00") + timedelta(hours=k)
        db.chart(p, MODE, t, "CPAP", "", num=False)
        db.chart(p, FIO2, t, 25, "%")
        db.chart(p, PEEP, t, 6, "cmH2O")
    for t, v in (("2023-03-03 11:00", 62), ("2023-03-03 23:00", 38), ("2023-03-04 01:00", 58), ("2023-03-04 05:00", 66)):
        db.chart(p, BST, ts(t), v, "mg/dL")
    for t, v in (("2023-03-02 05:00", 6.0), ("2023-03-03 05:00", 6.4), ("2023-03-04 05:00", 9.8)):
        db.lab(p, "tbil", ts(t), v)
    db.lab(p, "na", ts("2023-03-04 05:00"), 131)
    db.lab(p, "k", ts("2023-03-04 05:00"), 4.6)
    db.lab(p, "crp", ts("2023-03-04 05:00"), 0.3)
    db.lab(p, "hb", ts("2023-03-04 05:00"), 14.2)
    db.lab(p, "glucose", ts("2023-03-04 05:00"), 64)
    for k in range(24):
        t = ts("2023-03-03 07:00") + timedelta(hours=k)
        db.inp(p, IV, t, 4, 1.25)
        db.emar(p, "TPN neonatal custom bag(Parenteral nutrition)", t, rate_hr=4)
        if k % 3 == 0:
            db.inp(p, OG, t, 3, 1.25)
        if k % 3 == 1:
            db.out(p, URINE, t, 9)
    db.out(p, STOOL, ts("2023-03-03 14:00"), 3)
    for d in range(4):
        db.emar(p, "Caffeine citrate 20mg/mL inj(Caffeine citrate)", ts("2023-03-01 08:00") + timedelta(days=d))
    for h in range(0, 80, 12):
        db.emar(p, "Ampicillin 500mg inj(Ampicillin)", ts("2023-03-01 04:00") + timedelta(hours=h))
    for h in range(0, 56, 24):
        db.emar(p, "Gentamicin 10mg/mL inj(Gentamicin)", ts("2023-03-01 10:00") + timedelta(hours=h))
    db.proc(p, UVC, ts("2023-03-01 03:00"), 1.3)
    db.proc(p, BRAIN, ts("2023-03-03 14:00"), 1.25)
    db.add("MICROBIOLOGYEVENTS", MICROEVENT_ID=db.nid("m"), SUBJECT_ID=101, HADM_ID="H101", MICRO_SPECIMEN_ID=9101, CHARTDATE="2023-03-01", CHARTTIME=f(ts("2023-03-01 03:00")),
           SPEC_TYPE_DESC="BLOOD CULTURE", TEST_NAME="Blood Culture", STOREDATE="2023-03-03", STORETIME=f(ts("2023-03-03 09:00")), STAY_ID=101)

    # ---------- stay 102: 만삭 패혈증 의심
    p = (102, "H102", 102)
    adm = ts("2023-05-10 10:00")
    db.patient(p, "M", adm, ts("2023-05-10 11:00"), ts("2023-05-25 10:00"), ts("2023-06-02 10:00"), "Emergency department", ["P369", "P221"])
    for t, kgv in (("2023-05-10 12:00", 3.30), ("2023-05-11 06:00", 3.25), ("2023-05-12 06:00", 3.22)):
        db.chart(p, WT, ts(t), kgv, "kg")
    vitals(db, p, ts("2023-05-10 12:00"), 44, lambda k, t: {"hr": 165 + rnd.randint(-5, 25) if k > 20 else 150, "rr": 55 + rnd.randint(-5, 10),
                                                            "spo2": 95 + rnd.randint(-2, 3), "bt": 38.1 if k in (30, 31) else round(37.0 + rnd.uniform(-0.2, 0.3), 1),
                                                            "sbp": 58, "dbp": 32, "map": 42})
    for k in range(44):
        t = ts("2023-05-10 12:00") + timedelta(hours=k)
        db.chart(p, MODE, t, "SIMV", "", num=False)
        db.chart(p, FIO2, t, 40, "%")
        db.chart(p, PEEP, t, 6, "cmH2O")
    db.proc(p, INTUB, ts("2023-05-10 11:30"), 3.3)
    for k in range(30):
        db.emar(p, "Dopamine 200mg/5mL inj(Dopamine)", ts("2023-05-11 02:00") + timedelta(hours=k), rate_min=5)
    for h in range(0, 48, 12):
        db.emar(p, "Ampicillin 500mg inj(Ampicillin)", ts("2023-05-10 12:00") + timedelta(hours=h))
        db.emar(p, "Gentamicin 10mg/mL inj(Gentamicin)", ts("2023-05-10 13:00") + timedelta(hours=h))
    for t, v in (("2023-05-10 12:00", 0.8), ("2023-05-12 05:00", 1.2)):
        db.lab(p, "cr", ts(t), v)
    db.lab(p, "crp", ts("2023-05-11 05:00"), 1.8)
    db.lab(p, "crp", ts("2023-05-12 05:00"), 3.5)
    db.lab(p, "k", ts("2023-05-12 05:00"), 6.8)
    db.lab(p, "glucose", ts("2023-05-12 05:00"), 190)
    db.lab(p, "plt", ts("2023-05-12 05:00"), 120)
    db.lab(p, "wbc", ts("2023-05-12 05:00"), 28.0)
    for k in range(24):
        t = ts("2023-05-11 07:00") + timedelta(hours=k)
        db.inp(p, IV, t, 13, 3.22)
        if k % 2 == 0:
            db.out(p, URINE, t, 4)
    db.add("MICROBIOLOGYEVENTS", MICROEVENT_ID=db.nid("m"), SUBJECT_ID=102, HADM_ID="H102", MICRO_SPECIMEN_ID=9102, CHARTDATE="2023-05-10", CHARTTIME=f(ts("2023-05-10 11:20")),
           SPEC_TYPE_DESC="BLOOD CULTURE", TEST_NAME="Blood Culture", STAY_ID=102)

    # ---------- stay 103: 35주, 체중 감소·고빌리루빈혈증
    p = (103, "H103", 103)
    birth = ts("2023-06-01 09:00")
    db.patient(p, "F", birth, ts("2023-06-01 10:00"), ts("2023-06-09 10:00"), ts("2023-06-12 10:00"), "Admission", ["P073", "P599"])
    for t, g in (("2023-06-01 11:00", 2500), ("2023-06-03 06:00", 2350), ("2023-06-04 06:00", 2270), ("2023-06-05 06:00", 2220)):
        db.chart(p, WT, ts(t), g, "g")
    vitals(db, p, ts("2023-06-01 11:00"), 92, lambda k, t: {"hr": 140 + rnd.randint(-10, 10), "rr": 45 + rnd.randint(-6, 6), "spo2": 97 + rnd.randint(-1, 2),
                                                            "bt": round(36.9 + rnd.uniform(-0.2, 0.2), 1)})
    for t, v in (("2023-06-04 05:00", 14.0), ("2023-06-05 05:00", 16.5)):
        db.lab(p, "tbil", ts(t), v)
    for k in range(0, 24, 3):
        db.inp(p, ORAL, ts("2023-06-04 08:00") + timedelta(hours=k), 40, 2.22)
        db.out(p, URINE, ts("2023-06-04 09:00") + timedelta(hours=k), 18)
    for k in (2, 10, 18):
        db.out(p, STOOL, ts("2023-06-04 07:00") + timedelta(hours=k), 5)
    return db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_nicu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    db = build()
    for t, rows in db.rows.items():
        with open(os.path.join(a.out, t + ".csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS[t])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        with open(os.path.join(a.out, t + ".csv.CHECKSUM"), "w") as fh:
            fh.write("synthetic\n")
    print("wrote %d tables to %s" % (len(db.rows), a.out))


if __name__ == "__main__":
    main()
