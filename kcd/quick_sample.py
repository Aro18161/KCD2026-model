# -*- coding: utf-8 -*-
"""
quick_sample.py — 생후 28일 이하 NICU 환아의 (person_id, 평가 날짜) 간이 무작위 추출.
큰 이벤트 테이블은 읽지 않고 ICUSTAYS·PATIENTS·ADMISSIONS 만 사용(빠름).

조건
  1) NICU 병동(first_careunit 에 NICU 포함)
  2) 신생아: 나이 0세 또는 나이 결측(K-MIMIC은 NICU 환아 나이를 비워 둠)
  3) 출생 시각 근사: 입원 후 48시간 안에 NICU 입실 → 입원 시각 = 출생 시각
  4) 평가 시점 = 매일 07:00. 직전 24시간 전체가 NICU 재실 중이고, 생후 28일 이하, 사망 전
  5) 한 환아당 1시점 무작위, 전체에서 N명 무작위
  ※ 활력징후·섭취 기록 건수는 확인하지 않음 → 생성 문서에 '기록 없음' 칸이 생길 수 있음

사용: python quick_sample.py "~/workspace/datasets/kmimic-emr-003/*" --n 30 --seed 0 --out quick_sample.csv
"""
import argparse
import glob
import os
import random
import re
from datetime import timedelta

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("data", help='폴더 또는 와일드카드(따옴표로 감싸기), 예: "~/workspace/datasets/kmimic-emr-003/*"')
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-age-days", type=int, default=28)
ap.add_argument("--birth-hours", type=float, default=48)
ap.add_argument("--cutoff-hour", type=int, default=7)
ap.add_argument("--out", default="quick_sample.csv")
a = ap.parse_args()


def read(d, name, cols):
    p = os.path.join(d, name + ".csv")
    df = pd.read_csv(p, dtype=str, low_memory=False, usecols=lambda c: c.lower() in cols)
    df.columns = [c.lower() for c in df.columns]
    return df


def parse(s):
    """가명화로 2262년을 넘는 연도는 400년 단위로 당겨 파싱. (Timestamp, 당긴 연수) 반환."""
    s = str(s).strip()
    m = re.match(r"^(\d{4})(.*)$", s)
    if not m or s in ("", "nan"):
        return None, 0
    y = int(m.group(1))
    k = ((y - 2261) // 400 + 1) * 400 if y > 2261 else 0
    t = pd.to_datetime("%04d%s" % (y - k, m.group(2)), errors="coerce")
    return (None, 0) if pd.isna(t) else (t, k)


def fmt(t, k):
    """당긴 연수를 되돌려 원자료와 같은 날짜 문자열로."""
    dt = t.to_pydatetime()
    return dt.replace(year=dt.year + k)


root = os.path.expanduser(a.data)
dirs = sorted(d for d in glob.glob(root) if os.path.exists(os.path.join(d, "ICUSTAYS.csv"))) if any(c in root for c in "*?[") else [root]
print("폴더 %d개" % len(dirs))
flow = {"NICU stay": 0, "신생아(0세·나이 결측)": 0, "출생 시각 근사 가능": 0, "적격 시점 1개 이상": 0}
rows = []
for d in dirs:
    icu = read(d, "ICUSTAYS", {"subject_id", "hadm_id", "stay_id", "first_careunit", "intime", "outtime"})
    pat = read(d, "PATIENTS", {"subject_id", "anchor_age"})
    adm = read(d, "ADMISSIONS", {"hadm_id", "admittime", "deathtime"})
    icu = icu[icu["first_careunit"].astype(str).str.upper().str.contains("NICU", na=False)]
    flow["NICU stay"] += len(icu)
    icu = icu.merge(pat.drop_duplicates("subject_id"), on="subject_id", how="left").merge(adm.drop_duplicates("hadm_id"), on="hadm_id", how="left")
    age = pd.to_numeric(icu["anchor_age"], errors="coerce")
    icu = icu[age.isna() | (age < 1)]
    flow["신생아(0세·나이 결측)"] += len(icu)
    for _, r in icu.iterrows():
        tin, k = parse(r["intime"])
        tout, _ = parse(r["outtime"])
        tadm, _ = parse(r.get("admittime"))
        tdie, _ = parse(r.get("deathtime"))
        if tin is None or tout is None or tadm is None:
            continue
        if not (0 <= (tin - tadm).total_seconds() <= a.birth_hours * 3600):
            continue
        flow["출생 시각 근사 가능"] += 1
        first = tin.normalize() + timedelta(hours=a.cutoff_hour)
        if first <= tin:
            first += timedelta(days=1)
        cands, t, day = [], first, 1
        while t <= tout:
            ok = (t - tin) >= timedelta(hours=24) and (t - tadm).days <= a.max_age_days and (tdie is None or t < tdie)
            if ok:
                cands.append((day, t))
            t += timedelta(days=1)
            day += 1
        if cands:
            flow["적격 시점 1개 이상"] += 1
            rows.append({"person_id": r["subject_id"], "cands": cands, "k": k, "stay_id": r["stay_id"], "hadm_id": r["hadm_id"], "data_dir": d, "tadm": tadm})

rnd = random.Random(a.seed)
by_person = {}
for x in rows:
    by_person.setdefault(x["person_id"], []).append(x)
persons = sorted(by_person)
pick = sorted(rnd.sample(persons, min(a.n, len(persons))))
out = []
for pid in pick:
    x = rnd.choice(by_person[pid])
    day, t = rnd.choice(x["cands"])
    real = fmt(t, x["k"])
    out.append({"person_id": pid, "index_date": real.strftime("%Y-%m-%d"), "index_datetime": real.strftime("%Y-%m-%d %H:%M"),
                "age_days": (t - x["tadm"]).days, "stay_id": x["stay_id"], "hadm_id": x["hadm_id"], "kcd_day": day, "data_dir": x["data_dir"]})
df = pd.DataFrame(out)
df.insert(0, "sample_no", range(1, len(df) + 1))
df.to_csv(a.out, index=False)
for k_, v in flow.items():
    print("  %-22s %d" % (k_, v))
print("적격 환아 %d명 중 %d명 추출 → %s" % (len(persons), len(df), a.out))
cmd = os.path.splitext(a.out)[0] + "_commands.sh"
with open(cmd, "w", encoding="utf-8") as f:
    for _, x in df.iterrows():
        f.write("python kcd_nicu.py run --data %s --stay %s --day %d --out generated_quick --mode ${KCD_MODE:-template}\n" % (x["data_dir"], x["stay_id"], x["kcd_day"]))
print("문서 생성 명령: bash %s" % cmd)
print("주의: 결과 파일에는 person_id·날짜가 있으므로 분석환경 밖으로 반출하지 마세요.")
