#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""selftest.py — 합성 샘플로 생성·검증·CDSS 동작 확인. 사용: python selftest.py [합성데이터 경로]"""
import os
import random
import sys

sys.argv = sys.argv[:2]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kcd_transfer as K  # noqa: E402

data = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data", "synthetic")
if not os.path.isdir(data):
    os.system("%s %s --out %s" % (sys.executable, os.path.join(HERE, "make_synthetic_sample.py"), data))
p = K.Pipeline(data, verbose=False)
ok = 0


def check(cond, msg):
    global ok
    if not cond:
        raise AssertionError(msg)
    ok += 1
    print("  OK", msg)


r1 = p.process("1", None, write=False)
rules1 = {a["rule_id"] for a in r1["alerts"]}
check(r1["metrics"]["hallucination_rate"] == 0 and r1["metrics"]["missing"] == 0, "템플릿 기록지: 할루시네이션 0, 누락 0")
check(all(x["cites"] for x in r1["sents"] if x["section"] != "S6" and not x["text"].startswith("해당")), "모든 본문 문장에 인용 존재")
check("AKI_KDIGO" in rules1, "stay1: AKI(KDIGO) 알림")
check("ABX_MISMATCH" in rules1, "stay1: 항생제-감수성 불일치 알림")
check("MICRO_PENDING" in rules1, "stay1: 배양 미보고 알림")
check(any(f.ftype == "infusion_episode" and f.status == "stopped" for f in r1["facts"]), "stay1: 승압제 에피소드 중단 상태")
r2 = p.process("2", None, write=False)
rules2 = {a["rule_id"] for a in r2["alerts"]}
check("RENAL_DOSE" in rules2, "stay2: eGFR 기반 에녹사파린 경고")
check("DUR_PAIR" in rules2, "stay2: 병용금기(ciprofloxacin+tizanidine)")
check("QT_RISK" in rules2, "stay2: 저칼륨 + QT 연장 약물")
r3 = p.process("3", None, write=False)
check(not [a for a in r3["alerts"] if a["severity"] == "high"], "stay3: 고위험 알림 없음")
check(any("섭취 기록 없음" in f.text for f in r3["facts"]), "stay3: 섭취 기록 부재를 0으로 쓰지 않음")
rng = random.Random(0)
V = K.Verifier(r1["ctx"], r1["facts"], alerts=r1["alerts"])
for kind in ("num_change", "drug_swap", "time_shift", "status_flip", "fabricate"):
    out = K.perturb(r1["sents"], r1["facts"], rng, kind)
    if out is None:
        continue
    S, truth = out
    v = [x for x in V.verify(S) if x["sid"] == truth["sid"]][0]
    check(v["label"] != "Supported", "B1 탐지: %s → %s" % (kind, v["label"]))
S, truth = K.perturb(r1["sents"], r1["facts"], rng, "delete")
st = [c for c in V.coverage(S) if c["fact_id"] == truth["fact_id"]][0]["status"]
check(st in ("Missing", "Partial"), "B2 탐지: 필수 fact 삭제 → %s" % st)
print("selftest 통과: %d개 항목" % ok)
