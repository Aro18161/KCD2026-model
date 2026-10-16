# -*- coding: utf-8 -*-
"""
run_reports.py — 표본 CSV(quick_sample.csv)의 환아를 한 명씩 처리해 통합 보고서(HTML)를 바로바로 저장.
전체가 끝나기를 기다리지 않고, 한 명이 끝날 때마다 index.html(목록)이 갱신되어 그 환아 보고서를 열 수 있음.

사용: python run_reports.py quick_sample.csv --out reports_live
      (옵션) --mode template|grounded|repair  --only 1,3,5-8  --resume  --profile 병원.json  --spec 명세.json|none
출력: reports_live/index.html          진행 상황·환아별 링크(실행 중에는 15초마다 자동 새로고침)
      reports_live/reports/report_<person_id>_<MMDD_HHMM>.html|md   환아별 통합 보고서
      reports_live/logs/<번호>.txt        실패 시 오류 내용
"""
import argparse
import html
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_argv = sys.argv
sys.argv = [sys.argv[0]]
import pandas as pd  # noqa: E402
import kcd_nicu as K  # noqa: E402
sys.argv = _argv

ap = argparse.ArgumentParser()
ap.add_argument("sample_csv")
ap.add_argument("--out", default="reports_live")
ap.add_argument("--mode", default="template", choices=["template", "grounded", "repair"])
ap.add_argument("--only", default=None, help="처리할 표본 번호. 예: 1,3,5-8")
ap.add_argument("--resume", action="store_true", help="이미 보고서가 있는 표본은 건너뜀")
ap.add_argument("--profile", default=None)
ap.add_argument("--spec", default=None)
a = ap.parse_args()


def parse_only(s):
    out = set()
    for part in str(s).split(","):
        part = part.strip()
        if "-" in part:
            x, y = part.split("-", 1)
            out.update(range(int(x), int(y) + 1))
        elif part:
            out.add(int(part))
    return out


def fix_dir(d):
    """다른 팟에서 만든 CSV 의 경로(/home/<다른 사용자>/workspace/...)를 현재 홈 기준으로 바꿈."""
    d = str(d)
    if os.path.isdir(d):
        return d
    if "/workspace/" in d:
        alt = os.path.join(os.path.expanduser("~/workspace"), d.split("/workspace/", 1)[1])
        if os.path.isdir(alt):
            return alt
    return d


df = pd.read_csv(a.sample_csv, dtype=str)
need = {"person_id", "stay_id", "kcd_day", "data_dir"}
miss = need - set(df.columns)
if miss:
    raise SystemExit("표본 CSV에 열이 없습니다: %s" % ", ".join(sorted(miss)))
if "sample_no" not in df.columns:
    df.insert(0, "sample_no", [str(i) for i in range(1, len(df) + 1)])
df = df[df["sample_no"].notna() & df["stay_id"].notna()].copy()
df["sample_no"] = df["sample_no"].astype(int)
if a.only:
    df = df[df["sample_no"].isin(parse_only(a.only))]
df = df.sort_values("sample_no")
llm = K.LLM()
if a.mode != "template" and not llm.ok:
    raise SystemExit("LLM 모드는 KCD_LLM_BASE_URL, KCD_LLM_MODEL 환경변수가 필요합니다.")
os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)

rows = {int(x["sample_no"]): dict(x, status="대기", link="", secs="") for _, x in df.iterrows()}
STATUS = os.path.join(a.out, "status.json")
prev = {}
if a.resume and os.path.exists(STATUS):
    try:
        prev = {int(k): v for k, v in json.load(open(STATUS, encoding="utf-8")).items()}
    except Exception:
        prev = {}
t_all = time.time()


def write_index(running):
    """index.html 을 임시 파일에 쓴 뒤 바꿔치기(열어 둔 브라우저가 반쯤 쓴 파일을 읽지 않도록)."""
    done = sum(1 for r in rows.values() if r["status"].startswith("완료"))
    fail = sum(1 for r in rows.values() if r["status"].startswith("실패"))
    H = ["<!doctype html><html lang='ko'><head><meta charset='utf-8'>",
         "<meta http-equiv='refresh' content='15'>" if running else "",
         "<title>NICU 보고서 목록</title><style>body{font-family:-apple-system,'Malgun Gothic',sans-serif;margin:20px;color:#222}"
         "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px;font-size:13px}th{background:#f3f3f3}"
         ".ok{color:#2e7d32}.fail{color:#c62828}.run{color:#e65100;font-weight:bold}.muted{color:#777}</style></head><body>",
         "<h1>NICU 환자별 보고서</h1>",
         "<p>%s · 완료 %d / 실패 %d / 전체 %d · 경과 %d분 · 생성 모드 %s</p>" % (
             "<b class='run'>실행 중(15초마다 새로고침)</b>" if running else "<b>끝남</b>", done, fail, len(rows),
             (time.time() - t_all) // 60, html.escape(a.mode)),
         "<table><tr><th>번호</th><th>person_id</th><th>평가 시점</th><th>보고서</th><th>상태</th><th>문장</th><th>근거 일치</th>"
         "<th>누락</th><th>확인 필요 항목</th><th>CDSS(높음/전체)</th><th>소요(초)</th></tr>"]
    for no in sorted(rows):
        r = rows[no]
        st = r["status"]
        cls = "ok" if st.startswith("완료") else ("fail" if st.startswith("실패") else ("run" if st == "진행 중" else "muted"))
        link = "<a href='%s' target='_blank'>열기</a>" % html.escape(r["link"]) if r["link"] else "-"
        H.append("<tr><td>%d</td><td>%s</td><td>%s</td><td>%s</td><td class='%s'>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            no, html.escape(str(r["person_id"])), html.escape(str(r.get("index_datetime") or r.get("index_date") or "")), link, cls, html.escape(st),
            r.get("n", "-"), r.get("sup", "-"), r.get("miss", "-"), r.get("gap", "-"), r.get("cdss", "-"), r["secs"]))
    H.append("</table><p class='muted'>보고서에는 실제 환자 정보가 있으므로 분석환경 밖으로 반출하지 마세요.</p></body></html>")
    allrows = dict(prev)
    allrows.update({k: v for k, v in rows.items()})
    with open(STATUS + ".tmp", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in allrows.items()}, f, ensure_ascii=False, default=str)
    os.replace(STATUS + ".tmp", STATUS)
    p = os.path.join(a.out, "index.html")
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        f.write("\n".join(H))
    os.replace(p + ".tmp", p)


write_index(True)
print("[목록] %s  ← 이 파일을 열어 두면 한 명씩 끝날 때마다 링크가 생깁니다" % os.path.abspath(os.path.join(a.out, "index.html")))
pipe, pipe_dir = None, None
for no in sorted(rows):
    x = rows[no]
    d = fix_dir(x["data_dir"])
    exist = [f for f in os.listdir(os.path.join(a.out, "reports")) if f.startswith("report_%s_" % x["person_id"]) and f.endswith(".html")] \
        if (a.resume and os.path.isdir(os.path.join(a.out, "reports"))) else []
    if exist:
        old = prev.get(no, {})
        x.update({k: old[k] for k in ("n", "sup", "miss", "gap", "cdss", "link", "secs") if old.get(k) not in (None, "")})
        x.update(status="완료(이전 실행)", link=x.get("link") or "reports/" + sorted(exist)[-1])
        write_index(True)
        continue
    x["status"] = "진행 중"
    write_index(True)
    t1 = time.time()
    try:
        if not os.path.isdir(d):
            raise SystemExit("데이터 폴더 없음: %s" % d)
        if d != pipe_dir:   # 같은 폴더의 다음 환아는 불러온 데이터를 재사용
            pipe, pipe_dir = None, None
            pipe = K.Pipeline(d, verbose=False, profile=a.profile, spec=a.spec)
            pipe_dir = d
        pipe.preload([x["stay_id"]])
        r = pipe.process(x["stay_id"], int(x["kcd_day"]), a.out, mode=a.mode, llm=llm if llm.ok else None)
        m = r["metrics"]
        rep = r.get("report") or ""
        x.update(status="완료", link=os.path.relpath(rep[:-3] + ".html", a.out) if rep else "", n=m.get("n_sentences"), sup=m.get("supported"),
                 miss=m.get("missing"), gap=m.get("checklist_gap"), cdss="%s/%s" % (m.get("alerts_high", 0), m.get("alerts_total", 0)))
        want = str(x.get("index_datetime") or "")[5:16]
        got = pd.Timestamp(r["ctx"].t0).strftime("%m-%d %H:%M")
        if want and want != got:
            x["status"] = "완료(주의: 평가 시점 %s, 생성 기준 %s)" % (want, got)
        print("[%d/%d] 표본 %d 완료 %.0f초 → %s" % (len([1 for v in rows.values() if v["status"] != "대기" and v["status"] != "진행 중"]), len(rows), no,
                                              time.time() - t1, os.path.abspath(rep[:-3] + ".html") if rep else "-"))
    except BaseException as e:   # 한 명 실패해도 다음 환아 계속(Ctrl+C 는 중단)
        if isinstance(e, KeyboardInterrupt):
            x["status"] = "중단"
            write_index(False)
            raise
        msg = str(e) if isinstance(e, SystemExit) else repr(e)
        with open(os.path.join(a.out, "logs", "%03d.txt" % no), "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        x["status"] = "실패: %s" % msg[:100]
        print("표본 %d 실패: %s (logs/%03d.txt)" % (no, msg[:100], no))
    x["secs"] = "%.0f" % (time.time() - t1)
    write_index(True)
write_index(False)
done = sum(1 for r in rows.values() if r["status"].startswith("완료"))
print("\n완료 %d / 전체 %d, %.0f분 → %s" % (done, len(rows), (time.time() - t_all) / 60, os.path.abspath(os.path.join(a.out, "index.html"))))
