# -*- coding: utf-8 -*-
"""
run_reports.py — 표본 CSV(quick_sample.csv)의 환아를 한 명씩 처리해 통합 보고서(HTML)를 바로바로 저장.
전체가 끝나기를 기다리지 않고, 한 명이 끝날 때마다 index.html(목록)이 갱신되어 그 환아 보고서를 열 수 있음.

사용: python run_reports.py quick_sample.csv --out reports_live
      (옵션) --mode template|grounded|repair  --only 1,3,5-8  --resume  --profile 병원.json  --spec 명세.json|none
      여러 LLM 비교: --models all|모델1,모델2  (all = 게이트웨이 /models 목록 전체) [--with-template]
        → 환아마다 모델을 차례로 돌려 reports_live/<모델>/ 에 저장, index.html 에 모델별 비교
출력: reports_live/index.html          진행 상황·환아별 링크(실행 중에는 15초마다 자동 새로고침)
      reports_live/reports/report_<person_id>_<MMDD_HHMM>.html|md   환아별 통합 보고서
      reports_live/logs/<번호>.txt        실패 시 오류 내용
"""
import argparse
import html
import json
import os
import re
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
ap.add_argument("--models", default=None, help="비교할 LLM 모델. all 이면 KCD_LLM_BASE_URL/models 목록 전체, 또는 쉼표 구분")
ap.add_argument("--with-template", action="store_true", help="--models 와 함께: LLM 없는 템플릿 결과도 비교 기준으로 생성")
a = ap.parse_args()


def list_models():
    import urllib.request
    base = os.environ.get("KCD_LLM_BASE_URL", "").rstrip("/")
    req = urllib.request.Request(base + "/models", headers={"Authorization": "Bearer " + os.environ.get("KCD_LLM_API_KEY", "EMPTY")})
    with urllib.request.urlopen(req, timeout=60) as r:
        J = json.loads(r.read().decode("utf-8"))
    ids = [m.get("id") for m in (J.get("data") or J.get("models") or []) if m.get("id")]
    skip = [i for i in ids if re.search(r"asr|whisper|tts|embed|rerank|speech|audio|vision-only", i, re.I)]   # 문서 작성용이 아닌 모델
    if skip:
        print("[모델] 대화형이 아니라서 제외: %s" % ", ".join(skip))
    return [i for i in ids if i not in skip]


def safe(name):
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name))


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
if a.models:
    if not os.environ.get("KCD_LLM_BASE_URL"):
        raise SystemExit("--models 는 KCD_LLM_BASE_URL(와 KCD_LLM_API_KEY) 환경변수가 필요합니다.")
    models = list_models() if a.models.strip() == "all" else [m.strip() for m in a.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("모델 목록이 비었습니다. KCD_LLM_BASE_URL/models 응답을 확인하세요.")
    mode = a.mode if a.mode != "template" else "grounded"
    RUNS = ([("template", "template", None)] if a.with_template else []) + [(safe(m), mode, m) for m in models]
    print("[모델] %s (생성 모드 %s)" % (", ".join(m for m in models), mode))
else:
    if a.mode != "template" and not llm.ok:
        raise SystemExit("LLM 모드는 KCD_LLM_BASE_URL, KCD_LLM_MODEL 환경변수가 필요합니다.")
    RUNS = [("", a.mode, llm.model if a.mode != "template" else None)]
MULTI = len(RUNS) > 1 or bool(a.models)
os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)


def key(no, run):
    return "%d|%s" % (no, run)


rows = {key(int(x["sample_no"]), r[0]): dict(x, sample_no=int(x["sample_no"]), run=r[0], run_mode=r[1], model=r[2] or "", status="대기", link="", secs="")
        for _, x in df.iterrows() for r in RUNS}
STATUS = os.path.join(a.out, "status.json")
prev = {}
if a.resume and os.path.exists(STATUS):
    try:
        prev = {(k if "|" in k else k + "|"): v for k, v in json.load(open(STATUS, encoding="utf-8")).items()}
    except Exception:
        prev = {}
t_all = time.time()


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def write_index(running):
    """index.html 을 임시 파일에 쓴 뒤 바꿔치기(열어 둔 브라우저가 반쯤 쓴 파일을 읽지 않도록)."""
    done = sum(1 for r in rows.values() if r["status"].startswith("완료"))
    fail = sum(1 for r in rows.values() if r["status"].startswith("실패"))
    H = ["<!doctype html><html lang='ko'><head><meta charset='utf-8'>",
         "<meta http-equiv='refresh' content='15'>" if running else "",
         "<title>NICU 보고서 목록</title><style>body{font-family:-apple-system,'Malgun Gothic',sans-serif;margin:20px;color:#222}"
         "table{border-collapse:collapse;margin-bottom:16px}td,th{border:1px solid #ccc;padding:4px 8px;font-size:13px}th{background:#f3f3f3}"
         ".ok{color:#2e7d32}.fail{color:#c62828}.run{color:#e65100;font-weight:bold}.muted{color:#777}tr.first td{border-top:2px solid #888}</style></head><body>",
         "<h1>NICU 환자별 보고서%s</h1>" % (" — 모델 비교" if MULTI else ""),
         "<p>%s · 완료 %d / 실패 %d / 전체 %d · 경과 %d분 · %s</p>" % (
             "<b class='run'>실행 중(15초마다 새로고침)</b>" if running else "<b>끝남</b>", done, fail, len(rows),
             (time.time() - t_all) // 60, html.escape(", ".join("%s(%s)" % (r[2] or "LLM 없음", r[1]) for r in RUNS)))]
    if MULTI:   # 모델별 집계
        H.append("<h2>모델별 집계</h2><p class='muted'>생성 문장 근거 일치 = LLM이 직접 쓴 문장만(원문 대체·고정 문구·CDSS 제외). 원문 대체가 많은 모델은 전체 근거 일치가 높아 보일 수 있으니 함께 보세요. "
                 "LLM 판정자: %s</p>" % html.escape(os.environ.get("KCD_JUDGE_MODEL") or "없음(규칙 검증만, 모든 모델 같은 기준)") +
                 "<table><tr><th>실행</th><th>모델</th><th>완료</th><th>실패</th><th>생성 문장 근거 일치</th><th>전체 근거 일치</th><th>원문 대체 있는 환아</th>"
                 "<th>LLM 호출 오류 합</th><th>필수 사실 누락 합</th><th>원자료 반영 평균</th><th>평균 소요(초)</th></tr>")
        for run, mode, model in RUNS:
            rs = [r for r in rows.values() if r["run"] == run]
            ok = [r for r in rs if r["status"].startswith("완료") and _num(r.get("n"))]
            n = sum(_num(r.get("n")) or 0 for r in ok)
            sup = sum(_num(r.get("sup")) or 0 for r in ok)
            raws = [_num(str(r.get("raw", "")).split("%")[0]) for r in ok]
            raws = [x for x in raws if x is not None]
            secs = [_num(r.get("secs")) for r in ok]
            secs = [x for x in secs if x is not None]
            gn = sum(_num(r.get("gen_n")) or 0 for r in ok)
            gsup = sum(_num(r.get("gen_sup")) or 0 for r in ok)
            H.append("<tr><td>%s</td><td>%s</td><td>%d</td><td>%d</td><td>%s</td><td>%s</td><td>%d</td><td>%d</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
                html.escape(run), html.escape(model or "LLM 없음(템플릿)"), len(ok), sum(1 for r in rs if r["status"].startswith("실패")),
                ("%.1f%% (%d문장)" % (100.0 * gsup / gn, gn)) if gn else "-", ("%.1f%%" % (100.0 * sup / n)) if n else "-",
                sum(1 for r in ok if str(r.get("fb", "0/0")).split("/")[-1] not in ("0", "")), int(sum(_num(r.get("err")) or 0 for r in ok)),
                int(sum(_num(r.get("miss")) or 0 for r in ok)) if ok else "-", ("%.1f%%" % (sum(raws) / len(raws))) if raws else "-",
                ("%.0f" % (sum(secs) / len(secs))) if secs else "-"))
        H.append("</table><h2>환아별</h2>")
    H.append("<table><tr><th>번호</th><th>person_id</th><th>평가 시점</th>%s<th>보고서</th><th>상태</th><th>문장</th><th>근거 일치</th>"
             "<th>누락</th><th>확인 필요 항목</th><th>원자료 반영(미반영 항목)</th><th>원문 대체(섹션/문장)</th><th>CDSS(높음/전체)</th><th>소요(초)</th></tr>" % ("<th>모델</th>" if MULTI else ""))
    order = {r[0]: i for i, r in enumerate(RUNS)}
    last_no = None
    for k in sorted(rows, key=lambda k: (rows[k]["sample_no"], order.get(rows[k]["run"], 0))):
        r = rows[k]
        st = r["status"]
        cls = "ok" if st.startswith("완료") else ("fail" if st.startswith("실패") else ("run" if st == "진행 중" else "muted"))
        link = "<a href='%s' target='_blank'>열기</a>" % html.escape(r["link"]) if r["link"] else "-"
        first = r["sample_no"] != last_no
        last_no = r["sample_no"]
        H.append("<tr class='%s'><td>%s</td><td>%s</td><td>%s</td>%s<td>%s</td><td class='%s'>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            "first" if (first and MULTI) else "", r["sample_no"] if first else "", html.escape(str(r["person_id"])) if first else "",
            html.escape(str(r.get("index_datetime") or r.get("index_date") or "")) if first else "",
            ("<td>%s</td>" % html.escape(r["model"] or "템플릿")) if MULTI else "", link, cls, html.escape(st),
            r.get("n", "-"), r.get("sup", "-"), r.get("miss", "-"), r.get("gap", "-"), r.get("raw", "-"), r.get("fb", "-"), r.get("cdss", "-"), r["secs"]))
    H.append("</table><p class='muted'>보고서에는 실제 환자 정보가 있으므로 분석환경 밖으로 반출하지 마세요.</p></body></html>")
    allrows = dict(prev)
    allrows.update(rows)
    with open(STATUS + ".tmp", "w", encoding="utf-8") as f:
        json.dump(allrows, f, ensure_ascii=False, default=str)
    os.replace(STATUS + ".tmp", STATUS)
    p = os.path.join(a.out, "index.html")
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        f.write("\n".join(H))
    os.replace(p + ".tmp", p)


write_index(True)
print("[목록] %s  ← 이 파일을 열어 두면 한 명씩 끝날 때마다 링크가 생깁니다" % os.path.abspath(os.path.join(a.out, "index.html")))
pipe, pipe_dir = None, None
for no in sorted({r["sample_no"] for r in rows.values()}):
    for run, mode, model in RUNS:
        x = rows[key(no, run)]
        od = os.path.join(a.out, run) if run else a.out
        rd = os.path.join(od, "reports")
        exist = [f for f in os.listdir(rd) if f.startswith("report_%s_" % x["person_id"]) and f.endswith(".html")] if (a.resume and os.path.isdir(rd)) else []
        if exist:
            old = prev.get(key(no, run), {})
            x.update({k: old[k] for k in ("n", "sup", "miss", "gap", "raw", "cdss", "link", "secs", "gen_n", "gen_sup", "fb", "err", "judge") if old.get(k) not in (None, "")})
            x.update(status="완료(이전 실행)", link=x.get("link") or os.path.relpath(os.path.join(rd, sorted(exist)[-1]), a.out))
            write_index(True)
            continue
        x["status"] = "진행 중"
        write_index(True)
        t1 = time.time()
        try:
            d = fix_dir(x["data_dir"])
            if not os.path.isdir(d):
                raise SystemExit("데이터 폴더 없음: %s" % d)
            if d != pipe_dir:   # 같은 폴더의 다음 환아·다음 모델은 불러온 데이터를 재사용
                pipe, pipe_dir = None, None
                pipe = K.Pipeline(d, verbose=False, profile=a.profile, spec=a.spec)
                pipe_dir = d
            pipe.preload([x["stay_id"]])
            use = None
            if mode != "template":
                use = K.LLM()
                use.model = model or use.model
                use.ok = bool(use.base and use.model)
            if MULTI:
                print("[표본 %d] %s(%s) 시작" % (no, model or "템플릿", mode))
            r = pipe.process(x["stay_id"], int(x["kcd_day"]), od, mode=mode, llm=use)
            m = r["metrics"]
            rep = r.get("report") or ""
            x.update(status="완료", link=os.path.relpath(rep[:-3] + ".html", a.out) if rep else "", n=m.get("n_sentences"), sup=m.get("supported"),
                     miss=m.get("missing"), gap=m.get("checklist_gap"), cdss="%s/%s" % (m.get("alerts_high", 0), m.get("alerts_total", 0)),
                     raw=("%s%% (%s)" % (m.get("raw_used_pct"), m.get("raw_items_unused"))) if m.get("raw_used_pct") is not None else "-",
                     gen_n=(m.get("gen_sents") or 0) + (m.get("parent_gen_sents") or 0), gen_sup=(m.get("gen_supported") or 0) + (m.get("parent_gen_supported") or 0),
                     fb="%s/%s" % ((m.get("gen_fallback_sections") or 0) + (m.get("parent_gen_fallback_sections") or 0),
                                   (m.get("fallback_sents_total") or 0) + (m.get("parent_fallback_sents_total") or 0)),
                     err=(m.get("gen_llm_errors") or 0) + (m.get("parent_gen_llm_errors") or 0), judge=m.get("judge_model", ""))
            want = str(x.get("index_datetime") or "")[5:16]
            got = pd.Timestamp(r["ctx"].t0).strftime("%m-%d %H:%M")
            if want and want != got:
                x["status"] = "완료(주의: 평가 시점 %s, 생성 기준 %s)" % (want, got)
            print("[%d/%d] 표본 %d%s 완료 %.0f초 → %s" % (sum(1 for v in rows.values() if v["status"] not in ("대기", "진행 중")), len(rows), no,
                                                    (" " + (model or "템플릿")) if MULTI else "", time.time() - t1,
                                                    os.path.abspath(rep[:-3] + ".html") if rep else "-"))
        except BaseException as e:   # 한 건 실패해도 다음 계속(Ctrl+C 는 중단)
            if isinstance(e, KeyboardInterrupt):
                x["status"] = "중단"
                write_index(False)
                raise
            msg = str(e) if isinstance(e, SystemExit) else repr(e)
            lg = os.path.join(a.out, "logs", "%03d%s.txt" % (no, ("_" + run) if run else ""))
            with open(lg, "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            x["status"] = "실패: %s" % msg[:100]
            print("표본 %d%s 실패: %s (%s)" % (no, (" " + run) if run else "", msg[:100], lg))
        x["secs"] = "%.0f" % (time.time() - t1)
        write_index(True)
write_index(False)
done = sum(1 for r in rows.values() if r["status"].startswith("완료"))
print("\n완료 %d / 전체 %d, %.0f분 → %s" % (done, len(rows), (time.time() - t_all) / 60, os.path.abspath(os.path.join(a.out, "index.html"))))
