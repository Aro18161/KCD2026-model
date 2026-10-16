# KCD 2026 — AI 요약·인수인계 기록지 생성 / 검증 / CDSS 모델 구조 설계

- 버전: v0.2 (2026-10-16 갱신, 최초 v0.1)
- 대상: 팀 내부 설계 기준 문서. 심사 발표 자료의 뼈대로 사용함.
- 전제: 대회 팀 활동 시간은 10/16(금) 14:00 ~ 10/17(토) 15:00, 심사는 10/17 15:00~18:00 (datathon.kr 일정표 기준). 실 구현 가능 시간은 약 24시간임. 모든 설계는 이 시간 안에 데모 가능한 수준으로 범위를 제한함.

---

## v0.2 갱신 사항 (2026-10-16)

### 가. 산출물 확정
- 산출물을 **ICU → 병동 전동기록지**로 확정함(팀 결정). 기준 시각 t0 = ICU 퇴실 시각(ICUSTAYS.OUTTIME).
- 대상 stay: ICU 생존 퇴실(icu_expire_flag=0)이고 같은 입원의 퇴원 시각이 ICU 퇴실 이후인 stay.
- 섹션: 1) 환자·전동 정보 2) 입원·ICU 입실 사유(진단·수술) 3) ICU 경과 요약(인공호흡·승압제·진정제·CRRT·시술·검사 추세·AKI·배양) 4) 전동 시점 상태(최종 활력징후·GCS·호흡 보조·최근 48시간 검사·24시간 섭취배설·배액관) 5) 전동 시 투약(투여 중·중단) 6) 병동 주의사항(CDSS).
- 기존 I-PASS 인수인계지 설계(§3-1)는 전동기록지 섹션으로 대체함. Fact Sheet·검증기·CDSS 원칙은 유지함.

### 나. KHDP에서 확인한 데이터 사실 (KHDP Open API, 2026-10-16 조회)
| 코드 | 데이터셋 | 접근 | 과제 활용 |
|---|---|---|---|
| K-MIMIC | K-MIMIC SNUH Subset: 서울대병원 2023년 ICU 입원, EMR 25개 테이블, 날짜는 상대시간 변환으로 기술됨 | credentialed(API 다운로드 불가) | 본 분석 대상(분석환경 내) |
| SYN-ICU | Synthetic K-MIMIC v2.0.0: SNUH K-MIMIC 기반 GAN 합성, 15개 테이블, 1,328 입원 | open, CC-BY-4.0 | 코드 개발·검증기 B3 평가(로컬) |
| KMM | K-MIMIC-MEDS: SYN-ICU의 MEDS 변환 | open | 미사용 |
| MIMIC-IV-Note-KR | MIMIC-IV-Note 퇴원요약 한국어 번역(gpt-oss-120b) | credentialed | 임상의 작성 기록 비교(§4-4) 후보 |
| ann-pt-summ | 의사 작성·LLM 생성 환자 요약 각 100건의 근거 없는 사실 전문가 주석 | credentialed | 검증기 B1 외부 검증 후보 |
| K-MIMIC-MORTAL | SNUH K-MIMIC 사망 환자 부분집합 | restricted | 해당 없음(전동 대상 아님) |

- SYN-ICU에서 확인한 형식: ITEMID 문자열(`001C_…` 활력, `001I_…` 섭취, `001O_…` 배설, `001P_…` 시술, `001L…` 검사), 한글·영문 혼재 라벨(`HR > 심박동수`), 진단 KCD8(점 없는 코드, 예: N179), 시술 ICD9CM(앞자리 0 누락, 예: 1.59 = 01.59), 승압제·진정제는 EMAR + EMAR_DETAIL 주입속도, INPUTEVENTS는 수액량(cc)만 있음, 가명화 날짜가 2262년을 넘는 행 존재(400년 단위 보정으로 처리).
- 실제 K-MIMIC은 PRESCRIPTIONS·MICROBIOLOGYEVENTS·TESTEVENTS·ED 테이블을 추가로 가짐. 해당 경로는 자체 합성 샘플(`kcd/make_synthetic_sample.py`)로 점검함.

### 다. 구현 상태
- 구현: `kcd/kcd_transfer.py` 단일 파일. 사용법·결과는 `README.md` 참고.
- CDSS 규칙(전동 특화): NEWS2(RCP 2017), SWIFT 재입실 위험(Gajic 2008), 승압제 투여 중·중단 24시간 이내, 진정제 지속주입 중, 전동 직전 인공호흡기 기록, CRRT 이력, AKI(KDIGO), 소변량 감소, 검사 위험값, Lactate 미정상화, 인슐린 전환, eGFR 기반 용량(CKD-EPI 2021), 병용금기, 저칼륨·저마그네슘 + QT 연장 약물, 배양 미보고, 항생제-감수성 불일치, 배액관 유지.
- CDSS가 계산한 값(NEWS2, SWIFT, 계산 eGFR)은 입력 fact를 출처로 갖는 파생 fact로 등록해 검증기 대조 대상에 포함함.
- 검증기 판정 기준: 인용한 fact와 수치·시각이 다르면 다른 기록 값과 일치해도 Contradicted(VeriFact의 "인용 근거 기준 지지" 원칙).
- SYN-ICU B3 결과: `docs/syn_perturb_eval.csv`.

---

## 0. 전제 조건과 제약

### 0-1. 대회 주제와의 연결
- 주제: "Agentic AI: Light and Shadow". 환영사의 요구 사항은 두 가지임. (1) 에이전트로 만들 것, (2) 에이전트가 어디서 실패하는지, 그것을 어떻게 알 수 있는지를 밝힐 것.
- 본 팀 기획의 대응: 생성 모델(빛) + 검증 모델(그림자를 드러내는 장치) + CDSS(에이전트가 행동까지 할 때의 책임 문제). 검증 모델이 발표의 중심축이 되어야 주제 적합성이 높음.

### 0-2. 데이터 현황 (조사 결과)
- K-MIMIC: 국내 10개 병원 71개 ICU, 2001~2023, ICU 입원 약 28.7만 건 (Korean J Anesthesiol 2026, doi 10.4097/kja.25752). 데이터톤 공개판은 큐레이션된 부분집합임. 개인정보보호법에 따라 폐쇄 분석환경(IMPACT 플랫폼)에서만 접근 가능함.
- 데이터톤 공개판 테이블(2024 공개판 문서 기준, sites.google.com/view/k-mimic/modules):
  - Core: ADMISSIONS, PATIENTS, TRANSFERS
  - Hosp: DIAGNOSES_ICD, D_LABITEMS, D_TESTITEMS, EMAR, EMAR_DETAIL, LABEVENTS, MICROBIOLOGYEVENTS, PRESCRIPTIONS, PROCEDURES_ICD, SERVICES, TESTEVENTS
  - ICU: CHARTEVENTS, DATETIMEEVENTS(간호 활동 기록), D_ITEMS, ICUSTAYS, INPUTEVENTS, OUTPUTEVENTS, PROCEDUREEVENTS
  - ED: EDSTAY, MEDRECON, TRIAGE
  - HIS: HOSPITAL
- **자유서술 노트 테이블(경과기록, 간호기록, 퇴원요약)이 없음.** 텍스트 성격 컬럼은 EMAR_DETAIL.NOTE_TXT, EMAR.EVENT_TXT, LABEVENTS.COMMENTS, MICROBIOLOGYEVENTS.COMMENTS/ORDER_COMMENTS, CHARTEVENTS/DATETIMEEVENTS.VALUE 정도임. D_ITEMS 라벨은 간호 서술·간호 활동에서 유래함.
- 코드 체계: 진단 ICD-9/10, 시술 ICD-9-CM, 검사·약제 EDI 코드, PRESCRIPTIONS.GSN이 ATC로 기술됨. LOINC 없음. 약품명·일부 라벨에 한글 혼재.
- MIMIC-IV(+MIMIC-IV-Note): 임상의가 실제 작성한 퇴원요약(Brief Hospital Course 포함)과 영상판독문이 있음(영어). PhysioNet 자격 인증이 있는 팀원이 있을 때만 사용 가능함. LLM 사용은 PhysioNet 정책상 Azure OpenAI(사람 검토 opt-out), Amazon Bedrock, Google Vertex AI Gemini, Anthropic Claude만 허용되며 ChatGPT·OpenAI 공개 API는 금지임.

### 0-3. 설계에 미치는 영향
1. 기획 2번("기존 의사·간호사 기록지의 누락을 우리 검증기로 확인")은 K-MIMIC만으로는 비교 대상 기록지가 없어 수행 불가함. 대안 두 가지를 병행 설계함: (가) MIMIC-IV-Note 퇴원요약을 임상의 작성 기록지로 사용, (나) K-MIMIC에서는 "비접지(naive) LLM 에이전트가 쓴 기록지"를 비교군으로 사용.
2. 입력이 전부 구조화 데이터이므로 사실 검증을 퍼지한 NLI 대신 **결정론적 조회(테이블 행 매칭)** 로 수행할 수 있음. 이는 검증기의 신뢰도를 높이는 유리한 조건임.
3. 폐쇄환경의 GPU·LLM·인터넷 가용성이 미확인임. 생성 모델은 **LLM 없이도 동작하는 경로(템플릿 렌더러)** 를 기본으로 두고, LLM은 교체 가능한 부품으로 설계함.

---

## 1. 전체 아키텍처

```
K-MIMIC / MIMIC-IV 테이블
        │
        ▼
[D] 공통 데이터 계층 ─ Timeline Builder → Fact Extractor → Fact Sheet(JSON, 출처 포함)
        │                                             │
        ▼                                             │
[A] 생성 모델 ─ Content Planner → Surface Realizer ──┐  │
        │        (LLM 또는 템플릿)                     │  │
        ▼                                             │  │
   기록지 초안 (문장마다 fact_id 인용)                  │  │
        │                                             ▼  ▼
[B] 검증 모델 ─ B1 할루시네이션 판정 ← claim 분해, 근거 조회(결정론 + LLM judge)
        │        B2 누락 판정       ← must-include 집합 커버리지
        │        B3 검증기 자체 검증 ← 합성 오류 주입 + 임상의 주석
        ├── 불합격 → [A]로 섹션 단위 재생성 (최대 2회)
        ▼
   검증 완료 기록지 + 검증 리포트(claim별 판정, 누락 목록)
        │
        ▼
[C] CDSS ─ 규칙 엔진(결정론) + 지식베이스(DUR, 임계값, 용량) + LLM 설명문 → 알림(출처·규칙 ID 포함)
        │        CDSS 알림 문장도 [B1]로 재검증
        ▼
   최종 산출물: 기록지 + 검증 배지 + CDSS 알림 + 감사 로그(모든 도구 호출 기록)
```

### 핵심 설계 원칙
1. **Fact Sheet 중심.** 세 모델이 모두 같은 구조화 중간표현(Fact Sheet)을 소비함. 생성은 Fact Sheet를 문장으로 옮기고, 검증은 문장을 Fact Sheet(와 원본 행)에 대조하고, CDSS는 Fact Sheet 위에서 규칙을 평가함.
2. **결정론 우선, LLM 최소.** 사실 선택·수치·시간은 코드가 결정함. LLM은 문장화와 서술형 판단에만 사용함. LLM이 없어도 정확한(단조로운) 기록지가 나오도록 함.
3. **출처 추적.** 모든 fact는 (table, row_id, charttime)을 가짐. 기록지의 모든 문장은 fact_id를 인용함. CDSS 알림은 trigger fact_id와 rule_id를 가짐.
4. **검증기는 작성자 불문.** 사람이 쓴 기록지, naive 에이전트가 쓴 기록지, 본 파이프라인 기록지에 동일한 검증기를 적용함. "누가 썼든 같은 잣대"가 발표의 핵심 메시지임.
5. **검증기 자체도 검증.** "우리 검증기가 우리 생성기를 더 낫다고 판정"은 순환논증임. 합성 오류 주입 세트와 임상의 주석으로 검증기 성능을 먼저 보고함.

---

## 2. 공통 데이터 계층 [D]

### 2-1. 시간창(window) 정의
- 인수인계지(ICU shift handoff): 기준 시각 t0에서 직전 12시간(또는 8시간) 사건 + 현재 상태 스냅숏 + 입실 이후 핵심 경과 1~3개.
- 입원경과요약(hospital course): 입원~기준 시각(퇴원) 전체. MIMIC-IV 퇴원요약 BHC와 비교할 때만 사용함.
- 기본 산출물은 **ICU 인수인계지**로 함(K-MIMIC이 ICU 데이터이고, 시간창이 짧아 fact 수가 다루기 쉬움).

### 2-2. Timeline Builder
- 입력: stay_id(또는 hadm_id), t0, window.
- 처리: 각 테이블에서 window 내 행을 조회하고 공통 이벤트 스키마로 정규화함.
- 공통 이벤트: `{event_id, src_table, src_row_id, subject_id, hadm_id, stay_id, time, kind, concept, value_num, value_text, unit, flag}`
- kind: vital, lab, med_order, med_admin, infusion, output, procedure, micro, imaging, diagnosis, transfer, service, triage, nursing_log.

### 2-3. 개념 정규화 사전 (1일차 작성, 수작업 + 정규식)
- 검사: D_LABITEMS.LABEL/EDI_CODE → 내부 개념 약 40개(Na, K, Cr, BUN, Hb, Plt, WBC, CRP, lactate, glucose, INR, aPTT, bilirubin, troponin, ABG 항목 등). 한글·영문 라벨을 모두 사전에 수록함.
- 활력징후·장치: D_ITEMS.LABEL → HR, SBP/DBP/MAP, RR, SpO2, Temp, GCS, FiO2, PEEP, 인공호흡 모드, 승압제(INPUTEVENTS 품목), 요량.
- 약제: PRESCRIPTIONS.DRUG(한글 상품명 혼재) → EDI 코드 → ATC(GSN) → 성분명. ATC 앞 자리로 약효군(승압제 C01CA, 항생제 J01, 항응고제 B01, 인슐린 A10A 등)을 판정함.
- 사전에 없는 항목은 `concept = "UNMAPPED:<label>"` 로 보존하고 생성 시에는 원문 라벨을 그대로 사용함(누락보다 원문 보존을 우선함).

### 2-4. Fact Extractor → Fact Sheet
- 원시 이벤트에서 "기록지에 쓸 수 있는 단위 사실"을 생성함. 사실 유형과 생성 규칙:

| fact_type | 생성 규칙(요약) | 출처 테이블 |
|---|---|---|
| current_vitals | window 마지막 값 + window 내 min/max | CHARTEVENTS |
| vital_trend | MAP<65 지속, SpO2<90, HR>130 등 임계 돌파 구간 | CHARTEVENTS |
| lab_latest / lab_abnormal | 최신값, 참고범위(REF_RANGE) 밖, FLAG | LABEVENTS |
| lab_trend | ΔCr≥0.3/48h, lactate 상승·청소율, Hb 하강 ≥2 | LABEVENTS |
| critical_value | 기관 critical value 표 기준(K≥6.0, Na≤120, glucose≤50 등) | LABEVENTS |
| med_new / med_stopped / med_changed | window 내 시작·중단·용량 변경 | PRESCRIPTIONS, EMAR |
| infusion_status | 승압제·진정제·인슐린 drip 현재 속도와 변화 | INPUTEVENTS |
| vent_status | 호흡 지원 방식과 설정 변화 | CHARTEVENTS, PROCEDUREEVENTS |
| procedure | window 내 시술(삽관, 중심정맥관, CRRT 등) | PROCEDUREEVENTS, PROCEDURES_ICD |
| micro_result / micro_pending | 배양 결과(균·감수성), 결과 미보고 건 | MICROBIOLOGYEVENTS |
| imaging_done | 영상검사 시행 사실(판독문 없음을 명시) | TESTEVENTS, D_TESTITEMS |
| io_balance | 수분 균형, 요량(mL/kg/h) | INPUTEVENTS, OUTPUTEVENTS |
| transfer / service | 병동 이동, 진료과 변경 | TRANSFERS, SERVICES |
| diagnosis | 입원 진단(ICD) — 청구 코드임을 명시 | DIAGNOSES_ICD |
| ed_context | 내원 경로, triage 소견 | EDSTAY, TRIAGE, MEDRECON |
| nursing_log | 간호 활동 기록 | DATETIMEEVENTS |

- 각 fact에 **salience** 를 부여함(규칙 기반, 3단계):
  - critical(3): critical_value, 승압제 시작/증량, 삽관·CRRT 등 시술, 혈액배양 양성, 전동, 사망.
  - major(2): med_new/stopped, lab_abnormal(참고범위 밖), lab_trend, vent 설정 변경, micro_pending.
  - minor(1): 정상 범위 최신값, 일상 간호 활동.
- `must_include = salience >= 2` 를 기본값으로 함. 이 집합이 B2(누락 검증)의 기준이 됨.

### 2-5. Fact Sheet 스키마(요약)

```json
{
  "stay_id": 123, "hadm_id": 456, "t0": "2023-03-01T07:00:00", "window_hours": 12,
  "patient": {"age": 67, "sex": "F", "los_icu_days": 3.2, "careunit": "MICU"},
  "facts": [
    {"fact_id": "F001", "fact_type": "lab_trend", "concept": "creatinine",
     "value": {"from": 1.1, "to": 1.9, "unit": "mg/dL", "delta_hours": 36},
     "time": "2023-03-01T05:30:00", "salience": 3, "must_include": true,
     "provenance": [{"table": "LABEVENTS", "row_id": 9981}, {"table": "LABEVENTS", "row_id": 9712}],
     "text_ko": "크레아티닌 1.1→1.9 mg/dL (36시간)", "text_en": "Creatinine 1.1→1.9 mg/dL over 36 h"}
  ]
}
```

---

## 3. 모델 A — 요약지/인수인계지 생성

### 3-1. 산출물 형식
- 1순위: ICU 인수인계지, I-PASS 구조(Illness severity / Patient summary / Action list / Situation awareness & contingency / Synthesis). 간호 인수인계에는 SBAR 매핑도 제공함(동일 Fact Sheet, 다른 템플릿).
- 2순위(MIMIC-IV 비교용): 입원경과요약(BHC 형식).
- 출력 언어: 한국어 기본, 영문 병기 가능(fact에 text_ko/text_en을 둠). 국제 멘토 심사 고려 시 영문 템플릿으로 전환함.

### 3-2. 단계
- A1 Timeline Builder, A2 Fact Extractor: 2장 공통 계층 그대로 사용.
- A3 Content Planner(결정론): 섹션별로 포함할 fact를 선택함.
  - must_include 전부 포함. 나머지는 salience·최신성 순으로 섹션별 상한(예: 섹션당 6개) 내에서 선택.
  - Illness severity는 규칙으로 산출함(승압제 사용·인공호흡·lactate≥4·GCS≤8 중 하나면 "unstable", 등). LLM이 중증도를 판단하지 않음.
  - Action list / contingency는 "미보고 배양", "다음 검사 예정", "승압제 weaning 중" 등 fact_type에서 규칙으로 도출함. 근거 fact가 없는 행동 제안은 생성하지 않음.
- A4 Surface Realizer(교체 가능):
  - 경로 1(LLM 가용): Fact Sheet(선택된 fact만) → 섹션별 문장 생성. 제약: (a) 입력 fact 외 정보 추가 금지, (b) 모든 문장 끝에 인용 `[F001]`, (c) 수치·단위·시간은 fact 값을 그대로 복사, (d) 해석 표현("악화 추세")은 lab_trend 등 해석 fact가 있을 때만 허용, (e) 섹션에 fact가 없으면 "해당 사항 없음".
  - 경로 2(LLM 불가): Jinja2 템플릿 렌더러. fact_type별 문장 템플릿으로 결정론 생성. 할루시네이션 0을 보장하는 참조 기준선(reference arm)으로도 사용함.
  - 권장 운용: 수치 문장은 템플릿, 섹션 첫 줄의 종합 서술(Synthesis)만 LLM. 할루시네이션 노출 면적을 최소화함.
- A5 Verify-Repair 루프(에이전트 요소): 초안 → 모델 B → 불합격 claim이 있는 섹션만 피드백과 함께 재생성(최대 2회). 2회 후에도 불합격이면 해당 문장을 템플릿 문장으로 대체하고 리포트에 기록함.

### 3-3. 모델 선택(환경별)
| 환경 | 선택 | 비고 |
|---|---|---|
| 폐쇄환경, GPU 있음, 외부 API 없음 | 오픈 모델 로컬 서빙(vLLM/Ollama). 한국어 출력이면 EXAONE 3.5/4.0, Qwen3 8B~32B 중 설치 가능한 것 | 미세조정 없음. 프롬프트 + 스키마 제약 + 인용 강제 |
| 폐쇄환경, GPU 없음 | 경로 2 템플릿 전용 + 소형 모델(1~4B, CPU)로 Synthesis 1문장만 시도 | 데모 품질은 템플릿으로 확보 |
| MIMIC-IV 사용(외부 API 허용) | PhysioNet 허용 서비스만 사용: Azure OpenAI(사람 검토 opt-out), Amazon Bedrock, Vertex AI Gemini, Anthropic Claude | ChatGPT·OpenAI 공개 API 금지 |
- 미세조정(LoRA)은 대회 후 과제로 분류함. 24시간 내에는 학습 데이터(정답 기록지)가 없고 검증 시간도 부족함.

### 3-4. 프롬프트 골격(A4 경로 1)
```
[System] 당신은 ICU 인수인계지를 작성하는 보조자입니다. 아래 JSON의 facts만 사용합니다.
규칙: 1) facts에 없는 정보·추정·권고를 쓰지 않음 2) 모든 문장 끝에 사용한 fact_id를 [F###] 형식으로 인용
3) 수치·단위·시각은 fact의 값을 그대로 복사 4) 섹션에 해당 fact가 없으면 "해당 사항 없음"으로 기재
5) 출력은 지정된 섹션 제목과 순서를 따름
[User] <section_plan: 섹션별 fact_id 목록> <facts: 선택된 fact JSON>
```

### 3-5. 예상 실패 모드(발표의 "그림자" 자료로 수집)
- 인용은 달았으나 fact 값과 다른 수치(인용 불일치).
- 두 fact를 합쳐 인과관계를 창작("승압제 증량으로 MAP 회복").
- 한글 상품명을 성분명으로 잘못 변환.
- 섹션 비어 있음을 메우려는 일반론 문장("지속적인 모니터링 필요").

---

## 4. 모델 B — 정보누락·할루시네이션 검증

### 4-1. B1 할루시네이션(정밀도) 검증
1. Claim 분해: 기록지 문장 → 원자 claim 목록. 각 claim: `{claim_id, sentence_id, cited_fact_ids, claim_type, entity, value, unit, time_ref, polarity, text}`. 분해는 LLM이 가용하면 LLM, 아니면 문장 단위 + 정규식(수치·단위·약제명 추출)으로 수행함.
2. 근거 조회(2단계):
   - 1단계 인용 검사: cited_fact_ids가 있으면 해당 fact와 claim의 entity·value·time을 대조함(단위 변환 허용, 수치 오차 허용범위는 소수점 표기 수준, 시간은 window 내). 일치하면 `Supported(cited)`.
   - 2단계 전역 조회: 인용이 없거나 불일치면 Fact Sheet 전체와 Timeline 원본 행에서 entity 기준으로 후보를 조회함. 수치 claim은 결정론 매칭으로 판정함.
3. 서술형 claim 판정(LLM judge, VeriFact 방식): "혈역학적으로 불안정", "감염 호전"처럼 수치가 아닌 claim은 후보 fact 상위 k개를 근거로 LLM이 판정함. 라벨은 VeriFact와 동일하게 3분류함.
   - Supported: 근거가 claim을 뒷받침함.
   - Not Supported: 근거와 모순되거나 부분만 일치함(가장 위험, "contradiction"으로 별도 집계).
   - Not Addressed: 기록에 근거가 없음(검증 불가. 임상의 기록에서는 구조화 데이터에 없는 정보일 수 있으므로 "할루시네이션"으로 단정하지 않고 "unverifiable"로 보고함).
4. 지표: 할루시네이션율 = (Not Supported + Not Addressed) / 전체 claim. 모순율 = Not Supported / 전체 claim. 인용 불일치율 = 인용 있으나 1단계 불일치 / 인용 claim.

### 4-2. B2 정보누락(재현율) 검증
1. 기준 집합 M: Fact Sheet에서 `must_include = true` 인 fact 전부(2-4 salience 규칙). 규칙 근거는 인수인계 표준(I-PASS 요소: 중증도, 활성 문제, 최근 변화, 미결 사항, 비상 대책)과 critical value 관리 원칙임.
2. 커버리지 판정: M의 각 fact m에 대해 "기록지가 m을 전달하는가"를 판정함.
   - 결정론: m의 entity와 value가 어떤 claim에 포함되면 Covered.
   - LLM judge: 결정론 미매칭 시 "기록지 전체 + m" 을 입력으로 Covered / Partial / Missing 판정.
3. 지표: 누락율 = |Missing| / |M|. 가중 누락 점수 = Σ salience(Missing) / Σ salience(M). critical 누락 건수는 별도 보고함.
4. 확장(P2, 시간 있을 때): MED-OMIT 방식으로 "해당 fact가 빠지면 감별진단·처치 판단이 바뀌는가"를 LLM이 평가해 salience를 보정함. 대회 발표에서는 규칙 기반 M을 주 결과로, LLM 보정은 민감도 분석으로 제시함.

### 4-3. B3 검증기 자체 검증(필수)
1. 합성 오류 주입 세트(자동 생성, 1일차 밤):
   - 할루시네이션 주입: 수치 변경(K 3.2→5.2), 약제 치환(norepinephrine→vasopressin), 시간 이동, 부정 반전(양성↔음성), 사건 창작(없는 시술 추가), 인과 창작.
   - 누락 주입: must_include fact를 임의로 1~3개 삭제한 기록지 생성.
   - 각 변형에 정답 라벨이 자동 부여되므로 B1·B2의 precision/recall을 수백 건 규모로 측정 가능함.
2. 임상의 주석(2일차 오전, 팀 내 임상 인력 2명):
   - claim 100~200개를 Supported / Not Supported / Not Addressed로 주석, 누락 판정 50건 주석. 검증기와의 일치율, Cohen's κ, 주석자 간 일치율을 보고함(VeriFact는 임상의 간 최고 일치 88.5%, 시스템-임상의 92.7%를 보고함. 본 결과는 그 수치와 나란히 제시함).
3. 검증기의 실패 유형 수집: 개념 정규화 실패(한/영 라벨), 서술형 claim의 judge 오판, 시간창 경계 문제. 발표의 "그림자" 섹션에 사용함.

### 4-4. 기존 기록지(임상의 작성)에 적용하는 방법
- MIMIC-IV-Note 퇴원요약의 Brief Hospital Course 섹션을 입력으로, MIMIC-IV 구조화 테이블에서 같은 방식으로 Fact Sheet를 만들고 B2 누락율을 계산함. B1은 "unverifiable" 비율로 보고하고 할루시네이션으로 표현하지 않음.
- 발표 문구는 "임상의 기록이 나쁘다"가 아니라 "시간 압박 아래 작성된 인수인계·요약 문서에서 구조화 필수 항목의 N%가 빠지며, 동일한 검증기가 사람·AI 기록을 같은 기준으로 감사할 수 있다"로 함.
- K-MIMIC에서는 임상의 기록지가 없으므로 비교군을 (i) naive 에이전트 기록지(원시 테이블 텍스트 덤프 → LLM 요약, 접지 없음), (ii) 템플릿 기록지로 구성함.
- 1일차에 폐쇄환경에서 2026 공개판 테이블 목록을 반드시 재확인함(노트 테이블이 추가되었을 가능성).

---

## 5. 모델 C — CDSS

### 5-1. 원칙
- 판단은 규칙 엔진(결정론)이 하고 LLM은 설명문만 작성함. LLM이 의사결정 경로에 들어가지 않음.
- 입력은 검증을 통과한 Fact Sheet임. 알림은 `{alert_id, rule_id, severity, trigger_fact_ids, rule_text, reference, suggested_action}` 을 가짐.
- LLM이 쓴 설명문은 B1로 재검증함(알림 전제가 fact에 있는지).
- 런타임 외부 호출 없음. 지식베이스는 CSV/JSON으로 사전 적재함.

### 5-2. 규칙 패밀리(24시간 내 구현 가능 범위)
| 규칙군 | 판정식 | 입력 | 지식베이스 |
|---|---|---|---|
| 병용금기·연령금기·노인주의 | 활성 처방 쌍/환자 나이 대조 | PRESCRIPTIONS(EDI/ATC), PATIENTS | 식약처 DUR 품목정보·성분정보(data.go.kr 15059486, 15056780) 사전 다운로드. 불가 시 고위험 20쌍 수작업 목록 |
| 신기능 기반 용량 경고 | eGFR(CKD-EPI 2021) 계산 → 신배설 약제 용량 대조 | LABEVENTS(Cr), PATIENTS, PRESCRIPTIONS | 약제별 eGFR 구간 용량표(수작업 10~15개) |
| AKI 탐지(KDIGO) | Cr ↑≥0.3 mg/dL/48h 또는 ≥1.5×기저/7일 또는 요량 <0.5 mL/kg/h 6h | LABEVENTS, OUTPUTEVENTS, INPUTEVENTS(체중) | KDIGO 기준 |
| 패혈증 선별 | SOFA 증가 ≥2, qSOFA, 승압제+lactate>2 | CHARTEVENTS, LABEVENTS, INPUTEVENTS | Sepsis-3 |
| 항생제–배양 불일치 | 현재 항생제가 분리균 감수성 'R' | MICROBIOLOGYEVENTS, PRESCRIPTIONS | ATC J01 매핑 |
| 모니터링 공백(TDM) | vancomycin 72h 내 혈중농도 없음, heparin drip에 aPTT/anti-Xa 없음, 인슐린 drip에 혈당 측정 간격 >2h | PRESCRIPTIONS, INPUTEVENTS, LABEVENTS | 기관 프로토콜 상식 수준 |
| Critical value 미대응 | critical_value 발생 후 N시간 내 재검·관련 처방 없음 | LABEVENTS, PRESCRIPTIONS | critical value 표 |
| 전해질·QT | K<3.0 또는 Mg 저하 + QT 연장 약제 | LABEVENTS, PRESCRIPTIONS | QT 연장 약제 목록 |
| (P2) 악화 위험 예측 | 24h 사망·재입실 GBM 모델 | 시계열 피처 | 교정 곡선 없이는 발표에 넣지 않음 |

- 우선순위: AKI, 패혈증, 병용금기, 신기능 용량, TDM 공백 5종을 P1으로 구현함. 나머지는 시간 여유 시.

### 5-3. 설명문 생성
- 입력: 알림 구조체 + trigger fact. 출력: 1~2문장, fact_id 인용 필수. 템플릿 대체 가능.

### 5-4. CDSS의 "그림자" 분석
- 생성 모델 오류가 CDSS로 전파되는 사례를 의도적으로 구성함(예: 창작된 승압제 문장이 패혈증 알림을 유발). Fact Sheet 기반 설계에서는 기록지 문장이 아닌 fact를 입력으로 쓰므로 전파가 차단됨을 대비 실험으로 보임(기록지 텍스트를 CDSS 입력으로 쓰는 naive 구성과 비교).
- 알림 피로: 환자당 알림 수 분포를 보고함.

---

## 6. 에이전트 오케스트레이션과 감사 로그
- 상태기계: Plan → Extract → Draft → Verify → Repair → CDSS → Report. 구현은 순수 Python 상태기계로 충분함(LangGraph 등 프레임워크는 설치 가능할 때만).
- 도구: `get_timeline`, `build_fact_sheet`, `render_section`, `verify_claims`, `check_coverage`, `run_cdss_rules`. 도구 수를 작게 유지함.
- 감사 로그: 모든 도구 호출의 입력 해시·출력·소요시간·모델명·프롬프트 버전을 JSONL로 기록함. "에이전트가 대신 행동할 때 누가 책임지는가"에 대한 기술적 답(추적 가능성)으로 발표함.

---

## 7. 평가 설계

### 7-1. 비교군
| arm | 구성 | 목적 |
|---|---|---|
| A0 naive agent | 원시 테이블 텍스트 덤프 → LLM 자유 요약 | 접지 없는 에이전트의 실패 노출 |
| A1 grounded | Fact Sheet → LLM, 검증 루프 없음 | 접지 효과 |
| A2 grounded + repair | A1 + Verify-Repair | 검증 루프 효과 |
| A3 template | 템플릿 전용 | 할루시네이션 0 참조선 |
| H clinician | MIMIC-IV-Note BHC (가용 시) | 사람 기록의 누락율 |

### 7-2. 지표
- 기록지: 할루시네이션율, 모순율, 인용 불일치율, 누락율, 가중 누락 점수, critical 누락 건수, 길이, 생성 시간.
- 검증기: 합성 세트 precision/recall(오류 유형별), 임상의 주석 일치율·κ.
- CDSS: 환자당 알림 수, 임상의 검토 표본(50건)에서 적절 알림 비율.

### 7-3. 표본
- K-MIMIC: ICU stay 100~300건(무작위, 재현 가능한 seed). 각 stay에서 t0를 1~2개 선택.
- MIMIC-IV(가용 시): 퇴원요약 100건.

---

## 8. 24시간 실행 계획

### 8-1. 우선순위
- P0(심사에 반드시 필요): 데이터 계층, Fact Sheet, 템플릿 렌더러, B1·B2 결정론 경로, 합성 오류 세트 평가, A0 vs A2 비교, 발표 자료.
- P1: LLM 렌더러, LLM judge, Verify-Repair 루프, CDSS 규칙 5종, 임상의 주석.
- P2: MIMIC-IV-Note 비교, MED-OMIT식 salience 보정, 악화 예측 모델.

### 8-2. 시간표(안)
| 시간 | 작업 |
|---|---|
| 10/16 ~19:00 | 환경 확인(GPU·LLM·인터넷·pip·2026 테이블 목록), 코호트 추출, Fact Sheet 스키마 확정, 개념 사전 초안 |
| 19:00~23:00 | Timeline Builder, Fact Extractor, salience 규칙, 템플릿 렌더러(I-PASS) |
| 23:00~03:00 | B1(인용 검사 + 전역 조회), LLM judge(가용 시), claim 분해기 |
| 03:00~06:00 | B2 커버리지, 합성 오류 주입기, 검증기 평가 실행 |
| 06:00~09:00 | CDSS 규칙 5종, DUR 사전 적재, 임상의 주석 세트 준비 |
| 09:00~12:00 | 4 arm 실험 실행, 표·그림, 실패 사례(그림자) 선별, 감사 로그 시연 준비 |
| 12:00~14:30 | 발표 자료, 데모 시나리오 1건 리허설 |
| 14:30~15:00 | 최종 점검 |

### 8-3. 6인 역할(안)
- 데이터 계층 2명(임상 1 + 데이터 1): 개념 사전, Fact Extractor, salience 규칙.
- 생성 1명: 템플릿·LLM 렌더러, repair 루프.
- 검증 1명: B1·B2·합성 세트.
- CDSS 1명(임상 배경): 규칙·지식베이스·알림 검토.
- 평가·발표 1명: 실험 러너, 그림, 슬라이드, 데모.
- 수면은 2교대로 확보함(03:00~06:00 / 06:00~09:00).

---

## 9. 리스크와 대응
| 리스크 | 대응 |
|---|---|
| 폐쇄환경에 LLM·GPU 없음 | 템플릿 경로로 전 파이프라인 동작. LLM 의존 부분은 "P1"로 표기 |
| K-MIMIC에 임상의 기록지 없음 | naive 에이전트 비교군으로 대체. MIMIC-IV-Note는 자격 보유자 있을 때만 |
| 검증기 순환논증 지적 | 합성 세트 + 임상의 주석을 먼저 제시. 검증기 실패 사례도 공개 |
| 한글 상품명·라벨 정규화 실패 | UNMAPPED 보존 정책, 사전 미매핑 비율을 지표로 보고 |
| 과장된 주장 | "임상의보다 낫다" 표현 금지. 측정한 지표만, 표본 크기와 함께 제시 |
| 시간 초과 | P0만으로도 발표 가능하도록 23:00까지 A3 + B 결정론 경로를 먼저 완성 |

---

## 10. 1일차에 환경에서 확인할 사항
1. 폐쇄환경 사양: GPU 유무·메모리, 설치된 LLM 서빙(vLLM/Ollama/HF 모델 캐시), 외부 API 허용 여부, pip 설치 가능 여부, 파일 반입 가능 여부(DUR CSV, 사전 파일).
2. K-MIMIC 2026 공개판 테이블 목록과 텍스트 컬럼 존재 여부(노트성 테이블 추가 여부).
3. PRESCRIPTIONS의 DRUG 표기 언어와 EDI/ATC 채움 비율, D_LABITEMS·D_ITEMS 라벨 언어 비율.
4. MIMIC-IV-Note 접근 가능 팀원과 허용 LLM 경로.
5. 산출물 언어(한국어/영어) 결정.

---

## 부록 A. 스키마

### Claim
```json
{"claim_id": "C012", "sentence_id": "S4", "cited_fact_ids": ["F001"],
 "claim_type": "lab_trend", "entity": "creatinine",
 "value": {"from": 1.1, "to": 1.9, "unit": "mg/dL"}, "time_ref": "last_36h",
 "polarity": "positive", "text": "크레아티닌이 36시간 동안 1.1에서 1.9로 상승함"}
```

### Verdict
```json
{"claim_id": "C012", "label": "Supported", "method": "cited_match",
 "evidence_fact_ids": ["F001"], "evidence_rows": [{"table": "LABEVENTS", "row_id": 9981}],
 "judge_model": null, "reason": "entity/value/time match within tolerance"}
```

### Coverage
```json
{"fact_id": "F017", "must_include": true, "salience": 3,
 "status": "Missing", "method": "deterministic+judge", "matched_claim_ids": []}
```

### Alert
```json
{"alert_id": "AL03", "rule_id": "AKI_KDIGO_STAGE1", "severity": "high",
 "trigger_fact_ids": ["F001"], "rule_text": "Cr rise >= 0.3 mg/dL within 48h",
 "reference": "KDIGO 2012", "suggested_action": "신기능 재평가, 신독성 약제 검토",
 "explanation": "크레아티닌이 36시간 동안 0.8 mg/dL 상승함 [F001]",
 "explanation_verdict": "Supported"}
```

## 부록 B. LLM judge 프롬프트 골격(B1 서술형 claim)
```
[System] 당신은 임상 기록 사실 검증자입니다. 주어진 claim이 evidence로 뒷받침되는지 판정합니다.
라벨: Supported(evidence가 claim을 뒷받침) / Not Supported(evidence와 모순 또는 일부만 일치) / Not Addressed(evidence에 관련 정보 없음)
evidence에 없는 의학 지식으로 추론하지 않습니다. JSON으로만 답합니다: {"label": ..., "reason": ...}
[User] claim: <text> / evidence: <상위 k개 fact의 text + 값 + 시각>
```

## 부록 C. 참고 자료
- 대회 일정·주제: https://datathon.kr/ , https://datathon.kr/registration/entry/
- K-MIMIC 논문: https://ekja.org/journal/view.php?doi=10.4097%2Fkja.25752
- K-MIMIC 테이블 문서: https://sites.google.com/view/k-mimic/modules
- K-MIMIC 간호 데이터 SNOMED CT 매핑(Stud Health Technol Inform 2025;329:48-52): https://pubmed.ncbi.nlm.nih.gov/40775817/
- VeriFact(RAG + LLM judge, Supported/Not Supported/Not Addressed): https://arxiv.org/abs/2501.16672
- MED-OMIT(누락 평가, 감별진단 영향 기반 중요도): https://arxiv.org/abs/2311.08303
- 식약처 DUR 품목정보 OpenAPI: https://www.data.go.kr/data/15059486/openapi.do , 성분정보: https://www.data.go.kr/data/15056780/openapi.do
- PhysioNet LLM 사용 정책: https://physionet.org/news/post/gpt-responsible-use
