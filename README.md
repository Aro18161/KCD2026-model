# KCD 2026 — ICU → 병동 전동기록지 생성·검증·CDSS

K-MIMIC 데이터로 ICU에서 병동으로 전동하는 환자의 전동기록지를 만들고(A), 기록지의 할루시네이션·누락을 검증하고(B), 전동 안전성 CDSS 알림(C)을 붙이는 파이프라인임. 설계 근거는 `docs/model_architecture.md` 참고.

## 1. 파일

| 파일 | 역할 |
|---|---|
| `kcd/probe_kmimic.py` | 선행 코드. 데이터 디렉터리의 파일·컬럼·날짜 형식·사전(label) 구조만 출력. 환자 행은 출력하지 않음 |
| `kcd/kcd_transfer.py` | 본체. 단일 파일, pandas만 필요. 생성(A)·검증(B1/B2/B3)·CDSS(C)·감사 로그 |
| `kcd/selftest.py` | 합성 샘플로 17개 항목 자체 점검 |
| `kcd/make_synthetic_sample.py` | K-MIMIC 25개 테이블 컬럼 구조의 소형 합성 데이터(시나리오 3건) 생성 |

## 2. 분석환경에서 실행 순서

```bash
python probe_kmimic.py datasets/kmimic-emr-001/268 --out kmimic_probe
python kcd_transfer.py list    --data datasets/kmimic-emr-001/268
python kcd_transfer.py run     --data datasets/kmimic-emr-001/268 --stay auto --out out
python kcd_transfer.py batch   --data datasets/kmimic-emr-001/268 --n 50 --out out
python kcd_transfer.py perturb --data datasets/kmimic-emr-001/268 --n 50 --out out
```

Jupyter에서는 `%run kcd_transfer.py run --data ... --stay auto --out out` 으로 실행함. 셀에 코드 전체를 붙여넣은 경우 자동 실행되지 않으며 마지막 셀에서 `kcd_main("run --data datasets/kmimic-emr-001/268 --stay auto --out out")` 을 호출함.

시간 컬럼이 날짜가 아닌 숫자(입원 후 경과일 등)로 되어 있으면 `--time-unit D`(일)·`h`·`min` 중 하나를 지정함. 판단은 `probe_kmimic.py` 출력의 `DATE[...]` 표시 여부로 함.

## 3. LLM 사용(선택)

LLM이 없으면 템플릿 렌더러로 전 과정이 동작함. 분석환경에 OpenAI 호환 서버(vLLM, Ollama 등)가 있으면 다음과 같이 켬.

```bash
export KCD_LLM_BASE_URL=http://localhost:8000/v1
export KCD_LLM_MODEL=모델명
python kcd_transfer.py run  --data DIR --stay auto --mode repair --out out
python kcd_transfer.py arms --data DIR --n 10 --out out
```

- `--mode grounded`: Fact Sheet → LLM 문장화(인용 강제), 검증 루프 없음(A1)
- `--mode repair`: A1 + 검증 불합격 문장 재생성(최대 2회), 실패 시 템플릿 문장으로 대체, 누락 필수 fact 보완(A2)
- `--mode naive`: 원시 테이블 덤프 → LLM 자유 작성, 접지 없음(A0 비교군)
- `arms`: 네 방식(A0·A1·A2·A3 템플릿)을 같은 stay에 적용해 지표 비교

## 4. 출력(stay별 폴더)

- `note_<mode>.md`: 전동기록지. 문장마다 근거 fact 인용, 검증 불합격 문장에 ⚠️ 표시, 끝에 검증 요약
- `factsheet.json`: 사실 단위 목록(출처 테이블·행 ID, 중요도, 필수 여부)
- `verify_<mode>.json`: 문장별 판정(Supported/Contradicted/NotAddressed/Unverifiable)과 필수 fact 커버리지
- `alerts.json`: CDSS 알림(rule_id, 심각도, 근거 fact, 참고문헌)
- `audit.jsonl`: 단계별 입력 해시·출력·소요 시간·모델명·프롬프트 버전
- 상위 폴더: `summary_template.csv/md`, `cdss_rule_counts.json`, `perturb_eval.csv`, `arms.csv`

실제 환자 출력물은 분석환경 밖으로 반출하지 않음. 외부 공유는 summary·perturb_eval 집계 지표만 사용함.

## 5. 로컬 검증 결과 (Synthetic K-MIMIC, SYN-ICU v2.0.0)

SYN-ICU는 SNUH K-MIMIC 형식의 GAN 합성 공개 데이터(CC-BY-4.0, doi:10.23421/2025.0723, KHDP 제공)임. 실제 K-MIMIC과 같은 ITEMID 형식(`001C_1021_25105` 등), 한글 라벨, KCD8 진단코드, EMAR 기반 승압제 기록을 가짐.

- 전동 대상 stay 1,143건 중 200건 템플릿 생성: 문장 평균 27.8개, 필수 fact 평균 20.6개, 검증기 오판 0건
- B3 합성 오류 주입(200 stay × 유형별 2회): `docs/syn_perturb_eval.csv`

| 주입 유형 | 탐지율 |
|---|---|
| 수치 변경 | 99.7% |
| 약물 치환 | 100% |
| 시각 변경 | 99.8% |
| 투여 상태 반전 | 100% |
| 사건 창작 | 100% |
| 필수 fact 삭제 | 86.3% |
| 원본 문장 오탐 | 0% |

삭제 미탐지 사례는 같은 값이 다른 문장에 중복 기재된 경우임(예: Hb 최저값과 최신값이 같음, 인공호흡 경과와 전동 시점 호흡 상태).

## 6. 한계

- 검증기 B1의 결정론 경로는 수치·시각·코드·약물명·사건 어휘(`EVENT_VOCAB`)를 대조함. 어휘 밖의 서술형 창작은 `Unverifiable`로 분류되며 LLM judge가 있을 때만 판정됨.
- 템플릿 기록지의 할루시네이션 0은 구조상 당연한 결과이며, 검증기 성능 근거는 B3 결과임.
- 병용금기 목록은 예시 4쌍임. 식약처 DUR 병용금기 CSV(성분A, 성분B, 사유)를 `--dur-csv`로 지정해 교체함.
- SWIFT 점수의 입실 경로는 TRANSFERS로 근사함. NEWS2의 의식 항목은 GCS<15를 CVPU로 근사함.
- 위 성능 수치는 합성 데이터 기준이며 실제 K-MIMIC에서 다시 측정해야 함.
