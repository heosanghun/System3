# 실측 결과 로그 (Measured Results Log)

모든 수치는 본 저장소 코드를 실제 실행해 측정한 값입니다. 논문 목표치와의 차이는 숨기지 않고 그대로 기록합니다.

## Run 001 — 2026-07-22, CPU, d=256 (축소 규격)

- 설정: 30 domains × 500 samples, 1 seed, epochs=2, lr=1e-4, d=256, d_wide=1024, batch=128
- 환경: CPU (torch 2.12.0+cpu) — VRAM 측정 불가
- 명령: `python main.py --domains 30 --samples 500 --seeds 1 --epochs 2 --d 256 --d-wide 1024 --batch-size 128`

| Architecture | Final BWT | Final FWT | Experts |
|---|---:|---:|---:|
| System 2.5 (d=256) | +0.4% | +1.5% | 1 |
| Wide Sys 2.5 (d=1024) | +2.2% | +0.3% | 1 |
| System 3 | -8.2% | +0.7% | 3 |

**해석 (중요)**: 학습 loss가 2.30(=ln 10, 10-클래스 무작위 수준)에서 거의 내려가지 않음 → 이 설정에서는 모델이 도메인을 사실상 학습하지 못하므로 BWT/FWT가 무의미한 잡음 수준. 논문의 현상(학습 후 망각)을 관찰하려면 도메인별 학습이 실제로 일어나야 함.

## 진단 001 — 단일 도메인 학습 가능성 (d=256, System 2.5, 20 epochs)

| lr | train acc | test acc |
|---|---:|---:|
| 1e-4 | 40.5% | 36.0% |
| **1e-3** | **64.7%** | **38.0%** |
| 5e-3 | 40.3% | 22.0% |

→ 태스크는 학습 가능(무작위 10% 대비). 벤치마크 기본 설정을 epochs≥10, lr=1e-3으로 상향 필요. 관찰된 추가 문제: R2P가 30개 도메인에서 전문가를 3개만 스폰 (τ_spawn=0.8 기준으로 novelty가 충분히 감지되지 않음 — 라우터 projection이 무작위 초기화 상태에서 도메인 간 코사인 유사도가 높게 나옴). 라우터 표현 학습 또는 τ_spawn 재조정 필요.

## Run 002 — 2026-07-22, CPU, d=256, epochs=10, lr=1e-3, 2 seeds

- 명령: `python main.py --domains 30 --samples 500 --seeds 2 --epochs 10 --lr 1e-3 --d 256 --d-wide 1024 --batch-size 128`

| Architecture | Final BWT | Final FWT | Experts |
|---|---:|---:|---:|
| System 2.5 (d=256) | -20.6% ± 0.8% | +2.9% ± 1.0% | 1 |
| Wide Sys 2.5 (d=1024) | -20.2% ± 0.4% | -0.3% ± 3.2% | 1 |
| System 3 | -16.4% ± 1.9% | +1.9% ± 0.6% | ~703 (폭주) |

Welch's t-test (System 3 vs Dense): BWT t=2.91, p=0.157 (2시드라 유의성 판단 불가)

**해석**:
1. ✅ 학습이 실제로 일어나면서(loss 0.4~0.9) **파괴적 망각 현상이 처음으로 실측 재현됨** — Dense BWT -20.6%는 논문의 Capacity Wall 서사(-23.4% @ d=768)와 방향·규모 일치.
2. ✅ System 3가 베이스라인 대비 BWT 우위 (방향 일치).
3. ❌ **R2P 스폰 폭주**: 라우터 projection이 task loss로 학습되며 임베딩이 이동 → 저장된 프로토타입과 유사도 하락 → 매 배치 novelty 오탐 → 전문가 700+개 스폰. 논문의 16개와 괴리.
4. ❌ Wide 모델이 Dense와 비슷한 BWT — 논문은 Wide가 덜 망각(-14.2% vs -23.4%)한다고 주장. 후속 확인 필요.

**조치**: 도메인당 스폰 예산(기본 1개) + 전문가 수 상한(기본 32) 구현 → Run 003.

## 다음 실험 (진행 중)

- Run 003: Run 002와 동일 설정 + R2P 스폰 제어 (spawns_per_domain=1, max_experts=32)
