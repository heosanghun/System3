# System 3: 평생 지속 추론을 위한 희소 암묵적 혼합 전문가 프레임워크 (Sparse Implicit Mixtures)

본 저장소는 ICLR 2027 제출 규격에 맞춰 작성 중인 **System 3** 프레임워크의 PyTorch 구현체입니다. 단일 가중치 암묵적 모델(System 2.5)의 용량 장벽(Capacity Wall)을 Sparse Implicit MoE로 극복하는 것을 목표로 합니다.

---

## ⚠️ 구현 상태 (정직성 고지)

- 본 저장소의 모든 성능 지표는 `main.py` 실행 시 **실측**됩니다. 이전 버전에 존재하던 지표 하드코딩(논문 목표치 덮어쓰기), 시뮬레이션된 수렴 곡선, 보정된 전문가 수는 **전부 제거**되었습니다.
- DEQ 코어에 그래디언트가 흐르지 않던 역전파 버그(커스텀 autograd Function에 파라미터 미등록)를 표준 IFT adjoint(backward hook) 방식으로 수정했습니다. 이제 System 2.5 / Wide / System 3 모두 전이 가중치·라우터·헤드가 실제로 학습됩니다.
- 논문 Table 1의 수치는 아직 본 코드로 재현되지 않았습니다. 아래 표는 **논문의 목표(claimed) 수치**이며, 실측 결과는 GPU 벤치마크 완료 후 갱신됩니다.
- 현재 데이터는 합성(synthetic) 30-도메인 스트림입니다. 논문이 명시한 실제 벤치마크(FinQA·GLUE·LegalBench·ImageNet 패치 + 동결 RoBERTa/ViT 인코더)와의 정합은 진행 중입니다.

## 📊 논문 목표 수치 (Table 1 — 실측 재현 전)

| 평가 대상 아키텍처 | BWT (%) | FWT (%) | Peak VRAM (GB) | 전문가 수 |
| :--- | :---: | :---: | :---: | :---: |
| System 2.5 (d=768) | -23.4 ± 2.1 | +0.4 ± 0.2 | 16.3 | 1 |
| Wide Sys 2.5 (d=3072) | -14.2 ± 1.8 | +1.1 ± 0.4 | 19.8 | 1 |
| LoraMoE (16 전문가, 명시적 MoE) | -2.1 ± 0.9 | +3.2 ± 0.5 | 23.5 (OOM 근접) | 16 (고정) |
| **System 3 (제안 방법)** | **-1.8 ± 0.6** | **+6.7 ± 0.8** | **18.2** | **16 (동적 스폰)** |

> 위 수치는 doc/System3.pdf 논문 원고의 목표값입니다. 본 코드의 실측 결과로 대체 예정입니다.

---

## 🔍 핵심 구성 요소

1. **Contractive Gated Mixture (CGM)**: 라우팅 가중치 $g_i(x)$가 오직 입력 $x$에만 종속 ($z$-독립) → 혼합 야코비안이 전문가 야코비안의 볼록 결합이 되어 전역 Banach 수축성 보장 (Proposition 1).
2. **Sparse FP-EWC**: 도메인 경계에서 라우터를 동결한 뒤(post-hoc consolidation) 라우팅된 샘플에 대해서만 조건부 FIM을 추정 — SLLN의 i.i.d. 조건 복원 (Theorem 1).
3. **R2P (Router-Recruitment Policy)**: 질의 novelty가 $\tau_{spawn}=0.8$ 미만이면 새 전문가를 동적 스폰. 로드밸런싱 + z-loss로 라우팅 붕괴 방지.
4. **C-FIRE**: power iteration 기반 스펙트럴 정규화로 각 전문가의 Lipschitz 상수를 $L \le 0.95$로 유지 (Algorithm 2).

---

## 🚀 실행 방법

```bash
# 논문 규격 전체 벤치마크 (30 도메인 × 500 샘플, 5 시드) — GPU 권장
python main.py

# 빠른 스모크 테스트 (CPU 가능)
python main.py --domains 4 --samples 100 --seeds 2 --epochs 1 --d 128 --d-wide 256
```

출력: 시드 평균 ± 표준편차의 BWT/FWT/VRAM/전문가 수 표, System 3 대 베이스라인 Welch's t-test, 실측 그래프(`evaluation_results.png`).

## 📂 파일 구조

- `data_generator.py`: 4개 페이즈(금융/법률/NLP/비전 유사 구조)를 대표하는 30개 합성 도메인 스트림 생성 (도메인당 500샘플 = 400 Train / 50 Val / 50 Test).
- `deq_solver.py`: Anderson/Picard 고정점 솔버 + IFT adjoint 역전파 (backward hook, O(1) 메모리).
- `router.py`: Contrastive Router, R2P 동적 스폰, C-FIRE 수축성 정규화, 로드밸런싱/z-loss.
- `models.py`: System 2.5 / Wide DEQ / System 3 (CGM Sparse MoE) 아키텍처.
- `trainer.py`: 도메인 연속 학습 루프 + FP-EWC / Sparse FP-EWC 매니저.
- `evaluate.py`: BWT/FWT/VRAM/수렴 반복수 실측 + Welch's t-test 구현.
- `main.py`: 다중 시드 벤치마크 실행 및 결과 집계/시각화.

## 🗺️ 로드맵 (ICLR 제출까지)

1. ~~DEQ 역전파 버그 수정 및 지표 하드코딩 제거~~ (완료)
2. GPU 서버에서 논문 규격(30 도메인 × 5 시드) 실측 벤치마크 실행
3. 실측값으로 논문 Table 1/Figure 2 및 [FILL] 플레이스홀더 확정
4. 실제 데이터셋 벤치마크(RoBERTa/ViT 인코더) 정합 또는 논문 실험 서술 수정
5. Static 16-Expert CGM ablation, d-sweep, λ-sweep, τ_spawn ablation 구현
