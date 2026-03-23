# TransformerCPI2.0 - Modified for Modern PyTorch

## 출처 (Original Source)

이 프로젝트는 [myzhengSIMM/transformerCPI2.0](https://github.com/myzhengSIMM/transformerCPI2.0)을 기반으로 합니다.

**Original Paper:**
> Chen, L., et al. (2023). "Sequence-based drug design as a concept in computational drug design". *Nature Communications*, 14, 4217. https://doi.org/10.1038/s41467-023-39856-w

**Original Repository:**
> https://github.com/myzhengSIMM/transformerCPI2.0

---

## 프로젝트 개요

TransformerCPI2.0은 단백질 서열과 화합물 SMILES를 입력받아 결합 친화도를 예측하는 딥러닝 모델입니다. 본 저장소는 원본 프로젝트를 최신 PyTorch 환경(특히 NVIDIA Blackwell GPU 아키텍처)에서 실행 가능하도록 수정한 버전입니다.

---

## 주요 수정 사항 (Modifications)

### 1. **최신 PyTorch 호환성 개선**
- **목적**: NVIDIA Blackwell 아키텍처 기반 GPU에서 동작 가능하도록 수정
- **변경사항**:
  - [predict.py](predict.py): `torch.load()` 호출 시 `weights_only=False` 파라미터 추가
  - PyTorch 최신 버전과의 호환성 확보

### 2. **Reverse Screening 기능 추가**
- **새 파일**: [reversebatch.py](reversebatch.py)
- **기능**:
  - FASTA 파일에 포함된 여러 단백질에 대해 하나의 화합물(SMILES)과의 결합 친화도를 일괄 예측
  - 단백질 서열 길이 제한 기능 (기본값: 8094 이상 스킵)
  - CUDA 에러 발생 시 자동 중단 또는 계속 진행 옵션
  - CSV 형식으로 결과 저장 (Index, Protein, SequenceLength, Score, Status, ErrorType, ErrorMessage)
  - 진행 상황 표시 (tqdm 사용)

**사용 예시:**
```bash
python reversebatch.py \
  --fasta proteins.fasta \
  --smiles "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" \
  --output results.csv \
  --device cuda \
  --skip-len-threshold 8094
```

### 3. **환경 설정 파일 대폭 수정**
- **파일**: [environment.yml](environment.yml)
- **주요 변경사항**:
  - Python 버전: 3.8 → **3.10**
  - PyTorch 설치 방식: conda → **pip** (최신 버전 자동 설치)
  - 환경 이름: `trnascpi` → `transcpi-gpu`
  - `numpy<2` 버전 제한 추가 (호환성)
  - 불필요한 패키지 제거 (pytorch-lightning, einops, axial-positional-embedding 등)
  - 필수 의존성 패키지 추가 (boto3, botocore, s3transfer, urllib3<2 등)

**환경 설정:**
```bash
conda env create -f environment.yml
conda activate transcpi-gpu
```

---

## 파일 다운로드

모델 및 임베딩 파일은 원본 저장소의 Releases에서 다운로드해야 합니다:
- **Bert.pkl**: 단백질 임베딩 지원
- **DrugRepurpose.pt**: 약물 재창출(Drug Repurposing) 작업용 모델
- **Virtual_Screening.pt**: 가상 스크리닝(Virtual Screening) 작업용 모델

다운로드 링크: https://github.com/myzhengSIMM/transformerCPI2.0/releases

---

## 사용 방법

### 1. 기본 예측 (단일 단백질-화합물 쌍)
```bash
python predict.py
```
- 코드 내부에서 SMILES 및 단백질 서열을 직접 수정하여 사용

### 2. Reverse Screening (다중 단백질 스크리닝)
```bash
python reversebatch.py \
  --fasta uniprotkb_taxonomy_id_9606_AND_reviewed_2026_03_17.fasta \
  --smiles "CS(=O)(C1=NN=C(S1)CN2C3CCC2C=C(C4=CC=CC=C4)C3)=O" \
  --output batch_scores.csv \
  --device cuda \
  --limit 1000
```

**주요 옵션:**
- `--fasta`: 입력 FASTA 파일 경로
- `--smiles`: SMILES 문자열 또는 SMILES가 담긴 텍스트 파일
- `--output`: 결과 CSV 파일 경로 (기본값: batch_scores.csv)
- `--device`: 실행 장치 (auto/cpu/cuda, 기본값: auto)
- `--limit`: 처리할 단백질 개수 제한 (선택사항)
- `--skip-len-threshold`: 스킵할 최대 서열 길이 (기본값: 8094)
- `--continue-after-cuda-error`: CUDA 에러 발생 시 계속 진행 (권장하지 않음)

### 3. 기타 원본 기능
```bash
# 돌연변이 분석
python mutation_analysis.py

# 치환 분석
python substitution_analysis.py
```

---

## 시스템 요구사항

### 최소 요구사항
- **Python**: 3.10
- **PyTorch**: 최신 버전 (pip를 통해 자동 설치)
- **CUDA**: CUDA 지원 GPU (권장)
- **RAM**: 16GB 이상 권장
- **GPU VRAM**: 8GB 이상 권장

### 주요 의존성
- `torch`, `torchvision`, `torchaudio`
- `tape-proteins==0.5`
- `rdkit`
- `biopython`
- `numpy<2`
- `pandas`, `scipy`, `scikit-learn`

전체 의존성 목록은 [environment.yml](environment.yml)을 참조하세요.

---

## 라이선스 및 인용

본 수정 버전은 연구실 내부 사용을 목적으로 작성되었습니다. 원본 프로젝트의 라이선스를 준수하며, 본 모델을 사용한 연구 결과를 발표할 경우 반드시 원본 논문을 인용해주세요.

**Citation:**
```bibtex
@article{chen2023sequence,
  title={Sequence-based drug design as a concept in computational drug design},
  author={Chen, Lifan and others},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={4217},
  year={2023},
  publisher={Nature Publishing Group UK London},
  doi={10.1038/s41467-023-39856-w}
}
```

---

## 문제 해결 (Troubleshooting)

### CUDA Out of Memory
- `--skip-len-threshold` 값을 낮춰 긴 서열을 스킵
- 배치 크기가 큰 경우 더 작은 GPU로는 실행 불가능할 수 있음

### "device-side assert" 에러
- 긴 단백질 서열에서 주로 발생
- `--skip-len-threshold 8094` 옵션으로 긴 서열 자동 스킵
- 기본적으로 첫 CUDA 에러에서 중단됨 (안전)

### 환경 설치 문제
```bash
# 환경 삭제 후 재설치
conda env remove -n transcpi-gpu
conda env create -f environment.yml
conda activate transcpi-gpu
```

---

## 관련 프로젝트

- **TransformerCPI**: https://github.com/myzhengSIMM/transformerCPI
- **Original TransformerCPI2.0**: https://github.com/myzhengSIMM/transformerCPI2.0

---

## 기여자

본 수정 버전은 연구실 내부 사용을 위해 작성되었습니다.

**수정 내용:**
- PyTorch 최신 버전 호환성 개선
- Reverse screening 스크립트 ([reversebatch.py](reversebatch.py)) 개발
- Conda 환경 설정 파일 현대화

**Original Author:**
- Lifan Chen (SIMM, Chinese Academy of Sciences)
## Data Source
The protein sequence data (`uniprotkb_taxonomy_id_9606_AND_reviewed_2026_03_17.fasta`) provided in this repository is sourced from the UniProt Knowledgebase (UniProtKB).

* **Organism:** Homo sapiens (Human, TaxID: 9606)
* **Dataset:** Reviewed (Swiss-Prot)
* **Date Downloaded:** 2026-03-17
* **License:** This data is distributed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) License](https://creativecommons.org/licenses/by/4.0/).