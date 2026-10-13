# Roberts et al. (2025)와 우리 AIV H7 결과 비교 (2026-10-13, 심사 대응)

**논문에 넣을 3줄 결론.**
1. Roberts et al. (2025, PLoS Comput Biol 21(4):e1012995)은 **같은 AIV 자료**(133개 유전체, Lu et al. 2014 자료, Müller et al. 2017의 날짜 MCC 나무 하나를 고정 입력)를 정확 구조 합류 모형의 베이즈 MCMC(이동 이력 표본추출)로 다시 분석했다. 다만 숙주는 5개 deme(ANS·CHA·GAL·PAS·MEX)으로 나눴고, 우리는 3목(PAS→CHA, MEX→GAL)으로 합쳤다.
2. 기본 사전분포에서 그들의 뿌리는 "높은 확률로 Anseriformes"(다른 네 deme에도 작은 확률)이고, 이동 이력의 대부분(≥ 80%)이 ANS에 놓인다. 이는 우리 masco 나무의 결과(정확·MASCOT 모두 ans 뿌리 1.000)와 맞고, sisco·lisco 나무의 정확 최우 결과(gal 뿌리 0.999)와는 다르다.
3. 그들이 어느 MCC 나무(MASCOT/SISCO/LISCO 분석 중 어느 것)를 썼는지 본문에 없고, 공개 코드 저장소에도 AIV 입력 파일이 없다. 그래서 "같은 나무에서 결론이 같다/다르다"는 말할 수 없고, "요약 나무에 따라 뿌리가 바뀐다"는 우리 결론과 모순되지 않는다는 데까지만 말할 수 있다.

출처: 원문 HTML(https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012995, 2026-10-13 접속), 코드 https://github.com/IanPRoberts/StructCoalescent(파일 목록 확인, 같은 날). PLOS 온라인 판에는 쪽 번호가 없어서 절 제목·그림·표 번호로 위치를 적는다. 아래는 원문에서 확인한 내용만 적었다. 숫자가 그림에만 있고 본문에 없으면 "본문에 없음"으로 적었다.

## 서지
- Roberts I, Everitt RG, Koskela J, Didelot X. Bayesian inference of pathogen phylogeography using the structured coalescent model. PLoS Comput Biol 21(4): e1012995, 2025-04-21. 데이터·코드: github.com/IanPRoberts/StructCoalescent(Data Availability 문장).
- 자료 출처 문헌: 참고문헌 [71] Lu L, Lycett SJ, Leigh Brown AJ. Determining the phylogenetic and phylogeographic origin of highly pathogenic avian influenza (H7N3) in Mexico. PLoS One 9(9): e107330, 2014. 나무 출처: [32] Müller NF, Rasmussen DA, Stadler T. The structured coalescent and its approximations. Mol Biol Evol 34(11): 2970–81, 2017. 비교 대상: [30] De Maio et al. 2015 BASTA(PLoS Genet).

## 원문에서 확인한 사실 (절 "Determining the phylogenetic origin of Avian Influenza in Mexico", 그림 7, 표 2, S7–S9 그림)
- **자료.** Central·North America의 2000년대 초 AIV 전파를 다룬 이전 연구[71]의 133개 유전체. 지리가 아니라 **조류 목**으로 표본을 나눴다: Anseriformes(ANS), Charadriiformes(CHA), Galliformes(GAL), Passeriformes(PAS), 그리고 멕시코 가금 발병(MEX). deme 5개. 그래서 이동은 종간 전파를 뜻한다고 적었다. ANS 잎이 133개 중 93개라고 적었다(우리 자료의 ans 93과 같다).
- **나무.** "dated maximum clade credibility tree"를 입력으로 고정했고 출처는 [32](Müller 2017)다. 나무는 S7 그림. 원 연구[71]는 최대가능도로 날짜 계통수를 만들고 DTA(이산 형질 분석)로 퍼짐을 모형화했다고 적었다. **[32]의 세 MCC 나무(MASCOT·SISCO·LISCO 분석에서 나온 것) 중 무엇인지는 본문에 없다.** 공개 저장소 파일 목록에도 AIV 나무·XML이 없다.
- **분석.** 정확 구조 합류 모형, 이동 이력을 표본추출하는 MCMC. 사슬 5개 × 2세트(각 48 h): 기본(경험적 베이즈 감마) 사전분포와, BASTA 분석과 같은 Exp(1) 사전분포. 시작은 DTA로 뽑은 이동 이력.
- **수렴(표 2).** 기본 사전분포는 ESS·PSRF가 양호(다변량 PSRF 1.025). Exp(1)은 수렴 실패(다변량 PSRF 4.48, 25개 모수 중 21개 PSRF > 1.1)이고 이동 횟수를 평균 214회로 잡았다. 최소 변화(parsimony) 복원은 8회다(S8·S9 그림의 M = 8). 그래서 Exp(1)은 기각.
- **뿌리.** 기본 사전분포: 뿌리가 "높은 확률로 Anseriformes deme", 나머지 네 deme에 작지만 0이 아닌 확률. **수치는 본문에 없다**(그림 7). 비교로 BASTA(Exp(1))는 거의 균등, 원 DTA 분석은 Anseriformes 확률 1이라고 적었다.
- **이동 방향.** 기본 사전분포의 이동 이력은 대개 8–20회 이동이고, 거의 모든 이력이 길이의 ≥ 80%를 ANS에 둔다(S9 그림). 저자들은 표본이 ANS로 쏠린 탓일 수 있다고 적었다. 그림 7은 60% 합의 이동 이력이고 화살표 굵기가 역방향 이동률 크기에 비례한다. **어느 목이 MEX/GAL을 퍼뜨렸는지, 방향별 이동률 수치는 본문에 없다**(그림 7에만 있음, 텍스트로 확인 불가).

## 비교 표

| 항목 | Roberts et al. 2025 | 우리 masco | 우리 sisco | 우리 lisco |
|---|---|---|---|---|
| 잎 | 133 (같은 자료) | 133 | 133 | 133 |
| 나무 | Müller 2017의 날짜 MCC 나무 하나(어느 것인지 미기재), 고정 | Müller 2017 MASCOT MCC, 고정 | SISCO MCC, 고정 | LISCO MCC, 고정 |
| 숙주 구분 | 5 deme: ANS, CHA, GAL, PAS, MEX | 3목: ans 93 / cha 32(PAS 2 포함) / gal 8(MEX 2 포함) | 같음 | 같음 |
| 모형 | 정확 구조 합류, deme별 Ne + 비대칭 역방향 이동, 베이즈(이동 이력 MCMC) | 정확(TT) 최우추정 + MASCOT 최우추정, 비대칭·대칭 | 같음 | 같음 |
| 뿌리 | ANS 높은 확률(수치 없음), 다른 deme 작은 확률 | **정확·MASCOT 모두 ans 1.000** (ans 뿌리 허브) | 정확 **gal 0.999** / MASCOT cha 0.95 | 정확 **gal 0.999** / MASCOT cha 0.95 |
| 이동 요약 | 8–20회, 이력 ≥ 80%가 ANS; 방향별 수치 본문에 없음 | 허브: cha←gal 16.5/년, gal←ans 0.57/년(역방향) | gal 뿌리: ans←gal 1.62, cha←gal 0.46/년 | gal 뿌리: ans←gal 1.61, cha←gal 0.48/년 |
| 최소 변화 수 | 8 (5 deme) | Fitch 8 (3목) | – | – |
| 근사와의 대조 | BASTA(Exp(1)) 뿌리 거의 균등, 비수렴 | MASCOT 다중시작이 자기 최적을 놓쳤음, 정제 후 같은 봉우리 | MASCOT 최적과 정확 최적이 다른 봉우리 | 같음 |

## 해석 (우리 논문에 쓸 수 있는 범위)
- **같은 자료다.** 잎 수(133), ANS 93개, 목 단위 deme, Müller 2017의 날짜 MCC 나무라는 점이 모두 같다. 차이는 deme 구분(5 대 3)과 추론 방식(베이즈 대 최우)이다.
- 그들의 ANS 뿌리는 우리 masco 결과와 원 DTA 결과와 같은 방향이다. sisco·lisco의 정확 최우 결과(gal 뿌리)와는 다르다. 그들이 MASCOT MCC 나무를 썼다면 우리 masco와 일치하는 것이고, 다른 나무였다면 "같은 자료라도 요약 나무·deme 구분·사전분포에 따라 뿌리가 바뀐다"는 우리 결론의 사례가 하나 더 생긴다. 어느 쪽인지 원문으로는 정할 수 없다. 저자에게 물으면 알 수 있다.
- 그들의 베이즈 분석은 모수를 적분한 사후 뿌리 확률이고, 우리 값은 최우 모수에서의 조건부 뿌리 분포다. 또 PAS·MEX를 따로 둔 5 deme 모형은 우리 3목 모형과 모형 공간이 다르다. 그래서 숫자를 직접 비교하지 말고 방향만 비교해야 한다.
- 우리 원고와 관련해 주의할 점: 이전 실데이터2 레인 기록(LOG 2026-10-06 16:40)은 Roberts 2025를 ST239 자료로만 보고 "날짜 나무 비공개라 탈락"으로 적었다. 같은 논문에 AIV 분석이 있다는 사실은 빠져 있었다. 원고의 AIV 절에 이 선행 정확-구조합류 분석을 인용해야 하고, "공개 분석은 ans 뿌리"라는 문장(v1.5에서 근거 없어 삭제함)은 이제 Roberts 2025와 Lu 2014(DTA)를 근거로 되살릴 수 있다. 단, Roberts의 뿌리 확률은 수치 없이 "높은 확률"로만 인용한다.
