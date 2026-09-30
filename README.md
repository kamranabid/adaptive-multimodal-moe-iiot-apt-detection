# adaptive-multimodal-moe-iiot-apt-detection
Code for adaptive multimodal sparse Mixture-of-Experts APT detection in IIoT using network traffic and provenance data. Includes training, episode-level evaluation, statistical analysis, and figures.
## Contents

- [Research question](#research-question)
- [Repository contents](#repository-contents)
- [Dataset and setup](#dataset-and-setup)
- [Method and mathematical specification](#method-and-mathematical-specification)
- [Evaluation protocol](#evaluation-protocol)
- [Verified results](#verified-results)
- [Reproducibility and scope](#reproducibility-and-scope)
- [Citation and license](#citation-and-license)

## Research question

APT evidence may be distributed across traffic, host activity, and time. The study asks whether specialized representations, adaptive sparse routing, and episode-level aggregation can discriminate APT activity in CICAPT-IIoT under marked class imbalance. The implemented final predictor is an **integrated ensemble**: the sparse Mixture-of-Experts (MoE) term is one component of its score, alongside pairwise ranking, regularized stacking, rank consensus, and nonlinear hard-case fusion. The results below belong to this complete integrated output, not to the sparse gate in isolation.

## Repository contents

The following layout is recommended when uploading the supplied research artifacts. Keep the original notebook filename or rename it consistently in this README.

```text
.
├── README.md
├── kamran_paper_5 (21)(2).ipynb
├── Proposed_Multimodal_MoE_FINAL_RESULTS.zip
├── Proposed_Multimodal_MoE_Results_1200DPI.zip
└── LICENSE                         # add only after choosing a license
```

`Proposed_Multimodal_MoE_FINAL_RESULTS.zip` contains statistical-validation CSV/JSON files, result tables, publication figures, an audit note, and a results paragraph. `Proposed_Multimodal_MoE_Results_1200DPI.zip` contains the earlier evidence and high-resolution figures. The first package is the paper-facing source of final numbers. The raw CICAPT-IIoT dataset is downloaded separately; do not commit the multi-gigabyte data to the repository.

## Dataset and setup

The notebook uses the [CICAPT-IIoT Kaggle mirror](https://www.kaggle.com/datasets/waqarkha/cicapt-iiot) with the handle `waqarkha/cicapt-iiot`. The recorded local dataset inventory was approximately **9.14 GB**. Access terms and the dataset's authoritative provenance should be checked on the dataset page before redistribution.

### Run in Google Colab

1. Open the notebook in Colab and connect to a runtime with enough free Google Drive space and memory for the dataset and generated intermediate files.
2. Run the initial setup cell to mount Google Drive and install `kagglehub` and `pandas` as written in the notebook. Authenticate to Kaggle through the supported Colab/Kaggle mechanism if the download requests it. Never commit API tokens.
3. The notebook sets its project root to:

   ```python
   /content/drive/MyDrive/IIoT_APT_Research
   ```

   It downloads the dataset into the `CICAPT_IIoT` subdirectory. To use another root, update the `DRIVE_ROOT`/`ROOT` constants consistently in the notebook **before running its cells**.
4. Execute the notebook in order from a clean runtime. Intermediate preparation, temporal features, model scores, and final evidence depend on files produced by earlier cells. Do not run the final reporting cells against unrelated cached artifacts.
5. Compare newly generated reports with the final result package. Record package versions, random seed, input checksums, and any intentional changes in a reproducibility log.

The notebook uses Python with `numpy`, `pandas`, `scipy`, `scikit-learn`, `pyarrow`, `joblib`, `matplotlib`, and, for some earlier experiments, PyTorch. Colab's bundled versions may change. For strict replication, record the versions from the runtime that produced the reported results; the supplied notebook does not provide a locked environment or a standalone command-line entry point.

### Data flow

```text
CICAPT-IIoT traffic + provenance + attack annotations
        ↓ phase/time alignment and five-second windows
Network / provenance / fused / temporal feature matrices
        ↓ five specialized classifiers
Out-of-fold window expert scores
        ↓ retrospective episode construction and temporal summaries
Episode expert ranks + top-two gate + integrated score
        ↓ training-fold calibration and grouped evaluation
Out-of-fold episode probabilities, metrics, comparisons, figures
```

The paired feature set contains **119,896 five-second windows**, including **229 positive windows**. Retrospective grouping yields **2,070 episodes**, including **52 positive** and **2,018 negative** episodes. These counts describe the implemented preparation and should be rechecked after any data or preprocessing change.

## Method and mathematical specification

### 1. Aligned representations and experts

For window $i$, let $x_i^{(N)}$ be network features, $x_i^{(P)}$ provenance features, and $x_i^{(T)}$ temporal features. The base inputs are

$$
X_i^{(N)}=x_i^{(N)},\qquad
X_i^{(P)}=x_i^{(P)},\qquad
X_i^{(F)}=[x_i^{(N)},x_i^{(P)}],\qquad
X_i^{(TF)}=[x_i^{(N)},x_i^{(P)},x_i^{(T)}].
$$

Five classifiers produce window scores $s_{ie}\in[0,1]$ for experts $e\in\{N,P,F,TF,H\}$: Network, Provenance, Fusion, TemporalFusion, and HardNegative. The first four are `ExtraTreesClassifier` pipelines with median imputation, 160 trees, `max_features="sqrt"`, and balanced subsample weights. The Provenance expert uses `min_samples_leaf=2`; the other three use 1. The fifth expert uses fusion features and weighted training examples:

$$
w_i^{(H)}=
\begin{cases}
2, & y_i=1,\\
3, & y_i=0\ \text{and}\ s_{iF}\ge Q_{0.90}(\{s_{jF}:y_j=0,\ j\in\mathcal T\}),\\
1, & \text{otherwise},
\end{cases}
$$

where $\mathcal T$ is the current training partition. The cutoff and weights are determined inside training data. Outer held-out windows are scored by experts fitted without their episodes. Inner episode-stratified folds provide out-of-fold base predictions to the meta models.

### 2. Retrospective episode construction

Within each dataset phase, positive windows are sorted by time. A positive window begins a new episode if it is more than **60 seconds** after the preceding positive window. A negative window at time $t_i$ is assigned to the phase-specific **300-second** bin $\lfloor t_i/300\rfloor$. For an episode $E$, its evaluation label is

$$
y_E=\max_{i\in E} y_i.
$$

**Interpretation:** these episode boundaries use $y_i$ and therefore depend on ground truth. They define a retrospective experiment, not a label-free online detector. A future deployment version must replace this rule, train and tune with the replacement, and evaluate it on independent data.

### 3. Episode-level temporal summaries

Let $S_{E,e}=(s_{1e},\ldots,s_{|E|e})$ be the time-ordered scores of expert $e$ in episode $E$. The implementation records their maximum, top-three mean, 90th percentile, mean, standard deviation, early and late maxima, and linear time-index slope. Its robust summary is

$$
R_{E,e}=0.50\,\operatorname{mean}(\operatorname{top}_3 S_{E,e})
+0.20\,\max S_{E,e}
+0.20\,Q_{0.90}(S_{E,e})
+0.10\,\operatorname{mean}(S_{E,e}).
$$

For episodes with fewer than three windows, `top_3` uses all available scores. The recorded context statistics are passed to the meta layer. Training-fitted empirical rank maps transform each expert's robust score to $r_{E,e}\in[0,1]$; held-out episodes use the corresponding training map.

### 4. Expert competence, consensus, and sparse routing

Let $\pi$ be the positive-episode prevalence in the training fold and $AP_e$ the training AP for ranked expert $e$. Competence and redundancy are estimated as

$$
c_e=\operatorname{clip}\!\left(\frac{AP_e-\pi}{1-\pi},0,1\right),\qquad
\rho_e=\frac{1}{m-1}\sum_{j\ne e}\left|\operatorname{corr}\bigl(\operatorname{rank}(r_e),\operatorname{rank}(r_j)\bigr)\right|,
$$

with $m=5$ experts. The normalized utility weights are

$$
\tilde u_e=(c_e+10^{-4})^2\operatorname{clip}(1-0.50\rho_e,0.10,1)+0.002,
\qquad u_e=\frac{\tilde u_e}{\sum_j\tilde u_j}.
$$

The robust consensus score uses the weighted ranks, their median, and the mean of the largest $q=\lceil0.40m\rceil=2$ ranks:

$$
C_E=\operatorname{clip}\!\left(0.70\sum_eu_er_{E,e}
+0.15\operatorname{median}_e r_{E,e}
+0.15\operatorname{mean}(\operatorname{top}_q\{r_{E,e}\}),0,1\right).
$$

A logistic gate is fitted on training episode ranks and context features. Its training target identifies the expert with the smallest per-episode binary log loss. If its probability vector is $a_E=(a_{E,1},\ldots,a_{E,m})$, only the largest two components are retained:

$$
K_E=\operatorname{Top2}(a_E),\qquad
 g_{E,e}=\frac{a_{E,e}\mathbf 1[e\in K_E]}
 {\sum_{j\in K_E}a_{E,j}},\qquad
 M_E=\sum_{e=1}^{m}g_{E,e}r_{E,e}.
$$

A single-class gate, where encountered, assigns all weight to that class. The sparse score $M_E$ is only **one component** of the complete predictor.

### 5. Integrated score and calibration

The meta layer also computes a regularized logistic stacking probability $L_E$, pairwise rank score $P_E$, nonlinear hard-case probability $H_E$, and average $T_E$ of the two highest expert ranks. The pairwise logistic model learns from training-episode differences $z_{E^+}-z_{E^-}$, including sampled difficult negatives. The implemented integrated score is

$$
Z_E=\operatorname{clip}\left(
0.42P_E+0.26L_E+0.14C_E+0.10M_E+0.05T_E+0.03H_E,
0,1\right).
$$

The coefficients sum to one. This fixed combination is the output reported as **Proposed Multimodal MoE** in the final evidence. A logistic Platt calibrator is fitted using inner out-of-fold training scores:

$$
\hat p_E=\sigma\!\left(\alpha\operatorname{logit}\bigl(\operatorname{clip}(Z_E,\epsilon,1-\epsilon)\bigr)+\beta\right),
\qquad \sigma(v)=\frac{1}{1+e^{-v}}.
$$

In the notebook, $\epsilon=10^{-6}$ for calibration. Calibration is refitted within each outer training fold before scoring held-out episodes. The primary reported AP, PR-AUC, and ROC-AUC use the pooled **out-of-fold integrated-score** predictions; they must not be replaced by the fold-selected alternative-head result or a different threshold profile.

### 6. Alarm thresholds and classification measures

For a probability threshold $\tau$, $\hat y_E(\tau)=\mathbf 1[\hat p_E\ge\tau]$. The original training-fold threshold policy searches observed candidate probabilities subject to $FPR\le0.02$ and precision $\ge0.70$, maximizing

$$
J(\tau)=0.50F_2(\tau)+0.30MCC(\tau)+0.20\operatorname{Precision}(\tau).
$$

If no threshold satisfies both constraints, the code first relaxes the precision condition while retaining the FPR condition. Its selected thresholds are learned on training folds and applied to their held-out episodes. A **separate post-cross-validation** analysis characterizes a balanced policy ($\tau=0.096005$) using pooled out-of-fold scores with precision $\ge0.85$, FPR $\le0.005$, and MCC as its objective. That balanced threshold is descriptive because selection used the evaluated pooled predictions.

For confusion counts $TP,FP,FN,TN$, the principal threshold metrics are

$$
\operatorname{Precision}=\frac{TP}{TP+FP},\quad
\operatorname{Recall}=\frac{TP}{TP+FN},\quad
F_\beta=\frac{(1+\beta^2)TP}{(1+\beta^2)TP+\beta^2FN+FP},
$$

$$
FPR=\frac{FP}{FP+TN},\quad
\operatorname{BalancedAccuracy}=\frac12\left(\frac{TP}{TP+FN}+\frac{TN}{TN+FP}\right),
$$

$$
MCC=\frac{TP\cdot TN-FP\cdot FN}
{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}.
$$

AP summarizes precision at successive positive retrieval ranks; PR-AUC is trapezoidal area under the precision–recall curve. They use different numerical integration rules and need not be identical. ROC-AUC is the area under the true-positive-rate versus false-positive-rate curve. Calibration measures include

$$
\operatorname{Brier}=\frac1n\sum_{E=1}^{n}(\hat p_E-y_E)^2,
\qquad
\operatorname{ECE}=\sum_{b=1}^{B}\frac{|I_b|}{n}
\left|\operatorname{acc}(I_b)-\operatorname{conf}(I_b)\right|,
$$

where the code uses **ten quantile-based bins** for ECE, with a fallback to equally spaced bins if necessary.

## Evaluation protocol

- **Unit:** episode. All windows in an episode share the same outer-fold assignment.
- **Outer assessment:** four stratified episode folds; 13 positive episodes per held-out fold.
- **Base expert training scores:** two inner episode folds.
- **Meta training scores:** three inner episode folds.
- **Seed:** `20260929` in the final full-data evaluation.
- **Primary ranking output:** pooled out-of-fold calibrated scores for the fixed integrated predictor.
- **Uncertainty:** 5,000 stratified episode bootstrap resamples for intervals; 10,000 paired permutations for AP differences, with Holm adjustment for multiple comparisons.
- **Comparison:** internal methods share the episodes and evaluation protocol. Scores reported by different published papers on other datasets are not directly comparable.

A positive AP difference is described as statistically supported only when the paired bootstrap interval remains above zero **and** the adjusted permutation analysis supports the comparison. The predefined AP non-inferiority margin is 0.05. The threshold operating profiles are reported separately from ranking discrimination.

## Verified results

### Primary episode ranking

| Metric | Pooled out-of-fold result | Bootstrap mean and 95% interval |
|---|---:|---:|
| Average precision (AP) | **0.9495** | 0.9492 [0.8876, 0.9948] |
| PR-AUC | **0.9494** | 0.9490 [0.8873, 0.9947] |
| ROC-AUC | **0.9614** | 0.9611 [0.9040, 0.9999] |

The mean outer-fold AP was approximately **0.9512 ± 0.0861**. Pooled AP and mean fold AP are different summaries.

### Threshold-dependent results

| Policy | Precision | Recall | F1 | MCC | FPR | TP / FP / FN / TN |
|---|---:|---:|---:|---:|---:|---|
| Training-fold selected thresholds | 0.6389 | 0.8846 | 0.7419 | 0.7445 | 0.01288 | 46 / 26 / 6 / 1992 |
| Post-CV balanced characterization, $\tau=0.096005$ | **0.9412** | **0.9231** | **0.9320** | **0.9304** | **0.00149** | 48 / 3 / 4 / 2015 |
| Post-CV high-precision profile, $\tau=0.939698$ | 1.0000 | 0.8654 | 0.9278 | 0.9287 | 0.00000 | 45 / 0 / 7 / 2018 |

The balanced characterization also yielded F2 **0.9266**, balanced accuracy **0.9608**, and specificity **0.9985**. It changes the decision rule, not AP, PR-AUC, or ROC-AUC.

### Same-protocol model comparisons

| Internal comparator | Comparator AP | Integrated minus comparator AP | Paired bootstrap 95% interval | Holm-adjusted permutation $p$ |
|---|---:|---:|---:|---:|
| Hard-case nonlinear fusion | 0.9659 | −0.0165 | [−0.0619, 0.0237] | 0.4990 |
| Regularized stacking | 0.9573 | −0.0078 | [−0.0244, 0.0013] | 0.4688 |
| Pairwise rank fusion | 0.9297 | +0.0197 | [−0.0051, 0.0558] | 0.0060 |
| Sparse utility MoE | 0.7730 | **+0.1765** | **[0.0687, 0.2920]** | **0.0006** |
| Robust rank consensus | 0.7455 | **+0.2039** | **[0.1007, 0.3113]** | **0.0006** |
| Top-k episode baseline | 0.6069 | **+0.3426** | **[0.2024, 0.4524]** | **0.0006** |

The hard-case and stacking components have numerically higher pooled AP than the integrated score; do not claim universal superiority. The pairwise rank fusion interval crosses zero, so its positive AP point difference is not used as a confirmed superiority result. Non-inferiority under the prespecified 0.05 AP margin is supported against regularized stacking and pairwise rank fusion.

The pooled probability diagnostics are Brier score **0.0033**, log loss **0.0221**, expected calibration error **0.0079**, and calibration slope approximately **1.0197**. These characterize the current grouped dataset, not external deployment calibration.

## Reproducibility and scope

1. **Keep data separate.** Download CICAPT-IIoT independently and retain the dataset version and file checksums used in each run.
2. **Retain fold boundaries.** Episode IDs and all windows within an episode must remain on one side of every train/validation split. Fit imputers, experts, rank maps, meta models, and calibrators using only the appropriate training partition.
3. **Avoid stale caches.** The notebook writes intermediate artifacts to Google Drive. Use a new project root or remove only known intermediate products when changing data, features, or model settings; otherwise earlier cached outputs may be reused.
4. **Distinguish model outputs.** The primary published metrics correspond to the integrated score's out-of-fold prediction column. The notebook also calculates alternative heads and fold-selected results with different metrics.
5. **Distinguish threshold protocols.** The balanced threshold was chosen post-CV on pooled predictions. For a prospective performance claim, lock the threshold on development data and evaluate it once on untouched data.
6. **Resolve label-dependent episodes.** Ground-truth labels currently determine positive episode boundaries. Implement a label-free temporal/entity grouping rule and rerun the full pipeline before presenting this method as a deployable detector.
7. **Report compute and environment.** Record Python/package versions, hardware, wall-clock time, memory use, and exact notebook revision. These are not fully specified in the current result package.

