# Cross-sport injury-risk classification

## Dataset limitations

1. **Mixed target:** `0 = Healthy`, `1 = Low Risk`, `2 = High Risk/Injured`. Class 2 includes the seven days before injury as well as injury observations. Predicting these labels is not the same as predicting confirmed injury events.
2. **Unknown labeling rules:** Low Risk refers to fatigue or load. Models might reproduce a scoring rule if labels were calculated from these predictors. Exact rules and clinical verification are unavailable.
3. **Unverified collection:** The publisher describes 156 athletes monitored over six months, but original study records, recruitment and consistent sensor protocols across sports have not been verified.
4. **Unclear timing:** Session numbers have no dates. Regular 1–2-day spacing, seven-day windows, distinct injury episodes and measurement timing relative to injury cannot be reconstructed.
5. **Documentation discrepancies:** Fatigue reaches 270.19 despite a stated maximum of 85; recovery and sleep fall below stated minima. Recorded training load differs from intensity × duration. Missingness, gender proportions and two class counts also differ. Preserve questionable values until a correction is justified. Details: [DATA_AUDIT.md](DATA_AUDIT.md).
6. **Few independent athletes:** The three-sport subset has 13,043 sessions from 132 athletes. Sessions are related within athletes: splits, CV and uncertainty estimates must account for this grouping.
7. **Questionable sport context:** Surfaces are almost uniformly distributed across sports, including Grass and Turf for Basketball. This needs explanation but does not prove synthetic data. Three sports cannot establish transfer to all sports or independent cohorts; `Other` is excluded because its disciplines are unspecified.
8. **Reuse details:** Publisher version and license remain unverified. The raw file's checksum is recorded in the audit; verify the license before redistributing data.
9. **Exploratory evaluation:** The preparation notebook describes the full cohort. The binary task was adopted after inspecting three-class test results and reuses A's test athletes, so it is not a fresh confirmatory evaluation. Keep feature policies fixed and use training-only tuning. Familiar and transfer models also differ in training size and sport mix; score gaps cannot isolate sport coverage alone.

## Question and experiments

**How well do models trained on athletes from two sports detect supplied High Risk/Injured labels for athletes in a third sport?**

The main target is now binary: original classes 0 and 1 become **0 = Not high risk**, and original class 2 becomes **1 = High Risk/Injured**. False alarms are preferable to missed positives. F2 is the working selection metric; precision, recall, average precision and error counts show the trade-off. Beta=2 is a modeling preference, not an established operational cost ratio. The original three-class results remain the starting experiment.

| Experiment | Training | Testing |
|---|---|---|
| A: familiar sports | Training athletes from all three sports | Held-out athletes from each sport |
| B: unseen Track | All Football + Basketball athletes | All Track athletes |
| C: unseen Basketball | All Football + Track athletes | All Basketball athletes |
| D: unseen Football | All Basketball + Track athletes | All Football athletes |

Create these splits when implementing each modeling task. Athletes must not cross training and test boundaries. Tune with athlete-grouped folds using only the permitted training sports; fit imputers and encoders within those folds. A sport serving as a test domain in one experiment may train another experiment, provided target-test performance does not guide model choices.

Report B–D on their full target sports. For matched comparisons with A, additionally score the same transfer predictions on A's held-out target athletes; this requires no additional model fits. Report A per sport, rather than comparing a particular target sport only with a pooled score.

Models: dummy references, decision tree and random forest. Binary A compares models with and without one-hot `sport_type`. B–D use only the **without-sport** feature set, retaining the same search spaces, budgets and cutoff rules. Fit all imputers and categorical encoders on source-training folds only. Full plan: [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Notebooks

[1_data.ipynb](1_data.ipynb) contains light cleaning and full-cohort exploration: structure, missingness, label balance, range discrepancies, sport distributions, demographics, surfaces and correlations.

It retains 25 predictors and keeps athlete ID, session ID and sport separate. Missing measurements remain for fold-fitted imputation in the modeling notebooks.

[2_model_A.ipynb](2_model_A.ipynb) implements familiar-sport classification. It holds out 20% of athletes separately within each sport: 104 training athletes and 28 test athletes. A majority-class dummy, decision tree and random forest use the same five athlete-grouped CV folds. The tree searches 18 settings; the forest samples 12 settings with 200 trees. Selection uses macro-F1, with imputation and one-hot encoding fitted inside each training fold. Test results include pooled and per-sport scores, per-class metrics and confusion matrices. Sport is excluded in that initial three-class experiment.

[3_model_A_binary.ipynb](3_model_A_binary.ipynb) is the binary revision, preserving A's athlete split. It compares sport-included and sport-excluded pipelines, selects hyperparameters using grouped CV F2, and then selects cutoffs from grouped out-of-fold training predictions. These two selection stages reuse folds, so their reported selection scores are optimistic; test results are reported separately. Both “never alert” and “always alert” dummies are included. All settings and cutoffs are fixed before test evaluation. Test tables include false alarms, misses, precision, recall, F2, F1, average precision and per-sport results. With sport, the forest reaches F2 0.700 and recall 0.894 (44 misses, 624 false alarms); including sport provides little benefit on this split. The tree has higher recall but more false alarms.

The transfer notebooks repeat binary A's training workflow without `sport_type`. Five-fold grouped CV and cutoff selection use only source athletes; selected pipelines refit on all source athletes before testing on the entire target sport. The held-out sport is never used for fitting or tuning.

| Notebook | Training athletes / sessions | Test athletes / sessions |
|---|---|---|
| [4_model_B_track.ipynb](4_model_B_track.ipynb) | Football + Basketball: 90 / 8,896 | Track: 42 / 4,147 |
| [5_model_C_basketball.ipynb](5_model_C_basketball.ipynb) | Football + Track: 91 / 8,984 | Basketball: 41 / 4,059 |
| [6_model_D_football.ipynb](6_model_D_football.ipynb) | Basketball + Track: 83 / 8,206 | Football: 49 / 4,837 |

All three transfer notebooks have been executed. Forest F2 is 0.717 on Track, 0.679 on Basketball and 0.696 on Football. These full-target results are not paired transfer gaps against A's smaller test subsets; a matched comparison remains to be done.

Each notebook runs independently. Charts and tables display inline; there are no figure exports, results folders, split manifests or metadata JSONs. Model A keeps its test predictions in memory with athlete IDs for later matched comparisons.

## Running

Use Python 3.11 or later. From `final_proj`:

```bash
python -m pip install -r requirements.txt
python -m jupyter lab 1_data.ipynb
```

Restart the kernel and run all cells. Launching from the parent course folder also works. The raw CSV is preserved unchanged.

Dataset: [Multimodal Sports Injury Dataset](https://www.kaggle.com/datasets/anjalibhegam/multimodal-sports-injury-dataset). Definitions come from the description supplied in the planning chat; numerical summaries use the local CSV.
