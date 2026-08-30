# Improving the NFHS-5 child malnutrition prediction model

This guide is tailored to your current project in [src/01_prepare_data.py](src/01_prepare_data.py) and [src/02_run_models.py](src/02_run_models.py). Your current pipeline already does several good things:

- It builds labels for stunting and wasting using WHO cutoffs.
- It uses engineered features such as age groups, birth-order terms, and sanitation risk.
- It uses SMOTE, cross-validation, threshold tuning, and an ensemble.
- It already trains XGBoost and other models.

That means you are not starting from zero. Your main opportunity is to make the learning process more precise, more robust, and more tailored to tabular healthcare data.

## First important truth

A baseline of about 58% accuracy is not bad for a difficult public-health prediction problem, especially when the data are limited and the outcome is noisy. However, achieving 80% accuracy may be very hard unless you gain much richer predictors or more informative labels.

A more realistic target for this type of DHS-based child malnutrition task is often:

- 60–65%: solid baseline
- 65–72%: very good with strong preprocessing and tuning
- 72–78%: excellent if the data quality is high and the feature set is strong
- 80%+: possible only with very rich data, better feature engineering, or a much more specialized modeling setup

So the goal should be: improve steadily, not chase an unrealistic number blindly.

---

# 1. Data preprocessing improvements

## 1.1 Missing value handling

### What is it?
Missing values are values that are absent in the dataset. In healthcare data, missingness is common because some surveys, measurements, or records are incomplete.

### Why does it improve accuracy?
Many models cannot handle missing values directly. If missing values are not treated carefully, the model may learn poor patterns or become biased. Good missing-value handling can preserve information and reduce noise.

### How it works internally
There are several strategies:

- Mean/median imputation: replace missing values with the average or middle value.
- Mode imputation: replace categorical missing values with the most common category.
- KNN imputation: use similar rows to estimate the missing values.
- Iterative imputation: predict one missing feature from the others.
- Missingness indicators: add a binary flag showing whether a value was missing.

### When to use it
Use it whenever your dataset contains missing values in important features.

### Advantages and disadvantages

| Method | Advantages | Disadvantages |
|---|---|---|
| Mean/median | Simple and fast | Can distort distributions |
| Mode | Good for categorical variables | Can create false common patterns |
| KNN imputer | Often more accurate | Slower and sensitive to scale |
| Iterative imputer | Stronger than simple imputation | More computationally expensive |
| Missingness indicator | Preserves the signal of missingness | Can increase dimensionality |

### How to implement in your project
In your current pipeline, you already fill continuous missing values with the mean and replace categorical missing values with "unknown". That is a good start. You can improve it by:

1. Adding missingness indicator columns like `birth_weight_missing`, `anc_visits_missing`.
2. Using median instead of mean for skewed variables such as `birth_weight` or `anc_visits`.
3. Trying KNN imputation for a stronger filling strategy.
4. Using different strategies for different feature types.

### Expected improvement
Usually a gain of about 1–5 percentage points is realistic if missing values were hurting the model.

### Simple real-life example
Imagine a doctor is trying to predict whether a child is at risk. If the child’s birth weight is missing, replacing it with the average birth weight may be better than leaving it blank. But if missingness itself carries information, such as a mother not reporting weight due to a stressful situation, that missingness itself can be a useful signal.

---

## 1.2 Outlier detection

### What is it?
Outliers are extreme values that are very different from most other values.

### Why does it improve accuracy?
Outliers can distort the learning process. A few extreme values may pull the model toward the wrong pattern.

### How it works internally
Common methods include:

- Z-score: values far from the mean are flagged.
- IQR rule: values outside the interquartile range are flagged.
- Isolation Forest: a tree-based anomaly detector.
- Winsorizing: cap extreme values at a chosen percentile.

### When to use it
Use it for variables such as age, height, weight, wealth score, or breastfeeding duration, where extreme values can happen but should not dominate the model.

### Advantages and disadvantages

| Method | Strength | Weakness |
|---|---|---|
| Z-score | Simple | Sensitive to skewed data |
| IQR | Robust | May miss rare anomalies |
| Isolation Forest | Good for complex outliers | More complex |
| Winsorizing | Easy to apply | Can remove real variation |

### How to implement in your project
In your current feature pipeline, you can:

1. Detect outliers for continuous variables such as `birth_weight`, `mother_bmi`, `anc_visits`, `wealth_score`, and `child_age_months`.
2. Winsorize extreme values to the 1st and 99th percentiles.
3. Apply scaling after outlier handling.

### Expected improvement
A gain of about 0–3 percentage points is common, especially when the model is sensitive to extreme values.

### Simple real-life example
If one child is recorded with a birth weight that is clearly impossible, that single error can mislead the model. Removing or correcting it helps the model learn the real patterns.

---

## 1.3 Duplicate removal

### What is it?
Duplicate rows are exact or near-exact repeated records.

### Why does it improve accuracy?
Duplicates can make the model over-weight repeated patterns and reduce generalization.

### How it works internally
The process is simple: identify repeated rows and remove the extras.

### When to use it
Use it when the data have repeated survey records or copies from merges.

### Advantages and disadvantages

| Advantage | Disadvantage |
|---|---|
| Reduces bias from repeated patterns | Can remove legitimate multiple observations if not carefully handled |

### How to implement in your project
You can inspect duplicates by the ID columns such as `caseid` or the combination of household and child features.

### Expected improvement
Small, usually 0–1 percentage points, but it helps data quality.

### Simple real-life example
If one patient record appears five times in a hospital database, the model may think that patient pattern is more common than it really is.

---

## 1.4 Data normalization and standardization

### What is it?
Scaling transforms features so they are on a similar range.

### Why does it improve accuracy?
Some models are sensitive to feature scale. For example, SVM and neural networks train better when features are on comparable scales.

### How it works internally
- Standardization: convert features to mean 0 and standard deviation 1.
- MinMax scaling: convert features into a range such as 0 to 1.
- Robust scaling: useful when outliers exist.

### When to use it
Use it for SVM, KNN, neural networks, and sometimes logistic regression.

### Advantages and disadvantages

| Method | Good for | Weakness |
|---|---|---|
| StandardScaler | General use | Sensitive to outliers |
| RobustScaler | Outlier-heavy data | Slightly less intuitive |
| MinMaxScaler | Neural nets, bounded features | Not ideal for skewed data |

### How to implement in your project
Your code already uses `StandardScaler`. That is good. For your dataset, `RobustScaler` may be better than `StandardScaler` if you keep outliers. Try both.

### Expected improvement
Usually 1–4 percentage points depending on the model.

### Simple real-life example
Imagine one feature is salary measured in dollars and another is age in years. A model may over-focus on the larger-valued feature. Scaling makes them comparable.

---

## 1.5 Feature encoding

### What is it?
Encoding converts categorical variables into a numeric form that a machine learning model can process.

### Why does it improve accuracy?
Many models cannot directly use text or category labels. Good encoding preserves useful category information.

### How it works internally
Common approaches:

- One-hot encoding: creates one binary column per category.
- Ordinal encoding: maps categories to numbers.
- Target encoding: uses the target mean for each category.
- Frequency encoding: uses the frequency of the category.

### When to use it
Use it for variables like `education`, `residence`, `water_source`, `toilet_type`, `mother_marital_status`, and `child_age_group`.

### Advantages and disadvantages

| Method | Advantages | Disadvantages |
|---|---|---|
| One-hot | Simple and stable | Can create many columns |
| Ordinal | Compact | Assumes category order |
| Target | Powerful for high-cardinality data | Can overfit if not regularized |
| Frequency | Simple | Can leak information if not careful |

### How to implement in your project
Your code already uses `pd.get_dummies`, which is good. You can make it more careful by:

1. Using target encoding only for high-cardinality variables.
2. Avoiding overly sparse one-hot encoding for variables with too many categories.
3. Combining rare categories into an "other" bucket.

### Expected improvement
Usually 0–3 percentage points, but it matters a lot if the categorical variables contain strong signal.

### Simple real-life example
If a variable says `water_source = tap`, `well`, or `river`, the model needs a numerical representation. One-hot encoding says “this row is tap, not well, not river.”

---

# 2. Feature engineering

## 2.1 Creating new features

### What is it?
Feature engineering means creating new input variables from the existing ones.

### Why does it improve accuracy?
The model can learn patterns more easily when the features reflect the real-world problem better.

### How it works internally
A new feature can capture a hidden relationship. For example, using a child-age group can be more useful than using raw age in months.

### When to use it
Use it when you understand the domain well and know that combinations or categories matter.

### Advantages and disadvantages

| Advantage | Disadvantage |
|---|---|
| Often gives big gains | Requires domain knowledge |
| Helps the model learn non-linear patterns | Can overfit if too many features are added |

### How to implement in your project
Your current code already has good engineered features. You can add more, such as:

- `mother_bmi_to_age_ratio`
- `child_age_times_birth_order`
- `wealth_and_education_interaction`
- `sanitation_risk_index` (already present)
- `birth_weight_to_age_ratio`
- `breastfeeding_duration_per_age`
- `mother_height_to_bmi_ratio`

### Expected improvement
Often 2–8 percentage points if the new feature captures an important real pattern.

### Simple real-life example
If the model only sees age and weight separately, it may miss that the relationship between them is important. A new feature like `weight_for_age` can help.

---

## 2.2 Feature interactions

### What is it?
Interactions are features formed by combining two or more existing features.

### Why does it improve accuracy?
Some risks are not explained by single variables alone; they appear when two factors combine.

### How it works internally
The model learns that the effect of one variable changes depending on another variable.

### When to use it
Use it when the relationship is not additive.

### Advantages and disadvantages

| Advantage | Disadvantage |
|---|---|
| Captures combined effects | Can explode the feature space |
| Useful in medicine and social science | More prone to overfitting |

### How to implement in your project
Examples:

- `child_age_months * birth_order`
- `education * wealth_quintile`
- `residence * sanitation_risk_index`
- `mother_bmi * breastfeeding_duration`

### Expected improvement
Often 1–5 percentage points.

### Simple real-life example
The danger of malnutrition may be high when a child is both young and from a poor household. The effect is bigger than either factor alone.

---

## 2.3 Domain-specific feature creation

### What is it?
This means building features that reflect the real-world process behind the problem.

### Why does it improve accuracy?
In healthcare and nutrition, domain knowledge can be more valuable than raw statistical tricks.

### How it works internally
You use knowledge from the domain to encode relevant concepts. For example, a child’s nutrition risk may be shaped by feeding history, sanitation, maternal health, and household wealth.

### When to use it
Use it whenever domain knowledge is available and meaningful.

### Advantages and disadvantages

| Advantage | Disadvantage |
|---|---|
| Highly interpretable | Requires expertise |
| Often boosts performance | Hard to discover automatically |

### How to implement in your project
You already have a strong start with features like:

- sanitation risk index
- maternal BMI category
- child age group
- weaning indicator

You can add more domain-driven features such as:

- `high_risk_mother`: mother is underweight and has low antenatal visits
- `multiple_risk_factors`: number of risky conditions present
- `household_sanitation_risk`
- `maternal_nutrition_risk`

### Expected improvement
Often 2–6 percentage points when the domain features reflect real causes.

### Simple real-life example
In medicine, the combination of fever, cough, and low oxygen is more informative than any one symptom alone.

---

## 2.4 Polynomial features

### What is it?
Polynomial features add terms like $x^2$, $x^3$, or $x_1x_2$.

### Why does it improve accuracy?
Many real relationships are curved, not straight. Polynomial features allow the model to learn curves.

### How it works internally
Instead of assuming a straight-line relationship, the model sees squared or multiplied versions of features.

### When to use it
Use it when the relationship seems curved or when you want to capture increasing risk at higher values.

### Advantages and disadvantages

| Advantage | Disadvantage |
|---|---|
| Captures non-linearity | Can overfit quickly |
| Useful for logistic regression and linear models | Creates many features |

### How to implement in your project
You already use `birth_order_sq` and `child_age_months_sq`. That is a good start. You can add more carefully selected polynomial terms.

### Expected improvement
Often 0–3 percentage points if used selectively.

### Simple real-life example
If nutrition risk rises faster for older children, the relationship is not linear; a squared term can help capture that.

---

## 2.5 Feature transformation

### What is it?
Transformations change the scale or shape of a variable so it is easier to model.

### Why does it improve accuracy?
Many models work better when variables are more symmetric and less skewed.

### How it works internally
Examples:

- Log transform for skewed positive values
- Box-Cox transform
- Quantile transform
- Rank transform

### When to use it
Use it for skewed variables such as birth weight or household size if they are highly uneven.

### Advantages and disadvantages

| Method | Advantage | Disadvantage |
|---|---|---|
| Log transform | Reduces extreme influence | Not for zero/negative values |
| Quantile transform | Makes distributions more uniform | Harder to interpret |

### How to implement in your project
Try log-transforming variables such as `breastfeeding_duration` or `anc_visits` if they are skewed.

### Expected improvement
Around 0–2 percentage points.

### Simple real-life example
Very large incomes can dominate a model. Taking the log reduces the influence of extreme cases.

---

# 3. Feature selection

## 3.1 SHAP

### What is it?
SHAP measures how much each feature contributes to a prediction.

### Why does it improve accuracy?
It helps you remove weak, noisy, or redundant features and focus on truly informative ones.

### How it works internally
SHAP assigns a contribution value to each feature for each prediction. It is based on game theory and gives a fair estimate of each feature’s effect.

### When to use it
Use it after training a strong model, especially XGBoost, LightGBM, or CatBoost.

### Advantages and disadvantages

| Advantage | Disadvantage |
|---|---|
| Very interpretable | Can be computationally expensive |
| Helps remove low-value features | Requires a trained model |

### How to implement in your project
After training XGBoost, compute SHAP values and drop features with very low contribution.

### Expected improvement
Usually 0–3 percentage points if the original feature set contains noise.

### Simple real-life example
If a model says “the weather” is the most important feature for predicting traffic, but you know the weather is not actually causing the issue, SHAP helps reveal the confusion.

---

## 3.2 Recursive Feature Elimination (RFE)

### What is it?
RFE repeatedly removes the weakest features and retrains the model.

### Why does it improve accuracy?
It removes weak or redundant features that can hurt generalization.

### How it works internally
The model is trained, the least important feature is removed, and the process repeats.

### When to use it
Use it when you have many features and want a smaller simpler set.

### Advantages and disadvantages

| Advantage | Disadvantage |
|---|---|
| Simple and effective | Can be slow |
| Reduces overfitting | May remove useful features if the model is unstable |

### How to implement in your project
Use RFE with a tree-based or logistic-regression base model and keep only the top 20–30 features.

### Expected improvement
Often 0–2 percentage points.

### Simple real-life example
If a toolbox has 100 tools but only 20 are used regularly, removing the rest makes the work easier.

---

## 3.3 Mutual Information

### What is it?
Mutual information measures how much information a feature gives about the target.

### Why does it improve accuracy?
It helps identify features that are genuinely relevant to the target.

### How it works internally
It measures the dependency between a feature and the target.

### When to use it
Use it to identify informative features before model training.

### Advantages and disadvantages

| Advantage | Disadvantage |
|---|---|
| Works well for nonlinear relationships | Not as interpretable as SHAP |

### How to implement in your project
Compute mutual information scores for each feature and retain the top-ranked ones.

### Expected improvement
Usually small but helpful, around 0–2 percentage points.

### Simple real-life example
If the feature “fever” carries a lot of information about infection, it should be kept.

---

## 3.4 Feature importance

### What is it?
Feature importance ranks features by how much they help the model make predictions.

### Why does it improve accuracy?
It helps you understand which variables matter and which ones are distracting.

### How it works internally
Tree-based models provide importances based on how often and how effectively a feature splits the data.

### When to use it
Use it after training tree models or ensembles.

### Advantages and disadvantages

| Advantage | Disadvantage |
|---|---|
| Fast and easy | Can be biased or unstable |

### How to implement in your project
Your code already saves feature importance from Random Forest. Keep using it, but compare it with SHAP for better reliability.

### Expected improvement
Mostly indirect, but it can help yield 0–2 percentage points when used to simplify and improve the feature set.

### Simple real-life example
If a model treats a random ID column as important, feature importance reveals that the feature is useless and should be removed.

---

## 3.5 Correlation analysis

### What is it?
Correlation measures how strongly two variables move together.

### Why does it improve accuracy?
Highly correlated features can add redundant information and make models less stable.

### How it works internally
It measures linear relationships between variables.

### When to use it
Use it for numeric variables to reduce redundancy.

### Advantages and disadvantages

| Advantage | Disadvantage |
|---|---|
| Simple | Only captures linear relationships |

### How to implement in your project
Inspect correlations among continuous features and remove redundant ones if needed.

### Expected improvement
Small, usually below 2 percentage points.

### Simple real-life example
If height and weight are both present and strongly correlated, adding both may not help much.

---

# 4. Hyperparameter tuning

## 4.1 Grid Search

### What is it?
Grid search tries a fixed set of hyperparameter combinations.

### Why does it improve accuracy?
The right hyperparameters can dramatically change model behavior.

### How it works internally
It evaluates all combinations from a predefined grid.

### When to use it
Use it when you want a thorough but manageable search.

### Advantages and disadvantages

| Advantage | Disadvantage |
|---|---|
| Simple and exhaustive | Can be slow |

### How to implement in your project
Use it for smaller models like logistic regression or KNN.

### Expected improvement
Often 1–5 percentage points.

---

## 4.2 Random Search

### What is it?
Random search samples hyperparameter values randomly.

### Why does it improve accuracy?
It searches a wider range more efficiently than grid search.

### How it works internally
It samples combinations rather than testing every grid point.

### When to use it
Use it for models with many hyperparameters like XGBoost or Random Forest.

### Advantages and disadvantages

| Advantage | Disadvantage |
|---|---|
| Faster than grid search | Less exhaustive |

### Expected improvement
Often 1–4 percentage points.

---

## 4.3 Bayesian Optimization

### What is it?
Bayesian optimization learns which hyperparameter regions are promising and focuses the search there.

### Why does it improve accuracy?
It finds strong hyperparameters faster than brute-force search.

### How it works internally
It builds a probabilistic model of performance and uses it to choose the next settings.

### When to use it
Use it for expensive models such as XGBoost, CatBoost, LightGBM, and neural networks.

### Advantages and disadvantages

| Advantage | Disadvantage |
|---|---|
| Efficient | More complex |

### Expected improvement
Often 2–6 percentage points if the search space is large.

---

## 4.4 Optuna

### What is it?
Optuna is a modern hyperparameter optimization framework.

### Why does it improve accuracy?
It gives you efficient, flexible, and automated tuning.

### How it works internally
It uses pruning, sampling, and history-based search to find strong settings quickly.

### When to use it
Use it for your XGBoost, LightGBM, CatBoost, and neural network tuning.

### Advantages and disadvantages

| Advantage | Disadvantage |
|---|---|
| Great efficiency | Slight setup complexity |

### How to implement in your project
Use Optuna to tune:

- `max_depth`
- `learning_rate`
- `n_estimators`
- `subsample`
- `colsample_bytree`
- `min_child_weight`
- `reg_lambda`
- `reg_alpha`

### Expected improvement
Often 2–6 percentage points.

---

## Important hyperparameters by model

### Random Forest
Important parameters:

- `n_estimators`: number of trees
- `max_depth`: how deep each tree can grow
- `min_samples_split`: minimum samples to split a node
- `min_samples_leaf`: minimum samples in a leaf
- `max_features`: features considered at each split

Why they matter:
- More trees usually help stability.
- Deeper trees can capture more detail but can overfit.
- Higher leaf size reduces overfitting.

### XGBoost
Important parameters:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `min_child_weight`
- `reg_lambda`
- `reg_alpha`
- `gamma`

Why they matter:
- `learning_rate` controls how quickly the model adapts.
- `subsample` and `colsample_bytree` reduce overfitting.
- Regularization parameters help generalize.

### LightGBM
Important parameters:

- `num_leaves`
- `max_depth`
- `learning_rate`
- `feature_fraction`
- `bagging_fraction`
- `min_data_in_leaf`

Why they matter:
- LightGBM is very powerful on tabular data and can be faster and more accurate than XGBoost on some datasets.

### CatBoost
Important parameters:

- `iterations`
- `depth`
- `learning_rate`
- `l2_leaf_reg`
- `border_count`

Why they matter:
- CatBoost handles categorical variables very well without too much preprocessing.

### SVM
Important parameters:

- `C`
- `kernel`
- `gamma`

Why they matter:
- `C` controls the penalty for errors.
- `gamma` controls the flexibility of the decision boundary.

### KNN
Important parameters:

- `n_neighbors`
- `weights`
- `metric`

Why they matter:
- More neighbors smooth predictions.
- Distance weighting often helps.

### Logistic Regression
Important parameters:

- `C`
- `solver`
- `penalty`

Why they matter:
- Lower `C` creates stronger regularization.

### Neural Networks
Important parameters:

- `hidden_layer_sizes`
- `learning_rate`
- `dropout`
- `batch_size`
- `epochs`
- `weight_decay`

Why they matter:
- Architecture controls capacity.
- Regularization reduces overfitting.

---

# 5. Handling class imbalance

## 5.1 SMOTE

### What is it?
SMOTE creates synthetic minority-class samples by interpolation between neighbors.

### Why does it improve accuracy?
Many models learn poorly when one class is rare. SMOTE balances the classes and improves the minority-class learning signal.

### How it works internally
It takes a minority-class point and creates a new point between it and one of its neighbors.

### When to use it
Use it when the positive class is rare, as in malnutrition prediction.

### Advantages and disadvantages

| Advantage | Disadvantage |
|---|---|
| Helps the model learn the minority class | Can create unrealistic synthetic examples |

### How to implement in your project
Your code already uses SMOTE. Keep it, but evaluate it with different `k_neighbors` values.

### Expected improvement
Often 1–5 percentage points.

---

## 5.2 Borderline-SMOTE

### What is it?
A version of SMOTE that focuses on minority samples near the decision boundary.

### Why does it improve accuracy?
These borderline cases often matter most for classification performance.

### How it works internally
It generates synthetic samples near the boundary where the classifier struggles most.

### When to use it
Use it when the minority class is hard to learn and the class boundary is important.

### Advantages and disadvantages

| Advantage | Disadvantage |
|---|---|
| Often better than vanilla SMOTE | Slightly more complex |

### Expected improvement
Often 0–3 percentage points over plain SMOTE.

---

## 5.3 ADASYN

### What is it?
ADASYN generates more synthetic samples for hard-to-learn minority samples.

### Why does it improve accuracy?
It focuses on the minority points that are hardest to classify.

### How it works internally
It generates more synthetic points in regions where the classifier struggles.

### When to use it
Use it if your minority class is very underrepresented and the model struggles on the boundary.

### Advantages and disadvantages

| Advantage | Disadvantage |
|---|---|
| Can help difficult minority classes | Can create noisy synthetic samples |

### Expected improvement
Often similar to or slightly better than SMOTE depending on the dataset.

---

## 5.4 SMOTEENN

### What is it?
A combination of SMOTE and Edited Nearest Neighbors.

### Why does it improve accuracy?
It both creates minority samples and removes noisy majority samples.

### How it works internally
It oversamples the minority class and then cleans the dataset by removing noisy examples.

### When to use it
Use it when the dataset is noisy and imbalanced.

### Advantages and disadvantages

| Advantage | Disadvantage |
|---|---|
| Strong noise reduction | Can remove useful examples |

### Expected improvement
Often 0–2 percentage points over SMOTE alone.

---

## 5.5 SMOTETomek

### What is it?
A combination of SMOTE and Tomek links.

### Why does it improve accuracy?
It balances the dataset while removing borderline noise.

### How it works internally
It creates synthetic minority samples and removes unclear near-boundary pairs.

### When to use it
Use it when you want a cleaner boundary.

### Expected improvement
Often 0–2 percentage points.

---

## 5.6 Class weights

### What is it?
Class weights tell the model to penalize mistakes on the minority class more strongly.

### Why does it improve accuracy?
It makes the model pay more attention to the rare class.

### How it works internally
The loss function is adjusted so that minority-class errors cost more.

### When to use it
Use it for logistic regression, SVM, random forest, and XGBoost.

### Expected improvement
Often 1–3 percentage points.

### Simple real-life example
If only 10% of children are malnourished, the model may ignore them. Class weights force the model to care more about the rare but important class.

---

# 6. Ensemble learning

## 6.1 Bagging

### What is it?
Bagging trains many models on different bootstrap samples and averages their predictions.

### Why does it improve accuracy?
It reduces variance and makes the model more stable.

### How it works internally
Random Forest is a bagging-based ensemble.

### When to use it
Use it for tabular data with many features.

### Advantages and disadvantages

| Advantage | Disadvantage |
|---|---|
| Reduces overfitting | Can be computationally heavy |

### Expected improvement
Often good baseline improvement, especially for tabular data.

---

## 6.2 Boosting

### What is it?
Boosting trains models sequentially, each correcting the previous mistakes.

### Why does it improve accuracy?
It learns complex patterns well and often gives very strong results on tabular data.

### How it works internally
XGBoost, LightGBM, and CatBoost are boosting methods.

### When to use it
Use it for structured tabular data such as your DHS dataset.

### Advantages and disadvantages

| Advantage | Disadvantage |
|---|---|
| Often best for tabular data | Can overfit if not tuned carefully |

### Expected improvement
Often the most impactful approach for your project.

---

## 6.3 Voting

### What is it?
Voting combines predictions from multiple models.

### Why does it improve accuracy?
It can reduce the errors of any single model.

### How it works internally
It averages or majority-votes the outputs of several models.

### When to use it
Use it when you have several models with different strengths.

### Expected improvement
Often 0–3 percentage points.

---

## 6.4 Stacking

### What is it?
Stacking trains a meta-model on the predictions of multiple base models.

### Why does it improve accuracy?
It learns how to combine the strengths of several models.

### How it works internally
Base models produce predictions; a second-level model learns the best combination.

### When to use it
Use it when you have strong base learners and want the best possible performance.

### Advantages and disadvantages

| Advantage | Disadvantage |
|---|---|
| Often very strong | More complex and prone to leakage |

### Expected improvement
Often 1–5 percentage points when done carefully.

---

## 6.5 Blending

### What is it?
Blending is a simpler form of stacking where a weighted average is used instead of training a meta-model.

### Why does it improve accuracy?
It combines complementary strengths without much complexity.

### When to use it
Use it as a simple first ensemble approach.

### Expected improvement
Often 0–2 percentage points.

---

## Best choice for tabular healthcare datasets
For your project, the best choices are:

1. Boosting models such as XGBoost, LightGBM, or CatBoost.
2. A simple stacking ensemble of 2–4 strong models.
3. A voting ensemble as a baseline.

Why? Because tabular healthcare data usually benefits from gradient-boosted trees more than deep learning, especially when sample size is moderate and features are structured.

---

# 7. Model optimization

## 7.1 Early stopping

### What is it?
Training stops when validation performance stops improving.

### Why does it improve accuracy?
It prevents overfitting.

### How it works internally
The model is monitored on a validation set and training is stopped once performance stops improving.

### When to use it
Use it for neural networks and boosting models with validation data.

### Expected improvement
Often 1–3 percentage points.

---

## 7.2 Cross-validation

### What is it?
Cross-validation evaluates the model on multiple train-test splits.

### Why does it improve accuracy?
It gives a more reliable estimate of generalization performance.

### How it works internally
The dataset is split into folds and each fold serves as a validation set once.

### When to use it
Use it always for fair model evaluation.

### Expected improvement
Mainly improves trust in the result, not raw accuracy directly.

---

## 7.3 Threshold optimization

### What is it?
Instead of using the default threshold of 0.5 for class predictions, you choose the threshold that gives the best F1 or recall.

### Why does it improve accuracy?
A different threshold can greatly improve the balance between precision and recall.

### How it works internally
You try many thresholds and pick the one that maximizes a chosen metric.

### When to use it
Use it for imbalanced classification tasks.

### Expected improvement
Often 1–4 percentage points in practical decision performance.

### Simple real-life example
In medicine, a slightly lower threshold may catch more true cases, even if it results in a few more false alarms.

---

## 7.4 Calibration

### What is it?
Calibration makes predicted probabilities match real probabilities.

### Why does it improve accuracy?
A model with good accuracy can still produce unreliable probabilities.

### How it works internally
Methods like Platt scaling or isotonic regression adjust the predicted probabilities.

### When to use it
Use it when you care about probability quality, not just classes.

### Expected improvement
Mostly improves probability quality, not necessarily raw accuracy.

---

## 7.5 Probability threshold selection

### What is it?
Choosing the best probability cutoff for the positive class.

### Why does it improve accuracy?
Different applications need different trade-offs between recall and precision.

### How it works internally
You test thresholds on validation data and select the best one.

### When to use it
Use it for malnutrition screening, where false negatives may be more harmful than false positives.

### Expected improvement
Often 1–3 percentage points in the metric that matters.

---

# 8. Advanced machine learning models

## 8.1 CatBoost

### Why it often outperforms traditional models
CatBoost handles categorical variables very well and often needs less preprocessing than XGBoost.

### Best for
Tabular datasets with many categorical features.

### In your project
It may work very well because you have variables such as `education`, `residence`, `water_source`, and `cooking_fuel`.

---

## 8.2 LightGBM

### Why it often outperforms traditional models
LightGBM is very efficient and often reaches strong performance on tabular data with less tuning than many other models.

### Best for
Large tabular datasets or datasets with many features.

### In your project
A strong candidate after XGBoost.

---

## 8.3 TabNet

### Why it often outperforms traditional models
TabNet is a neural network designed specifically for tabular data. It uses attention-like mechanisms and can learn useful sparse feature selection.

### Best for
Structured tabular data where deep learning can help.

### In your project
Worth trying if you have enough data and compute resources.

---

## 8.4 TabTransformer

### Why it often outperforms traditional models
TabTransformer uses transformers to learn relationships between tabular features and can capture complex interactions.

### Best for
High-dimensional tabular datasets with strong interactions.

### In your project
Promising but may need careful tuning and more data than you currently have.

---

# 9. Deep learning improvements

## 9.1 Better neural network architectures

### What is it?
Use deeper or more suitable network structures.

### Why does it improve accuracy?
A better architecture can capture more complex patterns.

### How it works internally
Different layers learn different representations.

### When to use it
Use it when the data are complex and non-linear.

### Expected improvement
Potentially significant but depends heavily on the dataset.

---

## 9.2 Batch normalization

### What is it?
Normalizes outputs from one layer before passing them to the next.

### Why does it improve accuracy?
It stabilizes training and speeds convergence.

### When to use it
Useful in neural networks.

---

## 9.3 Dropout

### What is it?
Randomly drops units during training.

### Why does it improve accuracy?
It reduces overfitting and forces the network to learn more robust patterns.

---

## 9.4 Learning rate scheduling

### What is it?
Changes the learning rate during training.

### Why does it improve accuracy?
It helps the model converge more smoothly.

---

## 9.5 Regularization

### What is it?
Adds penalties to discourage large or unstable weights.

### Why does it improve accuracy?
It reduces overfitting.

---

## 9.6 Optimizers

### What is it?
Optimizers control how the neural network updates its weights.

### Common choices
- Adam: fast and robust
- AdamW: better regularization behavior
- SGD: can work well with good scheduling

### Expected improvement
Often helpful in deep learning, but less important than feature quality and tree-based modeling for your current dataset.

---

## 9.7 Activation functions

### What is it?
Activation functions decide whether a neuron should fire.

### Common choices
- ReLU: common default
- Leaky ReLU: helps with dead neurons
- GELU: often strong in modern architectures

### Why it matters
Some activation functions help optimization and learning stability.

---

# 10. Explainability

## 10.1 SHAP

SHAP is one of the best tools for explaining tree-based models. It gives insight into which features drive the prediction.

### Why it helps improve the model
It helps you find:

- weak features to remove
- misbehaving features
- surprising relationships
- possible leakage

### How to use it
Use SHAP after fitting XGBoost, LightGBM, or CatBoost.

---

## 10.2 LIME

### What is it?
LIME explains a prediction by approximating the model locally with a simpler interpretable model.

### Why it helps
It is useful for explaining single predictions.

### Best use case
When you want to understand why a particular child was flagged as high risk.

---

## 10.3 Feature importance

### Why it helps
It tells you whether the model is using sensible evidence.

### How it helps improve the model
If the model relies on a feature that should not matter, you may have a data leakage or an engineering issue.

---

# Ranking the techniques by likely impact for your project

Here is a practical ranking from highest to lowest impact for your NFHS-5 malnutrition prediction task.

| Rank | Technique | Likely impact |
|---|---|---|
| 1 | Better feature engineering and domain-specific features | Very high |
| 2 | Stronger gradient-boosted models (CatBoost/LightGBM/XGBoost) with Optuna tuning | Very high |
| 3 | Class imbalance handling with class weights and/or SMOTE variants | High |
| 4 | Threshold optimization and probability calibration | High |
| 5 | Missing value and outlier handling | High |
| 6 | Ensemble learning (stacking/voting) | High |
| 7 | Feature selection with SHAP and mutual information | Medium-high |
| 8 | Cross-validation and careful evaluation | Medium |
| 9 | Normalization/standardization | Medium |
| 10 | Polynomial features and interaction terms | Medium |
| 11 | Deep learning models such as TabNet/TabTransformer | Medium to low for this dataset unless you have much more data |
| 12 | Advanced explainability tools | Medium, mostly for interpretation and debugging |

---

# Step-by-step roadmap to improve your current model

## Phase 1: Make the data cleaner (1–3 days)

1. Review missing values carefully.
2. Add missingness indicators.
3. Winsorize extreme values.
4. Remove or correct obvious duplicates.
5. Compare StandardScaler vs RobustScaler.

What you should expect:
- Small to medium gains.
- Better stability.

## Phase 2: Improve the feature set (2–4 days)

1. Add more domain-specific features.
2. Add interaction terms.
3. Add polynomial terms only where they make sense.
4. Use SHAP and mutual information to remove noise.

What you should expect:
- Often the biggest gains after model choice.

## Phase 3: Tune the best tree-based models (3–5 days)

1. Tune XGBoost with Optuna.
2. Try LightGBM.
3. Try CatBoost.
4. Compare them with the same cross-validation setup.

What you should expect:
- Usually the biggest single gain after data quality and engineering.

## Phase 4: Improve the class balance handling (1–2 days)

1. Compare SMOTE, Borderline-SMOTE, ADASYN, and class weights.
2. Select the best one for your data.
3. Tune the threshold for recall and F1.

What you should expect:
- Better performance on the minority class.

## Phase 5: Build stronger ensembles (1–2 days)

1. Create a voting ensemble of the best three models.
2. Try a simple stacking ensemble.
3. Compare the ensemble to the single best model.

What you should expect:
- Useful improvement if the models make different errors.

## Phase 6: Try advanced tabular models (2–4 days)

1. Train CatBoost and LightGBM first.
2. Try TabNet or TabTransformer if you have enough compute and data.
3. Keep them only if they beat the gradient-boosted tree baselines.

What you should expect:
- Sometimes strong, but not always better on a modest health dataset.

## Phase 7: Check explanation and error analysis (1 day)

1. Use SHAP to inspect predictions.
2. Look for data leakage or poor feature quality.
3. Analyze false positives and false negatives.

What you should expect:
- Better understanding and sometimes better model choices.

---

# Practical recommendation for your project

If I were improving your model today, I would follow this order:

1. Improve preprocessing and missingness handling.
2. Add more domain-specific engineered features.
3. Tune XGBoost first with Optuna.
4. Compare LightGBM and CatBoost.
5. Apply class weights and threshold tuning.
6. Build a voting or stacking ensemble.
7. Use SHAP for feature selection and debugging.

The most likely path to a meaningful improvement is:

- better features
- better tuning
- better imbalance handling
- better threshold selection

---

# Realistic expectation for your project

Because this is a DHS-based health prediction task with a modest feature set and a noisy target, a strong realistic target is:

- 58% → 65–72% with careful data cleaning, feature engineering, and tuning
- 72–78% is possible if the data quality and features are strong
- 80% is possible only in a best-case scenario and may require richer data or a more specialized setup

So the improvement is very possible, but the path should be methodical rather than speculative.

---

# How much time will this take?

A practical plan is:

- 2–3 days for preprocessing and feature engineering
- 3–5 days for tuning XGBoost, LightGBM, and CatBoost
- 1–2 days for class imbalance and threshold tuning
- 1–2 days for ensemble building and explainability

In total, a serious improvement cycle is usually about 1–2 weeks for a solo researcher working carefully.

If you want, I can next turn this guide into a concrete implementation plan for your repository by proposing the exact code changes to add in [src/01_prepare_data.py](src/01_prepare_data.py) and [src/02_run_models.py](src/02_run_models.py).
