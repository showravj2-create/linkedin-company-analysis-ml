# LinkedIn Company Analytics: Predictive Modeling and Unsupervised Learning

## Overview

This project investigates whether publicly available company and LinkedIn attributes can explain differences in organizational size and LinkedIn presence.

The analysis combines:

- exploratory data analysis
- feature engineering
- statistical correlation analysis
- supervised machine learning
- unsupervised clustering
- basic natural language processing
- model diagnostics and critical evaluation

Rather than treating model accuracy as the only objective, the project emphasizes **method selection, reproducibility, comparison with baselines, and interpretation of limitations**.

## Research Questions

1. How are LinkedIn follower counts distributed across companies?
2. Which observable company characteristics are associated with follower counts?
3. How predictable is company size from LinkedIn followers and employee representation?
4. Can companies be grouped into meaningful clusters according to LinkedIn presence?
5. What information can be extracted from company specialties and slogans?

## Dataset

The dataset contains approximately 1,000 LinkedIn company profiles with organizational, geographic, funding, and textual attributes.

Key variables include:

- LinkedIn followers
- employees represented on LinkedIn
- company size
- founding year
- industry
- location
- funding information
- investors
- company description
- specialties
- slogan

## Methodology

### 1. Data preparation

- Duplicate records are removed.
- Numerical variables are converted to appropriate numeric types.
- Missing values are handled explicitly through model pipelines rather than treating missing numerical values as zero.
- Company age is derived from founding year.
- Follower counts are transformed using `log1p` to reduce extreme skew.

### 2. Exploratory analysis

The notebook examines:

- follower distributions
- company-size distributions
- geographic patterns
- Pearson and Spearman correlations
- relationships among company age, employees, and followers

### 3. Follower prediction

Follower count is modeled on the logarithmic scale.

Models:

- median baseline
- Linear Regression
- Random Forest Regressor

Metrics:

- MAE
- RMSE
- R²

A residual analysis is also performed for the best non-baseline model.

### 4. Company-size classification

Company size is simplified into reproducible categories.

Models:

- Logistic Regression
- Random Forest Classifier

Metrics:

- accuracy
- precision
- recall
- F1-score
- confusion matrix

A stratified train/test split is used to preserve class proportions.

### 5. Unsupervised clustering

K-Means is applied to log-transformed follower and employee counts.

The number of clusters is not chosen arbitrarily. Candidate values of `k` are compared using silhouette scores before fitting the final model.

### 6. NLP exploration

Company specialties are explored using a WordCloud.

Company slogans are analyzed using TextBlob polarity as a lightweight sentiment baseline.

These NLP analyses are explicitly treated as exploratory because short marketing text can be highly context-dependent.

## Key Research Lessons

The project demonstrates several important modeling considerations:

- Highly skewed social-media metrics can make raw-scale regression difficult.
- A simple baseline is useful for judging whether a more complex model adds value.
- Classification performance should not be summarized by accuracy alone.
- Clustering requires justification of the number of clusters.
- Predictive relationships should not be interpreted as causal relationships.
- Simple NLP techniques can provide useful exploratory signals but have important semantic limitations.

## Limitations

The dataset represents a limited observational snapshot of LinkedIn companies. Follower and employee counts are not controlled measurements, and the available features do not capture all factors that influence company visibility or organizational size.

The project therefore focuses on **association and prediction**, not causal inference.

## Future Research Directions

Potential extensions include:

- gradient boosting and XGBoost
- cross-validation and hyperparameter optimization
- SHAP-based model interpretation
- robust and quantile regression
- TF-IDF and transformer-based text representations
- temporal collection of company profiles
- richer analysis of funding and organizational growth
- causal or quasi-experimental analysis where suitable data become available

## Repository Structure

```text
LinkedIn-Data-analysis-and-prediction-model/
├── data/
│   └── LinkedIn-company-info.csv
├── notebooks/
│   └── linkedin_company_analysis_research.ipynb
├── figures/
├── README.md
└── requirements.txt
```

## Reproducibility

```bash
pip install -r requirements.txt
```

Then open the notebook and update `DATA_PATH` if the dataset is stored in a different location.

## Author

**Showrav Das**

BSc in Mathematics | Python | Machine Learning | Data Analysis

GitHub: https://github.com/showravj2-create
