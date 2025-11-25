Team Member A – Data Processing and Feature Engineering
Day 1: Data Exploration and Quality Assessment

Member A begins by exploring the dataset in detail.
This includes checking for missing values, outliers, duplicated records, and inconsistent timestamps.
The fraud versus non-fraud distribution is reviewed, and all 141 features are analyzed and grouped.
The goal is to produce a clear data quality report and identify the necessary preprocessing steps.

Day 2: Data Cleaning and Preprocessing

Member A builds the preprocessing pipeline.
Categorical features are encoded using target encoding or frequency encoding.
Numerical features are standardized or normalized.
Missing values are handled, and time-related fields are cleaned and formatted.
The output is a working preprocessing pipeline ready for feature engineering.

Day 3: Feature Engineering

Additional features are created to enhance predictive power.
Examples include transaction counts over recent periods, average transaction amounts, previous-month issue counts, deviation from historical spending patterns, and time-based features such as hour or weekday indicators.
The enriched dataset becomes ready for the modeling phase.

Day 4: Final Dataset Preparation

The dataset is finalized and split into training, validation, and test sets.
Imbalanced data is handled using class weight adjustment or basic oversampling.
Member A documents all features and delivers the complete dataset to Member B.

Team Member B – Model Development and Evaluation
Day 1: Baseline Model Development

Member B creates baseline models using a preliminary dataset.
Models such as Logistic Regression, Random Forest, and XGBoost are trained with default settings.
Performance is evaluated using AUC, Precision, Recall for fraud, and F1 score.
A candidate model is selected for further tuning.

Day 2: Imbalance Handling and Initial Tuning

Different strategies for imbalanced data are applied, including oversampling, undersampling, or adjusting class weights.
Classification threshold tuning is performed to increase fraud recall.
XGBoost or similar models are tested with small adjustments to key parameters such as depth, learning rate, and estimator count.

Day 3: Advanced Model Optimization

Hyperparameter tuning is expanded using GridSearch or Optuna.
If temporal order exists in the data, time-based cross-validation is used.
The best model—often XGBoost or LightGBM—is selected based on stable validation performance.

Day 4: Model Interpretation and Documentation

Model explanations are produced using SHAP.
Important features influencing fraud predictions are summarized.
Clear documentation is prepared for both technical and business audiences,
including a description of model behavior and deployment recommendations.

Final Deliverables After 6 Working Days

A complete preprocessing pipeline produced by Member A.

A finalized dataset with engineered features.

An optimized and validated fraud detection model built by Member B.

SHAP-based interpretability analysis and feature importance insights.

A technical report summarizing the model, workflow, and deployment recommendations.

A plan for monitoring and improving the model over time.

Two Reserved Buffer Days

These days are not assigned to specific tasks.
They can be used to resolve unexpected data issues, run additional experiments, fine-tune the model further, or complete extended documentation depending on project needs.
