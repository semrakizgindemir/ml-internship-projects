# Applied Machine Learning Internship Projects

A collection of end-to-end machine learning studies built around real-world tabular and image data. The repository explores several problem types rather than presenting a single model: regression, anomaly detection, association-rule mining, and image classification.

## Projects

### Turkish market sales forecasting

- Trains and compares regression models for sales prediction.
- Uses an AutoML workflow and saves the best-performing model and forecast visualizations.
- Focuses on reproducible model selection and result reporting.

### Customer anomaly detection

- Compares Isolation Forest, Local Outlier Factor, and DBSCAN.
- Analyzes 49,316 customers and reports both individual-model and consensus anomalies.
- Exports scored customer records and a compact performance summary.

### Association-rule mining

- Discovers frequently co-occurring products and evaluates rules with support, confidence, and lift.
- Produces scatter plots, ranked rule charts, a network graph, and a lift heatmap.

### Calorie prediction

- Compares linear and tree-based regression approaches.
- Uses cross-validation and hyperparameter tuning.
- The best recorded Gradient Boosting model achieved a test R² of approximately **0.95**.

### Food-101 image classification

- Contains early experiments for food image classification and model training.

## Repository structure

```text
Turkish_Market_Sales_Dataset_regression/
food2/
turkish_market_sales_anomali_tespiti/
turkish_market_sales_association_rule_mining/
turkish_market_sales_kalori_tahmini_regresyon/
```

## Tools and concepts

Python · Jupyter Notebook · pandas · scikit-learn · H2O AutoML · regression · anomaly detection · clustering · association rules · computer vision

> This repository documents an active learning and internship process. Each folder contains its own notebook, generated outputs, or evaluation artifacts.
