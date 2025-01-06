
# Telecom Churn Analysis

Justin Lee

This notebook is prepared for SyriaTel, a leading mobile telecommunications company in Syria since 2000. The company aims to minimize revenue losses caused by customers who discontinue their services prematurely. This analysis seeks to identify and understand patterns that may predict customer churn.

## Data Understanding
This dataset is from SyriaTel and tells the churn rate of customers across. This dataset contains 20 predictor variables mostly about customer usage patterns. There are 3333 records in this dataset, out of which 483 customers are churners and the remaining 2850 are non-churners. The ratio of churners in this dataset is 14%.

An important initial limitation to note here is the class imbalance of churn/non-churners.
## Data Preparation
In order to prepare our data for analysis, we must first import the train_test_split class to evaluate our model on unseen data. We will declare churn as our target variable and split the data into 70% training and 30% for testing. We will then ensure the numeric columns exclude our churn column to avoid data leakage, create a clear and unbiased correlation analysis, and to create more accurate modeling and evaluation.
## Modeling
For our first model we will begin with logistic regression because it is simple, interpretable, efficient, and provides a reference point for model comparison. It sets the foundation for exploring more advanced models if additional complexity is justified by improved performance.

To address the class imbalance noted earlier, we will use SMOTE (synthetic minority over-sampling technique). SMOTE is used in our iterated model to handle class imbalance, which improves the model's ability to detect the minority class (churn) and results in a more balanced and effective classifier.

Both models will have an evaluation on the classification report, confusion matrix, ROC-AUC score, and a visualization of the ROC curve. The classification report shows us the precision, recall and F1-score for overall evaluation. The confusion matrix shows us the detailed error breakdown. The ROC-AUC score measures our classification power. The ROC curve is a visual representation of performance across thresholds.
## Baseline Model Evaluation
Our baseline model predicted an accuracy score of 86% of the total samples but this may be misleading due to imbalanced datasets. This score may be favoring the majority class because we saw that we have 857 samples for non-churn cases and only 143 samples for churn cases.

Non-churn cases: A precision score of 87% means that out of all predictions made for non-churn, 87% were correct. A recall score of 99% means that our model correctly identified 99% of actual non-churn cases. An F1-score of 0.92 indicates excellent balance between precision and recall for the majority class.

Churn cases: A precision score of 58% means that out of all predictions made for churn, only 58% were correct. A recall score of 8% means that our model correctly identified only 8% of actual churn cases. An F1-score of 0.14 indicates a poor performance for the minority class as both precision and recall are low.

The confusion matrix provides additional insight: True Negatives (849): The model correctly predicted 849 non-churn instances. False Positives (8): Only 8 non-churn instances were incorrectly predicted as churn. False Negatives (132): A large number of churn cases were misclassified as non-churn. True Positives (11): Only 11 churn cases were correctly predicted. This highlights the model's strong bias toward the majority class.

ROC-AUC Score: 0.78 This indicates the model has a decent ability to rank positive samples higher than negative ones in terms of predicted probabilities. However, the low recall for churn cases suggests that the threshold for predicting churn may need adjustment.

We will now iterate our model to oversample the minority class, thus reducing the class imbalance observed in our baseline model.
## Iterated Model Evaluation
Our new model's accuracy is 70%. The overall accuracy decreased from the baseline model's 86%, but this is expected because the model is now focusing more on the minority class (churn). Accuracy is less important in this case, as we prioritize recall for churn.

Non-churn cases: A precision score of 94% means that out of all predictions made for non-churn, 94% were correct. A recall score of 69% means that our model correctly identified 69% of actual non-churn cases. An F1-score of 0.8 indicates that despite a drop in recall the model still performs well for the majority class.

Churn cases: A precision score of 29% means that out of all predictions made for churn, 29% were correct. This indicates a high false positive rate. A recall score of 76% tells us that the model correctly identified 76% of actual churn cases, a significant improvement over the baseline model's 8% recall. An F1-score of 0.42 shows an improvement over the baseline model's F1-score of 0.14 for churn cases. This indicates better balance between precision and recall.

The confusion matrix provides more insights. True Negatives (595): The model correctly identified 595 non-churn cases. False Positives (262): The model incorrectly predicted 262 non-churn cases as churn, contributing to the drop in precision for non-churn cases. False Negatives (35): The model missed only 35 churn cases, a significant improvement over the baseline (132 false negatives). True Positives (108): The model correctly identified 108 churn cases, up from just 11 in the baseline model. This reflects a clear improvement in identifying churn cases, which aligns with the primary business objective.

ROC-AUC Score: 0.778 This is comparable to the baseline (0.782), showing that the model's overall ability to distinguish between churn and non-churn remains strong despite the adjustments for class imbalance.
## Conclusion
Our baseline model had high accuracy of 86% but extremely poor recall of 8% for the minority class (churn). There was a strong bias toward the majority class (non-churn), leading to most churn cases being misclassified. Indicators of churn (international plans, customer service calls, total day minutes) were not effectively captured in predictions.

Addressing class imbalance with SMOTE significantly improved recall for churn (from 8% to 76%), aligning better with the business goal of identifying churn cases. However, the precision for churn dropped to 29%, meaning many non-churn cases were misclassified as churn (false positives). The ROC-AUC remained consistent across models (~0.78), suggesting that the overall separability of classes was maintained.

From the logistic regression coefficients and feature analysis there are some factors that can predict higher churn probabilities. Customers with international plans are more likely to churn. This could indiciate dissatisfaction with plan costs or perceived lack of value. A high number of customer service calls correlate with churn, likely reflecting customer dissatisfaction or unresolved issues. Additionally, customers with higher call usage (total day minutes) are more likely to churn. This could be due to higher bills or dissatisfaction with pricing.

This analysis highlights that churn is influenced by clear patterns, such as having an international plan or high customer service interactions. While the SMOTE-enhanced model improved recall for churn cases, further iterations (e.g., advanced models, threshold optimization) are needed to enhance precision and overall effectiveness. Combining these insights with retention strategies can significantly reduce churn rates and improve customer satisfaction.
## Limitations & Next Steps
The dataset's inherent imbalance limits the model's ability to learn patterns for churn without intervention like SMOTE or class weighting.

The iterated model prioritized recall for churn at the expense of precision, leading to high false positives. This could result in inefficient resource allocation for retention campaigns.

The models assumed linear relationships between features and churn. Non-linear models might capture more complex patterns.

Features like tenure, customer lifetime value, or qualitative feedback were not included in the dataset and could provide additional predictive power.

Logistic Regression uses a fixed decision threshold (default = 0.5). This threshold may not align with the business priority of minimizing false negatives (missed churn cases).

For next steps, I'd be interested testing on more advanced models like Random Forest or Gradient Boosting, which handle imbalanced datasets better and can capture non-linear relationships.

I'd also be interested in adjusting the decision threshold. We could lower the classification threshold to 0.3-0.4 to achieve a better precision-recall balance for churn cases.

To improve data collection we could include features like tenure, average monthly charges, customer satisfaction ratings, or billing disputes to better capture factors driving churn.

Lastly, we could deploy the model in a live environment to predict churn and measure the effectiveness of interventions.

## Sources
[Presentation]([url](https://docs.google.com/presentation/d/1E7GlQ2_4VpIh27G2hbc_jzbMn7FHRnq3aiZCbCcFMA4/edit#slide=id.g3224e56f0ab_0_604))
[Data Source]([url](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset/data))
