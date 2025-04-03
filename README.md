The objective of this project is to evaluate the performance of various classification algorithms, specifically:

* K Nearest Neighbor
* Logistic Regression
* Decision Trees
* Support Vector Machines

The comparison will involve analysing the training duration and accuracy of each model. This is to identify which model is most effective at predicting customer acceptance of a long-term deposit bank product through a telephone-based marketing campaign. Additionally, insights into the computational efficiency and predictive capabilities of each classifier will be gathered to provide a comprehensive assessment.

Furthermore, the project will also consider factors such as precision and recall, along with the F1 score for each model to better understand their performance in terms of handling false positives and false negatives. Such metrics are critical in ensuring the reliability of the predictions made by the classifiers. The results from this study will contribute to optimizing marketing strategies, ensuring that the bank can target the right customers more effectively and increase the success rate of their marketing campaigns.

## Data Preparation Steps

Given the imbalanced nature of the dataset, several steps were taken to prepare the data for analysis:

The "Y" feature was renamed to "deposit" for better clarity and understanding.
Features 1 - 7 (which include job, marital status, education, default status, housing, loan, and contact method) were selected to create a feature set.
To tailor data preparation steps for different types of data, ColumnTransformer was used. This tool allows specific transformations or sequences of transformations to be applied exclusively to numerical columns and others solely to categorical columns.
The LabelEncoder was utilized to encode the labels in the target column "deposit".
Finally, once the data was prepared, it was split into a training set and a test set using the train_test_split function. The test set was composed of 30% of the data, ensuring sufficient data for training and accurate evaluation of model performance.

## Baseline Model Comparison Overview

For the initial baseline model, a DecisionTreeClassifier was chosen. This classifier is capable of conducting multi-class classification on a dataset, using various feature subsets and decision rules at different classification stages.

This model is set to be evaluated against the Logistic Regression model, which is utilized to model the relationship between a dependent variable and one or more independent variables.

Logistic Regression in machine learning is quite intriguing as it often excels in scenarios where a Decision Tree might falter, particularly when extensive time and expertise are available. However, a notable drawback of Decision Trees is the high demand for sample sizes, making them resource-intensive.

During the training, fitting, and prediction phases with both models on the dataset, the following metrics were recorded for each model:

| Model Name  	        | Accuracy                              | Precision	                    | Recall 	                | F1_Score                  | Fit Time (ms) 
|-------------	        |:------------------------------------	|:-------------------------:	|:----------------------:	|:----------------------:	|:----------------------:	|
| Decision Tree       	| 0.887513                              | 0.443792                  	| 0.499954                  |  0.470202                 | 128                       |
| Logistic Regression   | 0.887594                              | 0.443797                     	| 0.500000                  |  0.470225                 | 193                       |
|             	        |                                      	|                           	|                        	|                           |                           |

This comparison aims to shed light on the strengths and limitations of each model when applied to the specific dataset.

## Comparative Analysis of Models

In this comparison, we assess the performance differences across four models: Logistic Regression, K Nearest Neighbor (KNN), Decision Tree, and Support Vector Machine (SVM). Each model was configured with default settings, then fit to the dataset and evaluated in terms of fit time and accuracy.

The following table presents the training time, training accuracy, and test accuracy for each of the models:

| Model Name        	| Train Time (s)                      | Train Accuracy                | Test Accuracy 	                | 
|-------------------	|:---------------------------	|:---------------------:	|:----------------------:	|
| Logistic Regression   | 0.609                         | 0.8872047448926502        | 0.8875940762320952                 |  
| KNN                   | 1min 38                          | 0.8846033783080711        | 0.8807963097839281                  |  
| Decision Tree	        | 0.333                         | 0.8911935069890049        | 0.884761673545359                 |  
| SVM                   | 26.2                          | 0.8873087995560335        | 0.8875131504410455                 |  
|                       |                               |                           |                        	| 

Upon examining these results, it is apparent that Logistic Regression performs optimally compared to the other models. It records the shortest training time, as well as the highest scores for both training and testing accuracy. This comparison highlights the efficiency and effectiveness of Logistic Regression in handling this specific dataset under the conditions tested.
