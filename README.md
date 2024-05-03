# Diabetes Prediction Using Support Vector Machine (SVM)

This project aims to predict diabetes in individuals using the PIMA diabetes dataset and a Support Vector Machine (SVM) classifier. Diabetes prediction is crucial for early diagnosis and intervention, thereby potentially improving patient outcomes.

#  Technologies Used

Python: The entire project is implemented in Python, leveraging its robust libraries for data manipulation, analysis, and machine learning.

NumPy: Utilized for creating and manipulating numerical arrays, essential for data preprocessing.

Pandas: Employed for data manipulation and analysis, particularly for handling structured data in the form of data frames.

scikit-learn (sklearn):

StandardScaler: Used for standardizing the dataset to ensure consistent scale across features.

train_test_split: Utilized for splitting the dataset into training and testing sets to evaluate model performance.

svm.SVC: Implemented the Support Vector Classifier for building the predictive model.

accuracy_score: Utilized to evaluate the accuracy of the trained model predictions.

Matplotlib/Seaborn: While not explicitly mentioned in the code, these libraries can be useful for visualizing data distributions and model performance metrics.

#  Project Overview

1)Data Collection and Analysis:
-The PIMA diabetes dataset is loaded into a pandas data frame for analysis.

-Statistical measures of the dataset are obtained using describe() function.

-Data is explored by examining the distribution of outcome labels and feature means grouped by outcome.

-The dataset is divided into features (X) and labels (Y), followed by standardization of features using StandardScaler.
2)Model Training and Evaluation:
-The dataset is split into training and testing sets using train_test_split.

-An SVM classifier with a linear kernel is trained on the training data.

-Model performance is evaluated using accuracy scores on both training and testing sets.

3)Prediction System:

-A simple predictive system is implemented to predict diabetes status for new input data.

-Input data is standardized using the same scaler used for training.

-The trained SVM model predicts the diabetes status for the input data.

-A human-readable interpretation of the prediction result is provided.

#  Outcome

The project demonstrates the application of SVM for diabetes prediction, achieving a certain level of accuracy on both training and testing data.
By standardizing the input data and training an SVM model, the project provides a basis for predicting diabetes status for new individuals based on their health parameters.
