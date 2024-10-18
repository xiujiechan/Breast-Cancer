Breast Cancer Diagnosis with Machine Learning

Description:

This project aims to develop a machine learning model to classify breast cancer as malignant or benign based on various diagnostic measures. The dataset used is the Breast Cancer Wisconsin dataset.

Dataset:

The dataset contains features derived from a digitized image of a fine needle aspirate (FNA) of a breast mass. It includes characteristics of the cell nuclei present in the image. Here are the key attributes:

  ·ID: Unique identifier for each sample.

  ·Target: Diagnosis result (M = malignant, B = benign).

  ·Features: Various measurements 

Dependencies:

  ·pandas

  ·numpy

  ·scikit-learn

  ·matplotlib

  ·Flask 

Usage

1.Data Preprocessing:

Load data: train = pd.read_csv("path/to/wdbc.data", header=None)

Set column names and preprocess.

2.Model Training:

Standardize and split data.

Train a model using algorithms like Random Forest, SVM, etc.

3.Model Evaluation:

Validate the model using metrics like accuracy, precision, recall, ROC-AUC.

Plot the ROC curve for visual evaluation.

Installation

git clone https://github.com/xiujiechan/Breast-Cancer
cd Breast-Cancer
pip install -r requirements.txt

Run the Application
python app.py

Results
Detailed results of model performance, including confusion matrix, classification report, and visual plots.

Contributing
Feel free to fork the repository, raise issues, and submit pull requests.
