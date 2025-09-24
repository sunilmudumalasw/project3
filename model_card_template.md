# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model used is a Random Forest Classifier.
## Intended Use
Intended use is to predict the salary of a person given some socio economic details.
## Training Data
The data used for the model training is the Census Income retrieved from the UC Irvine Machine Learning Repository.
## Evaluation Data
The leading and trailing spaces were removed, the data was split in an 80-20 ratio for train and test.

## Metrics
The metrics used for the evaluation of the model are: Precision, Recall & Fbeta.

## Ethical Considerations
Gender, race, native country are present at the dataset and this could lead to some sort of discrimination.
## Caveats and Recommendations
Data is based on USA, baised to USA.