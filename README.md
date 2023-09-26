# WaterPotabilityClassifier
Notebook of EDA and ML Model trained on Water Potability Dataset

### PROBLEM STATEMENT
Given the Water Potability Dataset perform EDA and build a machine learning model to 
predict potability. Identify key variables responsible for water potability and create an API 
for inference.

### APPROACH
1. Understanding the Data:
First, I loaded the dataset into a Pandas Data Frame. Then I ran some basic data analysis 
functions provided by Pandas to get some basic insights about the data such as the 
number of rows & columns, data types, duplications, null values etc.

3. EDA (Univariate & Bivariate):
For detailed EDA, I followed a two-step process. First, I performed Univariate Analysis in 
which I inspected each variable separately and its spread. I found that almost all the 
variables were normally distributed with Solids being slightly right skewed.
After this, I performed Bivariate Analysis by checking the correlation between different 
variables. I constructed a heatmap to better visualize the result from which I concluded 
that there was no strong correlation between any variables.

4. Handling Missing Values:
Missing Values was an important aspect of this project. I faced a challenge in deciding 
how to approach handling them. After reading from several resources, I found multiple 
ways to handle or replace these missing or null values. Popular approaches I found were 
removing these rows altogether, using Mean Median Imputation, replacing with an 
arbitrary number and nearest neighbor imputation etc. 
The mean-median values of our dataset were very close with minor skewing except 
Solids. Thus, I decided to try two approaches, one removing all null rows altogether and 
the other median imputation.
I applied both approaches and found that removing all null rows resulted in a higher 
accuracy as shown in the notebook. I utilized the Lazy Predict Library to run several 
models altogether to get a general idea and trend of accuracies.

5. Model Training:
For the actual model training, I decided to use a model pipeline with 5 popular ML models 
to check which of them perform best so that I can narrow down before performing 
hyperparameter tuning. 
The amount of data was low so it was not a reasonable choice to go towards neural 
network. Based on the Lazy Predict Results and my personal experience and knowledge 
working with ML Models before, I decided to use RandomForestClassifier, SVC, 
ExtraTreesClassifier, KNeighborsClassifier and XGBClassifier in the pipeline. These 
models are available in the sklearn and xgboost libraries respectively.
I evaluated these models based on accuracy score using metrics provided by sklearn
library. From which I chose the top 3 models including Random Forest, Extra Trees & 
XGB Classifier.

7. Hyperparameter Tuning:
In the three selected models I added one more model by the name of CatBoostClassifier
available in the CatBoost Library. I had used this model in several previous projects and 
competitions on Kaggle and found it to always give me high accuracy results.
For Hyperparameter tuning, I used RandomSearchCV technique provided by sklearn.
I set up a dictionary with these 4 models and iterable lists of options for their major 
parameters. The Random Search algorithm works by performing randomized fit 
operations on the models with different combinations of parameters.
The process ran for around 3 hours after which I found the best parameters for all the 4 
models.

9. Evaluation & Final Model Selection:
Using this information, I created a new pipeline with these 4 models and fit them on the 
Test Data Set using the hyper tuned parameters to evaluate their performance on Test 
Data. For this, I again used Accuracy Score provided by sklearn metrics.
Random Forest Classifier achieved the highest accuracy score, and I selected it as the 
final model.
All ensemble models and Random Forest in particular, perform well as they leverage an 
ensemble of multiple decision trees to generate predictions or classifications. By 
combining the outputs of these trees, the random forest algorithm delivers a consolidated 
and more accurate result.

I made a Confusion Matrix for the final model’s predictions on the test data which revealed 
that that the number of False Negatives is high. This may be because the ratio of positive 
and negative records was not equal and there was a 60-40 ratio with less positive records.

10. Feature Importance:
The RandomForestClassifier provides feature importances, which indicate the relative 
importance of each input feature in making predictions. The higher the feature importance 
value, the more influential that feature is in the model's predictions. By examining the 
feature importances, we can gain insights into which features have the strongest impact 
on the target variable. This information can be valuable in understanding the underlying 
relationships in the data and identifying key factors that drive the predictions.
From these importances, I found that Sulfate followed by pH seems to have a 
considerably higher effect on output compared to the rest of the features.
While Sulfate itself is generally not harmful at typical concentrations, elevated levels can 
lead to a laxative effect, especially in infants, resulting in diarrhea and discomfort. 
Additionally, the bacteria that reduce sulfate in the digestive system can produce
hydrogen sulfide gas, which, when present in high concentrations, can cause 
gastrointestinal issues and an unpleasant "rotten egg" odor, impacting overall well-being.
Extremely low or high pH levels in water can affect human health. Low pH (acidic water) 
can dissolve metals like lead and copper from pipes, posing health risks. High pH (alkaline 
water) can taste bitter and potentially cause gastrointestinal discomfort. Maintaining water 
within the optimal pH range of 6.5 to 8.5 is crucial to ensure it's safe for human 
consumption and doesn't harm health through corrosiveness or unpleasant taste.

### API
I made a Flask App to perform inference with the model. The API endpoint can be reached 
at 
http://<yourip>:5000/predict
All the variables need to be sent as a JSON object in the body of the POST request.
Example:
{
 "ph": "6.34",
 "Hardness": "190.10",
 "Solids": "41085.96",
 "Chloramines": "8.967",
 "Sulfate": "384.54",
 "Conductivity": "538.745",
 "Organic_carbon": "14.65",
 "Trihalomethanes": "73.21",
 "Turbidity": "4.627"
}
