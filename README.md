# ID2223Lab1_TitanicSurvival
There 5 different files in total, their functions are described as follows

# titanic_data_cleaning.py
This file downloads data about titanic passengers from the web, and processes and cleanses the data. 
Fill in the data where Age and Fare are missing, and the filled values are their respective averages. 
The data for Name and Ticket are discarded because these columns have no predictive power. 
Fill the missing data of Cabin with 'U' for 'unknown'. 
Fill Embarked's missing data with the most widely distributed value, which is 'S'.
Then, convert the remaining categorical variables into numerical variables.
Finally, output the correlation of each feature with 'survived', so as to select parameters for prediction.

# titanic_feature_pipeline.py
After completing the data cleaning, create a feature group named 'titanic_modal' on hopsworks.

# titanic_training_pipeline.py
Divide 80% of the data in the feature group into a training set and 20% into a test set, use the Logistic Regression Algorithm to train the data, and upload the training results to the modal. (The K-Neighbor Algorithm classifier was also implemented, but it was discarded because of its poor effect).

# titanic-feature-pipeline-daily.py
A synthetic data passenger generator and can update the feature pipeline to allow it to add new synthetic passengers.

# titanic-batch-inference-pipeline.py
A batch inference pipeline which is used to predict if the synthetic passengers survived or not, and can show the most recent synthetic passenger prediction and outcome, as well as a confusion matrix with historical prediction performance.

# \huggingface-spaces-titanic\app.py
This app is used to build an interactive UI for entering feature values and predicting if a passenger would survive the titanic or not.

# \huggingface-spaces-titanic-monitor\app.py
This app is used to build an dashboard UI showing a prediction of survival for the most recent passenger added to the Feature Store and the outcome (label) if that passenger survived or not. Include a confusion matrix to show historical model performance.

# Hugging Face Spaces public URL of Titanic Project for Interactive UI.
https://huggingface.co/spaces/YuhangDeng123/Titanic

# Hugging Face Spaces public URL of Titanic Project for Dashboard UI.

# Hugging Face Spaces public URL of Iris Project for Interactive UI.

# Hugging Face Spaces public URL of Iris Project for Dashboard UI.






