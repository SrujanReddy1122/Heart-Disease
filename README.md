# Heart Disease Prediction

This is a small web app I made to predict whether a person has heart disease or not. You enter 20 details like age, blood pressure, cholesterol, smoking and sleep hours, click Classify, and it shows Yes or No along with the probability.

I made this to practice the full flow of a machine learning project: cleaning the data, training a model, and putting it on a website that anyone can use. It is only a learning project, so please don't use it for real medical decisions.

## What it does

- Takes 20 inputs from the user (numbers are typed in, and things like smoking or diabetes are picked with buttons)
- Gives a Yes / No result with the probability in percent
- Fills the missing values in the data (median for numbers, most common value for categories)
- Uses class weights because only about 20% of the records have heart disease
- Keeps the preprocessing and the model together in one pipeline so the app does the same steps as training

## Technologies used

- Python
- Pandas and NumPy
- Scikit-learn
- Flask
- HTML and CSS
- Pickle (to save the model)
- Render (for deployment)

## Files in this project

- `app.py` - the Flask app
- `train_model.py` - trains the model and saves it as `heart_model.pkl`
- `heart_disease.csv` - the dataset (10,000 rows)
- `templates/index.html` - the web page
- `requirements.txt` - libraries needed


## About the model

I used Logistic Regression with balanced class weights and split the data 80/20 for training and testing. The accuracy came out around 50% and the ROC AUC was around 0.49, which is basically random guessing. I also tried Random Forest and Gradient Boosting and got the same thing, so the problem is the dataset, not the code. The features in this data don't really have a pattern that predicts heart disease.

I'm still keeping the project because the main goal was to build and deploy the whole thing end to end.

## What I learned

- How to build a complete ML project from the data to a working website
- Why it is better to keep preprocessing and the model in one pipeline
- How to handle imbalanced data, and that accuracy alone can be misleading
- That I should check the data quality first before trusting any model
- How to connect a model to Flask and deploy it on Render

## Author

Srujanreddy Nimmala
LinkedIn: https://www.linkedin.com/in/srujanreddy-nimmala/
live link : https://heart-disease-3zso.onrender.com/
