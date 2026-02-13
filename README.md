# Personality-Prediction

🔎 Personality Prediction Project: Classification of Social Behavior
Project Overview
Welcome to the Personality Prediction Project, a data science exploration into the link between observed social behaviors and core personality traits. Our primary objective is to build a simple, highly interpretable classification model to predict whether an individual leans toward being an Extrovert or an Introvert.
This project serves as a practical demonstration of binary classification using Logistic Regression, proving that powerful insights can be extracted from accessible behavioral data.
🎯 Key Features Used
The model is trained on a synthetic dataset that mimics essential social habits. We use three numerical features as proxies for real-world social tendencies:
social_attendance: A score representing a person's frequency of attending or engaging in social events. A high score suggests high real-world social engagement.
post_frequency: The count of how often a person shares content or updates with their network. This indicates a tendency to broadcast their life or thoughts.
friends_circle_size: The total number of connections in a person's network, reflecting the breadth of their social circle.

💻 Methodology and Pipeline
The project follows a standard machine learning pipeline detailed within the Personality Prediction Project.ipynb notebook:
1. Data Loading & Initial Exploration
We start by loading the synthetic_data.csv. We use simple Pandas functions to quickly inspect the data:
df.head() and df.tail(): To verify data loading and structure.
df.shape and df.info(): To confirm the total number of entries, quickly identify data types, and check for missing values.
df.describe(): To obtain statistical summaries (mean, min, max, etc.) of the numerical features.
2. Data Preprocessing & Cleaning
Missing Values: We handle any sparse data through mean imputation, filling missing numerical values with the average of that feature to preserve the data's overall distribution.
Label Encoding: The categorical target variable (personality_type: Extrovert/Introvert) is converted into numerical labels (e.g., 1 and 0) for model compatibility.
Data Splitting: The dataset is split into Training (80%) and Testing (20%) sets to ensure an unbiased evaluation of the model's performance on unseen data.
3. Data Visualization
We utilize Matplotlib and Seaborn to visualize the data, employing:
Histograms: To understand the distribution of individual features.
Box Plots: To compare the range of features (e.g., friends_circle_size) between the Extrovert and Introvert groups.
4. Model Training & Evaluation
We selected Logistic Regression for its clear interpretability. After training the model, we evaluate its performance using key classification metrics such as Accuracy, Precision, Recall, and the Confusion Matrix.

🚀 How to Run the Project
Prerequisites
You need Python 3.x installed on your system.
1. Clone the Repository
Bash
git clone https://github.com/your-username/personality-prediction-project.git
cd personality-prediction-project


2. Install Dependencies
It is highly recommended to use a virtual environment (venv).
Bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Use `venv\Scripts\activate` on Windows

# Install all required libraries
pip install -r requirements.txt


3. Execute the Notebook
Start the Jupyter server and open the main file:
Bash
jupyter notebook


Open Personality Prediction Project.ipynb and run the cells sequentially to reproduce the analysis, model training, and final prediction.

✅ Conclusion: Prediction on a New Data Point
The final section of the notebook demonstrates the model's application by predicting the personality for a new, unseen user profile. The raw input, which includes features like Time_spent_Alone, Social_event_attendance, and Post_frequency, is first processed: categorical variables are encoded, and numerical features are scaled using the same preprocessing steps applied to the training data. The trained Logistic Regression model then uses this standardized input to make a classification. For the provided data, the model predicts the personality as Introvert. While the class probabilities were close ($\text{0.5303}$ for Extrovert and $\text{0.4697}$ for Introvert), this successful prediction confirms the entire machine learning pipeline—from data preparation to model deployment—is functioning correctly, allowing us to turn raw social behavior data into actionable personality insights.
