# 🎓 Student Performance Prediction — End-to-End Machine Learning Project

An end-to-end machine learning project that predicts a student's **math performance** from demographic and academic information.

The project was built as a complete machine learning workflow rather than a single training notebook. It covers **data ingestion, data preprocessing, model training, model evaluation, model serialization, prediction, and deployment through a Flask web application**.

---

## 📌 Project Overview

Student academic performance can be influenced by several demographic, educational, and socioeconomic factors.

This project uses student information such as:

- Gender
- Race / ethnicity
- Parental level of education
- Lunch type
- Test preparation course
- Reading score
- Writing score

to predict the student's **math score** using machine learning.

The trained model is integrated into a Flask web application where users can enter student information and receive a predicted math score.

---

## 🎯 Problem Statement

The objective of this project is to build a machine learning regression system capable of predicting a student's mathematics score based on other available demographic and academic attributes.

The project follows a modular pipeline so that the same preprocessing and trained model used during development can be reused during inference.

---

## 🔄 Machine Learning Workflow

The complete workflow can be summarized as:

```text
Raw Dataset
     │
     ▼
Data Ingestion
     │
     ▼
Train / Test Split
     │
     ▼
Data Transformation
     │
     ├── Numerical Features
     │       └── Scaling / preprocessing
     │
     └── Categorical Features
             └── Encoding / preprocessing
     │
     ▼
Model Training
     │
     ├── Linear Regression
     ├── Decision Tree
     ├── Random Forest
     ├── Gradient Boosting
     ├── AdaBoost
     ├── XGBoost
     └── CatBoost
     │
     ▼
Model Evaluation
     │
     ▼
Best Model
     │
     ▼
model.pkl + preprocessor.pkl
     │
     ▼
Flask Application
     │
     ▼
User Input → Prediction
```

---

## 🧠 Machine Learning Models

The project evaluates multiple regression algorithms instead of relying on a single model.

The model training workflow includes:

- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **AdaBoost Regressor**
- **XGBoost Regressor**
- **CatBoost Regressor**

This allows different algorithms to be compared and the better-performing model to be selected based on the evaluation process.

The project uses `scikit-learn`, `XGBoost`, and `CatBoost` for model development.

---

## 📈 Model Performance

The trained regression models were evaluated using regression performance metrics, with **R² (coefficient of determination)** used to assess how well the model explains the variance in the target variable.

The final model achieved:

| Metric | Score |
|---|---:|
| **R² Score** | **88.04%** |

An R² score of **0.8804** indicates that the model explains approximately **88.04% of the variance in students' mathematics scores on the evaluation data**.

> **Note:** The reported score is based on the evaluation data used during the project's model-training workflow. It should not be interpreted as 88.04% prediction accuracy.

## 📊 Input Features

The prediction pipeline expects the following features:

| Feature | Description | Type |
|---|---|---|
| `gender` | Student's gender | Categorical |
| `race_ethnicity` | Student's race/ethnicity group | Categorical |
| `parental_level_of_education` | Parent/guardian education level | Categorical |
| `lunch` | Type of lunch provided | Categorical |
| `test_preparation_course` | Whether the student completed a preparation course | Categorical |
| `reading_score` | Reading examination score | Numerical |
| `writing_score` | Writing examination score | Numerical |

The `CustomData` class converts these individual inputs into a Pandas DataFrame before they are passed into the prediction pipeline.

---

## ⚙️ Data Preprocessing

A separate preprocessing pipeline is used to transform the input data before model prediction.

The trained preprocessing object is saved as:

```text
artifacts/preprocessor.pkl
```

During prediction, the same saved preprocessor is loaded and applied to the incoming data before it reaches the trained model.

This is important because the model should receive data in the same representation used during training.

---

## 🏗️ Project Architecture

The project is organized into separate components instead of placing the entire workflow inside a single Python file.

```text
Ml_Project/
│
├── artifacts/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── train.csv
│   └── test.csv
│
├── logs/
│   └── Application logs
│
├── notebook/
│   └── Jupyter notebooks
│
├── src/
│   ├── Components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── Pipelines/
│   │   ├── Predict_pipeline.py
│   │   └── Training_pipeline.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── templates/
│   ├── index.html
│   └── home.html
│
├── App.py
├── Requirement.txt
├── Setup.py
├── .gitignore
└── Readme.md
```

> File/folder names may differ slightly depending on the current repository version.

---

## 🗂️ Project Components

### `src/components`

Contains the main components responsible for the machine learning workflow.

#### Data Ingestion

Responsible for loading the dataset and preparing the data for subsequent stages of the pipeline.

The processed dataset is used to create training and testing data.

#### Data Transformation

Responsible for preparing the raw features for machine learning.

This stage handles the transformation required for numerical and categorical variables and produces the serialized preprocessing object.

#### Model Training

The model-training component trains multiple regression algorithms and evaluates their performance.

The selected trained model is serialized and stored as:

```text
artifacts/model.pkl
```

---

## 🔮 Prediction Pipeline

The prediction process is implemented separately from model training.

The `PredictPipeline`:

1. Loads the saved model.
2. Loads the saved preprocessing object.
3. Transforms the incoming feature DataFrame.
4. Passes the transformed data to the trained model.
5. Returns the prediction.

Conceptually:

```python
features
    ↓
preprocessor.transform(features)
    ↓
trained_model.predict(...)
    ↓
predicted math score
```

The implementation loads:

```text
artifacts/model.pkl
artifacts/preprocessor.pkl
```

and applies the preprocessing transformation before generating the prediction.

---

## 🌐 Flask Web Application

The machine learning pipeline is connected to a Flask web application through `App.py`.

The application provides:

```text
/
```

for the main page and:

```text
/predict
```

for prediction requests.

When the user submits the form, the application:

1. Reads the submitted values.
2. Creates a `CustomData` object.
3. Converts the input into a DataFrame.
4. Sends the DataFrame to `PredictPipeline`.
5. Loads the saved model and preprocessor.
6. Generates the predicted math score.
7. Displays the result on the web page.

The Flask application is configured to run on:

```text
0.0.0.0
```

so it can accept connections beyond the local loopback interface when appropriately deployed.

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Data Science

- Pandas
- NumPy
- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost
- CatBoost

### Web Development

- Flask
- HTML / Jinja templates

### Model Serialization

- Dill

### Development Environment

- VS Code
- Jupyter Notebook
- Git / GitHub

The project's dependency file currently includes NumPy, Pandas, Matplotlib, Seaborn, CatBoost, XGBoost, Flask, Scikit-learn, Dill, and the local package installation.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/sanyam410/Ml_Project.git
```

### 2. Navigate into the project

```bash
cd Ml_Project
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r Requirement.txt
```

The project also contains `Setup.py` and uses editable installation through the requirements file.

If required, the package can also be installed with:

```bash
pip install -e .
```

---

## ▶️ Running the Application

After installing the dependencies, run:

```bash
python App.py
```

Flask will start the application.

Open the local address displayed by Flask in your browser, typically:

```text
http://127.0.0.1:5000/
```

You can then enter the required student information and submit the form to obtain a predicted math score.

---

## 📁 Saved Artifacts

The project stores trained machine learning objects inside the `artifacts/` directory.

### `model.pkl`

Serialized trained machine learning model used for prediction.

### `preprocessor.pkl`

Serialized preprocessing pipeline used to transform incoming user data into the format expected by the model.

Keeping these objects separate allows the application to perform inference without retraining the model every time it receives a request.

---

## 📝 Logging & Exception Handling

The project includes dedicated logging and exception-handling modules:

```text
src/logger.py
src/exception.py
```

Instead of handling errors entirely inside the Flask application, exceptions can be routed through the project's custom exception mechanism.

This makes debugging and tracking errors across different stages of the pipeline easier.

---

## 🔬 Development Workflow

The project was developed incrementally using Jupyter Notebook and VS Code.

The general development process was:

```text
Exploratory Analysis
        ↓
Data Understanding
        ↓
Data Preprocessing
        ↓
Feature Transformation
        ↓
Model Training
        ↓
Model Comparison
        ↓
Best Model Selection
        ↓
Model Serialization
        ↓
Prediction Pipeline
        ↓
Flask Application
        ↓
Git / GitHub
```

The final repository contains both exploratory notebooks and modular Python source code, allowing experimentation and application code to remain separated.

---

## 📌 Key Learning Outcomes

Through this project, the following machine learning and software-development concepts were practiced:

- End-to-end machine learning workflow
- Regression modelling
- Data preprocessing
- Categorical feature handling
- Numerical feature transformation
- Train-test splitting
- Model comparison
- Hyperparameter experimentation
- Model serialization
- Reusable prediction pipelines
- Flask integration
- Custom exception handling
- Application logging
- Python project packaging
- Git and GitHub workflow

---

## 🔮 Future Improvements

Possible improvements for future versions include:

- Add cross-validation for more robust model evaluation.
- Add additional regression metrics such as MAE and RMSE.
- Improve hyperparameter optimization and experiment tracking.
- Add automated testing for preprocessing and prediction.
- Add input validation on the Flask forms.
- Improve the frontend UI and user experience.
- Containerize the application using Docker.
- Deploy the application to a cloud platform.
- Add CI/CD using GitHub Actions.
- Add model monitoring and versioning.
- Improve reproducibility by pinning dependency versions.

---

## ⚠️ Project Status

This is a **learning and portfolio project** demonstrating the development of an end-to-end machine learning application.

The primary goal is to demonstrate the complete workflow from data preparation and model training to deployment and prediction rather than to provide a production-ready educational assessment system.

---



## ⭐ Acknowledgement

This project was developed as part of my hands-on learning journey in **Machine Learning and Data Science**, with a focus on understanding how an ML model moves beyond a notebook and becomes part of a usable application.

If you found the project useful or interesting, consider giving the repository a ⭐.

