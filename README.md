# 🏠 Homiq — House Price Prediction with Linear Regression

Homiq is a beginner-friendly machine learning project that predicts the price of a house from its features (area, rooms, parking, air conditioning, and more). It uses **Linear Regression** from **Scikit-learn**, and comes with two notebooks: one to **train and save** the model, and one to **load it and predict** a price from your own input.

---

## 📑 Table of Contents

1. [What is Linear Regression?](#-what-is-linear-regression)
2. [The Formula](#-the-formula)
3. [More Than One Feature](#-more-than-one-feature)
4. [Dataset](#-dataset)
5. [Project Structure](#-project-structure)
6. [Installation](#-installation)
7. [How the Training Notebook Works](#-how-the-training-notebook-works)
8. [Using the Model (Prediction Notebook)](#-using-the-model-prediction-notebook)
9. [Results](#-results)
10. [Limitations](#-limitations)
11. [Ideas for Improvement](#-ideas-for-improvement)
12. [License](#-license)

---

## 📘 What is Linear Regression?

Linear Regression belongs to **supervised learning**: the algorithm learns from data where the correct answer (the *output*) is already known. It is primarily used for **regression** tasks — predicting a continuous number such as a price — though it can be applied to some classification problems in certain contexts.

The algorithm fits a **linear model** to the data by minimizing the **mean squared error** between the observed outputs and the predicted values. Once the model is fitted, we can predict the output for a new input using the linear equation.

We can implement it with the **Scikit-learn** library, which provides a ready-made model for this task. We supply it with the data points, and it fits the line for us.

## 📐 The Formula

```
y = ax + b
```

| Symbol | Meaning |
|--------|---------|
| `y` | output (what we want to predict) |
| `a` | weight |
| `x` | input |
| `b` | bias |

This formula is for a **two-dimensional space**: only one input and one output.

## 🧮 More Than One Feature

In reality we usually have more than one input. In that case the formula becomes:

```
y = a1·x + a2·z + a3·p + ... + b
```

Every weight is multiplied by its corresponding dimension, the results are summed, and finally the bias is added.

In artificial intelligence, these dimensions are called **features**, and each feature has its own value. We collect a set of data, feed it into linear regression, and it fits a line (in higher dimensions, a hyperplane) onto the data. When we provide a new input, the model can estimate the output.

In this project the model has **12 features**, so the learned equation is:

```
price = w1·area + w2·bedrooms + ... + w12·furnishingstatus + b
```

---

## 📊 Dataset

`Data.csv` contains **545 houses**, each with a price and 12 features.

> The dataset is a public housing-prices dataset. If you publish this repository, add a link to the original source here.

| Column | Type | Description | Encoding used by the model |
|--------|------|-------------|----------------------------|
| `price` | number | **Target** — the price to predict | — |
| `area` | number | House area | as is |
| `bedrooms` | number | Number of bedrooms | as is |
| `bathrooms` | number | Number of bathrooms | as is |
| `stories` | number | Number of stories | as is |
| `mainroad` | yes/no | Connected to the main road | yes = 1, no = 0 |
| `guestroom` | yes/no | Has a guest room | yes = 1, no = 0 |
| `basement` | yes/no | Has a basement | yes = 1, no = 0 |
| `hotwaterheating` | yes/no | Has hot water heating | yes = 1, no = 0 |
| `airconditioning` | yes/no | Has air conditioning | yes = 1, no = 0 |
| `parking` | number | Number of parking spaces | as is |
| `prefarea` | yes/no | Located in a preferred area | yes = 1, no = 0 |
| `furnishingstatus` | category | Furnishing level | unfurnished = 0, semi-furnished = 1, furnished = 2 |

Quick facts: prices range from 1,750,000 to 13,300,000 (mean ≈ 4,767,000), there are no missing values, and the original currency is not specified in the data.

---

## 📁 Project Structure

```
House-Price-Prediction/
├── Data.csv                      # The dataset
├── House-Price-Prediction.ipynb  # Train, evaluate, and save the model
├── Test-File.ipynb               # Load the model and predict from user input
├── House-Prediction-model        # Saved trained model (created with Joblib)
├── requirements.txt              # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/leVoidRaptor/House-Price-Prediction.git
cd House-Price-Prediction

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install the dependencies
pip install -r requirements.txt

# 4. Start Jupyter
jupyter notebook
```

> **Why is `scikit-learn` pinned?** The saved model was created with scikit-learn 1.7.2. Loading it with a different version still works but prints a warning. To avoid the warning, either install the pinned version or simply re-run the training notebook to regenerate the model.

---

## 🔬 How the Training Notebook Works

`House-Price-Prediction.ipynb` goes through these steps:

1. **Importing libraries**
   - **NumPy** — numerical operations and array handling
   - **Pandas** — loading and manipulating tabular data
   - **Matplotlib** — creating visualizations
   - **`train_test_split`** — splitting data into training and testing sets
   - **`LinearRegression`** — the machine learning model
   - **Joblib** — saving the trained model
2. **Loading the dataset** — read the CSV into a Pandas DataFrame and display the first five rows to inspect it.
3. **Inspecting dataset information** — display column names, data types, and non-null counts to understand the structure of the data.
4. **Converting categorical values into numerical values** — models only understand numbers, so yes/no values become `1`/`0`, and furnishing status becomes `unfurnished = 0`, `semi-furnished = 1`, `furnished = 2`.
5. **Checking data types after conversion** — confirm that every column is now numeric.
6. **Separating features and target** — `price` is the target. `X` holds the input features and `Y` holds the house prices the model must predict.
7. **Checking the shapes of X and Y** — `X` is `(545, 12)` and `Y` is `(545,)`, which confirms the data was separated correctly.
8. **Splitting data and training** — 90% of the data is used for training and 10% for testing. `random_state=18` makes the split reproducible. The model is trained on the training data and evaluated on the test data using **R-squared**.
9. **Generating predictions** — use the trained model to predict prices for the test features.
10. **Comparing predicted and actual prices** — print each predicted price next to its real value.
11. **Visualizing predictions** — plot predicted and actual prices on one figure, with a legend and grid.
12. **Saving the trained model** — the model is saved with Joblib as `House-Prediction-model`, so it can be loaded later without retraining.

---

## 🔮 Using the Model (Prediction Notebook)

`Test-File.ipynb` loads the saved model and asks you for the details of a house:

- area, number of bedrooms, bathrooms, stories, and parking spaces
- yes/no questions: main road, guest room, basement, hot water heating, air conditioning, preferred area
- furnishing status: `unfurnished`, `semi-furnished`, or `furnished`

It builds a single row in the same feature order used during training and prints the estimated price:

```
Welcome to Homiq!
House Price Prediction System!

Estimated House Price:
13,483,472.88
```

You can also use the model directly in Python:

```python
import joblib

model = joblib.load("House-Prediction-model")

# Feature order:
# area, bedrooms, bathrooms, stories, mainroad, guestroom, basement,
# hotwaterheating, airconditioning, parking, prefarea, furnishingstatus
house = [[7420, 4, 2, 3, 1, 0, 0, 0, 1, 2, 1, 2]]
print(f"{model.predict(house)[0]:,.0f}")
```

> ⚠️ **Important:** the encoding at prediction time must match training exactly (`yes = 1`, `no = 0`, furnishing `0/1/2`) and the features must be in the same order. A mismatch silently gives wrong predictions.

---

## 📈 Results

Evaluated on the 55 houses (10%) held out as a test set:

| Metric | Value |
|--------|-------|
| R² score | **≈ 0.80** |
| Mean absolute error | ≈ 795,000 |
| Root mean squared error | ≈ 978,000 |

An R² of 0.80 means the model explains about 80% of the variation in house prices on unseen data.

---

## ⚠️ Limitations

- The test set is small (55 houses), so the score can change noticeably with a different `random_state`.
- Linear Regression assumes a straight-line relationship between features and price, and it cannot capture complex interactions.
- `furnishingstatus` is encoded as 0/1/2, which assumes each step adds the same amount to the price.
- The model only knows the houses in this dataset; it should not be used for real financial decisions.
- The dataset does not specify the currency or the location.

## 🚀 Ideas for Improvement

- Use **cross-validation** for a more reliable score
- Replace the line plot with a **predicted vs. actual scatter plot**
- Use **one-hot encoding** for `furnishingstatus`
- Try other models (Ridge, Lasso, Random Forest, Gradient Boosting) and compare
- Add input validation for numeric questions in the prediction notebook
- Build a small web interface (Streamlit or Flask) around the model

## 👤 Author

Made by [leVoidRaptor](https://github.com/leVoidRaptor).
