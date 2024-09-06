# Sowing Success: Predicting Optimal Crop Based on Soil Conditions

## Project Overview

In this project, we aim to help farmers **select the best crop** for their fields based on essential soil metrics. **Soil condition** plays a pivotal role in maximizing crop yield, and it can be assessed using factors such as **nitrogen (N)**, **phosphorous (P)**, **potassium (K)** content, and **pH levels**. While measuring these metrics can be costly, it's critical to determine the ideal crop for optimal growth based on soil characteristics.

You, as a **machine learning expert**, have been tasked with building a **multi-class classification model** that can predict the best crop for a particular field, based on its soil composition. Additionally, the farmer would like to know which feature (e.g., nitrogen, potassium, etc.) has the **most significant impact** on crop selection.

### Dataset

The dataset provided is called `soil_measures.csv`, which contains the following columns:

- **N**: Nitrogen content ratio in the soil
- **P**: Phosphorous content ratio in the soil
- **K**: Potassium content ratio in the soil
- **pH**: pH value of the soil
- **crop**: The target variable representing the optimal crop for the given soil metrics

Each row represents a set of soil measurements for a particular field and the corresponding best crop type.

### Project Goals

- **Build multi-class classification models** to predict the type of crop based on soil metrics.
- **Identify the most important soil metric** that influences crop selection.

---

## Project Structure

```
├── data/
│   └── soil_measures.csv     # Dataset containing soil metrics and crop type
├── src/
│   ├── data_preprocessing.py  # Script for data cleaning and preprocessing
│   ├── model_training.py      # Script for building and training classification models
│   ├── feature_importance.py  # Script for analyzing feature importance
├── results/
│   └── model_evaluation.pdf   # Evaluation report for the classification models
├── README.md                  # Project overview and instructions (this file)
└── requirements.txt           # List of required dependencies
```

---

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/crop-prediction.git
cd crop-prediction
```

### Step 2: Install Dependencies

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### Step 3: Run the Analysis

To preprocess the data, train the models, and evaluate feature importance, you can run the following scripts:

1. **Preprocess the Data**:
   ```bash
   python src/data_preprocessing.py
   ```

2. **Train Classification Models**:
   ```bash
   python src/model_training.py
   ```

3. **Analyze Feature Importance**:
   ```bash
   python src/feature_importance.py
   ```

### Dependencies

- **pandas**: For data manipulation.
- **scikit-learn**: For building classification models and analyzing feature importance.
- **matplotlib**/**seaborn**: For visualizing data and results.

---

## Key Steps

### 1. Data Preprocessing

- Clean and prepare the data for model training.
- Handle any missing or anomalous values.
- Split the data into training and testing sets.

### 2. Model Training

- Train multiple **multi-class classification models**, such as:
  - Decision Trees
  - Random Forest
  - Support Vector Machines (SVM)
  - Logistic Regression
- Evaluate model performance using accuracy, precision, recall, and F1-score.

### 3. Feature Importance Analysis

- Identify the **most important feature** contributing to the predictive performance of the model.
- Use methods such as:
  - Feature importance from **tree-based models** (e.g., Random Forests)
  - Coefficients from **Logistic Regression**
  - **Permutation importance** for non-parametric methods

---

## Results

The results will show which **soil metric** (nitrogen, phosphorous, potassium, or pH) has the **highest influence** on predicting the best crop. Additionally, the best-performing model will be identified, along with its evaluation metrics.

---

## Conclusion

This project delivers a machine learning model capable of predicting the **optimal crop** based on soil conditions, helping farmers maximize their yield. By identifying the most important soil feature, the model offers practical insights for more targeted and cost-effective soil testing.

---

## License

This project is licensed under the MIT License.

---

### Future Work

- **Model Deployment**: Implement the model into a user-friendly web app for farmers.
- **Expand Dataset**: Gather more data from different regions and seasons to improve model robustness.
- **Add Weather Metrics**: Consider incorporating weather data for more comprehensive crop predictions.

Feel free to contribute, suggest improvements, or explore additional features!
