# Predictive Modeling of Residential Property Values in Ames, Iowa

## Project Overview
This project addresses the challenge of accurately assessing residential property values using machine learning models. Using a comprehensive dataset of 2,930 home sales in Ames, Iowa, from 2006 to 2010, the analysis aims to achieve three key goals:
1.  Evaluate the **predictive accuracy** of machine learning models in a volatile market, such as the 2008 housing crisis.
2.  Identify the most **influential features** that drive home values.
3.  Provide **actionable insights** through a scenario analysis, empowering homeowners to make data-driven decisions.

## Methodology
The project followed a standard data science pipeline:
* **Data Cleaning & Preprocessing:** The dataset was cleaned by imputing missing values with the median for numerical features and a 'None' value for categorical features. The data was then preprocessed using a detailed pipeline that included log transformation for skewed features and standardization for all numerical data. One-hot encoding was applied to categorical features, increasing the dimensionality from 82 to 330 features.
* **Model Selection:** A Ridge Regression model was selected as a baseline due to its effectiveness in handling high dimensionality. A sequential, feed-forward Neural Network was also implemented to test for non-linear relationships.
* **Feature Analysis:** Feature importance was evaluated using the coefficients from the Ridge Regression model and confirmed through Principal Component Analysis (PCA).
* **Scenario Analysis:** Three hypothetical scenarios ('Luxury Upgrade,' 'Fixer Upper,' and 'Green Renovation') were created to simulate home improvements and demonstrate the models' predictive power in a practical context.

## Key Findings & Results
The analysis yielded several important insights:
* **Model Performance:** The Ridge Regression model demonstrated superior performance, with a Mean Absolute Error of **$14,276.62** and an R² score of **0.9060**. This showed that a simpler, linear model was more effective than the complex Neural Network (MAE: **$18,706.93**; R²: **0.8225**), suggesting that the underlying relationships in the data were largely linear.
* **Feature Importance:** The strongest positive driver of home value was a specific neighborhood (`Neighborhood_GrnHill`), while the strongest negative driver was a specific roof material (`Roof Matl_ClyTile`). This reinforced the classic real estate aphorism, "location, location, location."
* **Market Resilience:** The models' predictive power remained resilient during the 2008 housing crisis, indicating that a home's core features remain significant drivers of value even during periods of economic uncertainty.
* **Scenario Outcomes:** The scenario analysis revealed that strategic, high-end renovations (`Luxury Upgrade`) produced the highest average price gain (**$50,000**), while low-budget improvements (`Fixer Upper`) and eco-friendly renovations (`Green Renovation`) had more modest but still positive impacts. The analysis also showed an **inverse relationship** between a home's original price and the predicted gain from improvements.

## Dependencies
This project was built using a Jupyter Lab Notebook with Python kernel via Anaconda Navigator and the following libraries:
* `numpy`
* `pandas`
* `scikit-learn`
* `tensorflow`
* `matplotlib`
* `seaborn`


## How to Run the Project
1.  Clone this repository to your local machine.
2.  Ensure you have the required dependencies installed (pip install -r requirements.txt)
3.  Open the main Jupyter Notebook file in a local environment (e.g., JupyterLab or VS Code).
4.  Run all cells to execute the full data analysis pipeline and reproduce the results.
