# Predicting House Prices Using Real-Time Data

## Project Overview

This project aims to build a predictive model that can accurately estimate house prices based on various features such as location, size, number of bedrooms, and other relevant factors using real-time data from APIs. By leveraging multiple free-to-use APIs, we will gather data on properties, locations, and weather conditions. The analysis will be performed using Python libraries such as `pandas`, `numpy`, and `scikit-learn`.

## Objectives

- Fetch real estate data for various properties using the Zillow API.
- Obtain location data and geocode addresses using the OpenStreetMap API.
- Fetch weather data for the locations of the properties using a Weather API.
- Perform data cleaning and preparation to handle missing values and merge datasets.
- Conduct exploratory data analysis (EDA) to identify trends and correlations.
- Build and evaluate predictive models to estimate house prices.
- Summarize key findings and provide actionable insights.

## APIs Used

1. **Zillow API**: For fetching real estate data such as house prices, property details, and more.
2. **OpenStreetMap API**: For obtaining location data and geocoding addresses.
3. **Weather API**: For fetching weather data that might influence house prices.

## Libraries Used

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `requests`

## Steps to Implement

1. **Set Up Environment**:
   - Install necessary libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `requests`.
   - Obtain API keys for Zillow, OpenStreetMap, and a Weather API.

2. **Fetch Data**:
   - Use the Zillow API to fetch real estate data for various properties.
   - Use the OpenStreetMap API to get location data and geocode addresses.
   - Use the Weather API to fetch weather data for the locations of the properties.

3. **Data Cleaning and Preparation**:
   - Clean the fetched data to handle missing values, incorrect data types, and duplicates.
   - Merge the real estate, location, and weather data based on property addresses or coordinates.

4. **Exploratory Data Analysis (EDA)**:
   - **Descriptive Statistics**: Calculate mean, median, standard deviation, and other statistics for numerical features.
   - **Correlation Analysis**: Analyze the correlation between features and the target variable (house prices).
   - **Visualization**:
     - **Histograms**: Plot histograms for numerical features.
     - **Box Plots**: Create box plots to visualize the distribution of numerical features.
     - **Heatmaps**: Create heatmaps to visualize the correlation matrix.

5. **Feature Selection**:
   - Identify the most important features that influence house prices.
   - Use techniques like correlation analysis, feature importance from tree-based models, or recursive feature elimination.

6. **Model Building**:
   - Split the dataset into training and testing sets.
   - Train multiple regression models such as Linear Regression, Decision Tree, Random Forest, and Gradient Boosting.
   - Evaluate the models using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.

7. **Model Evaluation and Tuning**:
   - Compare the performance of different models.
   - Tune hyperparameters using techniques like Grid Search or Random Search.
   - Select the best-performing model based on evaluation metrics.

8. **Insights and Conclusions**:
   - Summarize the key findings from the predictive analysis.
   - Identify the most influential features for predicting house prices.
   - Provide actionable insights or recommendations based on the analysis.

9. **Documentation and Presentation**:
   - Document the entire process, including code, visualizations, and findings.
   - Create a presentation or report to showcase the results.
  
## My Pitch
We Shall provides a comprehensive analysis of house prices using predictive modeling techniques and real-time data from APIs. By leveraging multiple APIs and performing EDA, we can build a model that accurately predicts house prices based on various features. The insights and findings from this project can be used to inform real estate decisions, understand market trends, and drive further research.

## Example Code Snippet

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Fetch real estate data from Zillow API
zillow_api_key = 'your_zillow_api_key'
zillow_url = f'https://api.zillow.com/v1/property?apikey={zillow_api_key}'
zillow_response = requests.get(zillow_url)
zillow_data = zillow_response.json()

# Fetch location data from OpenStreetMap API
osm_url = 'https://nominatim.openstreetmap.org/search'
params = {'q': 'New York, USA', 'format': 'json'}
osm_response = requests.get(osm_url, params=params)
osm_data = osm_response.json()

# Fetch weather data from Weather API
weather_api_key = 'your_weather_api_key'
weather_url = f'http://api.weatherapi.com/v1/current.json?key={weather_api_key}&q=New York'
weather_response = requests.get(weather_url)
weather_data = weather_response.json()

# Example data processing and visualization
zillow_df = pd.DataFrame(zillow_data['properties'])
osm_df = pd.DataFrame(osm_data)
weather_df = pd.DataFrame([weather_data['current']])

# Merge dataframes
merged_df = pd.concat([zillow_df, osm_df, weather_df], axis=1)

# Feature selection
features = ['bedrooms', 'bathrooms', 'square_feet', 'latitude', 'longitude', 'temp_c']
X = merged_df[features]
y = merged_df['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model building
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plotting actual vs. predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices')
plt.show()
