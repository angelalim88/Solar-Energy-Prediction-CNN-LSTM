# Solar Energy Prediction with CNN-LSTM

## Overview
This project focuses on predicting solar energy output using a hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) model. The dataset includes solar irradiance data (DHI, DNI, GHI) and weather-related features (temperature, humidity, wind speed, etc.) from 2014 to 2017. The analysis involves data preprocessing, exploratory data analysis (EDA), and the implementation of a CNN-LSTM model to predict time-series solar energy output.

**Objectives**:
- Preprocess and merge solar irradiance and weather datasets to create a comprehensive dataset.
- Conduct EDA to identify patterns and relationships in solar energy output and weather conditions.
- Develop a hybrid CNN-LSTM model to predict solar energy output, leveraging CNN for feature extraction and LSTM for capturing temporal dependencies.
- Support renewable energy forecasting and optimization of solar energy systems.

## Dataset
The dataset consists of multiple CSV files:
- `train.csv`: Training data with timestamps and solar energy output (% Baseline).
- `test.csv`: Test data for model evaluation.
- `Solar_Irradiance_2014.csv`, `Solar_Irradiance_2015.csv`, `Solar_Irradiance_2016.csv`, `Solar_Irradiance_2017.csv`: Solar irradiance data containing columns such as:
  - `Year`, `Month`, `Day`, `Hour`, `Minute`: Timestamp components.
  - `DHI`, `DNI`, `GHI`: Diffuse, direct, and global horizontal irradiance.
  - `Clearsky DHI`, `Clearsky DNI`, `Clearsky GHI`: Clear sky irradiance values.
  - `Cloud Type`, `Dew Point`, `Solar Zenith Angle`, `Surface Albedo`, `Wind Speed`, `Relative Humidity`, `Temperature`, `Pressure`: Weather-related features.
- `Weather.csv`: Additional weather data merged with irradiance data using timestamps.

**Data Issues**:
- Missing values in some weather-related columns were handled using imputation techniques.
- Timestamp formatting inconsistencies were resolved to ensure accurate merging.

## Project Structure
The analysis is conducted in a Google Colab Notebook: `Code_DAC0187_SDMTinggi.ipynb`. The notebook is structured as follows:
1. **Library Imports**: Imports libraries including pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow, plotly, and missingno.
2. **Data Loading and Merging**: Loads and combines solar irradiance and weather datasets, converting timestamps to a unified datetime format.
3. **Data Preprocessing**: Handles missing values, encodes categorical features (e.g., Cloud Type), and normalizes numerical features.
4. **Exploratory Data Analysis**: Visualizes solar energy output over time and correlations between features using Plotly and Seaborn.
5. **Model Development**: Implements a hybrid CNN-LSTM model with Conv1D layers for feature extraction and LSTM layers for temporal modeling.
6. **Model Evaluation**: Evaluates the model using metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE).
7. **Prediction and Output**: Generates predictions on the test set, formats results, and saves them as `results_test_cnn.csv`.

## Model Evaluation
The CNN-LSTM model was evaluated using the following metrics:
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual solar energy output.
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual values.

The model effectively captured temporal dependencies and complex patterns in the data, providing accurate predictions for solar energy output.

![Training Validation](https://github.com/angelalim88/Solar-Energy-Prediction-CNN-LSTM/blob/main/images/training_validation_loss_mae.png)

## Evaluation Results
The hybrid CNN-LSTM model demonstrated strong performance in predicting solar energy output:
- The CNN layers successfully extracted relevant features from the sequential data.
- The LSTM layers effectively modeled temporal dependencies, resulting in robust predictions.
- The predictions were formatted to four decimal places and saved in a standardized CSV file for submission.

These results highlight the potential of the CNN-LSTM model for accurate solar energy forecasting, supporting applications in renewable energy management.

### Others
- **Baseline Solar Output**
![Baseline Solar Output](https://github.com/angelalim88/Solar-Energy-Prediction-CNN-LSTM/blob/main/images/baseline_solar_output.png)
- **Var Correlation**
![Var Correlation](https://github.com/angelalim88/Solar-Energy-Prediction-CNN-LSTM/blob/main/images/var_corelation.png)
- **Pairplot Scatterplots**
![Pairplot Scatterplots](https://github.com/angelalim88/Solar-Energy-Prediction-CNN-LSTM/blob/images/images/pairplot_scatterplots.png)