import zipfile
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.tsa.api as tsa
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os


# Model Run

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    
    def model_train_and_predict(self, county_information):

        for county in county_information.keys():

            #Seasonality Check
            print('Seasonality Check for :-', county)
            plt.figure(figsize=(12, 10))
            decomp = tsa.seasonal_decompose(county_information[county]['df'])
            decomp.plot()
            plt.show()

            #splitting dataset into train and test sets for validation
            print('splitting dataset into train and test sets for validation \n')

            train_var = 'train_' + county
            test_var = 'test_' + county

            locals()[train_var], locals()[test_var] = self.train_test_split_ts(data=county_information[county]['df'], train_size = 0.80, test_size = 0.20)

            #plotting the split
            print('plotting the split :-', county)
            plt.figure(figsize=(12, 10))
            self.plot_train_test_split(train_data=locals()[train_var], test_data=locals()[test_var], county=county)
            plt.show()

            # Finding Best Parameters with Auto-Arima
            print('Finding Best Parameters with Auto-Arima \n')
            auto_model = pm.auto_arima(locals()[train_var], start_p=0, d=1, start_q=0, max_p=4, 
                                    max_d=3, max_q=4, start_P=0, start_Q=0, max_P=3, 
                                    max_D=3, max_Q=3, m=12)
            auto_model.summary()

            # Validate Model with Forecasts for Test Data
            print('Validate Model with Forecasts for Test Data \n')
            model = SARIMAX(locals()[train_var], order=(1,1,0), 
                            seasonal_order=(0,1,0,12), enforce_invertibility=False, 
                            enforce_stationarity=False).fit()
            self.evaluate_model(model, county)

            df_forecast_var = 'df_' + county + '_forecast'

            locals()[df_forecast_var] = self.get_forecast(model=model, train_data=locals()[train_var], test_data=locals()[test_var], plot=True)

            # Future Predictions
            print('Future Predictions \n')
            # Fitting Model to All Observed Data
            model = SARIMAX(county_information[county]['df'], order=(1,1,0), 
                            seasonal_order=(0,1,0,12), enforce_invertibility=False, 
                            enforce_stationarity=False).fit()
            self.evaluate_model(model=model, county=county)

            #creating a df of predictions and plotting
            print('creating a df of predictions and plotting \n')

            df_preds_var = 'df_' + county + '_forecast'

            locals()[df_preds_var] = self.get_prediction(model=model, data=county_information[county]['df'], 
                                        test_data=locals()[test_var], county_name=county, plot=True)

            #saving predictions df to dict for later use
            print('saving predictions df to dict for later use \n')
            county_information[county]['Predictions'] = locals()[df_preds_var]
            
            
            #`county_information` is a dictionary containing county information

            df = round(county_information[county]['Predictions'], 0)
            folder_path = r"C:\Users\pmoff\OneDrive\Desktop\PWSkills\MLProjects\EV_E_Mobility\Chargers_predictions_csv"
            # Create the directory if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)
            output_file_name = county + "_Prediction.csv"
            output_file_path = os.path.join(folder_path, output_file_name)
            # Save the output DataFrame to CSV
            df.to_csv(output_file_path, index=False)
            # Confirmation message
            print(f"Output DataFrame saved to '{output_file_path}'")
            

            


            ##################################################################################


            # Evaluation of Models

    def evaluate_model(self, model, county):
        """Function returns the model summary and diagnostics information to aid 
        the evaluation of the given model's performance.
        -------------------------------
        Arguments:
        model: SARIMAX or ARIMA model object
        Model variable to evaluate (Time series models for both pmdarima and 
        statsmodels are supported. 
        """
        print('evaluate_model with Forecasts for Test Data', county ,'\n')
        print(model.summary())
        model.plot_diagnostics()
        plt.tight_layout();


        # Creating a Function for train_test_split & plotting

    def train_test_split_ts(self, data, train_size, test_size):
        
        """Function splits a given DataFrame into two sets based on the given 
        train and test sizes so that the data can be used for validation.
        -------------------------------
        Arguments:
        df: class: pandas.DataFrame
        The base dataframe that will be getting split.
        
        train_size: float
        The size of the desired training set (for example: 0.80)
        
        test_size: float
        The size of the desired training set (for example: 0.20)
        """
        
        train_end_idx = int(round(len(data)*train_size,0))
        train_set = data.iloc[0:train_end_idx,:]["EV's on the Road"]
        test_set = data.iloc[train_end_idx:,:]["EV's on the Road"]
        return train_set, test_set


        # Additionally, to visualize these splits we will need a plotting function.
    def plot_train_test_split(self, train_data, test_data, county):
        
        """Function plots the training and testing data for visual inspection.
        -------------------------------
        Arguments:
        train_data: pandas.Series
        The training set of data to be plotted.
        
        test_data: pandas.Series
        The test set of data to be plotted.
        
        county: str
        Name of the county that the training and testing data belongs to. This 
        string is used to set the title of the axes.
        """
        
        train_data.plot(label='Train Data')
        test_data.plot(label='Test Data')
        ax=plt.gca()
        ax.set_xlabel('Year')
        ax.set_ylabel('Electric Vehicles on the Road')
        ax.set_title(f'Electric Vehicles on the Road in {county} County')
        ax.legend();


    # Creating a Function for Getting Forecasts
    def get_forecast(self, model, train_data, test_data, plot=True):
        
        """Function gets forecasted values from a given model and plots them for 
        visual inspection. The length of the forecasts are dependent on the length 
        of the test data. The forecasted values are returned in a DataFrame format.
        -------------------------------
        Arguments:
        model:  SARIMAX or ARIMA model object
        Model that the forecast is to be received from. 
        
        train_data: pandas.Series
        The training set of data used in training the model.
        
        test_data: pandas.Series
        The testing set of data used for validating the model.
        
        plot: bool, default=True
        Option to plot the forecasted values along with observed values 
        (train_data and test_data).
        """
        
        #creating a df with the forecast information
        forecast_df = model.get_forecast(steps=len(test_data)).conf_int()
        forecast_df.columns = ['Lower Confidence Interval', 
                                'Upper Confidence Interval']
        forecast_df['Forecasts'] = model.get_forecast(steps=len(test_data))\
        .predicted_mean
        #plotting
        if plot==True:
            with plt.style.context('seaborn-whitegrid'):
                fig, ax = plt.subplots(figsize=(15, 10))
                sns.lineplot(data=train_data, color='black', ax=ax)
                sns.lineplot(data=forecast_df, x=forecast_df.index, 
                            y='Forecasts', color='blue', ax=ax, 
                            label='Forecasted Data', ls='--')
                sns.lineplot(data=test_data, color='purple', ax=ax, 
                            label='Actual Data', ls='-.')
                ax.fill_between(forecast_df.index, 
                                y1=forecast_df['Lower Confidence Interval'], 
                                y2=forecast_df['Upper Confidence Interval'],
                                color = 'green', alpha=0.3, 
                                label='Confidence Interval')
                ax.set_xlabel('Year')
                ax.legend(loc=2)
                plt.show();
        return forecast_df


    # Defining a Function for Getting Predictions for Future
    def get_prediction(self, model, data, test_data, county_name, plot=True):
        
        """Function gets predicted values from a given model and plots them for 
        visual inspection. The length of the predictions are dependent on the 
        length of the test data. The forecasted values are returned in a DataFrame 
        format.
        -------------------------------
        Arguments:
        model:  SARIMAX or ARIMA model object
        Model to be used for making predictions.
        
        df: pandas.DataFrame
        DataFrame that contains all observed data.
        
        test_data: pandas.Series
        The testing set of data used for validating the model (dictates the length
        of predictions).
        
        plot: bool, default=True
        Option to plot the predicted values along with observed values.
        """
        
        #creating a df with the prediction information
        prediction_df = model.get_forecast(steps=len(test_data)).conf_int()
        prediction_df.columns = ['Lower Confidence Interval', 
                                'Upper Confidence Interval']
        prediction_df['Predictions'] = model.get_forecast(steps=len(test_data))\
        .predicted_mean
        #plotting
        if plot==True:
            with plt.style.context('seaborn-whitegrid'):
                fig, ax = plt.subplots(figsize=(15, 10))
                sns.lineplot(data=data, ax=ax)
                sns.lineplot(data=prediction_df, x=prediction_df.index, 
                            y='Predictions', color='orange', ax=ax, 
                            label='Predicted Data', ls='--')
                ax.fill_between(prediction_df.index, 
                                y1=prediction_df['Lower Confidence Interval'], 
                                y2=prediction_df['Upper Confidence Interval'],
                                color = 'green', alpha=0.3, 
                                label='Confidence Interval')
                ax.set_xlabel('Year')
                ax.set_ylabel('Electric Vehicles on the Road')
                ax.set_title(f'Predicted Electric Vehicle Count for {county_name}')
                plt.show();
        return prediction_df