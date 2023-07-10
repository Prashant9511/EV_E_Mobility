End To End ML Project
created a environment
conda create -p venv python==3.8

conda activate venv/
Install all necessary libraries
pip install -r requirements.txt


# Forecasting the Demand for Electric Cars and Charging Stations in Washington State 

## Introduction

Climate change is a major concern globally, with transportation being the largest contributor to greenhouse gas emissions in the US.
Current US government aims to pass a bipartisan infrastructure deal focusing on clean energy and developing a nationwide electric vehicle charger network.
States like California and Washington are pushing for mandates to phase out gasoline vehicles and promote electric vehicles by 2035 and 2030, respectively.
Major car manufacturers like Ford, GM, and Volvo have announced plans to transition to fully electric vehicles in the next decade.

These changes would result in a significant increase in demand for electric vehicles.
The existing charging infrastructure in states like California and Washington needs to be assessed for its ability to support the growing demand for EVs.

## Objective

The objective of this analysis is to predict the demand for electric vehicles in each county in Washington and identify counties that require strategically positioned electric vehicle chargers. By utilizing time series modeling, we aim to recommend the top 10 counties for investment based on their high electric vehicle purchases and minimal charger availability.

## Data

We will analyze Washington state's electric vehicle title and registration activity data to track the number of electric vehicles in each county over time. Furthermore, we will examine the current charging infrastructure data provided by the National Renewable Energy Laboratory to identify counties with significant potential for financial success. 

[Electric vehicle title and registration activity data](https://data.wa.gov/Transportation/Electric-Vehicle-Title-and-Registration-Activity/rpr4-cgyd)

[Current charging infrastructure data](https://developer.nrel.gov/docs/transportation/alt-fuel-stations-v1/all/)
