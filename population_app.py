import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
import os

# Load the world map GeoJSON file
world = gpd.read_file('custom.geo.json')

# Load your population dataset
population = pd.read_csv('World-population-by-countries-dataset.csv')
population = population.dropna()

# Merge the world map data with the population data
world = world.merge(population, left_on='iso_a3', right_on='Country Code', how='left')

# Define a color scale for the choropleth map
color_scale = [
    [0.0, '#dceefb'],
    [0.2, '#b0d5f1'],
    [0.4, '#80b9e0'],
    [0.6, '#5396c5'],
    [0.8, '#2e6da4'],
    [1.0, '#0a4c8f']
]

# Function to predict population using the saved or trained model
def predict_population(country_name, years_to_predict):
    model = get_or_train_model(country_name)
    years_to_predict = np.array(years_to_predict).reshape(-1, 1)
    predictions = model.predict(years_to_predict)
    return predictions

# Function to retrieve historical population data for a country
def get_historical_population_data(country_name):
    country_data = population[population['Country Name'] == country_name]
    years = country_data.columns[2:].astype(int)
    population_data = country_data.iloc[:, 2:].values.ravel()
    return years, population_data

# Train the linear regression model if it doesn't exist or load it if it does
def get_or_train_model(country_name):
    model_filename = f'{country_name}_population_model.pkl'
    if os.path.exists(model_filename):
        # Model exists, load it
        return joblib.load(model_filename)
    else:
        # Model doesn't exist, train and create it
        country_data = population[population['Country Name'] == country_name]
        years = country_data.columns[2:].astype(int).values.reshape(-1, 1)
        population_data = country_data.iloc[:, 2:].values.ravel()

        model = LinearRegression()
        model.fit(years, population_data)
        joblib.dump(model, model_filename)
        return model

# Streamlit UI
st.title("Population Prediction and Visualization App")

# Initial population year selection using a slider
selected_year = st.slider("Select a Year", min_value=1960, max_value=2021, step=1)

# Create a choropleth map for the selected year
fig_map= px.choropleth(world,
                        locations='iso_a3',
                        color=f'{selected_year}',
                        hover_name='Country Name',
                        color_continuous_scale=color_scale,
                        title=f'World Population Map For The Year {selected_year}',
                        labels={'Population': f'{selected_year} Population'})

# Display the map with a time sliders
st.plotly_chart(fig_map)

# Melt the data to reshape it
melted_data = pd.melt(population, id_vars=['Country Name', 'Country Code'], var_name='Year', value_name='Population')

# Convert the 'Year' column to datetime format
melted_data['Year'] = pd.to_datetime(melted_data['Year'], format='%Y')

# Create a line chart for all countries
fig = px.line(melted_data, x='Year', y='Population', color='Country Name', 
                  title='Population of Each Country Over Time')

st.plotly_chart(fig)

st.info("**Note:** The population prediction model is trained on data up to the year 2021.")

# Input: Country Name
country_name= st.selectbox("Select a Country Name", population['Country Name'].unique())

# Input: Years to Predict
years_to_predict = st.number_input("Enter the year:", min_value=2022, step=1)

# Predict Population Button
if st.button("Predict Population"):
    if years_to_predict:
        predictions = predict_population(country_name, years_to_predict)
        if predictions is not None:
            st.write(f"Predicted Population for {country_name} in {years_to_predict}:")
            st.write(predictions)
                
            years, population_data = get_historical_population_data(country_name)
            historical_data = pd.DataFrame({'Year': years, 'Population': population_data})
                
                # Create an interactive line chart
            fig = px.line(historical_data, x='Year', y='Population', title=f'Population Trends in {country_name}')
            st.plotly_chart(fig)

        else:
            st.error(f"Model for {country_name} not found. Please choose another country.")
    else:
        st.warning("Please enter a valid year.")