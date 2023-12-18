import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# Streamlit App
st.title('Flight Price Prediction')

 # Baca file CSV
df = pd.read_csv('Clean_Dataset.csv')
df_economy = df[df['class'] == 'Economy']

# Menambahkan navigasi di sidebar
st.write("KELOMPOK 4")
st.write("- Nayaka Wiryatama")
st.write("- Rizky Ramadhan")
st.write("- Ery Febrian")
st.write("- Ismail Baihaqi")
page = st.sidebar.radio("Pilih halaman", ["Dataset", "Perbandingan Class", "Class Vs Ticket Price", "Economy", "Stops", "Departure and Arrival", "Source and Destination", "Duration", "Days Left", 
"Actual vs Predicted","Make Prediction"])

result = None
if page == "Dataset":
    st.header("Halaman Dataset")
    st.write(df)

elif page == "Perbandingan Class":
    st.title('Classes of Different Airlines')
    # Plotting the pie chart
    fig, ax = plt.subplots()
    df['class'].value_counts().plot(kind='pie', ax=ax, autopct='%.2f', colors=sns.color_palette('pastel'), textprops={'color':'black'})
    ax.set_title('Classes of Different Airlines', fontsize=15)
    ax.legend(['Economy', 'Business'])

    # Display the plot in Streamlit
    st.pyplot(fig)

elif page == "Class Vs Ticket Price":
    st.title('Class Vs Ticket Price')

    # Plotting the boxplot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='class', y='price', data=df, palette='hls', ax=ax)
    ax.set_title('Class Vs Ticket Price', fontsize=15)
    ax.set_xlabel('Class', fontsize=15)
    ax.set_ylabel('Price', fontsize=15)

    # Display the plot in Streamlit
    st.pyplot(fig)

elif page == "Flights Count":
    st.title('Flights Count of Different Airlines (Economy)')

    # Plotting the count plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df_economy, x='airline', order=df_economy['airline'].value_counts().index[::-1], ax=ax)
    ax.set_title('Flights Count of Different Airlines (Economy)', fontsize=15)
    ax.set_xlabel('Airline', fontsize=15)
    ax.set_ylabel('Count', fontsize=15)

    # Display the plot in Streamlit
    st.pyplot(fig)

elif page == "Class Vs Ticket Price":
    st.title('Airlines Vs Price (Economy)')

    # Plotting the boxplot
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.boxplot(data=df_economy, x='airline', y='price', ax=ax)
    ax.set_title('Airlines Vs Price (Economy)', fontsize=15)
    ax.set_xlabel('Airline', fontsize=15)
    ax.set_ylabel('Price', fontsize=15)

    # Display the plot in Streamlit
    st.pyplot(fig)

elif page == "Economy":
    st.title('Flights Count and Airlines Vs Price (Economy)')

    # Plotting the count plot
    st.subheader('Flights Count of Different Airlines (Economy)')
    fig_count, ax_count = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df_economy, x='airline', order=df_economy['airline'].value_counts().index[::-1], ax=ax_count)
    ax_count.set_title('Flights Count of Different Airlines (Economy)', fontsize=15)
    ax_count.set_xlabel('Airline', fontsize=15)
    ax_count.set_ylabel('Count', fontsize=15)
    st.pyplot(fig_count)

    # Plotting the boxplot
    st.subheader('Airlines Vs Price (Economy)')
    fig_boxplot, ax_boxplot = plt.subplots(figsize=(15, 5))
    sns.boxplot(data=df_economy, x='airline', y='price', ax=ax_boxplot)
    ax_boxplot.set_title('Airlines Vs Price (Economy)', fontsize=15)
    ax_boxplot.set_xlabel('Airline', fontsize=15)
    ax_boxplot.set_ylabel('Price', fontsize=15)
    st.pyplot(fig_boxplot)

elif page == "Business":
    st.title('Flights Count and Airlines Vs Price (Economy)')

    # Plotting the count plot
    st.subheader('Flights Count of Different Airlines (Economy)')
    fig_count, ax_count = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df_economy, x='airline', order=df_economy['airline'].value_counts().index[::-1], ax=ax_count)
    ax_count.set_title('Flights Count of Different Airlines (Economy)', fontsize=15)
    ax_count.set_xlabel('Airline', fontsize=15)
    ax_count.set_ylabel('Count', fontsize=15)
    st.pyplot(fig_count)

    # Plotting the boxplot
    st.subheader('Airlines Vs Price (Economy)')
    fig_boxplot, ax_boxplot = plt.subplots(figsize=(15, 5))
    sns.boxplot(data=df_economy, x='airline', y='price', ax=ax_boxplot)
    ax_boxplot.set_title('Airlines Vs Price (Economy)', fontsize=15)
    ax_boxplot.set_xlabel('Airline', fontsize=15)
    ax_boxplot.set_ylabel('Price', fontsize=15)
    st.pyplot(fig_boxplot)

elif page == "Stops":
    st.title('Stops Vs Ticket Price')

    # Plotting the boxplot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='stops', y='price', data=df, palette='hls', ax=ax)
    ax.set_title('Stops Vs Ticket Price', fontsize=15)
    ax.set_xlabel('Stops', fontsize=15)
    ax.set_ylabel('Price', fontsize=15)

    # Display the plot in Streamlit
    st.pyplot(fig)

elif page == "Departure and Arrival":
    st.title('Departure and Arrival Time Vs Ticket Price')

    # Plotting the boxplot for Departure Time
    st.subheader('Departure Time Vs Ticket Price')
    fig_departure, ax_departure = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='departure_time', y='price', data=df, palette='hls', ax=ax_departure)
    ax_departure.set_title('Departure Time Vs Ticket Price', fontsize=15)
    ax_departure.set_xlabel('Departure Time', fontsize=15)
    ax_departure.set_ylabel('Price', fontsize=15)
    st.pyplot(fig_departure)

    # Plotting the boxplot for Arrival Time
    st.subheader('Arrival Time Vs Ticket Price')
    fig_arrival, ax_arrival = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='arrival_time', y='price', data=df, palette='hls', ax=ax_arrival)
    ax_arrival.set_title('Arrival Time Vs Ticket Price', fontsize=15)
    ax_arrival.set_xlabel('Arrival Time', fontsize=15)
    ax_arrival.set_ylabel('Price', fontsize=15)
    st.pyplot(fig_arrival)

elif page == "Source and Destination":
    st.title('Source and Destination City Vs Ticket Price')

    # Plotting the boxplot for Source City
    st.subheader('Source City Vs Ticket Price')
    fig_source, ax_source = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='source_city', y='price', data=df, palette='hls', ax=ax_source)
    ax_source.set_title('Source City Vs Ticket Price', fontsize=15)
    ax_source.set_xlabel('Source City', fontsize=15)
    ax_source.set_ylabel('Price', fontsize=15)
    st.pyplot(fig_source)

    # Plotting the boxplot for Destination City
    st.subheader('Destination City Vs Ticket Price')
    fig_destination, ax_destination = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='destination_city', y='price', data=df, palette='hls', ax=ax_destination)
    ax_destination.set_title('Destination City Vs Ticket Price', fontsize=15)
    ax_destination.set_xlabel('Destination City', fontsize=15)
    ax_destination.set_ylabel('Price', fontsize=15)
    st.pyplot(fig_destination)

elif page == "Duration":
    st.title('Duration Vs Price')

    # Plotting the regression plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.regplot(x='duration', y='price', data=df, line_kws={'color': 'blue'}, ax=ax)
    ax.set_title('Duration Vs Price', fontsize=20)
    ax.set_xlabel('Duration', fontsize=15)
    ax.set_ylabel('Price', fontsize=15)

    # Display the plot in Streamlit
    st.pyplot(fig)

elif page == "Days Left":
    st.title('Days Left Vs Price')

    # Plotting the line plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df, x='days_left', y='price', color='blue', ax=ax)
    ax.set_title('Days Left Vs Price', fontsize=20)
    ax.set_xlabel('Days Left', fontsize=15)
    ax.set_ylabel('Price', fontsize=15)

    # Display the plot in Streamlit
    st.pyplot(fig)

elif page == "Actual vs Predicted":
    st.title('Actual Price Vs Predicted Price')

    # Example: Train a simple linear regression model
    X = df[['duration', 'days_left']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Create a DataFrame for result
    result = pd.DataFrame({'Price_actual': y_test, 'Price_pred': y_pred})

    # Ensure 'Price_actual' and 'Price_pred' columns are numeric
    result['Price_actual'] = pd.to_numeric(result['Price_actual'])
    result['Price_pred'] = pd.to_numeric(result['Price_pred'])

    # Plotting the regression plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.regplot(x='Price_actual', y='Price_pred', data=result, ax=ax)
    ax.set_title('Actual Price Vs Predicted Price', fontsize=20)
    ax.set_xlabel('Actual Price', fontsize=15)
    ax.set_ylabel('Predicted Price', fontsize=15)

    # Display the plot in Streamlit
    st.pyplot(fig)

elif page == "Make Prediction":
    st.title('Make Flight Price Prediction')
    model = RandomForestRegressor()  # Use the same model for training and prediction
    joblib.dump(model, 'random_forest_model.pkl')
    # User input for prediction
    st.sidebar.subheader("Input Features for Prediction")
    duration = st.sidebar.number_input("Duration (hours)", min_value=0, max_value=24, value=1)
    days_left = st.sidebar.number_input("Days Left for Departure", min_value=0, value=30)
    # Add more input features as needed for your model

    # Prepare the input data for prediction
    input_data = pd.DataFrame({'duration': [duration], 'days_left': [days_left]})
    # Add more input features as needed for your model

    # Load the saved model (assuming you saved it during training)
    model_path = r'C:\Users\hp\Documents\DAL Streamlit\ADLtugas\random_forest_model.pkl'
    try:
        model = joblib.load(model_path)
        if not hasattr(model, 'fit'):
            raise ValueError("The loaded model does not have a 'fit' method.")
         # Fit the model if it's not already fitted
        if not hasattr(model, 'estimators_') or len(model.estimators_) == 0:
        # Example: fit the model with your training data
            X_train, y_train = df[['duration', 'days_left']], df['price']
            model.fit(X_train, y_train)
    # Ensure the model is fitted before making predictions
        if hasattr(model, 'predict'):
            prediction = model.predict(input_data)
            st.subheader('Predicted Price:')
            st.write(prediction[0])  # Assuming the prediction is a single value
        else:
            st.warning("Model is not fitted.")
    except Exception as e:
        st.error(f"Error loading or fitting the model: {str(e)}")
#if hasattr(model, 'predict'):
#    prediction = model.predict(input_data)
#    st.subheader('Predicted Price:')
#    st.write(prediction[0])
else:
    st.warning("Model is not fitted.")
