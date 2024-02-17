
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt

# Set the deprecation option to suppress the warning
st.set_option('deprecation.showPyplotGlobalUse', False)

string = "Startup's Profit Prediction"
# setup page config — dynamic web page
st.set_page_config(page_title=string, page_icon="💲" , layout="centered", initial_sidebar_state="auto", menu_items=None)
# st.title is a widget element
st.title(string, anchor=None)

# st.sidebar.number_input — creates a side bar at with number input field
rnd_spend = st.sidebar.number_input("Insert R&D Spend")
Administration_spend = st.sidebar.number_input("Insert Administration cost Spend")
Marketing_Spend = st.sidebar.number_input("Insert Marketing cost Spend")
option = st.sidebar.selectbox("Select the region", ("California", "New York", "Florida"))

df = pd.read_csv("50_Startups.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, 4].values
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

model = LinearRegression()
model.fit(X_train, y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if option == "California":
    optn = 0
elif option == "New York":
    optn = 1
elif option == "Florida":
    optn = 2

# Predict the profit
y_pred = model.predict([[Marketing_Spend, Administration_spend, rnd_spend, optn]])

# Display the input values
st.subheader("Input Values")
st.write(f"R&D Spend: {rnd_spend}")
st.write(f"Administration Spend: {Administration_spend}")
st.write(f"Marketing Spend: {Marketing_Spend}")
st.write(f"Region: {option}")

# Button to predict and show profit
if st.button('Predict'):
    st.subheader("Predicted Profit")
    st.success(f"The Profit must be: {y_pred[0]:.2f}")

# Display the data table with highlighted profit
num_rows_to_display = 60
table_height = 300

# Apply conditional styling to highlight the predicted profit
df_display = df.head(num_rows_to_display).style.applymap(
    lambda x: f"background-color: yellow; font-weight: bold;" if x == y_pred[0] and df.columns.name == "Profit" else ""
)

# Display the data table with highlighted profit
st.markdown(f"""
    <div style="overflow-y: auto; height: {table_height}px;">
        {df.head(num_rows_to_display).to_html(classes='dataframe', escape=False)}
    </div>
""", unsafe_allow_html=True)

# Add some space before the button
st.markdown("<br><br>", unsafe_allow_html=True)

# Display a stylish line plot showing profit based on user inputs
if st.button('Show Profit Graph'):
    # Create a line plot with markers
    fig, ax = plt.subplots()
    ax.plot([''], [y_pred[0]], marker='o', color='red', linestyle='-', linewidth=2, markersize=8)
    ax.set_xlabel('Predicted Profit')
    ax.set_ylabel('Amount')
    ax.set_title('Predicted Profit Based on User Inputs')
    st.pyplot(fig)

# You can also display additional information or controls below the table
#st.write("Scroll down to see more rows.")






