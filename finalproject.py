import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress most warnings in TensorFlow
from pandas.api.types import is_numeric_dtype
pd.set_option('display.float_format', lambda x: '%.3f' % x)
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

st.title("Studying Pokemon and Their Characteristics")

st.markdown("The dataset I will use for this project comes from https://gist.github.com/armgilles/194bcff35001e7eb53a2a8b441e8b2c6")
st.markdown("My github link is https://github.com/lilyannday/finalproject")

uploaded_file = st.file_uploader("file upload",['csv'])

st.header("Finding the General Trend for Attack and Defense of a Pokemon")

st.markdown("We will be comparing each Pokemon's attack and defense to other Pokemon in their given generation and plot the general trend we see using linear regression.")
v = st.slider("Choose what generation Pokemon you want to study", 1,6)

if uploaded_file is not None:
    df= pd.read_csv(uploaded_file)
    
df = df.applymap(lambda x: np.nan if x == " " else x)

df2 = df[df["Generation"] == v]

x_column = np.array(df2[["Attack"]]).reshape(-1,1)
y_column = np.array(df2[["Defense"]]).reshape(-1,1)

reg = LinearRegression()
reg.fit(x_column,y_column)

coef = reg.coef_
coef_ = pd.DataFrame(coef)
coef2 = coef_.loc[0,0]
rounded1 = round(coef2, 3)

intercept = reg.intercept_
intercept_ = pd.DataFrame(intercept)
intercept2 = intercept_.loc[0,0]
rounded2 = round(intercept2, 3)

(f"The line of best fit is y = {rounded1} + {rounded2}x.")

x = x_column.reshape(-1,)
y = y_column.reshape(-1,)

c = [intercept2, coef2]
y_true = c[0] + c[1]*x

df3 = pd.DataFrame({"x":x, "y_true":y_true, "y":y})

brush = alt.selection_interval(empty='none')
chart_data = alt.Chart(df3).mark_circle(clip = True).encode(
    x = "x", 
    y = "y",
    color = alt.condition(brush,
                          alt.Color('y_true:Q', scale=alt.Scale(scheme='turbo',reverse=True)),
                          alt.value("lightgrey")),
).add_selection(
    brush,
).properties(
    width = 720,
    height = 450,
    title = "Attack vs Defense")

chart_true = alt.Chart(df3).mark_line().encode(
    x = "x",
    y = "y_true",
    color = alt.value("black")
    )

def f_string(d):
    my_list = [f"x{i}" for i in range(1,d+1)]
    return my_list

temp = f_string(20)
for i in range(1,21):
    df3[temp[i-1]] = df3["x"]**i

def poly_reg(df3, d):
    reg = LinearRegression()
    X = df3[f_string(d)]
    reg.fit(X,df3["y"])
    return reg

def make_chart(df3,d):
    df4 = df3.copy()
    reg = poly_reg(df3,d)
    X = df3[f_string(d)]
    df4["y_pred"] = reg.predict(X)
    chart = alt.Chart(df4).mark_line(clip = True).encode(
        x = "x1",
        y = alt.Y("y_pred"),
        color = alt.value("red"),
    )
    return chart

st.altair_chart(make_chart(df3,2)+chart_data+chart_true)
st.markdown("As we can see, sometimes the fit polynomial would be curved up or down depending on where the outliers lie.")
st.markdown("If you pick a random number for attack, you can use the line of best fit to find the corresponding defense number for any Pokemon in any generation.")

st.header("Figuring Out How Accurately We Can Predict a Type of Pokemon From the Dataset")

st.markdown("First, we can split up the Pokemon and count how many of each type of Pokemon.")
h = alt.Chart(df).mark_bar().encode(
    x = "Type 1",
    y = "count()",
    color = "mean(Total)",
    tooltip = ["count()"]
).properties(title = "Total Number of Each Type of Pokemon") 

st.altair_chart(h, use_container_width=True)
st.markdown("The most common type of Pokemon is Water Type.")

st.markdown("Since water type is the most common type of Pokemon, we will use this type and put our data into a neural network to decide whether or not the Pokemon is water type.")
df5 = df[df.notna().all(axis=1)].copy()
numeric_cols = [c for c in df5.columns if is_numeric_dtype(df5[c])]

scaler = StandardScaler()
scaler.fit(df[numeric_cols])

df5[numeric_cols] = scaler.transform(df5[numeric_cols])

df5["is_water"] = df5["Type 1"].map(lambda g_list: "Water" in g_list)

X_train = df5[numeric_cols]
y_train = df5["is_water"]

model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape = (10,)),
        keras.layers.Dense(16, activation="sigmoid"),
        keras.layers.Dense(16, activation="sigmoid"),
        keras.layers.Dense(1,activation="sigmoid")
    ]
)

model.compile(
    loss="binary_crossentropy", 
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    metrics=["accuracy"],
)

history = model.fit(X_train,y_train,epochs=100, validation_split = 0.2)

fig, ax = plt.subplots()
ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
ax.legend(['train', 'validation'], loc='upper right')
st.pyplot(fig)
st.markdown("We can see that the validation set outperforms the training set so there is no overfitting.")
st.markdown("The neural network can easily be adapted to predict any type of Pokemon.")

st.header("Which Legendary Pokemon is the Strongest?")
st.markdown("Legendary Pokemon tend to be stronger and more rare. I wanted to study which one was the strongest.")
df6 = df[df["Legendary"] == True]
box= st.selectbox("Choose a column to plot data on boxplot",numeric_cols)
fig2, ax2 = plt.subplots()
ax2 = sns.boxplot(x=box, palette="husl", data=df6)
ax2 = sns.swarmplot(x=box, data=df6, color=".25")
st.pyplot(fig2)
st.markdown("You can choose different categories to plot in the boxplot to find the median and narrow down your data. The results may vary depending on which columns you choose to analyze.")
st.markdown("I chose the columns HP, Special Attack, and Special Defense.")
st.markdown("The median for HP is around 90, special attack is 120, and special defense is 100. I will be studying the Pokemon with stats better than these medians")
highstats = df6[(df6["HP"] > 90) & (df6["Sp. Atk"] > 120) & (df6["Sp. Def"] > 100)]

legendary = alt.Chart(highstats).mark_tick(
    color = 'red',
    thickness = 4,
).encode(
    x = 'Total',
    y = 'Speed',
    tooltip = ['Name']
).interactive()

st.altair_chart(legendary)
st.markdown("There is one Pokemon that excels in total and speed and this Pokemon is Mewtwo. Therefore, Mewtwo is the strongest legendary Pokemon as exceeds the average statistics for many of the other legendary Pokemon.")    

st.header("References")
st.markdown("The polynomial regression section of my project was taken from https://christopherdavisuci.github.io/UCI-Math-10/Week6/Week5-Friday.html")
st.markdown("The Altair interaction was taken from https://christopherdavisuci.github.io/UCI-Math-10/Week3/First-Altair-examples.html#spotify-interactive")
st.markdown("The bar graph was taken from https://christopherdavisuci.github.io/UCI-Math-10/Week3/First-Altair-examples.html")
st.markdown("The neural network section of my project was taken from https://christopherdavisuci.github.io/UCI-Math-10/Week10/overfitting.html")

with st.sidebar:
    st.title("Studying Pokemon and Their Characteristics.")
    st.header("Finding the General Trend for Attack and Defense of a Pokemon")
    st.write("In this section, I graphed the attack and defense of each Pokemon and found the linear regression the data seemed to follow.")
    st.header("Figuring Out How Accurately We Can Predict a Type of Pokemon From the Dataset")
    st.write("Here I graphed ran my dataset through a neural network to test the accuracy of my validation set compared to my training set to find a specific type of Pokemon.")
    st.header("Which Legendary Pokemon is the Strongest?")
    st.write("I focused on all the different statistics given to me about the Pokemon to find which legendary Pokemon is strongest overall.")
