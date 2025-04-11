import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns 

st.title("Top Steel Producing Countries")

st.markdown("""Dataset from kaggle: https://www.kaggle.com/datasets/kabhishm/countries-by-steel-production-19672021?select=steel_prod.csv

This is a list of countries by steel production in 1967, 1980, 1990, 2000 and from 2007 to 2021, based on data provided by the World Steel Association. All countries with annual production of crude steel at least 2 million metric tons.
The data set has the production of steel in million metric tons by year and by country.""")


#reading CSV file 
steel = pd.read_csv('steel_prod.csv')

#Data Cleaning
#replacing null values with NaN
steel['1967']= steel['1967'].fillna(np.nan)

#Replacing `-` with NaN
steel['1967'] = steel['1967'].replace(0,np.nan)

#Cleaning function
def clean_col(col):
    col = col.fillna('0')
    col = col.replace('-','0')
    col = col.astype(float)
    return col

#Using cleaning function on a loop through all columns 
for c in steel.columns[2:]:
    col = steel[c]
    col = clean_col(col)
    steel[c]=col

#cleaning column[0]
steel['Rank']= clean_col(steel['Rank'])

#Changing index to be `Country/Region` + dropping `Rank` column
steel_clean = steel.set_index('Country/Region')
steel_clean = steel_clean.drop('Rank', axis=1)

#reversing column order 
steel_clean = steel_clean[steel_clean.columns[::-1]] #reversing the order of the columns

countries = list(steel_clean.index)

#VISUALIZE! 
style.use('fivethirtyeight')


fig, ax = plt.subplots(figsize=(14,10))

for c in steel_clean.index[1:7]: #looping throug the countries in the dataframe index
    ax.plot(steel_clean.loc[c], label = c, alpha = 0.3) #alpa = 0.2 for a faded look

country = st.selectbox('Country to highligh', list(steel_clean.index)[1:7])
ax.plot(steel_clean.loc[country], color = 'red')
        
plt.xticks( rotation = 90)

#Customizations
ax.tick_params(left = False, bottom = False)

#removing all 4 axes
for location in ['left', 'right', 'bottom', 'top']:
    ax.spines[location].set_visible(False)
plt.legend(loc = 'upper left')

plt.title(f'Top Countries Producing Steel Highlighting {country}')

st.pyplot(fig)

st.header("China Produces More Steel Than Any Other Country By a Long Shot")
st.markdown("""
            - In 2021 China produced over 1000 million metric tons of Steel
- That's more than 5 times the amount produces by the next largest producer: the EU
- The EU produced less than 200 million metric tons

Let's take a closer look at the other steel producing companies, excluding China""")


fig, ax = plt.subplots(figsize=(14,10))

for c in steel_clean.index[2:7]: #looping throug the countries in the dataframe index
    ax.plot(steel_clean.loc[c], label = c)


plt.xticks( rotation = 90)

#Customizations
ax.tick_params(left = False, bottom = False)

#removing all 4 axes
for location in ['left', 'right', 'bottom', 'top']:
    ax.spines[location].set_visible(False)
plt.legend(loc = 'upper left')

plt.title('Top Countries Producing Steel Excluding China')

st.pyplot(fig)


st.header("Observations")
st.markdown("""
        - Top 5 steel producing countries after China are:
    1. EU
    2. India
    3. Japan
    4. USA
    5. Russia
- Since the year 2000, these countries maintained their relative positions with one exception: __India__

Let's take a closer look at India"""    )



fig, ax = plt.subplots(figsize=(14,10))

for c in steel_clean.index[2:7]: #looping throug the countries in the dataframe index
    ax.plot(steel_clean.loc[c], label = c, alpha = 0.2) #alpa = 0.2 for a faded look

ax.plot(steel_clean.loc['India'], color = 'red')
        
plt.xticks( rotation = 90)

#Customizations
ax.tick_params(left = False, bottom = False)

#removing all 4 axes
for location in ['left', 'right', 'bottom', 'top']:
    ax.spines[location].set_visible(False)
plt.legend(loc = 'upper left')

plt.title('Top Countries Producing Steel Excluding China')

st.pyplot(fig)

st.markdown("""
            ## India: Steady Rise
- In the year 2000, India was the world's 6th largest steel producer
- in 2009, India surpassed Russia to become the 5th largest steel producer
- In 2014, it surpassed the US to become the 4th largest steel producer
- Between 2017-2018, India surpassed Japan to become the 3rd largest steel producer, a position it retains until today.



#### Next, let's see how China's steel production compares with the total world production""")
fig, ax = plt.subplots(figsize=(14,10))

for c in steel_clean.index[0:2]: #looping throug the countries in the dataframe index
    ax.plot(steel_clean.loc[c], label = c)


plt.xticks( rotation = 90)

#Customizations
ax.tick_params(left = False, bottom = False)

#removing all 4 axes
for location in ['left', 'right', 'bottom', 'top']:
    ax.spines[location].set_visible(False)
plt.legend(loc = 'upper left')

plt.title('World vs China Production of Steel')
plt.ylabel('In Million Metric Tons')

st.pyplot(fig)


st.markdown("""
           ## China's production appears to be the main driver in increasing world steel production
Let's see how World's production and China's production correlate""")


#Correlation Visualization

china_prod = steel_clean.iloc[1]
world_prod = steel_clean.iloc[0]

china_world_corr = china_prod.corr(world_prod)

fig, ax = plt.subplots(figsize=(14,10))
ax.scatter(china_prod, world_prod)
plt.xlabel("China's Steel Production")
plt.ylabel("World's Total Steel Produciton")
plt.title("World vs China Steel Production")
ax.axhline(y=1830, xmin=0.05, xmax=0.61,
           linewidth=95, color='black', alpha=0.1)
ax.text(350, 1750, "Correlation between World's production \nand China's production is \nextremely high at 0.99", fontsize=20, color = 'red', ha='center')
st.pyplot(fig)



#Visualize by year

st.markdown("""
Let's see how China steel production in 2021 compares to the other major steel producers.
            __Use the slider to see earlier years__
            """)

years = list(steel_clean.columns)
year = st.select_slider('Year', years, value="2021")
new = steel_clean[year].reset_index()

fig, ax = plt.subplots(figsize=(7,12))

ax.bar(new['Country/Region'][1:6], new[year][1:6], label = f"Top 5 Steel Producer in {year}")

#Customizations
ax.tick_params(left = False, bottom = False)

#removing all 4 axes
for location in ['left', 'right', 'bottom', 'top']:
    ax.spines[location].set_visible(False)
plt.legend(loc = 'upper left')


st.pyplot(fig)


st.header('Between the years 2000 and 2007, China overtook the EU as the worlds largest producer of steel')
