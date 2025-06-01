import streamlit as st
import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
style.use('fivethirtyeight')

st.set_page_config(
    page_title="Data Analytics Portfolio - Mina Elgabalawy",
    page_icon="👋",
)
st.sidebar.header("Home")

st.title("Mina Elgabalawy's Portfolio")
st.write("Data Storytelling")
st.markdown("[Github](https://github.com/1mtn)")
st.markdown('[Tableau Dashboards](https://public.tableau.com/app/profile/mina.elgabalawy6222/vizzes)')



st.header('A portfolio of converting this:')
#reading data 
tests = pd.read_csv('StudentsPerformance.csv')
#Creating shorter, easier to type column names
new_columns = ['gender', 'race', 'parent_ed', 'lunch', 'prep_course', 'math', 'reading', 'writing']

#assigning new column names
tests.columns = new_columns

st.dataframe(tests.head())


st.header('Into this:')

#grouping by `gender`
gender_group = tests.groupby('gender') 

#creating female and male groupings
female = gender_group.get_group('female')
male = gender_group.get_group('male')

female_scores = female[['math', 'reading', 'writing']]
female_avg_scores = female_scores.agg(['mean', 'min', 'max', 'median'])
print("Female Scores")


male_scores = male[['math', 'reading', 'writing']]
male_avg_scores = male_scores.agg(['mean', 'min', 'max', 'median'])
print("Male Scores")
m_median_math = male_avg_scores.loc['median', 'math']
m_median_read = male_avg_scores.loc['median', 'reading']
m_median_writing = male_avg_scores.loc['median', 'writing']


#Creating variables for medians
m_math_median = male_avg_scores.loc['median', 'math']
m_read_median =  male_avg_scores.loc['median', 'reading']
m_write_median =  male_avg_scores.loc['median', 'writing']

f_math_median = female_avg_scores.loc['median', 'math']
f_read_median =  female_avg_scores.loc['median', 'reading']
f_write_median =  female_avg_scores.loc['median', 'writing']

fig = plt.figure(figsize=(14,6))

#Male plots

plt.subplot(2,3,1)
plt.hist(male['math'])
plt.xticks([0,50,100, m_math_median])
plt.yticks([0,50,100, 150])
plt.ylim(0,160) #adding y limits for easier comparision 
plt.ylabel('Male')
plt.title('Math')
plt.axvline(m_math_median, c='yellow')
plt.legend(['Median'], loc='upper left')
#m_median_math


plt.subplot(2,3,2)
plt.hist(male['reading'])
plt.xticks([0,100,50, m_read_median])
plt.yticks([0,50,100, 150]) 
plt.ylim(0,160) #adding y limits for easier comparision 
plt.title('Reading')
plt.axvline(m_read_median, c='yellow')
plt.legend(['Median'], loc='upper left')


plt.subplot(2,3,3)
plt.hist(male['writing'])
plt.xticks([0,50,100, m_write_median])
plt.yticks([0,50,100, 150])
plt.ylim(0,160) #adding y limits for easier comparision 
plt.title('Writing')
plt.axvline(m_write_median, c='yellow')
plt.legend(['Median'], loc='upper left')

#Female Plots
plt.subplot(2,3,4)
plt.hist(female['math'])
plt.ylabel('Female')
plt.xlabel('Score')
plt.xticks([0,100,50, f_math_median])
plt.yticks([0,50,100, 150])
plt.ylim(0,160) #adding y limits for easier comparision 
plt.axvline(f_math_median, c='yellow')
plt.legend(['Median'], loc='upper left')

plt.subplot(2,3,5)
plt.hist(female['reading'])
plt.xlabel('Score')
plt.xticks([0,100,50, f_read_median])
plt.yticks([0,50,100, 150])
plt.ylim(0,160) #adding y limits for easier comparision 
plt.axvline(f_read_median, c='yellow')
plt.legend(['Median'], loc='upper left')

plt.subplot(2,3,6)
plt.hist(female['writing'])
plt.xlabel('Score')
plt.xticks([0,100,50, f_write_median])
plt.yticks([0,50,100, 150])
plt.ylim(0,160) #adding y limits for easier comparision 
plt.axvline(f_write_median, c='yellow')
plt.legend(['Median'], loc='upper left')

st.pyplot(fig)


st.markdown("""
            ## Click on one of the projects on left to see more
            ### Technologies used:
            - Python Libraries:
                - Pandas
                - NumPy
                - Matplotlib
                - scikit-learn
                - Seaborn
                - Streamlit
            - SQL
            - dbt
            - Tableau

            To tell the story of your data, email: `info@get-insight.today`
            """)
