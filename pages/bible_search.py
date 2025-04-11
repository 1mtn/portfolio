import streamlit as st
st.title("Word Frequency in King James Bible")

st.text("This app helps you visualize the frequency of 2 words in King James Bible so you can compare the relatvie frequency")

import pandas as pd
import csv
import matplotlib.pyplot as plt

#reading csv file. Skipping first 4 rows as  they contain copyright data. 

bible = pd.read_csv("kjv.csv", skiprows=4,index_col=0)

target_word  = 'Jesus'
target_word_2 = 'Lord'
target_word = st.text_input("First Word", value="Jesus")
target_word_2 = st.text_input("Second Word", value="Lord")


#converting all verses to all lowercase:

bible['frequency'] = bible['Text'].str.lower().str.split().apply(lambda words: words.count(target_word.lower()))

bible['frequency_2'] = bible['Text'].str.lower().str.split().apply(lambda words: words.count(target_word_2.lower()))


bible[["Text", "frequency", "frequency_2"]].head(1)



#grouping 'sum of frequency' by book name
result = bible.groupby("Book Name")['frequency'].sum().reset_index() #adding .reset_index() makes the result a dataframe. 
#This enables us to use sns
result_2 =  bible.groupby("Book Name")['frequency_2'].sum().reset_index()

#sorting results
result_sorted = result.sort_values('frequency', ascending = False)
result_sorted_2 = result_2.sort_values('frequency_2', ascending = False)


#Combining (joining) the 2 dataframes
combined = result_sorted.merge(result_sorted_2,
                              on='Book Name',
                              how='outer',
                            suffixes =('_result_sorted', '_result_sorted_2')
        )

#creating a variable for the Y-axis to be used in the chart title
y_value = 'Book Name'

#chart title, automatically adjust for word and y-axis
title_txt = f"Frequency of the word \"{target_word}\" and \"{target_word_2}\" by {y_value}"



#dropping rows where the frequency is 0 in both columns
combined_drop_empty =~ ((combined['frequency'] == 0 ) & (combined['frequency_2'] == 0))
combined_clean = combined[combined_drop_empty]

sorting = 'frequency'
sorting_options = {'frequency': target_word, 'frequency_2': target_word_2}
sorting = st.selectbox("Sort by", sorting_options.keys(), format_func=lambda x: sorting_options[x])

combined_sorted = combined_clean.sort_values(sorting, ascending = True)



fig = plt.figure(figsize=(10, 14))  #acoomodates for a large y-axis. Second number adjusts y-axis height

plt.barh(combined_sorted['Book Name'], combined_sorted['frequency'], label = target_word)
plt.scatter( combined_sorted['frequency_2'],combined_sorted['Book Name'], label = target_word_2)

plt.xlabel('Frequency')
plt.ylabel('Book Name')
plt.grid(True, linestyle='--', color='gray', alpha=0.3, which='major')
plt.title(title_txt)
plt.legend()
plt.show()

st.pyplot(fig)

st.text("This app uses text from King James Bible")
st.text("If both words selected are not mentioned in some books, those books won't be displayed in the graph ")