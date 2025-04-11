import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
style.use('fivethirtyeight')


#reading data 
tests = pd.read_csv('StudentsPerformance.csv')
#Creating shorter, easier to type column names
new_columns = ['gender', 'race', 'parent_ed', 'lunch', 'prep_course', 'math', 'reading', 'writing']

#assigning new column names
tests.columns = new_columns

st.title("Analysis of Student Performance in Exams Data")
st.header('A Kaggle Datasel')
st.write('Data from https://www.kaggle.com/datasets/spscientist/students-performance-in-exams/data')

st.header('Visualizing the Data')
fig, ax = plt.subplots(1,3,figsize = (14,4))
ax[0].hist(tests['math'])
#ax[0].set_xlabel('Score')
ax[0].set_title('Math')
ax[0].set_ylabel('Frequency')

ax[1].hist(tests['reading'])
ax[1].set_xlabel('Score')
ax[1].set_title('Reading')

ax[2].hist(tests['writing'])
ax[2].set_title('Writing')

st.pyplot(fig)

st.write('This is what the first few rows of the dataset look like')

st.dataframe(tests.head())

st.markdown("""
### Questions we'll try to answer from the data:
1. Gender:
    - Who has higher average scores? Males or females?
    - Who is scoring higher per test topic?
2. Test Prep:
    - Does whether a studen did test prep courses affect their average test scores?
    - Does it benefit either gender more?
    - Does benefit students score better in certain tests? If so, which tests?
3. Parent's Education:
    - Does the higher the parent's education, the higher their kids' scores?
    - Does the parent's education affect certain tests more than others?
    - Does it affect one gender over the other?

4. Race:
    - Does race seem to affect test scores?
    - If so how?

5. Lunch:
    - Does the type of lunch taken by a student affect their score?

6. A combination of all factors:
    - Looking at all factors together, which factor have the highest weight in driving students score?
    - Does it vary by gender/test type?
            """)

st.markdown("""- Let's see how the data splits by gender""")

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
# Observations
- Males have a higer median Math score than females
- Females have higher median scores in Reading and Writing

### next, let's dive deeper into what/if other factors affect test scores            
""")

st.markdown("""
            # Test Prep
### Does test prep have an effect on test scores?
""")

#grouping by `prep_course`
prep_group = tests.groupby('prep_course') 

#creating prep and no_prep groupings
prep = prep_group.get_group('completed')
no_prep = prep_group.get_group('none')


#Calculating Averages

#Prep
prep_scores = prep[['math', 'reading', 'writing']]
prep_avg_scores = prep_scores.agg(['mean', 'min', 'max', 'median'])
print("Test Prep Scores")
#prep_avg_scores

#No Prep
no_prep_scores = no_prep[['math', 'reading', 'writing']]
no_prep_avg_scores = no_prep_scores.agg(['mean', 'min', 'max', 'median'])
print("No Test Prep Scores")
#no_prep_avg_scores


#Creating variables for medians
p_math_median = prep_avg_scores.loc['median','math']
p_read_median = prep_avg_scores.loc['median','reading']
p_write_median = prep_avg_scores.loc['median','writing']
np_math_median = no_prep_avg_scores.loc['median','math']
np_read_median = no_prep_avg_scores.loc['median','reading']
np_write_median = no_prep_avg_scores.loc['median','writing']



fig = plt.figure(figsize=(14,6))

#Test Prep plots

plt.subplot(2,3,1)
plt.hist(prep['math'])
plt.xticks([0,100,50, p_math_median])
plt.yticks([0,100,200])
plt.ylim(0,200) #adding y limits for easier comparision 
plt.ylabel('Test Prep')
plt.title('Math')
plt.axvline(prep_avg_scores.loc['median', 'math'], c='yellow')
plt.legend(['Median'], loc='upper left')
#m_median_math


plt.subplot(2,3,2)
plt.hist(prep['reading'])
plt.xticks([0,100,50, p_read_median])
plt.yticks([0,100,200])
plt.ylim(0,200) #adding y limits for easier comparision 
plt.title('Reading')
plt.axvline(prep_avg_scores.loc['median', 'reading'], c='yellow')
plt.legend(['Median'], loc='upper left')


plt.subplot(2,3,3)
plt.hist(prep['writing'])
plt.xticks([0,100,50, p_write_median])
plt.yticks([0,100,200])
plt.ylim(0,200) #adding y limits for easier comparision 
plt.title('Writing')
plt.axvline(prep_avg_scores.loc['median', 'writing'], c='yellow')
plt.legend(['Median'], loc='upper left')

#No Test Plots
plt.subplot(2,3,4)
plt.hist(no_prep['math'])
plt.ylabel('No Test Prep')
plt.xlabel('Score')
plt.xticks([0,100,50, np_math_median])
plt.yticks([0,100,200])
plt.ylim(0,200) #adding y limits for easier comparision 
plt.axvline(no_prep_avg_scores.loc['median', 'math'], c='yellow')
plt.legend(['Median'], loc='upper left')

plt.subplot(2,3,5)
plt.hist(no_prep['reading'])
plt.xlabel('Score')
plt.xticks([0,100,50, np_read_median])
plt.yticks([0,100,200])
plt.ylim(0,200) #adding y limits for easier comparision 
plt.axvline(no_prep_avg_scores.loc['median', 'reading'], c='yellow')
plt.legend(['Median'], loc='upper left')

plt.subplot(2,3,6)
plt.hist(no_prep['writing'])
plt.xlabel('Score')
plt.xticks([0,100,50,np_write_median])
plt.yticks([0,100,200])
plt.ylim(0,200) #adding y limits for easier comparision 
plt.axvline(no_prep_avg_scores.loc['median', 'writing'], c='yellow')
plt.legend(['Median'], loc='upper left')

st.pyplot(fig)


st.markdown("""
            # Observations 
- Majority of students do not take test prep for all tests
- The __median test score for all tests is higher for those students that take test prep___
- The average test score for math is the __same__ for students that took a test prep and those who didn't
- The average score for reading is __higher__ for students that took test prep
- The average score for writing is __higher__ for students that took test prep
            """)

st.markdown("""
       ## Exploring effect of parent's education on student's test score     
            """)

#group by parent's education

parent_ed_grouped = tests.groupby('parent_ed')



#some high school
s_high_school = parent_ed_grouped.get_group('some high school')
s_high_school_scores = s_high_school[['math', 'reading', 'writing']]
s_high_school_avg = s_high_school_scores.agg(['mean', 'min', 'max', 'median'])
print('Some High School')
print(s_high_school_avg)


#high school
high_school = parent_ed_grouped.get_group('high school')
high_school_scores = high_school[['math', 'reading', 'writing']]
high_school_avg = high_school_scores.agg(['mean', 'min', 'max', 'median'])
print()
print('High School')
print(high_school_avg)


#some college
s_college = parent_ed_grouped.get_group('some college')
s_college_scores = s_college[['math', 'reading', 'writing']]
s_college_avg = s_college_scores.agg(['mean', 'min', 'max', 'median'])
print()
print('Some College')
print(s_college_avg)


#associate's degree
associate = parent_ed_grouped.get_group("associate's degree")
associate_scores = associate[['math', 'reading', 'writing']]
associate_avg = associate_scores.agg(['mean', 'min', 'max', 'median'])
print()
print("Associate's Degree")
print(associate_avg)


#bachelor's degree
bachelor = parent_ed_grouped.get_group("bachelor's degree")
bachelor_scores = bachelor[['math', 'reading', 'writing']]
bachelor_avg = bachelor_scores.agg(['mean', 'min', 'max', 'median'])
print()
print("Bachelor's Degree")
print(bachelor_avg)


#master's degree
master = parent_ed_grouped.get_group("master's degree")
master_scores = master[['math', 'reading', 'writing']]
master_avg = master_scores.agg(['mean', 'min', 'max', 'median'])
print()
print("Master's Degree")
print(master_avg)


#Creating variables for medians

#Some high School
s_high_school_math_median = s_high_school_avg.loc['median', 'math']
s_high_school_read_median = s_high_school_avg.loc['median', 'reading']
s_high_school_write_median = s_high_school_avg.loc['median', 'writing']

#high school
high_school_math_median = s_high_school_avg.loc['median', 'math']
high_school_read_median = s_high_school_avg.loc['median', 'reading']
high_school_write_median = s_high_school_avg.loc['median', 'writing']

#Some college
s_college_math_median = s_college_avg.loc['median', 'math']
s_college_read_median = s_college_avg.loc['median', 'reading']
s_college_write_median = s_college_avg.loc['median', 'writing']

#Associate's
associate_math_median = associate_avg.loc['median', 'math']
associate_read_median = associate_avg.loc['median', 'reading']
associate_write_median = associate_avg.loc['median', 'writing']

#Bachelor's Degree
bachelor_math_median = bachelor_avg.loc['median', 'math']
bachelor_read_median = bachelor_avg.loc['median', 'reading']
bachelor_write_median = bachelor_avg.loc['median', 'writing']

#Master's Degree
master_math_median = master_avg.loc['median', 'math']
master_read_median = master_avg.loc['median', 'reading']
master_write_median = master_avg.loc['median', 'writing']


#Visualizations

fig = plt.figure(figsize=(14,18))

#Test Prep plots

#Some high school
plt.subplot(6,3,1)
plt.hist(s_high_school['math'])
plt.xticks([0,100,50, s_high_school_math_median])
plt.yticks([0,50,100,200])
plt.ylim(0,75) #adding y limits for easier comparision 
plt.ylabel('Some High School')
plt.title('Math')
plt.axvline(s_high_school_math_median, c='yellow')
plt.legend(['Median'], loc='upper left')

plt.subplot(6,3,2)
plt.hist(s_high_school['reading'])
plt.xticks([0,100,50, s_high_school_read_median])
plt.yticks([0,50,100,200])
plt.ylim(0,75) #adding y limits for easier comparision 
plt.title('Reading')
plt.axvline(s_high_school_read_median, c='yellow')
plt.legend(['Median'], loc='upper left')

plt.subplot(6,3,3)
plt.hist(s_high_school['writing'])
plt.xticks([0,100,50, s_high_school_write_median])
plt.yticks([0,50,100,200])
plt.ylim(0,75) #adding y limits for easier comparision 
plt.title('Writing')
plt.axvline(s_high_school_write_median, c='yellow')
plt.legend(['Median'], loc='upper left')


#High school
plt.subplot(6,3,4)
plt.hist(high_school['math'])
plt.xticks([0,100,50, high_school_math_median])
plt.yticks([0,50,100,200])
plt.ylim(0,75) #adding y limits for easier comparision 
plt.ylabel('High School')
plt.axvline(high_school_math_median, c='yellow')
plt.legend(['Median'], loc='upper left')

plt.subplot(6,3,5)
plt.hist(high_school['reading'])
plt.xticks([0,100,50, high_school_read_median])
plt.yticks([0,50,100,200])
plt.ylim(0,75) #adding y limits for easier comparision 
plt.axvline(high_school_read_median, c='yellow')
plt.legend(['Median'], loc='upper left')

plt.subplot(6,3,6)
plt.hist(high_school['writing'])
plt.xticks([0,100,50, high_school_write_median])
plt.yticks([0,50,100,200])
plt.ylim(0,75) #adding y limits for easier comparision 
plt.axvline(high_school_write_median, c='yellow')
plt.legend(['Median'], loc='upper left')

#Some college
plt.subplot(6,3,7)
plt.hist(s_college['math'])
plt.xticks([0,100,50, s_college_math_median])
plt.yticks([0,50,100,200])
plt.ylim(0,75) #adding y limits for easier comparision 
plt.ylabel('Some College')
plt.axvline(s_college_math_median, c='yellow')
plt.legend(['Median'], loc='upper left')

plt.subplot(6,3,8)
plt.hist(s_college['reading'])
plt.xticks([0,100,50, s_college_read_median])
plt.yticks([0,50,100,200])
plt.ylim(0,75) #adding y limits for easier comparision  
plt.axvline(s_college_read_median, c='yellow')
plt.legend(['Median'], loc='upper left')

plt.subplot(6,3,9)
plt.hist(s_college['writing'])
plt.xticks([0,100,50, s_college_write_median])
plt.yticks([0,50,100,200])
plt.ylim(0,75) #adding y limits for easier comparision 
plt.axvline(s_college_write_median, c='yellow')
plt.legend(['Median'], loc='upper left')

#Associate's 
plt.subplot(6,3,10)
plt.hist(associate['math'])
plt.xticks([0,100,50, associate_math_median])
plt.yticks([0,50,100,200])
plt.ylim(0,75) #adding y limits for easier comparision 
plt.ylabel("Associate's Degree")
plt.axvline(associate_math_median, c='yellow')
plt.legend(['Median'], loc='upper left')

plt.subplot(6,3,11)
plt.hist(associate['reading'])
plt.xticks([0,100,50, associate_read_median])
plt.yticks([0,50,100,200])
plt.ylim(0,75) #adding y limits for easier comparision 
plt.axvline(associate_read_median, c='yellow')
plt.legend(['Median'], loc='upper left')

plt.subplot(6,3,12)
plt.hist(associate['writing'])
plt.xticks([0,100,50, associate_write_median])
plt.yticks([0,50,100,200])
plt.ylim(0,75) #adding y limits for easier comparision 
plt.axvline(associate_write_median, c='yellow')
plt.legend(['Median'], loc='upper left')

#Bachelor's
plt.subplot(6,3,13)
plt.hist(bachelor['math'])
plt.xticks([0,100,50, bachelor_math_median])
plt.yticks([0,50,100,200])
plt.ylim(0,75) #adding y limits for easier comparision 
plt.ylabel("Bachelor's Degree")
plt.axvline(bachelor_math_median, c='yellow')
plt.legend(['Median'], loc='upper left')

plt.subplot(6,3,14)
plt.hist(bachelor['reading'])
plt.xticks([0,100,50, bachelor_read_median])
plt.yticks([0,50,100,200])
plt.ylim(0,75) #adding y limits for easier comparision 
plt.axvline(bachelor_read_median, c='yellow')
plt.legend(['Median'], loc='upper left')

plt.subplot(6,3,15)
plt.hist(bachelor['writing'])
plt.xticks([0,100,50, bachelor_write_median])
plt.yticks([0,50,100,200])
plt.ylim(0,75) #adding y limits for easier comparision 
plt.axvline(bachelor_write_median, c='yellow')
plt.legend(['Median'], loc='upper left')

#Master's
plt.subplot(6,3,16)
plt.hist(master['math'])
plt.xticks([0,100,50, master_math_median])
plt.yticks([0,50,100,200])
plt.ylim(0,75) #adding y limits for easier comparision 
plt.ylabel("Master's Degree")
plt.axvline(master_math_median, c='yellow')
plt.legend(['Median'], loc='upper left')

plt.subplot(6,3,17)
plt.hist(master['reading'])
plt.xticks([0,100,50, master_read_median])
plt.yticks([0,50,100,200])
plt.ylim(0,75) #adding y limits for easier comparision 
plt.axvline(master_read_median, c='yellow')
plt.legend(['Median'], loc='upper left')

plt.subplot(6,3,18)
plt.hist(master['writing'])
plt.xticks([0,100,50, master_write_median])
plt.yticks([0,50,100,200])
plt.ylim(0,75) #adding y limits for easier comparision 
plt.axvline(master_write_median, c='yellow')
plt.legend(['Median'], loc='upper left')

st.pyplot(fig)

st.markdown("""
            # Observations 
- Parent's education appears to have an effect on student's test scores
- The higher the parent's education, the higher the median test score
- Number of students with parents who have `Some highschool`, `High School` or `Some College` education is about the same
- Less students have parents with `Associate's Degree`
- Even less students have parents with `Bachelor's Degree`
- Even less still are students whose parents havea `Master's Degree`
            """)