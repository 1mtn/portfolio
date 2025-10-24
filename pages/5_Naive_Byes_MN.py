import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset 
df = pd.read_csv('SMSSpamCollection',sep='\t',header=None, names=['label', 'sms'])


# Preprocess labels: ham=0, spam=1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})


# Extract texts and labels
texts = df['sms']
labels = df['label']

# Initialize and fit vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=37)

# Train the Multinomial Naive Bayes model
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Predict on test set for evaluation
y_pred = mnb.predict(X_test)

# Streamlit App
st.title('SMS Spam Classifier - Multinomial Naive Bayes Model')

st.header('Project Overview')
st.write("""
This app demonstrates a Multinomial Naive Bayes model for classifying SMS messages as spam or ham (non-spam).
The model is trained on the SMS Spam Collection dataset, which contains 5,574 labeled messages.
We use bag-of-words (CountVectorizer) for feature extraction and MultinomialNB for classification.
""")

st.header('Try It Yourself!')
sms_input = st.text_area('Enter the SMS text here and see if the model detects it as Ham or Spam:')
if st.button('Classify'):
    if sms_input:
        # Vectorize the input
        input_vec = vectorizer.transform([sms_input])
        # Predict
        prediction = mnb.predict(input_vec)[0]
        prob = mnb.predict_proba(input_vec)[0]
        result = 'Spam' if prediction == 1 else 'Ham'
        st.write(f'**Prediction:** {result}')
        st.write(f'**Confidence (Ham | Spam):** {prob[0]:.2f} | {prob[1]:.2f}')
    else:
        st.write('Please enter some text.')

st.header('Dataset Sample')
st.dataframe(df.head(10))

st.header('How The Model Works')
st.markdown("""
- Data was split into 80% __Training Set__ and 20% __Test Set__ 
- Using the __Training Set__
    - The text of all Spam and Non Spam SMS was split into a bag of words forming a __Vocabulary__
    - Each SMS was split into a list of words
    - For Spam SMS, the probability of the word appearing on spam SMS was calculated $P(Word|Spam)$
        - Calculted by dividing total times a word appeared in a spam SMS by the total number of words that appeared in a spam SMS 
    - For Non-spam messages, calculated the probability of each word appearing on a ham (non-spam) SMS $P(Word|Ham)$
        - Calculted by dividing total times a word appeared in a Ham SMS by the total number of words that appeared in a Ham SMS 

            
- Based on the training set, a general probability of an SMS being Spam $P(Spam)$ or $P(Ham)$ is calculated
- For ever new SMS coming, the probabilities are calculated as follows:
    -The probability of the new message being ham is multiplied by the probability of each word on the SMS being ham. The result is $P(Ham)$
    -The probability of the new message being Spam is multiplied by the probability of each word on the SMS being Spam. The result is $P(Spam)$
- __For a new SMS, if $P(Ham) > P(Spam)$ the message is classified as Ham and vice versa__
""")

st.header('Model Performance')
accuracy = accuracy_score(y_test, y_pred)
st.write(f'**Accuracy:** {accuracy:.2f}')
st.write("*(Accuracy measures the proportion of correctly classified messages (both ham and spam) out of all predictions.)*")


st.subheader('Classification Report')
st.text(classification_report(y_test, y_pred))
st.write("""
**Precision:** The proportion of predicted spam (or ham) that is actually spam (or ham). High precision means fewer false positives (e.g., ham flagged as spam).  
**Recall:** The proportion of actual spam (or ham) that is correctly identified. High recall means fewer false negatives (e.g., spam missed as ham).  
**F1-Score:** The harmonic mean of precision and recall, balancing both metrics—useful for imbalanced data like this data set (This dataset has 87% Ham SMS).
""")

st.subheader('Confusion Matrix')
cm = confusion_matrix(y_test, y_pred)
st.write(pd.DataFrame(cm, columns=['Predicted Ham', 'Predicted Spam'], index=['Actual Ham', 'Actual Spam']))
st.write("*(The confusion matrix shows the counts of true positives (spam correctly identified), true negatives (ham correctly identified), false positives (ham misclassified as spam), and false negatives (spam misclassified as ham).)*")


