import pandas as pd
import nltk
from nltk.corpus import stopwords
import string 



df = pd.read_csv('Reviews.csv')

def get_sentiment(Score) :
    if Score >= 4:
        return 'postive'

    elif Score == 3 :
        return 'neutral'

    else :
        return 'negative'

df['sentiment'] = df['Score'].apply(get_sentiment)

#Text preprocessing
nltk.download('stopwords')
stops_words =set(stopwords.words('english'))

def clean_review(text):
    text  = str(text).lower()
    text  = ''.join( c for c in text if c not in string.punctuation)
    words = text.split()
    words = [w for w in words if w not in stops_words]
    return ''.join(words)
df['clean_text'] = df['Text'].apply(clean_review)


# FEATURE EXTRACTION

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features= 5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']


# TRAIN /TEST SPLIT
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X , y ,test_size =.02 , random_state= 42 ,stratify=y)

# TRAIN A CLASSIFIER

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train ,y_train)

# EVALUATION

from sklearn.metrics import classification_report , confusion_matrix

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


sample_reviews = [
    "Great product, works perfectly!",
    "It's average, nothing special.",
    "Really bad, broke after one use."
]
X_sample = vectorizer.transform(sample_reviews)
predictions = model.predict(X_sample)
for review, pred in zip(sample_reviews, predictions):
    print(f"Review: {review}\nPredicted Sentiment: {pred}\n")

from sklearn.linear_model import LogisticRegression

# Train logistic regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(class_weight='balanced')
clf.fit(X_train, y_train)


# Evaluate
y_pred_lr = clf.predict(X_test)
print("Logistic Regression:\n", classification_report(y_test, y_pred_lr))

from sklearn.svm import SVC

# Train Linear SVM
svm_model = SVC(class_weight='balanced')
svm_model.fit(X_train, y_train)

# Evaluate
y_pred_svm = svm_model.predict(X_test)
print("SVM:\n", classification_report(y_test, y_pred_svm))
