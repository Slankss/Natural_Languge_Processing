import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt


news_data = []

# read news data from excel
med_data = pd.read_excel("medicine_news.xlsx")
tech_data = pd.read_excel("technical_news.xlsx")

# add news to news data list
news_data.append(med_data)
news_data.append(tech_data)

# Merge all news
all_news = pd.concat(news_data, ignore_index=True)
all_news.head()

all_news["News"] = all_news["News"].str.lower()

# Tren-Test Bölme
X_train, X_test, y_train, y_test = train_test_split(all_news['News'], all_news['Category '], test_size=0.1,random_state=49)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Naive Bayes Model Eğitimi
model = MultinomialNB()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)

train_accuracy = accuracy_score(y_train,y_train_pred)

y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test,y_pred)
test_news = pd.DataFrame({'NEWS': X_test, 'CATEGORY': y_test})
test_news['PREDICTED_CATEGORY'] = y_pred


print(test_news[['NEWS', 'CATEGORY', 'PREDICTED_CATEGORY']])
print("Train accuracy: {:.2f}%".format(train_accuracy * 100))
print("Test accuracy:", test_accuracy)

cm=confusion_matrix(y_test, y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.title('Confusion Matrix')
plt.show()
