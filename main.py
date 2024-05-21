import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

med_data = pd.read_excel("medicine_news.xlsx")
tech_data = pd.read_excel("technical_news.xlsx")

all_news = pd.concat([med_data,tech_data], ignore_index=True)

all_news["News"] = all_news["News"].str.lower()

news_content_train, news_content_test, news_category_train, news_category_test = train_test_split(all_news['News'], all_news['Category '], test_size=0.1,random_state=47)

vectorizer = TfidfVectorizer()
news_content_train = vectorizer.fit_transform(news_content_train)
news_content_test = vectorizer.transform(news_content_test)

model = MultinomialNB()
model.fit(news_content_train, news_category_train)

train_pred = model.predict(news_content_train)
train_accuracy = accuracy_score(news_category_train,train_pred)

test_pred = model.predict(news_content_test)
test_accuracy = accuracy_score(news_category_test,test_pred)

print("Train accuracy: {:.2f}%".format(train_accuracy * 100))
print("Test accuracy: {:.2f}%".format(test_accuracy * 100))

cm=confusion_matrix(news_category_test, test_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.title('Confusion Matrix')
plt.show()
