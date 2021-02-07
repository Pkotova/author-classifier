import pandas as pd
import numpy as np
'''' аuthor_a - yovkov, author_b - vazov'''

with open('yovkov.txt',encoding='utf-8') as аuthor_a, open('vazov.txt',encoding='utf-8') as author_b:

  lines_аuthor_a = []
  lines_vazov = []

  for line in аuthor_a:
    lines_аuthor_a.append(line)

  for line in author_b:
    lines_vazov.append(line)

df_аuthor_a = pd.DataFrame(lines_аuthor_a, columns = ['Text'])
df_аuthor_a['Label'] = 1 
df_аuthor_a['Text'] = df_аuthor_a['Text'].replace({'\t' : ''}, regex = True)\
                                     .replace({'\n' : ''}, regex = True)

df_аuthor_a['Text'].replace('', np.nan, inplace = True)
df_аuthor_a.dropna(subset = ['Text'], inplace = True)

df_author_b = pd.DataFrame(lines_vazov, columns = ['Text'])
df_author_b['Label'] = 0
df_author_b['Text'] = df_author_b['Text'].replace({'\t' : ''}, regex = True)\
                                   .replace({'\n' : ''}, regex = True)

df_author_b['Text'].replace('', np.nan, inplace = True)
df_author_b.dropna(subset = ['Text'], inplace = True)

df = df_аuthor_a.append(df_author_b)

from sklearn.model_selection import train_test_split

X = df['Text'].values
y = df['Label'].values

data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(data_X_train)

X_train = vectorizer.transform(data_X_train)
X_test  = vectorizer.transform(data_X_test)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(max_iter = 180)
classifier.fit(X_train, data_y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score = accuracy_score(data_y_test, y_pred)
print("Accuracy:", round(accuracy_score*100, 2), "%")

with open('test.txt', encoding='utf-8') as check:
  lines_text = []
  for line in check:
    lines_text.append(line)

lines_text_combined = ["".join(lines_text).replace("\n", "")\
                                          .replace("\t", "")]

result = vectorizer.transform(lines_text_combined)

if classifier.predict(result) == 1:
  print("Текстът е от прозиведение на Йордан Йовков.")
else:
  print("Текстът е от прозиведение на Иван Вазов.")