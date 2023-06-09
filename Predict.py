import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


# PRE PROCESSING DATA

# ganti directorynya 
df = pd.read_csv(r".\datadummy50k_new_grouped.csv")

# Cleaning the unused columns
df = df.drop(df.columns[[0]], axis=1)
# Create dictionaries
# Assuming you have a DataFrame called 'df' and you want to rename the column 'old_column' to 'new_column'
df.rename(columns={'Project Type': 'Project_Type'}, inplace=True)
df.rename(columns={'Sub Topic': 'Sub_Topic'}, inplace=True)


top_dict = dict(enumerate(df["Topics"].astype('category').cat.categories))
subtop_dict = dict(enumerate(df["Sub_Topic"].astype('category').cat.categories))
ptype_dict = dict(enumerate(df["Project_Type"].astype('category').cat.categories))

print(top_dict, subtop_dict, ptype_dict)

# Transform 'Workers' from strings into lists
df['Workers'] = df['Workers'].str.replace("[\'\[\]]","",regex=True)
df['Workers'] = df['Workers'].str.replace(", ","|",regex=True)
df['Workers'] = df['Workers'].apply(lambda s: [l for l in str(s).split('|')])


top_dict = dict(enumerate(df["Topics"].astype('category').cat.categories))
subtop_dict = dict(enumerate(df["Sub_Topic"].astype('category').cat.categories))
ptype_dict = dict(enumerate(df["Project_Type"].astype('category').cat.categories))
mlb = MultiLabelBinarizer()


print(top_dict, subtop_dict, ptype_dict)

string_col = ['Topics', 'Sub_Topic', 'Project_Type']

# Transform columns from string into integer

for col in string_col:
  df[col] = df[col].astype('category').cat.codes

for col in string_col:
  df[col] = df[col].astype('category')

# Creating list of labels
labels_list = df['Workers']
labels_list = list(labels_list)
mlb = MultiLabelBinarizer()
mlb.fit(labels_list)

N_LABELS = len(mlb.classes_)
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i, label))


# LOAD THE MODEL
base_model = tf.keras.models.load_model(r'.\model.h5')

# THE MODEL HERE

print("Choose Down Below\n")

topics = [
    ("Project Type", "Back End"),
    ("Topic", "Back End"),
    ("Sub Topic", ["Django", "Node.js", "Express.js"]),
    ("Project Type", "Front End"),
    ("Topic", "Front End"),
    ("Sub Topic", ["React", "Ember.js", "Angular"]),
    ("Project Type", "ML"),
    ("Topic", "Time-series"),
    ("Sub Topic", ["LSTM", "ARIMA"]),
    ("Project Type", "ML"),
    ("Topic", "Speech / Audio"),
    ("Sub Topic", ["Speech Recognition", "Music Information Retrieval"]),
    ("Project Type", "ML"),
    ("Topic", "NLP"),
    ("Sub Topic", ["Topic Modeling", "Sentiment Analysis"]),
    ("Project Type", "ML"),
    ("Topic", "Data Engineering"),
    ("Sub Topic", ["Data Warehousing"]),
    ("Project Type", "ML"),
    ("Topic", "Computer Vision"),
    ("Sub Topic", ["Object Detection"]),
    ("Project Type", "ML"),
    ("Topic", "Classification & Regression"),
    ("Sub Topic", ["Logistic Regression", "Linear Regression"]),
]

for topic in topics:
    if isinstance(topic[1], str):
        print(f"{topic[0]}\n{topic[1]}\n")
    else:
        print(f"{topic[0]}\n{', '.join(topic[1])}\n")
        print("-" * 17)

    




# Testing predictions
testype = input('Project Type : ') #Input here
testopic = input('Topics : ') #Input here
testopicsub = input('Sub Topic : ') #Input here
testdif = int(input('Input Diff : ')) #Input here

#Get key of a dictionary
def get_key(d, val):
    return [k for k, v in d.items() if v == val]

testX = [get_key(ptype_dict, testype)[0], get_key(top_dict, testopic)[0],
         get_key(subtop_dict, testopicsub)[0], testdif]
testX = np.asarray([testX])
yhat = base_model.predict(testX)[0]

# Converting the prediction into dataframe
predf = pd.DataFrame(yhat, index=mlb.classes_)
predf = predf.multiply(100).round(0).sort_values(by=0, ascending=False)
predf = predf[predf[0] >= 1]
print(predf)

# List Of Activate Talents

activetalents = ['Nyoman Satiya Najwa Sadha', 'Rikip Ginanjar', 'I Putu Ranantha Nugraha Suparta', 'Putu Gede Agung Karna Sampalan',
                 'Sarah Sema Khairunisa', 'Christopher Kristianto', 'Azis Sofyanto', 'Alvin Tan', 'Suci Rahmadani']

besteam = predf.filter(items=activetalents, axis=0)
besteam = besteam.sort_values(by=0, ascending=False).head(5)
besteam = besteam.set_axis(['Prediction'], axis='columns')

print(besteam)