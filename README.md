# Recommender System

## Installation

Follow These Steps to Install and Set up The Model:
1. Clone the github repository
```bash 
git clone https://github.com/CR23-TR03-PukulEnam-Recommender-System
```

2. Change Directory to Model Folder
```bash
cd model
```

3. Create Python Environment
```bash
python -m venv venv
```

4. Install the requirements
```bash
pip install requirements.txt
```

5. You can try train the data yourself first here
```bash
python Main.py
```
then try the prediction here
```bash
python Predict.py
```

## Data Collecting and Preprocessing
PukulEnam provides real past projects data, however after consideration our team decided to make a dummy data consist of 50.000 rows of past project data to train the model. The dataset comprises four columns of features: Topics, Subtopics, Difficulty, and Project Type. Additionally, there is a label column called Workers. But in the future development project manager can do transfer learning to fit real-life data for more accurate prediction.

![C23-TR03_ Presentation Slides](https://github.com/CR23-TR03-PukulEnam-Recommender-System/model/assets/72967822/a8ed8c08-b704-4131-9508-8312bd9968ee)


Data Preprocessing can be achieve using some Python Library such as:
- Pandas (For pre-processing data from imported CSV)
- SKLearn Model_Selection (For splitting train,validation and test data)
- SKLearn Preprocessing (For Encoding the label into One-Hot Code)
- Numpy (For transform shape of data)
- Tensorflow (For Creating Model and Dense Layer)

### Steps of Preprocessing Data

1. First load the Data CSV and Remove the unused column 
```python
# Load The CSV
df = pd.read_csv(r".\datadummy50k_new_grouped.csv")
# Cleaning the unused columns
df = df.drop(df.columns[[0]], axis=1) 
```

2. Then We Transform the Workers from strings to list, and renaming the other columns
```python 
# Transform 'Workers' from strings into lists
df['Workers'] = df['Workers'].str.replace("[\'\[\]]","",regex=True)
df['Workers'] = df['Workers'].str.replace(", ","|",regex=True)
df['Workers'] = df['Workers'].apply(lambda s: [l for l in str(s).split('|')])

df.rename(columns={'Project Type': 'Project_Type'}, inplace=True)
df.rename(columns={'Sub Topic': 'Sub_Topic'}, inplace=True)

```

3. For encoding the strings column into interger we use enumerate whole categorical columns and transform it to interger, the other columns will be transform into strings

```python
top_dict = dict(enumerate(df["Topics"].astype('category').cat.categories))
subtop_dict = dict(enumerate(df["Sub_Topic"].astype('category').cat.categories))
ptype_dict = dict(enumerate(df["Project_Type"].astype('category').cat.categories))

# Transform columns from string into integer
string_col = ['Topics', 'Sub_Topic', 'Project_Type']
for col in string_col:
  df[col] = df[col].astype('category').cat.codes

# Transform other columns into strings
for col in string_col:
  df[col] = df[col].astype('category')


```
4. For Label column encoding, we use MultilabelBinazer to turn our label to One-hot Encoding 
```python
# Creating list of labels
labels_list = df['Workers']
labels_list = list(labels_list)
mlb = MultiLabelBinarizer()
mlb.fit(labels_list)

N_LABELS = len(mlb.classes_)
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i, label))
```

5. Splitting The data to train and validate
```python

# Making the labels

train, test = train_test_split(df, test_size=0.05)
train, val = train_test_split(train, test_size=0.05)

train_labels = train.pop('Workers')
val_labels = val.pop('Workers')
test_labels = test.pop('Workers')

train_labels = list(train_labels)
val_labels = list(val_labels)
test_labels = list(test_labels)

#Encode each Split Labels
train_labels2 = mlb.transform(train_labels)
val_labels2 = mlb.transform(val_labels)
test_labels2 = mlb.transform(test_labels)
```

## Model Training and Evaluation

### Model Training
For Solving PukulEnam's problem we use Multi-label Classification that outputs each individual compatibility percentages, our model consist of 2 sequential Dense Layers, each layer uses Bias to capture relationships between inputs and outputs therefore increasing the Accuracy, first layer consist of 1000 Neuron, and second layer is 23 Neuron, same as total number of workers.

```python
# get the model
def get_model(n_inputs, n_outputs):
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(1000, input_dim=n_inputs,use_bias=True, kernel_initializer='he_uniform', activation='relu'))
	model.add(tf.keras.layers.Dense(23, use_bias=True, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=tf.keras.metrics.Precision()
    )
	return model

#Getting the shape of data 
n_inputs, n_outputs = train.shape[1], train_labels2.shape[1]
# get model
model = get_model(n_inputs, n_outputs)

#Train the data

# using val set
model.fit(x=train, y=train_labels2, validation_data=(val, val_labels2), epochs=20, verbose=1)

# Evaluating the model using the test set
loss, accuracy = model.evaluate(x=test, y=test_labels2)
print("Accuracy", accuracy)
```


For First Layer activation we used Relu, the reason being is to filter out unrelated feature for each input to get more accurate output, for the second layer activation we used Sigmoid, the reason being is to give the output estimate from 0-1 (In this example is 0-100%) 

![Precision Score Metrics](https://github.com/DaffaKhalishHP/RecommenderSystemPukulEnam/assets/72967822/9d551e1d-49b1-487a-835b-198ed9c0a459)

To Compile the model we use Binary Crossentropy with Adam Optimizer, to calculate the metrics we use default Precision Metrics from tensorflow, the reason is we want the model can label True Positive more and we want to compare it to True Negative

We Measure other Metrics such as F1 Score and Recall

![F1 Score Metrics](https://github.com/DaffaKhalishHP/RecommenderSystemPukulEnam/assets/72967822/01a84710-ed6e-4ead-8ea4-279a9faada46)

![Recall Metrics](https://github.com/DaffaKhalishHP/RecommenderSystemPukulEnam/assets/72967822/e8ba22a8-01b5-42ce-9f63-619a2b4a2db4)

## Using The Predict.py

After you run it we provide of list Type, Topics and Sub Topics. You can try it and the model will produce the team! Type your desire team. Note: If you enter the wrong input the program will raise error

![image](https://github.com/CR23-TR03-PukulEnam-Recommender-System/model/assets/72967822/343f4a39-edce-4262-b8b4-2eebc61dc6fc)

