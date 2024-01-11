import pandas as pd
import numpy as np

import text_hammer as th
from tqdm._tqdm_notebook import tqdm_notebook
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer,TFBertModel
from transformers import BertTokenizer, TFBertModel, BertConfig,TFDistilBertModel,DistilBertTokenizer,DistilBertConfig
import shutil
import tensorflow as tf
from keras.layers import Input, Dense,GlobalMaxPool1D,Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.initializers import TruncatedNormal
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.utils import to_categorical
from keras.models import load_model,Model
#from keras.optimizers.schedules import ExponentialDecay
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle

from transformers import TFBertModel


# importing the dataset
df_train = pd.read_csv('training.csv')
df_test = pd.read_csv('test.csv')
df_val = pd.read_csv('validation.csv')
df_full = pd.concat([df_train,df_test,df_val], axis = 0)

#preprocessing
def text_preprocessing(df, col_name):
    column = col_name
    df[column] = df[column].apply(lambda x: str(x).lower())
    df[column] = df[column].apply(lambda x: th.cont_exp(x))
    df[column] = df[column].apply(lambda x: th.remove_emails(x))
    df[column] = df[column].apply(lambda x: th.remove_special_chars(x))
    df[column] = df[column].apply(lambda x: th.remove_accented_chars(x))

    return df

df_cleaned = text_preprocessing(df_full, 'text')
df_cleaned = df_cleaned.copy()
df_cleaned['num_words'] = df_cleaned['text'].apply(lambda x:len(x.split()))
df_cleaned['label'] = df_cleaned['label'].astype('category')

max_len=70

encoded_dict  = {'anger':0,'fear':1, 'joy':2, 'love':3, 'sadness':4, 'surprise':5}
data_train,data_test = train_test_split(df_cleaned, test_size = 0.3, random_state = 42,stratify = df_cleaned['label'])
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
bert = TFBertModel.from_pretrained('bert-base-cased')

#not that imp
tokenizer.save_pretrained('bert-tokenizer')
bert.save_pretrained('bert-model')
shutil.make_archive('bert-tokenizer', 'zip', 'bert-tokenizer')
shutil.make_archive('bert-model','zip','bert-model')


x_train = tokenizer(
    text=data_train['text'].tolist(),
    add_special_tokens=True,
    max_length = 70,
    truncation=True,
    padding=True,
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)


x_test = tokenizer(
    text=data_test['text'].tolist(),
    add_special_tokens=True,
    max_length = 70,
    truncation=True,
    padding=True,
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)


input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

embeddings = bert(input_ids,attention_mask = input_mask)[0] #(0 is the last hidden states,1 means pooler_output)
out = GlobalMaxPool1D()(embeddings)
out = Dense(128, activation='relu')(out)
out = Dropout(0.1)(out)
out = Dense(32,activation = 'relu')(out)
y = Dense(6,activation = 'sigmoid')(out)

model = Model(inputs=[input_ids, input_mask], outputs=y)
model.layers[2].trainable = True
#model.summary()



optimizer = Adam(
    learning_rate=5e-05, # this learning rate is for bert model , taken from huggingface website 
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)

loss =CategoricalCrossentropy(from_logits = True)
metric = CategoricalAccuracy('balanced_accuracy'),

model.compile(
    optimizer = optimizer,
    loss = loss,
    metrics = metric)

#updated
x_train_formatted = {'input_ids': x_train['input_ids'], 'attention_mask': x_train['attention_mask']}
x_test_formatted = {'input_ids': x_test['input_ids'], 'attention_mask': x_test['attention_mask']}

trained_model=load_model('trained_model.h5',custom_objects={"TFBertModel": TFBertModel})
'''
train_history = trained_model.fit(
    x={'input_ids': x_train['input_ids'], 'attention_mask': x_train['attention_mask']},
    y=to_categorical(data_train['label']),
    validation_data = (
    {'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']}, to_categorical(data_test['label'])
    ),
    epochs=3,
    batch_size=36
)

trained_model.save('trained_model.h5')
'''
# Loading the trained model


predicted_raw = trained_model.predict({'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']})
predicted_raw[0]
y_predicted = np.argmax(predicted_raw, axis = 1)
acc=accuracy_score(data_test['label'],y_predicted)
print(classification_report(data_test['label'],y_predicted))
conf=confusion_matrix(data_test['label'],y_predicted)
sns.heatmap(conf,annot=True)
plt.show()
print(acc)