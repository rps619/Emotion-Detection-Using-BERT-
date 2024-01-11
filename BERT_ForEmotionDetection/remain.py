import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer,TFBertModel
from transformers import BertTokenizer, TFBertModel, BertConfig,TFDistilBertModel,DistilBertTokenizer,DistilBertConfig
from keras.models import load_model
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import TFBertModel


   
texts = input(str('Text Input : '))
#encoded_dict  = {'anger':0,'fear':1, 'joy':2, 'love':3, 'sadness':4, 'surprise':5}
encoded_dict   =  {'sadness':0,'joy':1, 'love':2, 'anger':3, 'fear':4, 'surprise':5}
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
bert = TFBertModel.from_pretrained('bert-base-cased')

x_val = tokenizer(
    text=texts,
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    padding='max_length', 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True) 


trained_model=load_model('trained_model.h5',custom_objects={"TFBertModel": TFBertModel})



validation = trained_model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})*100
for key , value in zip(encoded_dict.keys(),validation[0]):
    print(key,value)

