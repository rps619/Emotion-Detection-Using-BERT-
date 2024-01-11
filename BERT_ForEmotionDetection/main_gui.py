import pandas as pd
import numpy as np
import tkinter as tk
import time
from transformers import AutoTokenizer,TFBertModel
from transformers import BertTokenizer, TFBertModel, BertConfig,TFDistilBertModel,DistilBertTokenizer,DistilBertConfig
from keras.models import load_model
from transformers import TFBertModel


def computation(texts):

    encoded_dict  = {'anger':0,'fear':1, 'joy':2, 'love':3, 'sadness':4, 'surprise':5}
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    #bert = TFBertModel.from_pretrained('bert-base-cased')

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
    output_list=[]
    for key , value in zip(encoded_dict.keys(),validation[0]):
        print(key,value)
        output_list.append((key,value))

    return output_list


#texts = input(str('input the text'))




def on_button_click():
    
    #process()
    user_input = texts.get()  # Get the text from the input field
    output=computation(user_input)
    result_label.config(text=f"{output}")


# def process():
    
#     root=tk.Tk()
#     label= tk.Label(root, text="processing")
#     label.pack()
#     root.mainloop()
#     root.destroy()
#     root.quit()




#tkinter code starts here

window = tk.Tk()
window.title("Emotion Detector")
window.geometry("400x300")

texts= tk.Entry(window, width=40)
texts.pack(pady=10)  

button = tk.Button(window, text="Go", command=on_button_click)
button.pack()

# Create a label to display the result
result_label = tk.Label(window, text="")
result_label.pack()

# Start the tkinter event loop
window.mainloop()
