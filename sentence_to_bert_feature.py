# import
import json
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open("/home/miura/brain/src/encoding/LittlePrince/features/data/all_sentence.json") as f:
    dict_sentences = json.load(f)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')

model.to("cuda")

features = np.empty(768, dtype=np.float32)

for i in range(9):
    sentence_list = []
    sentence_list = dict_sentences[str(i+1)]
    input = tokenizer(sentence_list, return_tensors="pt", padding=True,truncation=True)

    input["input_ids"] = input["input_ids"].to("cuda")
    input["token_type_ids"] = input["token_type_ids"].to("cuda")
    input["attention_mask"] = input["attention_mask"].to("cuda")

    outputs = model(**input)
    last_hidden_states = outputs.last_hidden_state

    attention_mask = input.attention_mask.unsqueeze(-1)
    valid_token_num = attention_mask.sum(1)

    sentence_vec = (last_hidden_states*attention_mask).sum(1) / valid_token_num
    sentence_vec = sentence_vec.detach().cpu().numpy()    

    np.save("/home/miura/brain/src/encoding/LittlePrince/features/data/features_bert-cased/features_section_"+str(i+1)+".npy", sentence_vec)
