# import
import json
import numpy as np
from transformers import BertTokenizer, BertModel
# GoEmotions-pytorch のmodel.pyを同じディレクトリに置いておく
from model import BertForMultiLabelClassification
import torch

with open("/home/miura/brain/src/encoding/LittlePrince/features/data/all_sentence.json") as f:
    dict_sentences = json.load(f)

tokenizer = BertTokenizer.from_pretrained("/home/miura/brain/src/Finetuning_by_AliceAnnotation/ckpt/alice/goemotions-to-alice-5step/checkpoint-55")
model = BertForMultiLabelClassification.from_pretrained("/home/miura/brain/src/Finetuning_by_AliceAnnotation/ckpt/alice/goemotions-to-alice-5step/checkpoint-55")

for i in range(9):
    score = np.empty(80)
    sentence_list = []
    sentence_list = dict_sentences[str(i+1)]
    for txt in sentence_list:
        inputs = tokenizer(txt,return_tensors="pt")
        outputs = model(**inputs)
        scores =  1 / (1 + torch.exp(-outputs[0]))  # Sigmoid
        for item in scores:
            scores = []
            for idx, s in enumerate(item):
                sl = s.tolist()
                scores.append(sl)
        score = np.vstack([score, scores])
    score = np.delete(score, 0, 0)

    np.save("/home/miura/brain/src/encoding/LittlePrince/features/data/features_emotion/features_section_"+str(i+1)+".npy", score)
