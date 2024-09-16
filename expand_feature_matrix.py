# import
import json
import numpy as np

brain_measure_length = [286,302,344,307,269,347,329,296,372]

with open("/home/miura/brain/src/encoding/LittlePrince/features/data/timing.json") as f:
    dict_timing = json.load(f)

for section in range(9):
    feature_emotion = np.load("/home/miura/brain/src/encoding/LittlePrince/features/data/features_emotion/features_section_"+str(section+1)+".npy")
    feature_bert = np.load("/home/miura/brain/src/encoding/LittlePrince/features/data/features_bert-cased/features_section_"+str(section+1)+".npy")
    timing_list = []
    timing_list = dict_timing[str(section+1)]
    for i,timing in enumerate(timing_list):
        reading_TR = round((timing[1] - timing[0]) / 2)
        if i==0:
            feature_emotion_expanded = np.tile(feature_emotion[i],(reading_TR,1))
            feature_bert_expanded = np.tile(feature_bert[i],(reading_TR,1))
        else:
            feature_emotion_expanded = np.vstack([feature_emotion_expanded, np.tile(feature_emotion[i],(reading_TR,1))])
            feature_bert_expanded = np.vstack([feature_bert_expanded, np.tile(feature_bert[i],(reading_TR,1))])
            if i == len(timing_list)-1:
                remaining_time = brain_measure_length[section] - feature_emotion_expanded.shape[0]
                feature_emotion_expanded = np.vstack([feature_emotion_expanded, np.tile(feature_emotion[i],(remaining_time,1))])
                feature_bert_expanded = np.vstack([feature_bert_expanded, np.tile(feature_bert[i],(remaining_time,1))])

    feature_concated = np.hstack([feature_emotion_expanded,feature_bert_expanded])
    np.save("/home/miura/brain/src/encoding/LittlePrince/features/data/features_concated/features_section_"+str(section+1)+".npy", feature_concated)
