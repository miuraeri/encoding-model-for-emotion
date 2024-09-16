import pandas as pd
import textgrid2csv
import json

timing_dict = {}
for section in range(9):
    # load textgrid
    tgrid = textgrid2csv.read_textgrid("/home/miura/brain/src/encoding/LittlePrince/features/data/textgrids/lppEN_section" + str(section+1) + ".TextGrid")

    df=pd.DataFrame(tgrid)

    word_list = df["name"].values.tolist()
    start_list = df["start"].values.tolist()
    stop_list = df["stop"].values.tolist()

    empty_list=[i for i, x in enumerate(word_list) if x=='']

    for i,empty in enumerate(empty_list[::-1]):
        del word_list[empty]
        del start_list[empty]
        del stop_list[empty]
        if i != 0:
            del word_list[empty-1]
            del start_list[empty-1]
            del stop_list[empty-1]

    split_list=[i for i, x in enumerate(word_list) if x=="#"]

    timing_list=[]
    for i,split in enumerate(split_list):
        start_end_list=[]
        if i != len(split_list)-1:
            start_end_list.append(stop_list[split])
            start_end_list.append(start_list[split_list[i+1]])
            timing_list.append(start_end_list)
    
    # dictに追加
    timing_dict[str(section+1)] = timing_list

# JSONファイルに保存
with open('/home/miura/brain/src/encoding/LittlePrince/features/data/timing.json', 'w') as f:
    json.dump(timing_dict, f)

