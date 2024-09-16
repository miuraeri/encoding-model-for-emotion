# import
import pandas as pd
import textgrid2csv
import json

sentences_dict = {}
for section in range(9):
    # load textgrid
    tgrid = textgrid2csv.read_textgrid("/home/miura/brain/src/encoding/LittlePrince/features/data/textgrids/lppEN_section" + str(section+1) + ".TextGrid")

    # convert textgrid to dataframe
    df=pd.DataFrame(tgrid)
    # 文の列を抽出
    word_list = df["name"].values.tolist()

    # 一つの文章にする
    words = ""
    for word in word_list:
        words = words + word + " "

    # "#" で文章を区切る
    l = [x.strip() for x in words.split('#')]
    l = [y for y in l if y != ""]

    # 短縮形などの表現を修正
    for i,sentence in enumerate(l):
        sentence=sentence.replace("\\", "")
        sentence=sentence.replace(" t ", "'t ")
        sentence=sentence.replace("i ", "I ")
        sentence=sentence.replace(" m ", "'m ")
        sentence=sentence.replace(" d ", "'d ")
        sentence=sentence.replace(" ll ", "'ll ")
        sentence=sentence.replace(" s ", "'s ")
        sentence=sentence.replace(" ve ", "'ve ")
        sentence=sentence.replace(" re ", "'re ")
        l[i]=sentence

    # dictに追加
    sentences_dict[str(section+1)] = l

# JSONファイルに保存
with open('/home/miura/brain/src/encoding/LittlePrince/features/data/all_sentence.json', 'w') as f:
    json.dump(sentences_dict, f)
