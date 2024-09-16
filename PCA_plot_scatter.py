import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from adjustText import adjust_text
import os
import pickle

roi_list = ["inferiortemporal","lateraloccipital","lateralorbitofrontal","medialorbitofrontal","middletemporal","parsorbitalis","rostralmiddlefrontal","frontalpole"]
sub_id_list = [18,22,23,24,28,30,31,35,36,37,38,39,41,42,43,44,45,46,47,48,49,50,51,52,53,
               57,58,61,62,63,64,65,67,68,69,70,72,73,74,75,76,77,78,79,81,82,83,84,86,87,
               88,89,91,92,93,94,95,96,97,98,99,100,101,103,104,105,106,108,109,110,113,114,115]
# sub_id_list = [18,22,23,24,28,30,31,35,36,37,38,39,41,42,43,44,45,46,47,48,49,50,51,52,53] # alice all
# sub_id_list = [57,58,61,62,63,64,65,67,68,69,70,72,73,74,75,76,77,78,79,81,82,83,84,86,87,
#                88,89,91,92,93,94,95,96,97,98,99,100,101,103,104,105,106,108,109,110,113,114,115] # LPP all
# sub_id_list = [23,24,30,31,38,43,44,46,47,48,49,50,51,53,57,61,62,65,67,69,70,72,73,74,78,79,81,82,83,84,93,94,95,96,97,98,99,100,103,104,105,110,113,115] # female
# sub_id_list = [23,24,30,31,38,43,44,46,47,48,49,50,51,53] # alice female
# sub_id_list = [57,61,62,65,67,69,70,72,73,74,78,79,81,82,83,84,93,94,95,96,97,98,99,100,103,104,105,110,113,115] # LPP female
# sub_id_list = [18,22,28,35,36,37,39,41,42,45,52,58,62,63,64,68,75,76,77,86,87,88,89,91,92,101,106,108,109,114] # male
# sub_id_list = [18,22,28,35,36,37,39,41,42,45,52] # alice male
# sub_id_list = [58,62,63,64,68,75,76,77,86,87,88,89,91,92,101,106,108,109,114] # LPP male

annotation = pd.read_excel('/home/miura/brain/annotation_result.xlsx', sheet_name=0).columns.values
annotation.tolist()

out_dir = "/home/miura/brain/src/all_subject_mean/image/scatter_plot/all/all_subject/"
# out_dir = "/home/miura/brain/src/all_subject_mean/image/scatter_plot/all/female/"
# out_dir = "/home/miura/brain/src/all_subject_mean/image/scatter_plot/all/male/"
# out_dir = "/home/miura/brain/src/all_subject_mean/image/scatter_plot/alice/all_subject/"
# out_dir = "/home/miura/brain/src/all_subject_mean/image/scatter_plot/alice/female/"
# out_dir = "/home/miura/brain/src/all_subject_mean/image/scatter_plot/alice/male/"
# out_dir = "/home/miura/brain/src/all_subject_mean/image/scatter_plot/LittlePrince/all_subject/"
# out_dir = "/home/miura/brain/src/all_subject_mean/image/scatter_plot/LittlePrince/female/"
# out_dir = "/home/miura/brain/src/all_subject_mean/image/scatter_plot/LittlePrince/male/"

for roi in roi_list:
    for j,sub_id in enumerate(sub_id_list):
        if sub_id <= 53:
            subject = "sub-"+str(sub_id)
            model_path = "/home/miura/brain/model/with_concat/Ridge/encodingmodel_output_"+subject+"/"
            section = 5
        elif sub_id < 100:
            subject = "sub_EN0"+str(sub_id)
            model_path = "/home/miura/brain/src/encoding/LittlePrince/model/with_concat/Ridge/encodingmodel_output_"+subject+"/"
            section = 9
        else:
            subject = "sub_EN"+str(sub_id)
            model_path = "/home/miura/brain/src/encoding/LittlePrince/model/with_concat/Ridge/encodingmodel_output_"+subject+"/"
            section = 9
        for run in range(section):
            ROI_emotion = np.load(model_path+"run_"+str(run+1)+"/PCA_ROI/each_roi_matrix/"+roi+".npy")
            ROI_emotion[ROI_emotion < 0] = 0 
            if run == 0:
                ROI_emotion_mean = ROI_emotion
            else:
                ROI_emotion_mean += ROI_emotion

        ROI_emotion_mean = ROI_emotion_mean / section

        if j==0:
            ROI_emotion_mean_all_subject = ROI_emotion_mean
        else:
            ROI_emotion_mean_all_subject += ROI_emotion_mean

    ROI_emotion_mean_all_subject = ROI_emotion_mean_all_subject / len(sub_id_list)

    df_ROI_emotion_mean_all_subject = pd.DataFrame(ROI_emotion_mean_all_subject)
    df_ROI_emotion_mean_all_subject.columns = ["Voxel_{}".format(x+1) for x in range(len(df_ROI_emotion_mean_all_subject.columns))]
    df_ROI_emotion_mean_all_subject.index = annotation

    pca = PCA(n_components=2)
    pca.fit(df_ROI_emotion_mean_all_subject)

    ROI_emotion_mean_2d = pca.transform(df_ROI_emotion_mean_all_subject)
    df_ROI_emotion_mean_all_subject_2d = pd.DataFrame(ROI_emotion_mean_2d)
    df_ROI_emotion_mean_all_subject_2d.index = annotation
    df_ROI_emotion_mean_all_subject_2d.columns = ['PC1','PC2']

    df_ROI_emotion_mean_all_subject_2d['emotion_mean'] = pd.Series(ROI_emotion_mean_all_subject.mean(axis=1), index=df_ROI_emotion_mean_all_subject_2d.index)
    emotion_mean_max = df_ROI_emotion_mean_all_subject_2d['emotion_mean'].max()
    emotion_mean_min = df_ROI_emotion_mean_all_subject_2d['emotion_mean'].min()
    emotion_mean_scaled = (df_ROI_emotion_mean_all_subject_2d.emotion_mean-emotion_mean_min) / emotion_mean_max
    df_ROI_emotion_mean_all_subject_2d['emotion_mean_scaled'] = pd.Series(emotion_mean_scaled, index=df_ROI_emotion_mean_all_subject_2d.index)

    df_ROI_emotion_mean_all_subject_2d.to_pickle(out_dir+roi+"_PCA.pkl")

    fig, ax = plt.subplots()
    df_ROI_emotion_mean_all_subject_2d.plot(
            kind='scatter', ax=ax,
            x='PC2',y='PC1',
            cmap="Reds",
            # alpha=0.8,
            vmin=0,
            vmax=0.0001,
            # norm=LogNorm(vmin=-1e7,vmax=1e7),
            c=df_ROI_emotion_mean_all_subject_2d['emotion_mean'],
            s=abs(df_ROI_emotion_mean_all_subject_2d['emotion_mean'])*150000000,
            figsize=(35,25),
            fontsize = 40,
            linewidth = 5)

    texts=[]
    for i, emotion in enumerate(df_ROI_emotion_mean_all_subject_2d.index):
        plt_text = ax.annotate(  
            emotion,
        (df_ROI_emotion_mean_all_subject_2d.iloc[i].PC2, df_ROI_emotion_mean_all_subject_2d.iloc[i].PC1),
        fontsize = 15,
        path_effects=[patheffects.withStroke(linewidth=3, foreground='white', capstyle="round")]
        )
        texts.append(plt_text)
    adjust_text(texts)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    fig.savefig(out_dir+roi+"_scatter.png")
    plt.close(fig)
