import random
import argparse
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
# from himalaya.ridge import Ridge
# from himalaya.backend import set_backend
from statsmodels.stats.multitest import fdrcorrection as fdr
import numpy as np
from scipy import stats
import os
import pandas as pd

# GPU を使用するよう変更する
# backend = set_backend("torch_cuda")

# コマンドライン引数を設定
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--ridge_type", type=str,
                    choices=["Ridge","KernelRidge"],
                    default="Ridge",
                    help="input ridge type")
parser.add_argument("-m", "--mode", type=str,
                    choices=["without_concat","with_concat", "run_test"],
                    default="with_concat",
                    help="input feature used")
args = parser.parse_args()

ridge_type = args.ridge_type
mode = args.mode

# 関数の定義
'''
    ペアデータを作る
    
    <<parameters>>
    feature: ペアデータにするデータ
    width: 時間幅
    delay: 遅延
    
    <<return>>
    paired_data: ペアデータになったもの
'''
def make_pair_data(feature, width, delay):
    for i in range(width):
        a = np.roll(feature, delay+i, axis=0)
        if i==0:
            paired_data = a
        else:
            paired_data = np.concatenate((paired_data, a), axis=1)
    return paired_data


'''
    cvに使用するインデックスを決める
    
    <<parameters>>
    samplenum: サンプル数
    chunklen: 何個のデータを連続した塊と見るか
    n_cv: 何foldか
    
    <<return>>
    cv_inds: 辞書。n_cv回分のidが入っている
'''
def get_cv_inds(samplenum, chunklen, n_cv):
    # 塊を作ってシャッフル
    allinds = range(samplenum)
    indchunks = list(zip(*[iter(allinds)]*chunklen))
    random.shuffle(indchunks)
    inds = np.array(indchunks).flatten()
    
    cv_inds = []
    kf = KFold(n_splits=n_cv, shuffle=False)
    for trn_ind, val_ind in kf.split(inds):
        cv_inds.append({'trn_ind': inds[trn_ind], 'val_ind': inds[val_ind]})
    return cv_inds

'''
    pを求める
'''
def compute_p(corrs, n):
    t = np.dot(corrs, np.sqrt(n - 2)) / np.sqrt(1 - np.power(corrs, 2))
    p = (1 - stats.t.cdf(np.absolute(t), n - 2)) * 2
    return p


annotation = pd.read_excel('/home/miura/brain/annotation_result.xlsx', sheet_name=0).columns.values

sub_id_list = [18,22,23,24,28,30,31,35,36,37,38,39,41,42,43,44,45,46,47,48,49,50,51,52,53,
               57,58,61,62,63,64,65,67,68,69,70,72,73,74,75,76,77,78,79,81,82,83,84,86,87,
               88,89,91,92,93,94,95,96,97,98,99,100,101,103,104,105,106,108,109,110,113,114,115]

for sub_id in sub_id_list:
    if sub_id <= 53:
        subject = "sub-"+str(sub_id)
        section = 5
    elif sub_id < 100:
        subject = "sub_EN0"+str(sub_id)
        section = 9
    else:
        subject = "sub_EN"+str(sub_id)
        section = 9

    print(subject+":start")
    for run in range(section):
        if sub_id <= 53:
            n_cv = 5 # 何fold cross-validationをするか
            brain_path = '/home/miura/brain/pycortex_project/resource/Alice/' + subject + "/normalization/split_5data_line_trilinear/"
            brain_files= []

            for i in range(5):
                brain_files.append(subject + "_" + str(i+1) + ".npy")

            feature_path = '/home/miura/brain/Alice/features/' # 特徴量の .npy を作成したら設定
            if mode == "without_concat":
                feature_file = 'emotion_features_365.npy'
            elif mode == "with_concat":
                feature_file = 'concat_features_365.npy'
            else:
                feature_file = 'emotion_features_365.npy'

            model_dir = '/home/miura/brain/model/'+ mode + "/" + ridge_type +'/encodingmodel_output_sub-' + str(sub_id) +"/"
            out_data_dir = model_dir +"run_" + str(run+1) + "/PCA_ROI/matrix/"
            new_out_data_dir = out_data_dir + "each_emotion/" 
            os.makedirs(new_out_data_dir, exist_ok=True)

            # 特徴量の読み込み
            cut_timing = [47,162,241,305]
            with open(feature_path + feature_file, 'rb') as f:
                feature = np.load(f)

            for j in range(len(brain_files)):
                with open(brain_path + brain_files[j], 'rb') as f:
                    data = np.load(f)
                if run!=0 and j==0: # リッジ回帰が1回目でなくて最初のファイルを読み込むとき
                    brain_trn = data # 最初のファイルは訓練用として使う
                elif run==0 and j==1: # リッジ回帰が1回目で2つ目のファイルを読み込むとき
                    brain_trn = data # 最初のファイルはテスト用になるので、2つ目のファイルを訓練用として登録
                elif j==run:
                    brain_test = data
                else:
                    brain_trn = np.vstack([brain_trn, data])

            # 特徴量
            if run==0:
                feature_trn = feature[cut_timing[run]:]
                feature_test = feature[:cut_timing[run]]
            elif run == len(brain_files) - 1:
                feature_trn = feature[:cut_timing[run-1]]
                feature_test = feature[cut_timing[run-1]:]
            else:
                feature_trn = feature[:cut_timing[run-1]]
                feature_trn = np.vstack([feature_trn,feature[cut_timing[run]:]])
                feature_test = feature[cut_timing[run-1]:cut_timing[run]]
        else:
            n_cv = 9 # 何fold cross-validationをするか
            brain_path = "/home/miura/brain/pycortex_project/resource/littlePrince/" + subject + "/echo_1_cortex/normalization/"
            brain_files= []
            feature_files= []

            feature_path = '/home/miura/brain/src/encoding/LittlePrince/features/data/' # 特徴量の .npy を作成したら設定
            if mode == "without_concat":
                feature_file_path = '/home/miura/brain/src/encoding/LittlePrince/features/data/features_emotion/'
            elif mode == "with_concat":
                feature_file_path = '/home/miura/brain/src/encoding/LittlePrince/features/data/features_concated/'
            else:
                feature_file_path = '/home/miura/brain/src/encoding/LittlePrince/features/data/features_bert-cased/'

            for i in range(9):
                brain_files.append("run-" + str(i+1) + ".npy")
                feature_files.append("features_section_" + str(i+1) +".npy")

            model_dir = '/home/miura/brain/src/encoding/LittlePrince/model/'+ mode + "/" + ridge_type +'/encodingmodel_output_' + subject + "/"
            out_data_dir = model_dir + "run_" + str(run+1) + "/PCA_ROI/matrix/"
            new_out_data_dir = out_data_dir + "each_emotion/" 
            os.makedirs(new_out_data_dir, exist_ok=True)

            # 脳活動データ・特徴量の読み込み

            for j in range(9):
                with open(brain_path + brain_files[j], 'rb') as f:
                    brain_data = np.load(f)
                with open(feature_file_path + feature_files[j], 'rb') as f:
                    feature_data = np.load(f)
                if run!=0 and j==0: # リッジ回帰が1回目でなくて最初のファイルを読み込むとき
                    brain_trn = brain_data # 最初のファイルは訓練用として使う
                    feature_trn = feature_data
                elif run==0 and j==1: # リッジ回帰が1回目で2つ目のファイルを読み込むとき
                    brain_trn = brain_data # 最初のファイルはテスト用になるので、2つ目のファイルを訓練用として登録
                    feature_trn = feature_data
                elif j==run:
                    brain_test = brain_data
                    feature_test = feature_data
                else:
                    brain_trn = np.vstack([brain_trn, brain_data])
                    feature_trn = np.vstack([feature_trn, feature_data])

        n_voxels = brain_trn.shape[1] # ボクセル数
        width = 4
        delay = 2

        feature_trn_paired = make_pair_data(feature_trn, width, delay)
        feature_test_paired = make_pair_data(feature_test, width, delay)

        n_samples = feature_trn_paired.shape[0] # データ数
        chunklen = 25 # TR=2のデータなので、25くっつけたら50秒分くっつけたことになる

        cv_inds = get_cv_inds(n_samples, chunklen, n_cv)
            
        # 訓練データ、検証データを選ぶ
        trn_ind = cv_inds[n_cv-1]['trn_ind']
        val_ind = cv_inds[n_cv-1]['val_ind']

        X_trn = feature_trn_paired[trn_ind]
        y_trn = brain_trn[trn_ind]
        X_val = feature_trn_paired[val_ind]
        y_val = brain_trn[val_ind]

        # best alpha の読み込み
        with open(model_dir + "run_" + str(i+1) + "/best_alpha.txt") as f:
            ba = f.read()
        best_alpha = float(ba)

        # 採用したalphaで学習
        if ridge_type == "Ridge":
            rg = Ridge(alpha=best_alpha)    
        rg.fit(feature_trn_paired, brain_trn)

        # 予測
        pred_test = rg.predict(feature_test_paired)

        for k in range(80):
            # ここから0パディング
            feature_test_paired_0_pad = feature_test_paired
            # delayで4分割
            feature_test_paired_0_pad_4, feature_test_paired_0_pad_6, feature_test_paired_0_pad_8, feature_test_paired_0_pad_10= np.hsplit(feature_test_paired_0_pad, 4)

            # ある特徴量だけ0パディング
            feature_test_paired_0_pad_4[:,k]=0
            feature_test_paired_0_pad_6[:,k]=0
            feature_test_paired_0_pad_8[:,k]=0
            feature_test_paired_0_pad_10[:,k]=0

            feature_test_paired_0_pad_any_feature = np.hstack([feature_test_paired_0_pad_4,feature_test_paired_0_pad_6])
            feature_test_paired_0_pad_any_feature = np.hstack([feature_test_paired_0_pad_any_feature,feature_test_paired_0_pad_8])
            feature_test_paired_0_pad_any_feature = np.hstack([feature_test_paired_0_pad_any_feature,feature_test_paired_0_pad_10])    

            pred_test_without_any_feature = rg.predict(feature_test_paired_0_pad_any_feature)
            pred_test_only_any_feature = pred_test - pred_test_without_any_feature

            corr_test_only_any_feature = []
            for m in range(y_val.shape[1]): 
                r_af = np.corrcoef(brain_test.T[m,:], pred_test_only_any_feature.T[m,:])[0,1]
                corr_test_only_any_feature.append(r_af)
            corr_test_only_any_feature = np.array(corr_test_only_any_feature)

            p_test_only_any_feature = compute_p(corr_test_only_any_feature, corr_test_only_any_feature.shape[0]) # p値を求める。
            p_rejected_only_any_feature, _ = fdr(p_test_only_any_feature)

            # print('p_rejected shape:', p_rejected_only_any_feature.shape)
            # print('棄却されたボクセル数:', np.count_nonzero(p_rejected_only_any_feature))
            
            # np.save(new_out_data_dir+annotation[k], pred_test_only_any_feature)
            # np.save(new_out_data_dir+"p_rejected_"+annotation[k], p_rejected_only_any_feature)

            cortex_data_only_any_feature = np.mean(pred_test_only_any_feature,axis=0)
            cortex_data_only_any_feature[~p_rejected_only_any_feature] = 0

            if k==0:
                cortex_for_pca = cortex_data_only_any_feature
            else:
                cortex_for_pca =np.vstack([cortex_for_pca,cortex_data_only_any_feature])

        # データの保存
        np.save(out_data_dir+subject+"_section_"+str(run+1), cortex_for_pca)

    print(subject+":end")
