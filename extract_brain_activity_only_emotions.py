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
import pickle

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

# パスの設定
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
            out_data_dir = '/home/miura/brain/model/'+ mode + "/" + ridge_type +'/encodingmodel_output_sub-' + str(sub_id) +"/run_" + str(run+1) + "/feature_based_extract/"
            os.makedirs(out_data_dir, exist_ok=True)

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
                feature_files.append("features_section_" + str(i+1) + ".npy")

            model_dir = '/home/miura/brain/src/encoding/LittlePrince/model/'+ mode + "/" + ridge_type +'/encodingmodel_output_' + subject + "/"
            out_data_dir = model_dir + "run_" + str(run+1) + "/feature_based_extract/"
            os.makedirs(out_data_dir, exist_ok=True)

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

        # 相関&P値
        corr_test = [] # (n_voxels, )

        # ボクセルごとに相関を求める
        for m in range(y_val.shape[1]): 
            # r, _ = pearsonr(brain_test.T[m,:], pred_test.T[m,:])
            r = np.corrcoef(brain_test.T[m,:], pred_test.T[m,:])[0,1]
            corr_test.append(r)
            
        corr_test = np.array(corr_test)
        p_test = compute_p(corr_test, corr_test.shape[0]) # p値を求める。これはライブラリとかを使っても求められると思います

        # FDR補正 0.05
        p_rejected, _ = fdr(p_test)
        p_rejected_corr = np.asarray(p_rejected)

        # 棄却されたところのみ。可視化のために<0は0にしてしまっている
        corr_rejected = np.array(corr_test)
        corr_rejected[~p_rejected_corr] = 0
        corr_rejected[corr_rejected<0] = 0

        # 0パディング
        feature_test_paired_0_pad = feature_test_paired
        feature_test_paired_0_pad_4, feature_test_paired_0_pad_6, feature_test_paired_0_pad_8, feature_test_paired_0_pad_10= np.hsplit(feature_test_paired_0_pad, 4)

        emotion_0_pad = np.zeros((feature_test_paired_0_pad.shape[0],80))

        feature_test_paired_0_pad_emotion = np.hstack([emotion_0_pad, feature_test_paired_0_pad_4[:,80:]])
        feature_test_paired_0_pad_emotion = np.hstack([feature_test_paired_0_pad_emotion,emotion_0_pad])
        feature_test_paired_0_pad_emotion = np.hstack([feature_test_paired_0_pad_emotion,feature_test_paired_0_pad_6[:,80:]])
        feature_test_paired_0_pad_emotion = np.hstack([feature_test_paired_0_pad_emotion,emotion_0_pad])
        feature_test_paired_0_pad_emotion = np.hstack([feature_test_paired_0_pad_emotion,feature_test_paired_0_pad_8[:,80:]])
        feature_test_paired_0_pad_emotion = np.hstack([feature_test_paired_0_pad_emotion,emotion_0_pad])
        feature_test_paired_0_pad_emotion = np.hstack([feature_test_paired_0_pad_emotion,feature_test_paired_0_pad_10[:,80:]])

        # 言語に関する脳活動の予測
        pred_test_language = rg.predict(feature_test_paired_0_pad_emotion)

        # 相関&P値
        corr_test_language = [] # (n_voxels, )

        # ボクセルごとに相関を求める
        for m in range(y_val.shape[1]): 
            # r, _ = pearsonr(brain_test.T[m,:], pred_test.T[m,:])
            r = np.corrcoef(brain_test.T[m,:], pred_test_language.T[m,:])[0,1]
            corr_test_language.append(r)
            
        corr_test_language = np.array(corr_test_language)
        p_test_language = compute_p(corr_test_language, corr_test_language.shape[0]) # p値を求める。これはライブラリとかを使っても求められると思います

        # FDR補正 0.05
        p_rejected_language, _ = fdr(p_test_language)

        # 棄却されたところのみ。可視化のために<0は0にしてしまっている
        corr_rejected_language = np.array(corr_test_language)
        corr_rejected_language[~p_rejected_language] = 0
        corr_rejected_language[corr_rejected_language<0] = 0

        # 引き算
        pred_test_emotion = pred_test - pred_test_language

        # データの保存
        # モデル
        with open(model_dir + "run_" + str(run+1) +'/model.pkl','wb') as f:
            pickle.dump(rg, f)

        # 予測脳活動
        np.save(out_data_dir+'pred_test', pred_test)
        np.save(out_data_dir+'pred_test_language', pred_test_language)
        np.save(out_data_dir+'pred_test_emotion', pred_test_emotion)

        # 相関係数
        np.save(out_data_dir+'corr_test', corr_test)
        np.save(out_data_dir+'corr_test_language', corr_test_language)


        # 相関係数 補正済み
        np.save(out_data_dir+'corr_rejected', corr_rejected)
        np.save(out_data_dir+'corr_rejected_language', corr_rejected_language)

        # 棄却されたボクセル
        np.save(out_data_dir+'rejected_voxel', p_rejected)
        np.save(out_data_dir+'rejected_voxel_language', p_rejected_language)

    print(subject+":end")
