# encoding-model-for-emotion
## データの前処理
### textgridファイル(単語+読まれた時間のデータ)の処理
1. 文の区切りを示す文字列「#」をもとに、読まれた文章を抽出: `textgrid2sentence.py`
    - 別途 `textgrid2csv.py` が必要。
2. 文章が読まれた時間（[開始時間, 終了時間]）を抽出: `extract_read_timing.py`  
### 脳活動データの前処理
1. fMRIデータをVertex形式(2次元配列)へ変換: `preprocessing_brain_volume.py`
2. データの正規化: `normalization.py`
## 多段階ファインチューニング
1段階目: [GoEmotions-pytorch](https://github.com/monologg/GoEmotions-pytorch) を参照。コード実行時に`--taxonomy original`を指定。  
2段階目: 上記のconfigディレクトリに`alice.json`を追加。`--taxonomy alice`を指定。
## 特徴量の作成
言語特徴量: `sentence_to_bert_feature.py`  
感情特徴量: `sentence_to_emotion_feature.py`  
読まれた時間に合わせて特徴量を拡張: `expand_feature_matrix.py`  
## 符号化モデルの作成
符号化モデルの構築およびパラメータ・モデルの保存: `build_encoding.py`
## 感情のみに関する予測脳活動の抽出
### 各被験者の予測脳活動
感情のみに関する予測脳活動の抽出: `extract_brain_activity_only_emotions.py`  
各感情カテゴリーのみに関する予測脳活動の抽出: `extract_brain_activity_each_emotion.py`  
### 全被験者・男女別の平均予測脳活動
全被験者の予測脳活動データをsub-22のflatmapへマッピング: `mapping_all_subject.py`  
平均予測脳活動: `get_subject_mean.ipynb`  
## 関心領域別の各感情の反応強度調査
1. 各感情のみに関する予測脳活動の全タイムスタンプの平均を取り(1xボクセルの行列)、スタックする: `extract_brain_activity_each_emotion.py`参照
2. 各ROIでマスクをかけ (他のROIの列は全部値を0にする)、ROIごとの行列 (80xボクセル)を取り出す: `get_each_roi_matrix.py`
3. 2.の行列を使ってPCAでPC1とPC2の値を出し、もとの行列から各感情でのボクセル間の値の平均を出す
4. 散布図にプロット: `PCA_plot_scatter.py`  
