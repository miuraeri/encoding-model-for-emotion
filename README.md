# encoding-model-for-emotion

## Data Preprocessing

### Processing TextGrid Files (Words and Reading Time Information)
1. Extract the sentences that were read, based on the sentence boundary marker `#`: `textgrid2sentence.py`  
   - Requires `textgrid2csv.py` as a prerequisite.
2. Extract the time intervals during which each sentence was read ([start time, end time]): `extract_read_timing.py`

### Preprocessing Brain Activity Data
1. Convert fMRI data into vertex-based format (2D array): `preprocessing_brain_volume.py`
2. Normalize the data: `normalization.py`

## Multi-step Fine-tuning
- **First step**: Follow the implementation in [GoEmotions-pytorch](https://github.com/monologg/GoEmotions-pytorch).  
  Specify `--taxonomy original` when running the code.
- **Second step**: Add `alice.json` to the config directory above and specify `--taxonomy alice`.

## Feature Construction
- Linguistic features: `sentence_to_bert_feature.py`  
- Emotion features: `sentence_to_emotion_feature.py`  
- Expand features according to the sentence reading time: `expand_feature_matrix.py`

## Construction of Encoding Models
- Build encoding models and save parameters and trained models: `build_encoding.py`

## Extraction of Emotion-only Predicted Brain Activity

### Predicted Brain Activity for Each Subject
- Extract predicted brain activity related only to emotions: `extract_brain_activity_only_emotions.py`  
- Extract predicted brain activity related only to each emotion category: `extract_brain_activity_each_emotion.py`

### Average Predicted Brain Activity (All Subjects / By Gender)
- Map predicted brain activity of all subjects onto the flatmap of sub-22: `mapping_all_subject.py`  
- Compute average predicted brain activity: `get_subject_mean.ipynb`

## Analysis of Emotion Response Intensity by Region of Interest (ROI)
1. Average predicted brain activity across all time points for each emotion (resulting in a 1 × voxel matrix), and stack them across emotions  
   (see `extract_brain_activity_each_emotion.py`).
2. Apply ROI masks (set values outside the ROI to zero) and extract ROI-specific matrices (80 × voxel): `get_each_roi_matrix.py`
3. Perform PCA on the matrices from step 2 to obtain PC1 and PC2, and compute the average voxel-wise response for each emotion
4. Plot the results as a scatter plot: `PCA_plot_scatter.py`
