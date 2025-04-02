# Sound_Distance_Estimation
This is the baseline for sound distance estimation in IASP lab.

## Installation Guide
```python
cd Solution_on_3D_SELD
pip install -r requirements.txt
```
## Extract Audio Features
```python
dataset_dir = '...'
feat_label_dir = '...'

# Extract audio features
python utils/cls_tools/batch_feature_extraction.py
```
