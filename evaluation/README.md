# Evaluation

## Labeled image

I have manually labeled data
- `images` contains the original minimap images
- `labels` contains the label information in  PASCAL VOC format (XML), matching X, Y and champion id to the corresponding champion icons in `images`
- `labeled_images` contains the minimap images after `labels` data has been applied to `images`

### Setup for labelImg (if you wish to label yourself)

`pip install labelImg`

### Running labelImg software

`labelImg images/ league.names`

## Reproducing results

### Show image detection

`python eval.py ..\trained_models\fastrcnn_model_8hrs.pt 149 images labels`

Saves labeled images to `labeled_images`

### Score threshold increase

`python score_threshold_eval.py ..\trained_models\fastrcnn_model_8hrs.pt 149 images labels`

### IOU threshold increase

`python iou_threshold_eval.py ..\trained_models\fastrcnn_model_8hrs.pt 149 images labels`

### Precision vs recall

`python precision_recall_eval.py ..\trained_models\fastrcnn_model_8hrs.pt 149 images labels`
