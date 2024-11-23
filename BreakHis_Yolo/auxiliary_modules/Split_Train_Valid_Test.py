from ultralytics.data.utils import autosplit

autosplit(  
    path="C:\\Users\\Wilsiman Evangelista\\Desktop\\Breast_Cancer_Detection\\BreakHis_Yolo\\40X_multiclass\\tubular_adenoma",
    weights=(0.7, 0.2, 0.1),  # (train, validation, test) fractional splits
    annotated_only=False,  # split only images with annotation file when True
)