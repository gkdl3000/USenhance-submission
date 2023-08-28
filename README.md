# Ultrasound enhance model

## Environment
Linux         
RTX 3090

## Installation
Use anaconda environment
```
conda create -n USenhance python=3.7
conda activate USenhance
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install scikit-learn
pip install PyYAML
pip install visdom
pip install scikit-image
pip install opencv-python
```

## Train
- Create dataset 
   -  split low quality, high quality images to A, B folder. (A: low quality, B: high quality)
   -  train path/A/
   -  train path/B/
   -  val path/A/
   -  val path/B/ 

- Modify CycleGan.yaml options as follows
```
save_root: model save path
dataroot: train path
val_dataroot: val path
```
- Run train:
 ```
python train.py
```

## Inference and save images

- Modify CycleGan.yaml options as follows
```
model_root: model weight path to inference
model_root2: model2 weight path to averaging with above weight
infer_dataroot: low quality image path
infer_image_save: high quality image save path
```
- Run inference code
```
python multiinference.py
```
 
## Use visdom：
```
python -m visdom.server -p 6022
```
If other port parameters are used, you need to modify the port in yaml.

