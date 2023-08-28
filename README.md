# Ultrasound enhance model

## Main Reference Environment
1. Linux         (Titan RTX)
2. Python        (3.6.6)
3. torch         (1.9.0+cu111)
5. visdom        (0.1.8.9)
6. numpy         (1.19.2)
7. skimage       (0.15.0)
8. Yaml          (5.4.1)
9. cv2           (3.4.2)
10. PIL          (8.3.2)

## Usage
1. Create dataset
   -  train path/A/
   -  train path/B/
   -  val path/A/
   -  val path/B/ 
2. Start visdom：
 ```
python -m visdom.server -p 6019
```
If other port parameters are used, you need to modify the port in yaml.

3. Train:
 ```
python train.py
```

4. Inferece and save images
```
python inference.py
```