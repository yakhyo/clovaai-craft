## CRAFT: Character-Region Awareness For Text detection
[Paper](https://arxiv.org/abs/1904.01941) | [Pretrained Model](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ)

### Overview
PyTorch implementation for CRAFT text detector that effectively detect text area by exploring each character region and affinity between characters. The bounding box of texts are obtained by simply finding minimum bounding rectangles on binary map after thresholding character region and affinity scores. 

![Craft Example](./figures/craft_example.gif)

### Training
The code for training is not included in this repository, and we cannot release the full training code for IP reason.


### Test instruction using pretrained model
- Download the trained models
 
 *Model name* | *Used datasets* | *Languages* | *Purpose* | *Model Link* |
 | :--- | :--- | :--- | :--- | :--- |
General | SynthText, IC13, IC17 | Eng + MLT | For general purpose | [Click](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ)
IC15 | SynthText, IC15 | Eng | For IC15 only | [Click](https://drive.google.com/open?id=1i2R7UIUqmkUtF0jv_3MXTqmQ_9wuAnLf)
LinkRefiner | CTW1500 | - | Used with the General Model | [Click](https://drive.google.com/open?id=1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO)

* Run with pretrained model
``` python
python test.py
```

The result image and score maps will be saved to `./result` by default.

### Arguments
* `--trained_model`: pretrained model
* `--text_threshold`: text confidence threshold
* `--low_text`: text low-bound score
* `--link_threshold`: link confidence threshold
* `--cuda`: use cuda for inference (default:True)
* `--canvas_size`: max image size for inference
* `--mag_ratio`: image magnification ratio
* `--poly`: enable polygon type result
* `--show_time`: show processing time
* `--test_folder`: folder path to input images
* `--refine`: use link refiner for sentense-level dataset
* `--refiner_model`: pretrained refiner model

`config.yaml`:
```python
trained_model: 'craft_mlt_25k.pth'                  # pretrained model
text_threshold: 0.7                                 # text confidence threshold
low_text: 0.4                                       # text low-bound score
link_threshold: 0.4                                 # link confidence threshold
cuda: True                                          # Use cuda for inference
canvas_size: 1280                                   # image size for inference
mag_ratio: 1.5                                      # image magnification ratio
poly: False                                         # enable polygon type
show_time: False                                    # show processing time
test_folder: 'data'                                 # folder path to input images
refine: False                                       # enable link refiner
refiner_model: 'weights/craft_refiner_CTW1500.pth'  # pretrained refiner model
```