# Simple OCR Engine with Faster-RCNN
## 1. Introduction
The code is totally based on the excellent work, written by Yun Chen (https://github.com/chenyuntc/simple-faster-rcnn-pytorch). I also adopted the idea of using Resnet backbone from Bart Trzynadlowski
(https://github.com/trzy/FasterRCNN). Before I came into his implementation, I didn't know that freezing the parameters of batch normalization layers is critical to properly fine-tune the pre-trained model. The implementation written by Jianwei Yang (https://github.com/jwyang/faster-rcnn.pytorch) will be very useful but may not be really easy to start with for beginners.

## 2. Peformance on PASCAL VOC 2007 Dataset

Here, 'mAP_07' is the PASCAL VOC 2007 evaluation metric for calculating average precision. And, 'mAP' is the later version for more accurate calculation. 
I checked that the performance on VOC 2007 Dataset is similar to what was already reported and that the use of Resnet backbone considerably boots the performance. 

(1) VGG backbone:
| epoch | lr | mAP_07 | mAP | total_loss | rpn_loc_loss | rpn_cls_loss | roi_loc_loss | roi_cls_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.001 | 0.403672 | 0.394523 | 0.980073 | 0.065378 | 0.179688 | 0.360688 | 0.37432 |
| 1 | 0.001 | 0.519956 | 0.519763 | 0.7596 | 0.054442 | 0.138137 | 0.304269 | 0.262752 |
| 2 | 0.001 | 0.583169 | 0.588185 | 0.675677 | 0.052033 | 0.122896 | 0.273295 | 0.227453 |
| 3 | 0.001 | 0.590256 | 0.596801 | 0.614348 | 0.050033 | 0.112419 | 0.251262 | 0.200635 |
| 4 | 0.001 | 0.619948 | 0.631572 | 0.571489 | 0.048458 | 0.102462 | 0.236231 | 0.184339 |
| 5 | 0.001 | 0.616753 | 0.627509 | 0.53167 | 0.046819 | 0.095342 | 0.221056 | 0.168454 |
| 6 | 0.001 | 0.636117 | 0.646613 | 0.50211 | 0.04589 | 0.087776 | 0.210875 | 0.157569 |
| 7 | 0.001 | 0.645062 | 0.658311 | 0.477255 | 0.045149 | 0.081806 | 0.201914 | 0.148387 |
| 8 | 0.001 | 0.631926 | 0.645849 | 0.456026 | 0.044162 | 0.075981 | 0.193634 | 0.142248 |
| 9 | 0.001 | 0.658308 | 0.679825 | 0.432311 | 0.043285 | 0.070534 | 0.183996 | 0.134496 |
| 10 | 0.0001 | 0.686796 | 0.708004 | 0.34545 | 0.037704 | 0.054806 | 0.149821 | 0.10312 |
| 11 | 0.0001 | 0.692785 | 0.71294 | 0.329832 | 0.036723 | 0.050511 | 0.143937 | 0.098661 |
| 12 | 0.0001 | 0.693039 | 0.713714 | 0.318847 | 0.036284 | 0.048572 | 0.140012 | 0.093978 |
| 13 | 0.0001 | **0.696292** | **0.717402** | 0.314113 | 0.03607 | 0.047072 | 0.137908 | 0.093063 |

(2) Resnet backbone:
| epoch | lr | mAP_07 | mAP | total_loss | rpn_loc_loss | rpn_cls_loss | roi_loc_loss | roi_cls_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.001 | 0.184378 | 0.151529 | 1.142191 | 0.073632 | 0.182598 | 0.402623 | 0.483337 |
| 1 | 0.001 | 0.597307 | 0.607589 | 0.780584 | 0.061057 | 0.114066 | 0.35078 | 0.254681 |
| 2 | 0.001 | 0.694578 | 0.711223 | 0.602926 | 0.052473 | 0.087811 | 0.271607 | 0.191035 |
| 3 | 0.001 | 0.700652 | 0.720618 | 0.531989 | 0.048989 | 0.075502 | 0.238271 | 0.169227 |
| 4 | 0.001 | 0.73555 | 0.761293 | 0.474171 | 0.046207 | 0.065751 | 0.211559 | 0.150654 |
| 5 | 0.001 | 0.733022 | 0.758205 | 0.431338 | 0.043697 | 0.057803 | 0.193411 | 0.136427 |
| 6 | 0.001 | 0.730688 | 0.752982 | 0.398987 | 0.042086 | 0.050081 | 0.179734 | 0.127086 |
| 7 | 0.001 | 0.736201 | 0.763603 | 0.387164 | 0.040948 | 0.050202 | 0.171626 | 0.124389 |
| 8 | 0.001 | 0.730626 | 0.754919 | 0.360851 | 0.039732 | 0.044472 | 0.160092 | 0.116556 |
| 9 | 0.001 | 0.75053 | 0.777591 | 0.345579 | 0.038518 | 0.042075 | 0.15336 | 0.111626 |
| 10 | 0.0001 | 0.776109 | 0.810495 | 0.25431 | 0.029196 | 0.025776 | 0.11561 | 0.083728 |
| 11 | 0.0001 | 0.778772 | 0.810525 | 0.239215 | 0.027688 | 0.023442 | 0.108456 | 0.079629 |
| 12 | 0.0001 | 0.780056 | **0.813081** | 0.230916 | 0.026561 | 0.021572 | 0.105736 | 0.077047 |
| 13 | 0.0001 | **0.780067** | 0.809398 | 0.225224 | 0.026156 | 0.021053 | 0.103375 | 0.07464 |


## 3. OCR Application
The object detectors, like faster-RCNN, can be immediately trained for OCR applications, where the target objects on images are simply text. 
At first, I struggled, trying to dectect every single text character on an image of large size. This approach implies that there can be many kinds of labels (or classes) and also many objects (or class instances) to detect in the first place, which can make models slowly converge for training. Instead, I took a presumably well-known and classical, 2-step approach: (1) first, detect text area; (2) and then recognize text characters on each detected area. I think this approach has the advantage of reduced training complexity in that (1) there is only one class, i.e. whether it is text or not, on a large image; and that (2) there are many classes, e.g. alpha-numeric characters and other symbols, but on a small chopped box image. 

For each step, I generated random text charater sequences for training datasets. I think that this random character approach is okay with the text detection part, but it can result in quite a few erronious predictions with text recognition part, as there are many similar characters: e.g. (1) o vs. O vs. 0, (2) p vs. P, (3) s vs. S, (4) l vs. I vs. ] vs. [ vs. 1, etc. For the training of text recognition, using real dictionary words can improve the accuracy, as the model is trained to the most probable neighboring characters in the real world. 

On the other hand, I needed to refine or upgrade the model or its parameters to better perform on my customized OCR datasets:
- more refined anchor scales and ratios, to better detect small objects
- use of ROI Align scheme
- use of Resnet backbone (cf. dont' forget freezing batch norm layer parameters), if it's beneficial.

### 3.1 Train/Validation
(1) Text detection: training image with label (left) vs. predicted bounding boxes/labels/confidence (right), note that there is only one label, called 'text'.
![image](https://github.com/user-attachments/assets/7785564e-9803-4217-826c-ee68a0a6b053)
![image](https://github.com/user-attachments/assets/09e6bc17-8300-4e5a-a89f-10c29546a333)

(2) Text recognition: training image with label (left) vs. predicted bounding boxes/labels/confidence (right)
![image](https://github.com/user-attachments/assets/b12f3796-0b0d-4b22-bce7-0eb179562b87)
![image](https://github.com/user-attachments/assets/2112141f-04e5-4b3a-9d87-e6d1df81b1dc)


### 3.2 Inference
Here are some inference results (bounding boxes in read and labels on top of each) on real-world images or documents. 
![image](https://github.com/user-attachments/assets/f870f006-8390-48df-8f55-7fcb4e46c17e)

![image](https://github.com/user-attachments/assets/30ae3f2b-065c-4110-9379-109d339b675c)

![image](https://github.com/user-attachments/assets/428e1f95-725d-4347-ac6e-3b012f34050e)


