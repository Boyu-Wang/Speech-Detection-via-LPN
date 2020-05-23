# Speech Activity Detection via Landmark Pooling Network

This repo contains the dataset and code for our FG 19 paper [Are You Speaking: Real-Time Speech Activity Detection via Landmark Pooling Network] (FG19_speech_detection.pdf). The original paper was implemented in Tensorflow. This is a Pytorch version.

## LSW Dataset

The LSW dataset can be downloaded [here](http://vision.cs.stonybrook.edu/~boyu/LSW_dataset.zip): which includes aligned mouth image and aligned mouth landmark.



## Usage

The code is written in Python 3.6. The pytorch version is 1.2.

The code contain different models for speech activity classification on LSW dataset:

+ Appearance CNN
+ Landmark Pooling Network (LPN)
+ LPN + Appearance CNN.


To run the training code:

```
python main.py --model_type lpn
```

For testing:

```
python main.py --model_type lpn --is_train False
```

`model_type` can be changed to `appearance, appearance_lpn`.


The pytorch implementation has slightly better performance than original reported in the paper.

| Model                | Accuracy |
| :-------------       |:--------:|
| Appearance CNN       | 78.5     | 
| LPN                  | 75.6     |
| LPN + Appearance CNN | 81.0     |




Please cite our paper if you are using our dataset or code:


```
@INPROCEEDINGS{8756549,
  author={Boyu Wang and Xiaolong Wang},
  booktitle={2019 14th IEEE International Conference on Automatic Face   Gesture Recognition (FG 2019)}, 
  title={Are You Speaking: Real-Time Speech Activity Detection via Landmark Pooling Network}, 
  year={2019}}```