# Humpback Whale Identification
046003 Technion course project

Amir Saad & Sahar Carmel

An educational (closed) [Kaggle challenge](https://www.kaggle.com/c/whale-categorization-playground)

## Table of Contents
- [Introduction](#Introduction)
- [Related Work](#Related-Work)
- [Dataset Exploration](#Dataset-Exploration)
- [Integrating a semantic-segmentaion network](#Integrating-a-semantic-segmentaion-network-a-bit-brute-force)
- [Implementation](#Implementation)
- [Discussion](#Discussion)
- [Conclusion](#Conclusion)

## Introduction
In the Humpback Whale Identification kaggle competition we are challenged to build an algorithm to identifying whale species in images.
### Dataset description
The data base includes 4251 whale calsses:
- [ ] 4250 whale IDs
	- about half of these classes contain only 1 sample
- [ ] 'new_whale' class specifing unidentified whales
	- this class contains 630 images

** An 'inverse' histogram of the dataset:**

- x-axis: quantity

- y-axis: number of classes wich holds that quantity

![o](https://github.com/assctechnion/-HumpbackWhaleIdentification/blob/master/Documents/Num%20of%20categories%20by%20images.png)

**Images from the data set for example:**

![o](https://github.com/assctechnion/-HumpbackWhaleIdentification/blob/master/Documents/Whales%20pictures%20example.png)

### Score evaluation
Submission is done by a 'csv' file containing maximum 5 ID estimations for each 
image in the test set. The score is evaluated by:

![o](https://github.com/assctechnion/-HumpbackWhaleIdentification/blob/master/Documents/score.PNG)

Wich is the mean average precission 
Where U is the total number of images , n is the number of estimations for each image, and P(k) is the percission 
of the estimation defined by TP/(TP+FP). In our case we belive it is calculated simply by:

![o](https://github.com/assctechnion/-HumpbackWhaleIdentification/blob/master/Documents/simlifiedP.PNG)

### Hardware
The NNs where trained on a personal computer with GeforceRTX2060 GPU wich is equivalent TBD


## Related Work
## Dataset Exploration
## Integrating a semantic-segmentaion network: A bit brute force
Inspired yb a bounding box technuiqe that was used by other kernels we tried running a 
segmentaion network to label only the whale in each image.

We ran into a [pytorch implemented Semantic Segmentation network](https://github.com/CSAILVision/semantic-segmentation-pytorch)
<img src="./teaser/ADE_val_00000278.png" width="900"/>
<img src="./teaser/ADE_val_00001519.png" width="900"/>
[From left to right: Test Image, Ground Truth, Predicted Result]


## Implementation
This section is divided to kaggle submissions, thus describing our work in layers.
### Submission #0
#### Dataset
* Ignoring classes with less than 5 samples
* Randomly splitting to train-val with 25% val.

run `dataset_2_train_val_subfolders`

#### PreProcessing
* Random resize crop
* Random horizontal flip
* Normalization


#### Network Architecture
Resnet50 with last layer FC with 313 features (Number of whales with more then 5 occurances)
#### Results
Whale competiton score of **0.324**
This is a pretty good score for a very simple implementation!
**But** examining the network's output reveals that we went into a **loacl minima** where all
highest probabilities are 'new_whale'

| Image         | Id            |
| ------------- |:-------------:|
| 00029b3a.jpg  | new_whale w_01cbcbf w_073b15e w_0d2dc7e w_030294d 	 |
| 0003c693.jpg  | new_whale w_01cbcbf w_073b15e w_0d2dc7e w_0122d85      |
| 000bc353.jpg  | new_whale w_0fe48f3 w_01cbcbf w_0e25cf2 w_0122d85      |


#### Conclusions and ToDO's
- [ ] find a way out of new_whale local minima
### Submission #1

#### Dataset
* Taking in account all classes
* Randomly splitting to train-val with 25% val.

run `dataset_2_train_val_subfolders`

#### PreProcessing
* Random resize crop
* Random horizontal flip
* Normalization


#### Network Architecture
Resnet50 with last layer FC with 4251 features 
#### Results
Whale competiton score of 0.008

## Discussion
## Conclusion
