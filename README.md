# Humpback Whale Identification
046003 Technion course project

Amir Saad & Sahar Carmel

An educational (closed) [Kaggle challenge](https://www.kaggle.com/c/whale-categorization-playground)

## Table of Contents
- [Introduction](#Introduction)
- [Related Work](#Related-Work)
- [Dataset Exploration](#Dataset-Exploration)
- [Implementation](#Implementation)
- [Discussion](#Discussion)
- [Conclusion](#Conclusion)

## Introduction
## Related Work
## Dataset Exploration
## Implementation
This section is divided to kaggle submissions, thus describing our work in layers.
### Submission #0
#### Dataset
* Ignoring classes with less than 5 samples
* Randomly splitting to train-val with 25% val.

run `dataset_2_train_val_subfolders`

#### PreProcessing
*Random resize crop
*Random horizontal flip
*Normalization


#### Network Architecture
Resnet50 with last layer FC with 313 features (Number of whales with more then 5 occurances)
#### Results
Whale competiton score of 0.324
This is a pretty good score for a very simple implementation!
**But** examining the network's output reveals that we went into a **loacl minima** where all
highest probabilities are 'new_whale'

#### Conclusions and ToDO's
- [ ] find a way out of new_whale local minima
### Submission #1
## Discussion
## Conclusion
