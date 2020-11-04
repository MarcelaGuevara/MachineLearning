---
title: "Activity performance"
author: "Marcela Guevara"
date: "11/3/2020"
output:
  html_document:
    keep_md: yes
---




```
## Warning: package 'randomForest' was built under R version 4.0.3
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## Warning: package 'caret' was built under R version 4.0.3
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## 
## Attaching package: 'ggplot2'
```

```
## The following object is masked from 'package:randomForest':
## 
##     margin
```


## Summary
The objective of this report is trying to fit a model which predicts how well people does a particular activity. In order to do this, we are going to use data provided by Human Activity Recognition (HAR) from 5 activity classes.From the analysis of this data, we found out that the Random Forest method is a very good model to make predictions about the matter in which people did the activity.

## Loading and cleaning the data

First, we will load the datasets corresponding to the training and testing subsets.


```r
training=read.csv("pml-training.csv",na.strings = c("",NA))
testing=read.csv("pml-testing.csv",na.strings = c("",NA))
```

The outcome variable we want to predict is the "classe" variable, which is represented with a letter (A, B, C, D, E). So, we will transform it into a factor variable.


```r
training$classe=factor(training$classe)
```

If we take a look of the summary of the variables, some of them has a lot of NA values. An example of this is the "kurtosis_roll_belt" variable. Let's count how many NA values does this variable contain.


```r
table(is.na(training$kurtosis_roll_belt))
```

```
## 
## FALSE  TRUE 
##   406 19216
```

It has over 19000 NA values. To fit our model, this type of variables will be useless. That's why we will remove all variables which have more than 10000 NA values.


```r
#Remove columns with lots of NAs
index=c()
for (i in 1:length(training)){
    if (sum(is.na(training[i]))>10000){
        index[length(index)+1]=i
    }
}


training=training[,-index]    
testing=testing[,-index]
```

We will also remove the first 6 variables, since they are only identifiers.


```r
#Remove first 6 columnns
training=training[,-c(1,2,3,4,5,6)]
testing=testing[,-c(1,2,3,4,5,6)]
```

Since we finished cleaning our data, we can now start to preprocess it.

## Preprocessing

Rigth now, we have 53 predictors, which is a really large number. Maybe some variables could be somehow correlated, so we will perform a PCA preprocessing to keep the variables which explain most of the variability.


```r
preProc <- preProcess(training[,-54],method="pca")
preProc
```

```
## Created from 19622 samples and 53 variables
## 
## Pre-processing:
##   - centered (53)
##   - ignored (0)
##   - principal component signal extraction (53)
##   - scaled (53)
## 
## PCA needed 26 components to capture 95 percent of the variance
```

From 53 variables now we have 26 variables to explain of model. Up next, we have to apply this model to our training and testing sets.


```r
trainPC <- predict(preProc,training[,-54])
testPC <- predict(preProc,testing[,-54])
```

We are ready to fit our model.


## Fitting the model

To fit our model, we will use the Random Forest model, using the "classe" variable as the outcome, and all the PCA variables as predictors.


```r
model1=randomForest(factor(training$classe) ~ .,data=trainPC)
```

We will use this model to predict the training values and see the accuracy of the model.


```r
p1=predict(model1,trainPC)
confusionMatrix(p1,training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5580    0    0    0    0
##          B    0 3797    0    0    0
##          C    0    0 3422    0    0
##          D    0    0    0 3216    0
##          E    0    0    0    0 3607
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9998, 1)
##     No Information Rate : 0.2844     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##                                      
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

As we can see, this model makes an excellent fitting of our training data. And we don't have to make extra cross validation since the "randomForest" function does it internally as it performs the classification of the data.

Now, we can estimate our out-of-sample error. Since the model fits perfectly the training dataset, this could be caused due an overfitting in the model. So, the out-of-sample error could be large. However, since we are using the variables which explain most of the variability of the dataset, the model could perform well predicting values out of the training set, and causing an low out-of-sample error.

