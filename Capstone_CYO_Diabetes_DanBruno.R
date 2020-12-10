# Introduction
#  This report was produced for the HarvardX PH125.9x Data Science course. The Capstone 'Choose Your Own project' is the second and last of the two projects that have to be submitted to pass the course. In this project, we can chose our own dataset to analyze and implement machine learning to predict an outocme.

## Dataset
#We chose a diabetes dataset taken from the hospital Frankfurt, Germany. The dataset was obtained from https://www.kaggle.com/c/diabetes/overview. The data is structurd as follows:

# | Column Name               | Data Type | Description                 |
# | --------------------------| --------- | --------------------------- |
# | Id                        | Integer   | Unique Pation Identifier    |
# | Pregnancies               | Integer   | Number of Pregnancies       |
# | Glucose                   | Integer   | Glucouse                    |
# | BloodPressure             | Integer   | Blood Preasure              |
# | SkinThickness             | Integer   | Skin Thinkness              |
# | Insulin                   | Integer   | Insulin                     |
# | BMI                       | Decimal   | BMI                         |
# | DiabetesPedigreeFunction  | Decimal   | Diabetes Pedigree Function  |
# | Age                       | Integer   | Age                         |
# | Outcome                   | Integer   | Outcome                     |
# 
# The **Outcome** field is what we want to predict. We will use the rest of the fields, except for **Id**, as predictors.

## Objective
#The objective of this project is to learn which machine learning algorithm can achieve the best predictive accuracy. The machine learning algorithms that we will consider are: Classification and Regression Trees, Random Forest, Gradient Boosting Machine, k-Nearest Neighbors, Generalized Linear Model, Support Vector Machine with Radial Basis Function Kernel, and eXtreme Gradient Boosting. We set a target goal of **95% accuracy**.  

## Train & Test dataset breakdown
#The dataset is given in two different files with identical structure. The first file, named **train.csv**, contains the data that will be used to train the models. The second file, named **test.csv**, will be used to score the models and obtain accuracy metrics.

## Load Libraries
#We automate the installation of the necessary libraries for convinience and resuability." 

list.of.packages <- c("DBI", "dplyr", "tidyverse","Hmisc","odbc","reshape2","gridExtra","ggplot2","plotly","ggcorrplot","forecast","GGally","lessR","future.apply","fpp3","furrr","feasts","tinytex","knitr","caret","Rborist","glmnet","kableExtra","caretEnsemble","PerformanceAnalytics","mlbench","nnet","gbm")

new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]

if(length(new.packages)) install.packages(new.packages)

library(DBI)
library(dplyr)
library(tidyverse)
library(Hmisc)
library(reshape2)
library(gridExtra)
library(ggplot2)
library(plotly)
library(ggcorrplot)
library(GGally)
library(forecast)
library(fpp3)
library(lessR)
library(furrr)
library(feasts)
library(tinytex)
library(knitr)
library(caret)
library(Rborist)
library(glmnet)
library(kableExtra)
library(caretEnsemble)
library(PerformanceAnalytics)
library(mlbench)
library (nnet)
library(gbm)

# Methods & Analysis
# We conduct exploratory data analysis. We seek to visualize the data to understand patterns, and distribution. The insights we gain from this analysis will inform our decisions to create a machine learning model that can be effective at predicting ratings for new reviews.

# Exploratory Data Analysis
# Read the dataset
# We read the dataset which are in **csv** format. The files are located in the dataset subfolder."

data_train <- read.csv("dataset/train.csv")
data_test <- read.csv("dataset/test.csv")

# Dataset Schema
# We display the schema of the training data to ensure it matches our expectations."

str(data_train)

## Data Quality Checks
# We need to understand if there are any missing values in the dataset. We find that there are no missing values."

sapply(data_train, function(x) sum(is.na(x))) %>%
  melt() %>%
  kable() %>%
  kable_classic(full_width = F, html_font = "Calibri")

# Summary Statistics
# We begin to explore the dataset to understand the distribution of the data, patterns and trends."

summary(data_train)

# Row Counts
# Next we want to understand the distinct counts for each column. 
# We find that there are **1405** observations. The Id field is indeed unique to each observation. 
# There are only two outcomes 1 and 0 correspoting to **postive** diagnosis and **negative** diagnosis.  
# We learn that categorical machine learning models would be appropriate for this type of dataset.
# We understand that the **Outcome** column will need to be converted to **factor** to make it compatible with the machine learning models. 

data_train %>% summarise(n_patients = n_distinct(Id),
                         n_pregnancies = n_distinct(Pregnancies),
                         n_blood_preassure = n_distinct(BloodPressure),
                         n_skin_thicknes = n_distinct(SkinThickness),
                         n_insulin = n_distinct(Insulin),
                         n_bmi = n_distinct(BMI),
                         n_diabetes_pedigree_function = n_distinct(DiabetesPedigreeFunction),
                         n_age = n_distinct(Age),
                         n_outcome = n_distinct(Outcome)) %>%
  melt() %>%
  kable() %>%
  kable_classic(full_width = F, html_font = "Calibri")

# Next we seek to expand our understanding by breaking down distinct value counts by the two possible outcomes. 
# We find that in general the means of the values are higher when the patient is diagnosed with diabetes."

data_train %>% group_by(Outcome) %>%
  summarise(n_patients = n_distinct(Id),
            n_pregnancies = mean(Pregnancies),
            n_bp = mean(BloodPressure),
            n_skin_thicknes = mean(SkinThickness),
            n_insulin = mean(Insulin),
            n_bmi = mean(BMI),
            n_diabetes_pedigree = mean(DiabetesPedigreeFunction),
            n_age = mean(Age)) %>%
  kable()  %>%
  kable_classic(full_width = F, html_font = "Calibri")

# Boxplots
# We seek to better understand the distribution of the data for each outcome. For this we need to knwo more than the means. 
# We find that the medians are generally higher for all predictors when the patient is diagnozed with diabetes."

data_train %>% select(Pregnancies, 
                      Glucose, 
                      BloodPressure, 
                      SkinThickness, 
                      Insulin, 
                      BMI, 
                      DiabetesPedigreeFunction, 
                      Age, 
                      Outcome) %>%
  pivot_longer(., cols = c(Pregnancies, 
                           Glucose, 
                           BloodPressure, 
                           SkinThickness, 
                           Insulin, 
                           BMI, 
                           DiabetesPedigreeFunction, 
                           Age), 
               names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = factor(variable), y = value, fill = factor(Outcome))) +
  geom_boxplot() + 
  facet_wrap(~variable, scale="free") 

# Historgrams
# We want to understand the distribution of the predictor values. 
# We find **Blood Pressure** and **BMI** are not significantly different from a normal distribution, although they don't match a normal distribution perfectly. 
# The rest of the predictors are significantly different from a normal distribution."

data_train %>% select(-Id, -Outcome) %>%
               hist.data.frame()

#  Correlation Analysis
# We want to understand the predictive power of each field in the dataset. 
# We also seek to understand if there is any correlation between the predictors themselves." 

data_train %>% select(-Id) %>% ggcorr(label = TRUE)

# Preprocessing
# We remove the Id column since it is a unique identifier for each observation. 
# We also transform the **Outcome** column to **factor** as this is required by the Caret package when using classification algorithms. 
# We inspect the first 10 records of the transofmred trained dataset for quality assurance."

data_train_transformed <- data_train %>% select(-Id)
data_train_transformed$Outcome <- as.factor(ifelse(data_train$Outcome == 1,"Yes","No"))

# Isolate Outcome column into its own object for later use in measuring the performance fo the model
test_outcome <- as.factor(ifelse(data_test$Outcome == 1,"Yes","No"))

# Remove Id and Outcome from test dataset that will be used to score the trained models.
data_test_transformed <- data_test %>% select(-Id, -Outcome, -split)

data_train_transformed %>% head() %>%
  knitr::kable(caption = "Accuracy")  %>%
  kable_classic(full_width = F, html_font = "Calibri")

## Modeling

### Cross Validation

# We implement cross validation to reduce mitigate the issue of the model performing well due to chance and to avoid overfilling. 
# Set set **K** to **10** as this is standard approach. We set seeds before we train each model to ensure evaluations are deterministic. 
# We also set the seed values which we will use before we train each model to ensure that our results will be deterministic.

# Cross Validation using 10-fold cross validation 
control <- trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = "final", 
                        allowParallel = TRUE, 
                        classProbs = TRUE, 
                        summaryFunction = twoClassSummary)

metric <- "Accuracy"

seed <- 7

## Train Models

# We proceed to build different classification models to understand which ones can be more effective based on our dataset. 
# We also will conduct hyperparameter tuning. We identify tuning parameters for each model and we will perform a grid search 
# to find which values optimize our models best. The following table describes the models we will build along with their tuning parameters. 

# | Model                                                     | Method Value | Libraries | Tuning Parameters                                                             |
# | ----------------------------------------------------------| ------------ | ----------| ----------------------------------------------------------------------------- |
# | Classification and Regression Trees                       | rpart        | rpart     | cp                                                                            |
# | Random Forest                                             | Rborist      | Rborist   | predFixed, minNode                                                            |  
# | Gradient Boosting Machine                                 | gbm          | gbm       | interaction.depth, n.trees, shrinkage, n.minobsinnode                         |
# | k-Nearest Neighbors                                       | nnet         | nnet      | K                                                                             |
# | Generalized Linear Model                                  | glmnet       | glmnet    | Matrix, alpha, lambda                                                         |    
# | Support Vector Machine with Radial Basis Function Kernel  | svmRadial    | svmRadial | Sigma, C                                                                      |
# | eXtreme Gradient Boosting       	                        | xgbTree      | plyr      | nrounds, max_depth, eta, gamma, colsample_bytree, min_child_weight, subsample |

# Generalized Linear  Model
tune_grid_glm <- expand.grid(alpha = 0:1, lambda = seq(0, 10, 0.25))

set.seed(seed)

model_glm <- train(Outcome ~ ., 
                   data = data_train_transformed, 
                   method = "glmnet", 
                   metric = metric, 
                   trControl = control,
                   tuneGrid = tune_grid_glm,
                   trace = FALSE)

# k-Nearest Neighbors
tune_grid_knn <-data.frame(k = c(3,5,7))

set.seed(seed)

model_knn <- train(Outcome~., 
                   data = data_train_transformed, 
                   method = "knn", 
                   metric = metric, 
                   trControl = control,
                   tuneGrid = tune_grid_knn)

# Gradient boosting machine
tune_grid_gbm <- expand.grid(interaction.depth=c(1, 3, 5), 
                             n.trees = (0:50)*50,
                             shrinkage=c(0.01, 0.001),
                             n.minobsinnode=10)
set.seed(seed)

model_gbm <- train(Outcome~., 
                   data=data_train_transformed, 
                   method="gbm", 
                   metric=metric, 
                   trControl=control,
                   tuneGrid = tune_grid_gbm)

# Neural Networks
tune_grid_nnet <- expand.grid(size = seq(from = 1, to = 10, by = 1),
                              decay = seq(from = 0.1, to = 0.5, by = 0.1))

set.seed(seed)

model_nnet <- train(Outcome~.,
                    data = data_train_transformed, 
                    method = 'nnet', 
                    trControl = control,
                    metric = metric,
                    tuneGrid = tune_grid_nnet)

# Support Vector Machine using non-linear kernel
tune_grid_svm <- expand.grid(sigma= 2^c(-25, -20, -15,-10, -5, 0), C= 2^c(0:5))

set.seed(seed)

model_svm <-  train(Outcome ~ .,
                    data = data_train_transformed, 
                    method = "svmRadial",
                    tuneGrid = tune_grid_svm,
                    metric = metric,
                    trControl = control)

# Classification Trees
tune_grid_rpart <- data.frame(cp = seq(0.0, 0.1, len = 25))

set.seed(seed)

model_rpart <- train(Outcome~., 
                     data = data_train_transformed, 
                     method = "rpart", 
                     metric = metric, 
                     trControl = control,
                     tuneGrid = tune_grid_rpart)

# Random Forest
tune_grid_rf <- data.frame(predFixed = 2, minNode = c(3, 50))

set.seed(seed)

model_rf <- train(Outcome~., 
                  data=data_train_transformed, 
                  method="Rborist", 
                  metric=metric, 
                  trControl=control,
                  tuneGrid = tune_grid_rf)

# Summarize model accuracy for each model
training_results <- resamples(list(glm = model_glm,
                                   knn = model_knn,
                                   gbm = model_gbm,
                                   nnet = model_nnet,
                                   svm = model_svm,
                                   rpart = model_rpart,
                                   Rborist = model_rf))

# Compare accuracy of models
dotplot(training_results)
            
### Ensamble Models
# We want to understand if we can improve the accuracy by combining the previous models into an ensemble of models which can perform better than an individual model. 
# **caretEnsemble** is a package for making ensembles of caret models. In the **Caret** package, the function caretList is the preferred way to construct list of 
# caret models, because it will ensure the resampling indexes are identical across all models.

set.seed(seed)

model_list <- caretList(Outcome~., 
                        data=data_train_transformed, 
                        trControl=control,
                        metric = metric,
                        tuneList=list(
                          glm = caretModelSpec(method = "glmnet", tuneGrid = tune_grid_glm),
                          knn = caretModelSpec(method = "knn", tuneGrid = tune_grid_knn),
                          gbm = caretModelSpec(method = "gbm", tuneGrid = tune_grid_gbm),
                          nnet = caretModelSpec(method = "nnet", tuneGrid = tune_grid_nnet),
                          svm = caretModelSpec(method = "svmRadial", tuneGrid = tune_grid_svm),
                          rpart = caretModelSpec(method = "rpart", tuneGrid = tune_grid_rpart),
                          Rborist = caretModelSpec(method = "Rborist", tuneGrid = tune_grid_rf)
                        )
)
set.seed(seed)

model_ensemble <- caretEnsemble(model_list,
                                metric = metric, 
                                trControl = trainControl(
                                  method = "cv",
                                  number = 5,
                                  classProbs = TRUE,
                                  verboseIter = FALSE,
                                  returnData = FALSE
                                )
)

model_ensemble_results <- resamples(model_list)
dotplot(model_ensemble_results)

# Results
# We use the test data to score our trained models and obtained accuracy metrics.

predict_glm <- predict(model_glm, data_test_transformed)
accuracy_glm <- confusionMatrix(predict_glm, test_outcome)$overall["Accuracy"]

predict_rpart <- predict(model_rpart, data_test_transformed)
accuracy_rpart <- confusionMatrix(predict_rpart, test_outcome)$overall["Accuracy"]

predict_gbm <- predict(model_gbm, data_test_transformed)
accuracy_gbm <- confusionMatrix(predict_gbm, test_outcome)$overall["Accuracy"]

predict_nnet <- predict(model_nnet, data_test_transformed)
accuracy_nnet <- confusionMatrix(predict_nnet, test_outcome)$overall["Accuracy"]

predict_svm <- predict(model_svm, data_test_transformed)
accuracy_svm <- confusionMatrix(predict_svm, test_outcome)$overall["Accuracy"]

predict_knn <- predict(model_knn, data_test_transformed)
accuracy_knn <- confusionMatrix(predict_knn, test_outcome)$overall["Accuracy"]

predict_rf <- predict(model_rf, data_test_transformed)
accuracy_rf <- confusionMatrix(predict_rf, test_outcome)$overall["Accuracy"]

predict_ensemble <- predict(model_ensemble, data_test_transformed)
accuracy_ensemble <- confusionMatrix(predict_ensemble, test_outcome)$overall["Accuracy"]

results <- data_frame(Model = "NNET", Accuracy = accuracy_nnet)
results <- rbind(results, data_frame(Model = "K-Nearest Neighbor", Accuracy = accuracy_knn))
results <- rbind(results, data_frame(Model = "Generalized Linear Model", Accuracy = accuracy_glm))
results <- rbind(results, data_frame(Model = "Classification Trees", Accuracy = accuracy_rpart))
results <- rbind(results, data_frame(Model = "Gradient Boosting Machine", Accuracy = accuracy_gbm))
results <- rbind(results, data_frame(Model = "Support Vector Machine", Accuracy = accuracy_svm))
results <- rbind(results, data_frame(Model = "Random Forest", Accuracy = accuracy_rf))
results <- rbind(results, data_frame(Model = "Models Ensemble", Accuracy = accuracy_ensemble))

results %>% knitr::kable(caption = "Results Table")  %>%
  kable_classic(full_width = F, html_font = "Calibri")