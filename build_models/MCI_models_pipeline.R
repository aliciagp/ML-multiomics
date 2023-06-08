# Load libraries
library(caret)
library(doParallel)
library(data.table)
library(readxl)
library(writexl)

# Load data
data <- readRDS(paste0(data_dir, "file.rds"))

# Use all data for training
train <- data

# Apply downsampling to address classes imbalance
dim(train)
train_down <- downSample(train, train$Diagnosis2)
train_down$Class <- NULL
dim(train_down)
train <- train_down

# Check that classes are balanced now
table(train$Diagnosis2)


# Define the same train control for all the models
set.seed(1234)
fitControl <- trainControl(method = "repeatedcv",
                           number = 3,
                           repeats = 10,
                           classProbs = TRUE,
                           allowParallel = TRUE,
                           summaryFunction = twoClassSummary,
                           savePredictions = T)

# Define our own grid for hyperparameter tuning for machine learning models
grid_ridge <- expand.grid(.alpha=0,
                          .lambda=10^seq(-3, -1, length =25))
default <- round(sqrt(ncol(train)),0)
grid_rf <- expand.grid(.mtry=seq(default-20, default+20, 10))

grid_mlp <- expand.grid(size = seq(5, 15, 5),
                        decay = 10^seq(-3, -6, length = 4))



# Train the models
## Logistic regression
ridge <- train(x=train[, -match("Diagnosis2", colnames(train))],
               y=train$Diagnosis2,
               trControl = fitControl,
               metric="ROC",
               method = "glmnet",
               preProcess=c("center", "scale"),
               tuneGrid=grid_ridge,
               family="binomial")

saveRDS(ridge, paste0(models_dir, "ridge_MCI.rds"))


## Random forest
rf <- train(x=train[, -match("Diagnosis2", colnames(train))],
            y=train$Diagnosis2,
            trControl = fitControl,
            metric="ROC",
            method = "rf",
            preProcess=c("center", "scale"),
            tuneGrid=grid_rf,
            family="binomial")

saveRDS(rf, paste0(models_dir, "rf_MCI.rds"))


## Support vector machines
svm <- train(x=train[, -match("Diagnosis2", colnames(train))],
             y=train$Diagnosis2,
             trControl = fitControl,
             metric="ROC",
             method = "svmLinear",
             preProcess=c("center", "scale"),
             family="binomial")

saveRDS(svm, paste0(models_dir, "svm_MCI.rds"))


## mlp
mlp <- train(x=train[, -match("Diagnosis2", colnames(train))],
             y=train$Diagnosis2,
             trControl = fitControl,
             metric="ROC",
             method = "mlpWeightDecay",
             preProcess=c("center", "scale"),
             tuneGrid=grid_mlp,
             family="binomial")

saveRDS(mlp, paste0(models_dir, "mlp_MCI.rds"))










