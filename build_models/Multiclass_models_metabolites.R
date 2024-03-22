# Load libraries ----
library(caret)
library(parallel)
library(doParallel)
library(ggplot2)
library(ggcorrplot)
library(gridExtra)
library(grid)
library(DALEX)
library(ggVennDiagram)

cl <- makePSOCKcluster(4)
registerDoParallel(cl)
set.seed(1234)

# Load data ----
data <- readRDS(paste0(data_dir, "metabolomics_preprocessed.rds"))
head(data)
dim(data)
table(data$Diagnosis)


# Split data into train, test and validation ----
## Validation set ----
validation <- data[data$Studyname=="EDAR", ]
dim(validation)

## Train and test ----
set.seed(1234)
traintest <- data[data$Studyname!="EDAR", ]
index <- createDataPartition(traintest$Diagnosis, 
                             p = .7, 
                             list = FALSE, 
                             times = 1)

train <- traintest[index,]
test  <- traintest[-index,]

# Scale data
train[, 20:ncol(train)] <- apply(train[, 20:ncol(train)], 2, function(x) scale(x))
train$Diagnosis <- factor(train$Diagnosis, levels=c("NL", "MCI", "AD"))
index <- match(c("SubjectId", "Studyname", "LastFU_Diagnosis", "MCI_Convert", "Ptau_ASSAY_Zscore", "Ttau_ASSAY_Zscore"), colnames(train))
train <- train[, -index]

test[, 20:ncol(test)] <- apply(test[, 20:ncol(test)], 2, function(x) scale(x))
test$Diagnosis <- factor(test$Diagnosis, levels=c("NL", "MCI", "AD"))
test <- test[, -index]

validation[, 20:ncol(validation)] <- apply(validation[, 20:ncol(validation)], 2, function(x) scale(x))
validation$Diagnosis <- factor(validation$Diagnosis, levels=c("NL", "MCI", "AD"))
validation <- validation[, -index]


# trainControl ----
set.seed(1234)

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 3,
                           classProbs= TRUE, 
                           summaryFunction = multiClassSummary,
                           savePredictions = "all",
                           sampling="down",
                           returnResamp="all",
                           allowParallel = TRUE)


# Algorithms ----

## Logistic regression ----

### First tuning 

info <- getModelInfo("glmnet")
info <- info$glmnet

x=train[, -match("Diagnosis", colnames(train))]
y=train$Diagnosis
len=3
search="grid"

if (search == "grid") {
  numLev <- if (is.character(y) | is.factor(y)) 
    length(levels(y))
  else NA
  if (!is.na(numLev)) {
    fam <- ifelse(numLev > 2, "multinomial", "binomial")
  }
  else fam <- "gaussian"
  if (!is.matrix(x) && !inherits(x, "sparseMatrix")) 
    x <- Matrix::as.matrix(x)
  init <- glmnet::glmnet(x, y, family = fam, nlambda = len + 2, alpha = 0.5)
  lambda <- unique(init$lambda)
  lambda <- lambda[-c(1, length(lambda))]
  lambda <- lambda[1:min(length(lambda), len)]
  out <- expand.grid(alpha = seq(0.1, 1, length = len), 
                     lambda = lambda)
}    else {
  out <- data.frame(alpha = runif(len, min = 0, 1), lambda = 2^runif(len, min = -10, 3))
}


LR <- train(x=train[, -match("Diagnosis", colnames(train))],
            y=train$Diagnosis,
            trControl = fitControl,
            metric="Mean_Balanced_Accuracy",
            method = "glmnet",
            preProcess=c("center", "scale"),
            tuneGrid=out)

saveRDS(LR, paste0(models_dir, "/metabolites_multiclass_EDAR/metabolites_multiclass_LR_tuning_EDAR.rds"))


### Best tuning

info <- getModelInfo("glmnet")
info <- info$glmnet

x=train[, -match("Diagnosis", colnames(train))]
y=train$Diagnosis
len=5
search="grid"

if (search == "grid") {
  numLev <- if (is.character(y) | is.factor(y)) 
    length(levels(y))
  else NA
  if (!is.na(numLev)) {
    fam <- ifelse(numLev > 2, "multinomial", "binomial")
  }
  else fam <- "gaussian"
  if (!is.matrix(x) && !inherits(x, "sparseMatrix")) 
    x <- Matrix::as.matrix(x)
  init <- glmnet::glmnet(x, y, family = fam, nlambda = len + 2, alpha = 0.5)
  lambda <- unique(init$lambda)
  lambda <- lambda[-c(1, length(lambda))]
  lambda <- lambda[1:min(length(lambda), len)]
  out <- expand.grid(alpha = seq(0.1, 1, length = len), 
                     lambda = lambda)
}    else {
  out <- data.frame(alpha = runif(len, min = 0, 1), lambda = 2^runif(len, min = -10, 3))
}


LR <- train(x=train[, -match("Diagnosis", colnames(train))],
            y=train$Diagnosis,
            trControl = fitControl,
            metric="Mean_Balanced_Accuracy",
            method = "glmnet",
            preProcess=c("center", "scale"),
            tuneGrid=out)

saveRDS(LR, paste0(models_dir, "/metabolites_multiclass_EDAR/metabolites_multiclass_LR_EDAR.rds"))



## Random forest ----

### First tuning 
out <- data.frame(mtry=c(5,10,50,100))

RF <- train(x=train[, -match("Diagnosis", colnames(train))],
            y=train$Diagnosis,
            trControl = fitControl,
            metric="Mean_Balanced_Accuracy",
            method = "rf",
            preProcess=c("center", "scale"),
            tuneGrid=out,
            importance=T)

saveRDS(RF, paste0(models_dir, "/metabolites_multiclass_EDAR/metabolites_multiclass_RF_tuning_EDAR.rds"))


### Best tuning
out <- data.frame(mtry=seq(10,100,10))

RF <- train(x=train[, -match("Diagnosis", colnames(train))],
            y=train$Diagnosis,
            trControl = fitControl,
            metric="Mean_Balanced_Accuracy",
            method = "rf",
            preProcess=c("center", "scale"),
            tuneGrid=out,
            importance=T)

saveRDS(RF, paste0(models_dir, "/metabolites_multiclass_EDAR/metabolites_multiclass_RF_EDAR.rds"))



## Support vector machines ----

### First tuning 
out <- data.frame(C=c(1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100))

SVM <- train(x=train[, -match("Diagnosis", colnames(train))],
             y=train$Diagnosis,
             trControl = fitControl,
             metric="Mean_Balanced_Accuracy",
             method = "svmLinear",
             preProcess=c("center", "scale"),
             tuneGrid=out)

saveRDS(SVM, paste0(models_dir, "/metabolites_multiclass_EDAR/metabolites_multiclass_SVM_tuning_EDAR.rds"))


### Best tuning
out <- data.frame(C=seq(10,30,2))

SVM <- train(x=train[, -match("Diagnosis", colnames(train))],
             y=train$Diagnosis,
             trControl = fitControl,
             metric="Mean_Balanced_Accuracy",
             method = "svmLinear",
             preProcess=c("center", "scale"),
             tuneGrid=out)

saveRDS(SVM, paste0(models_dir, "/metabolites_multiclass_EDAR/metabolites_multiclass_SVM_EDAR.rds"))



## Multilayer perceptron ----

### First tuning 
out <- expand.grid(size=c(1,5,10), decay = c(0, 10^seq(-1, -4, length = 4)))

MLP <- train(x=train[, -match("Diagnosis", colnames(train))],
             y=train$Diagnosis,
             trControl = fitControl,
             metric="Mean_Balanced_Accuracy",
             method = "mlpWeightDecay",
             preProcess=c("center", "scale"),
             tuneGrid=out)

saveRDS(MLP, paste0(models_dir, "/metabolites_multiclass_EDAR/metabolites_multiclass_MLP_tuning_EDAR.rds"))

### Best tuning
out <- expand.grid(size=seq(5,20,5), decay = c(10^seq(-3, -6, length = 4)))

MLP <- train(x=train[, -match("Diagnosis", colnames(train))],
             y=train$Diagnosis,
             trControl = fitControl,
             metric="Mean_Balanced_Accuracy",
             method = "mlpWeightDecay",
             preProcess=c("center", "scale"),
             tuneGrid=out)

saveRDS(MLP, paste0(models_dir, "/metabolites_multiclass_EDAR/metabolites_multiclass_MLP_EDAR.rds"))



# Variable importance ----

## Logistic regression
set.seed(1234)
explainer_lr <- DALEX::explain(LR, 
                               label="lr", 
                               data =test[,-match("Diagnosis", colnames(test))], 
                               y = test$Diagnosis)

vip_lr <- model_parts(explainer = explainer_lr, B = 10, type="difference", N = NULL)
p <- plot(vip_lr, max_vars=20)
final_vip_lr <- p$data[order(p$data$dropout_loss.x, decreasing=T), ]

saveRDS(final_vip_lr, "final_vip_lr.rds")
saveRDS(vip_lr, "vip_lr.rds")


## Random forest
set.seed(1234)
explainer_rf <- DALEX::explain(RF, 
                               label="rf", 
                               data =test[,-match("Diagnosis", colnames(test))], 
                               y = test$Diagnosis)

vip_rf <- model_parts(explainer = explainer_rf, B = 10, type="difference", N = NULL)
p <- plot(vip_rf, max_vars=20)
final_vip_rf <- p$data[order(p$data$dropout_loss.x, decreasing=T), ]

saveRDS(final_vip_rf, "final_vip_rf.rds")
saveRDS(vip_rf, "vip_rf.rds")


## Support vector machines
set.seed(1234)
explainer_svm <- DALEX::explain(SVM, 
                                label="svm", 
                                data =test[,-match("Diagnosis", colnames(test))], 
                                y = test$Diagnosis)

vip_svm <- model_parts(explainer = explainer_svm, B = 10, type="difference", N = NULL)
p <- plot(vip_svm, max_vars=20)
final_vip_svm <- p$data[order(p$data$dropout_loss.x, decreasing=T), ]

saveRDS(final_vip_svm, "final_vip_svm.rds")
saveRDS(vip_svm, "vip_svm.rds")

## Multilayer perceptron 
set.seed(1234)

explainer_mlp <- DALEX::explain(MLP, 
                                label="mlp", 
                                data =test[,-match("Diagnosis", colnames(test))], 
                                y = test$Diagnosis)

vip_mlp <- model_parts(explainer = explainer_mlp, B = 10, type="difference", N = NULL)
p <- plot(vip_mlp, max_vars=20)
final_vip_mlp <- p$data[order(p$data$dropout_loss.x, decreasing=T), ]

saveRDS(final_vip_mlp, "final_vip_mlp.rds")
saveRDS(vip_mlp, "vip_mlp.rds")



