# Multiclass models proteins

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
library(limma)
library(sva)
library(swamp)

cl <- makePSOCKcluster(4)
registerDoParallel(cl)
set.seed(1234)

# Load data ----
data <- readRDS(paste0(data_dir, "proteomics_preprocessed.rds"))
head(data)
dim(data)


# Split data into train/test and validation ----

## Validation 
validation <- data[data$Studyname=="EDAR", ]
dim(validation)

## Train and test 
set.seed(1234)
traintest <- data[data$Studyname!="EDAR", ]
dim(traintest)


# Correct batch effect ----

## Validation set ----
val_covs <- validation[, 1:20]
val_met <- t(validation[, 21:ncol(validation)])
val_covs$set <- factor(val_covs$set, levels=c(1,2))

val_combat <- sva::ComBat(dat=val_met,batch=val_covs$set,
                          mod=model.matrix(~1,data=as.data.frame(val_covs[, c(2,4, 8, 10:13, 15:18)])),
                          par.prior=T)


pca <- prcomp(validation[, c(21:ncol(validation))], scale = T)
df <- as.data.frame(pca$x[, 1:2])
df$cohort <- validation$Studyname
df$batch <- validation$set
var <- (pca$sdev^2 / sum(pca$sdev^2))*100

p1 <- ggplot(df, aes(x=PC1, y=PC2, color=batch)) +
  geom_point() + 
  theme_classic() +
  theme(legend.key.size = unit(0.2, 'cm')) +
  labs(title="Validation set\nbefore batch correction")

pca <- prcomp(t(val_combat), scale = T)
df <- as.data.frame(pca$x[, 1:2])
df$cohort <- val_covs$Studyname
df$batch <- val_covs$set
var <- (pca$sdev^2 / sum(pca$sdev^2))*100

p2 <- ggplot(df, aes(x=PC1, y=PC2, color=batch)) +
  geom_point() + 
  theme_classic() +
  theme(legend.key.size = unit(0.2, 'cm')) +
  labs(title="Validation set\nafter batch correction")

grid.arrange(p1,p2,ncol=2)


## Train/Test set ----
tt_covs <- traintest[, 1:20]
tt_met <- t(traintest[, 21:ncol(traintest)])
tt_covs$set <- factor(tt_covs$set, levels=c(1,2))

tt_combat <- sva::ComBat(dat=tt_met,batch=tt_covs$set,
                         mod=model.matrix(~1,data=as.data.frame(tt_covs[, c(2,4, 8, 10:13, 15:18)])),
                         par.prior=T)


pca <- prcomp(traintest[, c(21:ncol(traintest))], scale = T)
df <- as.data.frame(pca$x[, 1:2])
df$cohort <- traintest$Studyname
df$batch <- traintest$set
var <- (pca$sdev^2 / sum(pca$sdev^2))*100

p1 <- ggplot(df, aes(x=PC1, y=PC2, color=batch)) +
  geom_point() + 
  theme_classic() +
  theme(legend.key.size = unit(0.2, 'cm')) +
  labs(title="Train/Test set\nbefore batch correction")

pca <- prcomp(t(tt_combat), scale = T)
df <- as.data.frame(pca$x[, 1:2])
df$cohort <- tt_covs$Studyname
df$batch <- tt_covs$set
var <- (pca$sdev^2 / sum(pca$sdev^2))*100

p2 <- ggplot(df, aes(x=PC1, y=PC2, color=batch)) +
  geom_point() + 
  theme_classic() +
  theme(legend.key.size = unit(0.2, 'cm')) +
  labs(title="Train/Test set\nafter batch correction")

grid.arrange(p1,p2,ncol=2)


# Split into train and test ----
dim(val_combat)
val_combat <- as.data.frame(t(val_combat))
dim(val_combat)
val_combat <- cbind(val_covs, val_combat)
dim(val_combat)
stopifnot(identical(rownames(val_combat), rownames(validation)))
stopifnot(identical(colnames(val_combat), colnames(validation)))
validation <- val_combat

dim(tt_combat)
tt_combat <- as.data.frame(t(tt_combat))
dim(tt_combat)
tt_combat <- cbind(tt_covs, tt_combat)
dim(tt_combat)
stopifnot(identical(rownames(tt_combat), rownames(traintest)))
stopifnot(identical(colnames(tt_combat), colnames(traintest)))

set.seed(1234)
index <- createDataPartition(tt_combat$Diagnosis,
                             p = .7,
                             list = FALSE,
                             times = 1)

train <- tt_combat[index,]
test  <- tt_combat[-index,]


# Scale data

train[, 21:ncol(train)] <- apply(train[, 21:ncol(train)], 2, function(x) scale(x))
train$Diagnosis <- factor(train$Diagnosis, levels=c("NL", "MCI", "AD"))
index <- match(c("SubjectId", "Studyname", "LastFU_Diagnosis", "MCI_Convert", "Ptau_ASSAY_Zscore", "Ttau_ASSAY_Zscore", "set"), colnames(train))
train <- train[, -index]

test[, 21:ncol(test)] <- apply(test[, 21:ncol(test)], 2, function(x) scale(x))
test$Diagnosis <- factor(test$Diagnosis, levels=c("NL", "MCI", "AD"))
test <- test[, -index]

validation[, 21:ncol(validation)] <- apply(validation[, 21:ncol(validation)], 2, function(x) scale(x))
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
set.seed(1234)
n <- round(nrow(train)/2, 0)
tuning <- sample(1:nrow(train), n)


## Logistic regression ----

### First tuning 
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


LR <- train(x=train[tuning, -match("Diagnosis", colnames(train))],
            y=train$Diagnosis[tuning],
            trControl = fitControl,
            metric="Mean_Balanced_Accuracy",
            method = "glmnet",
            preProcess=c("center", "scale"),
            tuneGrid=out)

saveRDS(LR, paste0(models_dir, "/proteins_multiclass_EDAR/proteins_multiclass_LR_tuning_EDAR.rds"))


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

saveRDS(LR, paste0(models_dir, "/proteins_multiclass_EDAR/proteins_multiclass_LR_EDAR.rds"))


## Random forest ----

### First tuning 
out <- data.frame(mtry=c(10, 50, 100, 150, 200, 250, 300))

RF <- train(x=train[tuning, -match("Diagnosis", colnames(train))],
            y=train$Diagnosis[tuning],
            trControl = fitControl,
            metric="Mean_Balanced_Accuracy",
            method = "rf",
            preProcess=c("center", "scale"),
            tuneGrid=out,
            importance=T)

saveRDS(RF, paste0(models_dir, "/proteins_multiclass_EDAR/proteins_multiclass_RF_tuning_EDAR.rds"))


### Best tuning
out <- data.frame(mtry=seq(200,500,50))

RF <- train(x=train[, -match("Diagnosis", colnames(train))],
            y=train$Diagnosis,
            trControl = fitControl,
            metric="Mean_Balanced_Accuracy",
            method = "rf",
            preProcess=c("center", "scale"),
            tuneGrid=out,
            importance=T)

saveRDS(RF, paste0(models_dir, "/proteins_multiclass_EDAR/proteins_multiclass_RF_EDAR.rds"))


## Support vector machines ----

### First tuning 
out <- data.frame(C=c(1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100))

SVM <- train(x=train[tuning, -match("Diagnosis", colnames(train))],
             y=train$Diagnosis[tuning],
             trControl = fitControl,
             metric="Mean_Balanced_Accuracy",
             method = "svmLinear",
             preProcess=c("center", "scale"),
             tuneGrid=out)

saveRDS(SVM, paste0(models_dir, "/proteins_multiclass_EDAR/proteins_multiclass_SVM_tuning_EDAR.rds"))


### Best tuning
out <- data.frame(C=c(1,2,3,4,5,40,50,60,70))

SVM <- train(x=train[, -match("Diagnosis", colnames(train))],
             y=train$Diagnosis,
             trControl = fitControl,
             metric="Mean_Balanced_Accuracy",
             method = "svmLinear",
             preProcess=c("center", "scale"),
             tuneGrid=out)

saveRDS(SVM, paste0(models_dir, "/proteins_multiclass_EDAR/proteins_multiclass_SVM_EDAR.rds"))


## Multilayer perceptron ----

### First tuning 
out <- expand.grid(size=c(1,5,10), decay = c(0, 10^seq(-1, -4, length = 4)))

MLP <- train(x=train[tuning, -match("Diagnosis", colnames(train))],
             y=train$Diagnosis[tuning],
             trControl = fitControl,
             metric="Mean_Balanced_Accuracy",
             method = "mlpWeightDecay",
             preProcess=c("center", "scale"),
             tuneGrid=out)

saveRDS(MLP, paste0(models_dir, "/proteins_multiclass_EDAR/proteins_multiclass_MLP_tuning_EDAR.rds"))

### Best tuning
out <- expand.grid(size=seq(10,30,10), decay = c(10^seq(-2, -4, length = 3)))

MLP <- train(x=train[, -match("Diagnosis", colnames(train))],
             y=train$Diagnosis,
             trControl = fitControl,
             metric="Mean_Balanced_Accuracy",
             method = "mlpWeightDecay",
             preProcess=c("center", "scale"),
             tuneGrid=out)

saveRDS(MLP, paste0(models_dir, "/proteins_multiclass_EDAR/proteins_multiclass_MLP_EDAR.rds"))


# Variable importance ----
## Logistic regression
set.seed(1234)

explainer_lr <- DALEX::explain(LR, 
                               label="lr", 
                               data =test[,-match("Diagnosis", colnames(test))], 
                               y = test$Diagnosis)

vip_lr <- model_parts(explainer = explainer_lr, B = 10, type="difference", N = NULL, parallel=T)
p <- plot(vip_lr, max_vars=20)
final_vip_lr <- p$data[order(p$data$dropout_loss.x, decreasing=T), ]

saveRDS(final_vip_lr, "final_vip_lr_proteins.rds")
saveRDS(vip_lr, "vip_lr_proteins.rds")

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

