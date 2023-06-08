# Load libraries
library(readr)
library(caret)

# Read files
prot1 <- read.delim(paste0(data_dir, '/file1.csv'), sep=",")
prot2 <- read.delim(paste0(data_dir, '/file2.csv'), sep=",")


# Remove SCI samples
prot1 <- prot1[prot1$Diagnosis!="SCI", ]
prot2 <- prot2[prot2$Diagnosis!="SCI", ]


# Remove duplicated donors
tab <- table(prot1$SubjectId)
tab <- tab[tab>1]

tab <- table(prot2$SubjectId)
tab <- tab[tab>1]


# Keep only relevant clinical features
## Select the relevant features
variables <- as.data.frame(read_csv(paste0(data_dir, "/Variables.csv"), col_names=FALSE))
variables <- variables$X1
clinical_features <- c("SubjectId", variables[1:26], "MCI_Convert")

## Remove the non-relevant clinical features
variablesToRemove <- setdiff(colnames(prot1)[1:132], clinical_features)
prot1 <- prot1[, -which(colnames(prot1) %in% variablesToRemove)]

variablesToRemove <- setdiff(colnames(prot2)[1:45], clinical_features)
prot2 <- prot2[, -which(colnames(prot2) %in% variablesToRemove)]


# Remove proteins with > 20% missing values
count1 <- colSums(is.na(prot1))
count1 <- (count1*100)/nrow(prot1)
covsToRemove1 <- names(which(count1>20))
covsToRemove1 <- covsToRemove1[covsToRemove1!="MCI_Convert"]

count2 <- colSums(is.na(prot2))
count2 <- (count2*100)/nrow(prot2)
covsToRemove2 <- names(which(count2>20))
covsToRemove2 <- covsToRemove2[covsToRemove2!="MCI_Convert"]

covsToRemove <- unique(c(covsToRemove1, covsToRemove2))
prot1 <- prot1[, -which(colnames(prot1) %in% covsToRemove)]
prot2 <- prot2[ ,-which(colnames(prot2) %in% covsToRemove)]


# KNN Imputation
names <- names(count1[which(count1>0)])
names <- names[-match("MCI_Convert", names)]
names <- setdiff(names, covsToRemove)

preProcValues <- preProcess(prot1[, names],
                            method = c("knnImpute"),
                            k = round(sqrt(ncol(prot1)), -1))

prot1_imputed <- predict(preProcValues, prot1, na.action = na.pass)

procNames <- data.frame(col = names(preProcValues$mean), mean = preProcValues$mean, sd = preProcValues$std)

for(i in procNames$col){
  prot1_imputed[i] <- prot1_imputed[i]*preProcValues$std[i]+preProcValues$mean[i] 
}



names <- names(count2[which(count2>0)])
names <- names[-match("MCI_Convert", names)]
names <- setdiff(names, covsToRemove)

preProcValues <- preProcess(prot2[, names],
                            method = c("knnImpute"),
                            k = nrow(prot2[ complete.cases(prot2), ]) + 1)

prot2_imputed <- predict(preProcValues, prot2, na.action = na.pass)

procNames <- data.frame(col = names(preProcValues$mean), mean = preProcValues$mean, sd = preProcValues$std)

for(i in procNames$col) {
  prot2_imputed[i] <- prot2_imputed[i]*preProcValues$std[i]+preProcValues$mean[i] 
}

prot2_imputed <- prot2_imputed[, c(match(colnames(prot1_imputed)[1:17], colnames(prot2_imputed)[1:17]), 18:ncol(prot2_imputed))]
stopifnot(identical(colnames(prot1_imputed)[1:17], colnames(prot2_imputed)[1:17]))


# Log10 transformation
prot1_imputed[, 18:ncol(prot1_imputed)] <- apply(prot1_imputed[, 18:ncol(prot1_imputed)], 2, function(x) log10(x))
prot2_imputed[, 18:ncol(prot2_imputed)] <- apply(prot2_imputed[, 18:ncol(prot2_imputed)], 2, function(x) log10(x))


# Scaling
prot1_imputed[, 18:ncol(prot1_imputed)] <- apply(prot1_imputed[, 18:ncol(prot1_imputed)], 2, function(x) scale(x))
prot2_imputed[, 18:ncol(prot2_imputed)] <- apply(prot2_imputed[, 18:ncol(prot2_imputed)], 2, function(x) scale(x))


# Merge both data frames
int <- intersect(colnames(prot1_imputed), colnames(prot2_imputed))

prot1_imputed <- prot1_imputed[, which(colnames(prot1_imputed) %in% int)]
prot2_imputed <- prot2_imputed[, which(colnames(prot2_imputed) %in% int)]

prot2_imputed <- prot2_imputed[, match(colnames(prot1_imputed), colnames(prot2_imputed))]
stopifnot(identical(colnames(prot1_imputed), colnames(prot2_imputed)))

prot <- as.data.frame(rbind(prot1_imputed, prot2_imputed))


# Scale at individual level
prot[, 18:ncol(prot)] <- apply(prot[, 18:ncol(prot)], 2, function(x) scale(x))
saveRDS(prot, paste0(results_dir, "proteomics_preprocessed.rds"))

