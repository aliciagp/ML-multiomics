# Load libraries ----
library(readr)
library(caret)
library(readxl)
library(gridExtra)


# Read files ----
## Proteomics file 1
prot1 <- read.delim(paste0(data_dir, '/old_cohort.csv'), sep=",")
dim(prot1)
head(prot1)

## Proteomics file 2
prot2 <- read.delim(paste0(data_dir, '/new_cohort.csv'), sep=",")
dim(prot2)
head(prot2)

dictionary <- prot1[, c("Study", "Studyname")]
dictionary <- dictionary[!duplicated(dictionary), ]
dictionary <- dictionary[order(dictionary$Study), ]

Studyname <- dictionary[match(prot2$Study, dictionary$Study), ]
stopifnot(identical(Studyname$Study, prot2$Study))

prot2 <- cbind(prot2[, 1:2], Studyname=Studyname$Studyname, prot2[, 3:ncol(prot2)])


# Remove SCI subjects ----
prot1 <- prot1[prot1$Diagnosis!="SCI", ]
prot2 <- prot2[prot2$Diagnosis!="SCI", ]


# Check for duplicated donors ----
tab <- table(prot1$SubjectId)
tab[tab>1]

tab <- table(prot2$SubjectId)
tab[tab>1]


# Remove the non-relevant clinical features ----
variables <- as.data.frame(read_csv(paste0(data_dir, "/Variables.csv"), col_names=FALSE))
variables <- variables$X1
clinical_features <- c("SubjectId", "Studyname", "LastFU_Diagnosis", variables[1:26], "MCI_Convert")

variablesToRemove <- setdiff(colnames(prot1)[1:132], clinical_features)
prot1 <- prot1[, -match(variablesToRemove, colnames(prot1))]

variablesToRemove <- setdiff(colnames(prot2)[1:46], clinical_features)
prot2 <- prot2[, -match(variablesToRemove, colnames(prot2))]


# Check for duplicated columns
count <- !duplicated(asplit(prot1, 2))
table(count)

count <- !duplicated(asplit(prot2, 2))
table(count)


# Data preprocessing ----

## Removing metabolites with > 20% missing ----
count1 <- colSums(is.na(prot1))
count1 <- (count1*100)/nrow(prot1)
covsToRemove1 <- names(which(count1>20))
covsToRemove1 <- covsToRemove1[covsToRemove1!="MCI_Convert"]

count2 <- colSums(is.na(prot2))
count2 <- (count2*100)/nrow(prot2)
covsToRemove2 <- names(which(count2>20))
covsToRemove2 <- covsToRemove2[covsToRemove2!="MCI_Convert"]

covsToRemove <- unique(c(covsToRemove1, covsToRemove2))
prot1 <- prot1[, -match(covsToRemove, colnames(prot1))]
prot2 <- prot2[ ,-match(covsToRemove, colnames(prot2))]


## KNN Imputation ----
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


prot2_imputed <- cbind(prot2_imputed[, 1:18], LastFU_Diagnosis=rep(NA, nrow(prot2_imputed)), prot2_imputed[, 19:ncol(prot2_imputed)])
covariates <- intersect(colnames(prot2_imputed)[1:19], colnames(prot1_imputed)[1:19])
proteins <- intersect(colnames(prot2_imputed)[20:ncol(prot2_imputed)], colnames(prot1_imputed)[20:ncol(prot1_imputed)])

prot1_imputed2 <- prot1_imputed[, match(c(covariates, proteins), colnames(prot1_imputed))]
prot2_imputed2 <- prot2_imputed[, match(c(covariates, proteins), colnames(prot2_imputed))]

stopifnot(identical(colnames(prot1_imputed2), colnames(prot2_imputed2)))

prot1_imputed2 <- cbind(set=rep("1", nrow(prot1_imputed2)), prot1_imputed2)
prot2_imputed2 <- cbind(set=rep("2", nrow(prot2_imputed2)), prot2_imputed2)


## Merge both data frames ----
prot <- as.data.frame(rbind(prot1_imputed2, prot2_imputed2))


## Log10 transformation ----
prot[, 21:ncol(prot)] <- apply(prot[, 21:ncol(prot)], 2, function(x) log10(x))


## All columns as numeric ----
prot[, 21:ncol(prot)] <- apply(prot[, 21:ncol(prot)], 2, function(x) as.numeric(as.character(x)))


## Save the file ----
saveRDS(prot, paste0(results_dir, "proteomics_preprocessed.rds"))

