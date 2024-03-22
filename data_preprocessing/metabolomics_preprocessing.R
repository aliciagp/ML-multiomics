# Load libraries ----
library(readr)
library(caret)
library(gridExtra)

# Read file ----
met <- as.data.frame(read_csv(file = paste0(data_dir, '/EMIF_data_log10_scaled_3sd_removed.csv')))[, -1]

# Remove SCI subjects ----
table(met$Diagnosis)
met <- met[met$Diagnosis!="SCI", ]
table(met$Diagnosis)


# Remove duplicated donors ----
tab <- table(met$SubjectId)
dupIds <- names(tab[tab>1])
index <- which(met$SubjectId %in% dupIds)
met <- met[-index, ]

# Remove non-relevant clinical features ---- 
variables <- as.data.frame(read_csv(paste0(data_dir, "/Variables.csv"), col_names=FALSE))
variables <- variables$X1
clinical_features <- c("SubjectId", "Studyname", "LastFU_Diagnosis", variables[1:26], "MCI_Convert")
met$Sex <- NULL
met$Group <- NULL
met$Diagggroups <- NULL
variablesToRemove <- setdiff(colnames(met)[1:127], clinical_features)
met <- met[, -match(variablesToRemove, colnames(met))]


# Preprocess colnames ----
colnames(met) <- gsub("X", "", colnames(met))
colnames(met) <- gsub("\\.$", "", colnames(met))


# Check for duplicated columns ----
count <- !duplicated(asplit(met, 2))
c <- table(count)

# Data preprocessing ----

## Removing metabolites with > 20% missing ----
count <- colSums(is.na(met))
count <- count*100/(nrow(met))
colsToRemove <- names(count)[count>20]
colsToRemove <- colsToRemove[colsToRemove!=c("LastFU_Diagnosis", "MCI_Convert")]
met <- met[, -match(colsToRemove, colnames(met))]
met$Diaggroups <- NULL


## Knn Imputation ----
count <- colSums(is.na(met))
count <- count*100/(nrow(met))
count <- sort(count, decreasing=T)
names <- names(count)[count>0]
names <- names[names!="LastFU_Diagnosis"]
names <- names[names!="MCI_Convert"]

preProcValues <- preProcess(met[, names],
                            method = c("knnImpute"),
                            k =round(sqrt(ncol(met)),0))

imputed <- predict(preProcValues, met, na.action = na.pass)

procNames <- data.frame(col = names(preProcValues$mean), mean = preProcValues$mean, sd = preProcValues$std)

for(i in procNames$col){
  imputed[i] <- imputed[i]*preProcValues$std[i]+preProcValues$mean[i] 
}


## All columns as numeric ----
imputed[, 19:ncol(imputed)] <- apply(imputed[, 19:ncol(imputed)], 2, function(x) as.numeric(as.character(x)))

# Save the file ----
saveRDS(imputed, paste0(results_dir, "metabolomics_preprocessed.rds"))

