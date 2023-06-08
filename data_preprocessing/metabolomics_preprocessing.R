# Load libraries
library(readr)
library(caret)


# Read file
met <- as.data.frame(read_csv(file = paste0(data_dir, '/EMIF_data_log10_scaled_3sd_removed.csv')))


# Remove SCI donors
met <- met[met$Diagnosis!="SCI", ]


# Remove duplicated donors
tab <- table(met$SubjectId)
dupIds <- names(tab[tab>1])

df <- met[which(met$SubjectId %in% dupIds), ]
index <- which(met$SubjectId %in% dupIds & met$Study==9)
met <- met[-index, ]


# Remove non-relevant or duplicated variables
index <- grep("Sex", colnames(met))
index <- c(index:ncol(met))
met <- met[, -index]


# Select the relevant clinical features
variables <- as.data.frame(read_csv(paste0(data_dir, "/Variables.csv"), col_names=FALSE))
variables <- variables$X1
clinical_features <- c("SubjectId", variables[1:26], "MCI_Convert")

variablesToRemove <- setdiff(colnames(met)[1:125], clinical_features)
met <- met[, -which(colnames(met) %in% variablesToRemove)]


# Preprocess colnames
colnames(met) <- gsub("X", "", colnames(met))
colnames(met) <- gsub("\\.$", "", colnames(met))


# Remove metabolites with > 20% missing
count <- colSums(is.na(met))
count <- count*100/(nrow(met))
colsToRemove <- names(count)[count>20]
colsToRemove <- colsToRemove[colsToRemove!="MCI_Convert"]
met <- met[, -which(colnames(met) %in% colsToRemove)]


# Knn Imputation
count <- colSums(is.na(met))
count <- count*100/(nrow(met))
names <- names(count)[count>0]
names <- names[names != "MCI_Convert"]

preProcValues <- preProcess(met[, names],
                            method = c("knnImpute"),
                            k =round(sqrt(ncol(met)),0))

imputed <- predict(preProcValues, met, na.action = na.pass)

procNames <- data.frame(col = names(preProcValues$mean), mean = preProcValues$mean, sd = preProcValues$std)

for(i in procNames$col){
  
  imputed[i] <- imputed[i]*preProcValues$std[i]+preProcValues$mean[i] 
}


# Log10 transformation (already applied)


# Scaling
imputed[, 18:ncol(imputed)] <- apply(imputed[, 18:ncol(imputed)], 2, function(x) scale(x))


# Save the file
saveRDS(imputed, paste0(results_dir, "metabolomics_preprocessed.rds"))



