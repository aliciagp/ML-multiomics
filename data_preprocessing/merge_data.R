# Load libraries
library(readr)


# Read the files
prot <- readRDS(paste0(data_dir, "proteomics_preprocessed.rds"))
met <- readRDS(paste0(data_dir, "metabolomics_preprocessed.rds"))


# Select the individuals for which we have both proteomics and metabolomics (paired data)
ids <- intersect(prot$SubjectId, met$SubjectId)

prot <- prot[match(ids, prot$SubjectId), ]
met <- met[match(ids, met$SubjectId), ]

stopifnot(identical(prot$SubjectId, met$SubjectId))
stopifnot(identical(colnames(prot)[1:17], colnames(met)[1:17]))


# Merge both metabolomcis and proteomics table
data <- as.data.frame(cbind(met[1:17], prot[, 18:ncol(prot)], met[, 18:ncol(met)]))


# Scale the data
data[, 18:ncol(data)] <- apply(data[, 18:ncol(data)], 1, function(x) scale(x))


# Split MCI into converters and non-converters
data$MCI_Convert[is.na(data$MCI_Convert)] <- -1

Diagnosis2 <- data$Diagnosis
Diagnosis2[which(data$MCI_Convert==0)] <- "MCI_NC"
Diagnosis2[which(data$MCI_Convert==1)] <- "MCI_C"
data <- cbind(data[, 1:4], Diagnosis2, data[, 5:ncol(data)])


# Save the file
saveRDS(data, paste0(data_dir, "/multiomics.rds"))
