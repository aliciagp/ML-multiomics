# Load library
library(tableone)

# Load merged multiomics data
matrix <- readRDS(paste0(data_path, "multiomics.rds"))

# Select MCI donors with conversion information available
matrix <- matrix[which(matrix$MCI_Convert %in% c(0,1)), ]
table(matrix$MCI_Convert)

# Select the variables you want to include in the summary table as well as the order of appearance
myvars <- colnames(matrix)[c(3, 4, 2, 8, 10, 11, 7, 15:18)]

# Define features will be treated as categorical variables
catVars <- c("Gender", "APOEdich_IMP", "AMYLOIDstatus")

# Create the summary table
tab <- CreateTableOne(vars = myvars, strata = "MCI_Convert" , data = matrix, factorVars = catVars)
tab <- print(tab, nonnormal = "", quote = FALSE, noSpaces = TRUE, printToggle = FALSE)
tab

# Save the table
# write.csv(tab, file = "demographics.csv")

