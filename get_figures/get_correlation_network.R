# Load libraries
library(readxl)
library(Hmisc)
library(qgraph)

# List the variables selected ordered
multiclass <- c("MCI_Convert", "MMSE", "PriorityLanguageZscore", "PriorityAttentionZscore", "PriorityMemoryDelayedZscore", "AB_Zscore", "AMYLOIDstatus", "oleamide", "methionine.sulfoxide")
conversion <- c("Local_PTAU", "Local_TTAU", "PI15", "CFP", "SNCA", "JPH3", "PPY", "PLA2G1B", "CRAT", "TNFRSF19")
selected <- c(multiclass, conversion)

# Load expression matrix
data <- readRDS(paste0(data_path, "multiomics.rds"))

# Select MCI converters or non-converters donors
data <- data[data$MCI_Convert %in% c(0,1), ]

# Get protein symbols
p <- read_excel(paste0(results_path, "Protein names.xlsx"))
index <- match(colnames(data), p$UniProt)
colnames(data)[which(!is.na(index))] <- p$EntrezGeneSymbol[index[!is.na(index)]]
data <- data[, match(selected, colnames(data))]

# Preprocess variable names
colnames(data)[1:10] <- c("cMCI", "MMSE", "Priority\nLanguage\nZscore", "Priority\nAttention\nZScore", "AB\nZscore", "AMYLOID\nstatus", "Local\nPTAU", "Local\nTTAU", "oleamide", "methionine\nsulfoxide")

# Get correlations
corMat <- cor_auto(data)

# Get correlations graph
graph <- qgraph(corMat,
                graph = "cor",
                sampleSize = nrow(data),
                layout = "spring",
                # correlations 
                minimum = "sig",
                cut=0,
                details=T,
                curveAll = F, 
                curveDefault = 0.5,
                # edges colors
                posCol="forestgreen",
                negCol="firebrick",
                color = c("white", rep("white", ncol(data)-1)),
                # nodes color and shape
                shape=rep(c("diamond", "square", "triangle", "circle"), c(1,7,2,8)),
                groups=rep(c("target", "clinical", "metabolites", "proteins"), c(1,7,2,8)),
                palette = 'colorblind',  # 'rainbow', 'pastel', 'gray', 'R', 'ggplot2'
                theme = "Hollywood", # 'classic', 'gray', 'Hollywood', 'Borkulo', 'gimme','TeamFortress', 'Reddit', 'Leuven', 'Fried'
                borders = T,
                # legend
                nodeNames = colnames(data),
                legend.mode = 'style3',
                legend.cex=0.4,
                # figure features
                font = 1.5,
                width = 7 * 1.4, # width of figure
                height = 5,
                # nodes size
                vsize = ifelse(colnames(data) == "cMCI", 9,7),
                border.width = 0.5,
                # nodes labels
                labels = colnames(data),
                label.cex = 1.2, # scalar on label size
                label.color = 'black', # string on label colors
                label.prop = 0.7,
                label.norm= "OOOOOO",
                label.font=1,
                fade=T,
                colFactor=0.1,
                GLratio=5,
                repulsion=0.88,
                vTrans=210)