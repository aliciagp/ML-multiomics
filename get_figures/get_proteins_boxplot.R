# Load libraries
library(ggplot2)
library(ggpubr)

# Load merged multiomics data
data <- readRDS(paste0(data_path, "multiomics.rds"))
rownames(data) <- data$SubjectId
data[, "SubjectId"] <- NULL
data <- data[data$MCI_Convert %in% c(0,1), ]
data$Diagnosis[data$MCI_Convert==0] <- "sMCI"
data$Diagnosis[data$MCI_Convert==1] <- "cMCI"
data$Diagnosis <- factor(data$Diagnosis, levels=c("sMCI", "cMCI"))
dim(data)

# Load the proteins selected as relevant for MCI conversion by all the algorithms
vars <- read_excel(paste0(results_path, "selected_variables_MCI.xlsx"))
vars <- table(unlist(vars))
vars <- names(vars[vars==4])
vars <- vars[c(3,4,5,6)]

# Get protein names
p <- read_excel(paste0(results_path, "Protein names.xlsx"))
p <- p[match(vars, p$UniProt), ]

# Create the final data frame with age, gender, diagnosis and the protein levels.
data2 <- data[, match(c("Age", "Gender", "Diagnosis", p$UniProt), colnames(data))]
colnames(data2)[4:6] <- p$EntrezGeneSymbol
head(data2)

# Compare the proteins levels between sMCI and cMCI
data3 <- stack(data2[, 3:ncol(data2)])
data3$dx <- rep(data2$Diagnosis, 4)


ggplot(data3, aes(x=dx, y=values, color=dx)) +
       geom_boxplot(outlier.size = 0.8) + 
       facet_nested_wrap(.~ind, ncol=2) +   
       theme_classic() + 
       ylab(expression(paste("protein levels ", "(", log[10], ")"))) +
       theme(text=element_text(size=16),
              legend.position="bottom", 
              legend.key.size=unit(0.4, "cm"),
              legend.title = element_text(size=10),
              axis.title=element_text(size=12)) +
       geom_jitter(size=0.8, shape=16, position=position_jitter(0.2)) + 
       xlab("MCI conversion") +
       stat_compare_means(label="p.signif", comparison=list(c("sMCI", "cMCI")), method="wilcox.test", label.x=1.4, label.y=c(3.0)) + 
       scale_color_brewer(palette="Set1", direction=-1) + 
       labs(color='MCI') +
       coord_cartesian(ylim=c(-3,3.5)) +
       scale_y_continuous(breaks=seq(-3,4,1))

# For pairwise comparisons of the proteins levels, Wilcoxon test was used (we get the same p-values as the ones represented in the boxplot). 
wilcox.test(data2$SNCA[data2$Diagnosis=="sMCI"], data2$SNCA[data2$Diagnosis=="cMCI"])

# The influence of the covariates was evaluated using the ANCOVA test
ancova <- aov(SNCA ~ Age + Gender + Diagnosis, data=data2)
summary(ancova)


