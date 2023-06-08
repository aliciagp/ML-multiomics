# Load libraries
library(ggplot2)
library(ggpubr)

# Load metabolomics preprocessed data
matrix <- readRDS(paste0(data_path, "metabolomics_preprocessed.rds"))

# Define four levels of AD pathology
matrix2 <- matrix
matrix2$Diagnosis[matrix2$MCI_Convert==1] <- "cMCI"
matrix2$Diagnosis[matrix2$MCI_Convert==0] <- "sMCI"
matrix2 <- matrix2[-which(matrix2$Diagnosis=="MCI"), ]
matrix2$Diagnosis <- factor(matrix2$Diagnosis, levels=c("NL", "sMCI", "cMCI", "AD"))

# Create the boxplot
ggplot(matrix2, aes(x=Diagnosis, y=oleamide, color=Diagnosis)) +
       geom_boxplot(outlier.size = 0.8, size=0.8) + 
       theme_classic() + 
       ylab(expression(paste("oleamide levels ", "(", log[10], ")"))) +
       theme(text=element_text(size=18), 
            legend.position="bottom", 
            legend.key.size=unit(0.6, "cm"),
            legend.title = element_text(size=10),
            axis.title=element_text(size=12)) +
       geom_jitter(size=0.8, shape=16, position=position_jitter(0.2)) + 
       stat_compare_means(label="p.signif", comparisons = list(c("NL", "sMCI"), 
                                                              c("sMCI", "cMCI"), 
                                                          c("cMCI", "AD"), 
                                                          c("NL", "AD")), 
                     label.y = c(2.2, 2.7, 3.2, 4.2),
                     method="wilcox.test") + 
       scale_color_brewer(palette="Spectral", direction=-1)  +
       scale_y_continuous(breaks=seq(-4,4.5))

# For pairwise comparisons of oleamide levels, Wilcoxon test was used (we get the same p-values as the ones represented in the boxplot). 
wilcox.test(matrix2$oleamide[matrix2$Diagnosis=="NL"], matrix2$oleamide[matrix2$Diagnosis=="sMCI"])

# The influence of the covariates was evaluated using the ANCOVA test
ancova <- aov(oleamide ~ Age + Gender + Diagnosis, data=matrix2)
summary(ancova)
