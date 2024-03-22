# Load libraries ----
library(ggVennDiagram)
library(ggplot2)
library(stringr)
library(readxl)

# Get top 20 variables per algorithm per data modality ----
path <- ""
sheets <- excel_sheets(path = path)
f <- list()

for (s in sheets) {
  file <- read_excel("supplementary_table_6.xlsx", sheet=s)
  f[[s]] <- file$variable
}


df <- read_excel("supplementary_table_6.xlsx", sheet = "MCI_varImp")
mci <- list()
mci[["LR"]] <- df$LR
mci[["RF"]] <- df$RF
mci[["SVM"]] <- df$SVM
mci[["MLP"]] <- df$MLP


# Venn Diagram ----

## Metabolites
mvip <- list(LR=f$metabolites_LR_varImp, RF=f$metabolites_RF_varImp, 
             SVM=f$metabolites_SVM_varImp, MLP=f$metabolites_MLP_varImp)

png(filename=paste0(figures_dir, "mvip.png"), width=8, height=8, unit="cm", res=500)
ggVennDiagram(mvip, label="count", set_size = 4, label_size=4) + 
  scale_fill_distiller(palette="Blues", direction=1) + 
  theme(legend.position="bottom") 
dev.off()

## Proteins
pvip <- list(LR=f$proteins_LR_varImp, RF=f$proteins_RF_varImp, 
             SVM=f$proteins_SVM_varImp, MLP=f$proteins_MLP_varImp)

png(filename=paste0(figures_dir, "pvip.png"), width=8, height=8, unit="cm", res=500)
ggVennDiagram(pvip, label="count", set_size = 4, label_size=4) + 
  scale_fill_distiller(palette="Greens", direction=1) + 
  theme(legend.position="bottom")
dev.off()

## MCI
mcivip <- list(LR=df$LR, RF=df$RF, 
               SVM=df$SVM, MLP=df$MLP)

png(filename=paste0(figures_dir, "mcivip.png"), width=8, height=8, unit="cm", res=500)
ggVennDiagram(mcivip, label="count", set_size = 4, label_size=4) + 
  scale_fill_distiller(palette="Reds", direction=1) + 
  theme(legend.position="bottom")
dev.off()
