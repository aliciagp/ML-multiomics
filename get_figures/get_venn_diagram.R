# Load libraries
library(ggVennDiagram)
library(gridExtra)

# Load the models
ridge <- readRDS(paste0(models_dir, "ridge_MCI.rds"))
rf <- readRDS(paste0(models_dir, "rf_MCI.rds"))
svm <- readRDS(paste0(models_dir, "svm_MCI.rds"))
mlp <- readRDS(paste0(models_dir, "mlp_MCI.rds"))

# Load the top 20 most relevant features per algorithm
MC_top_prot <- read_excel(paste0(results_path, "selected_variables_multiclass.xlsx"), sheet="Proteins")
MC_top_met <- read_excel(paste0(results_path, "selected_variables_multiclass.xlsx"), sheet="Metabolites")
MCI_top <- read_excel(paste0(results_path, "selected_variables_MCI.xlsx"))


# Plot the overlap of the top 20 most relevant features per scenario
## Multiclass models with proteins and clinical covariates
MC_top_prot <- list(LR=MC_top_prot$LR, RF=MC_top_prot$RF, SVM=MC_top_prot$SVM, MLP=MC_top_prot$MLP)

p1 <- ggVennDiagram(MC_top_prot, label_alpha = 0, label_color="black", label="count", edge_size=0.2, set_size=4, label_size=4) +
                    theme(legend.position="bottom", 
                          plot.title = element_text(size=12, hjust=0.5, vjust=5), 
                          legend.key.size = unit(0.6, 'cm'),
                          legend.text = element_text(size=10)) +
                    labs(title="Multiclass models\ntop proteins") +
                    scale_fill_continuous(low = "white", high = "lightblue3") +
                    scale_colour_manual(values=rep("grey27", 4)) +
                    scale_x_continuous(expand = expansion(mult = .1))

## Multiclass models with metabolites and clinical covariates
MC_top_met <- list(LR=MC_top_met$LR, RF=MC_top_met$RF, SVM=MC_top_met$SVM, MLP=MC_top_met$MLP)

p2 <- ggVennDiagram(MC_top_met, label_alpha = 0, label_color="black", label="count", edge_size=0.1, set_size=4, label_size=4) +
                    theme(legend.position="bottom", 
                          plot.title = element_text(size=12, hjust=0.5, vjust=5),
                          legend.key.size = unit(0.6, 'cm'),
                          legend.text = element_text(size=10)) +
                    labs(title="Multiclass models\ntop metabolites") +
                    scale_fill_continuous(low = "white", high = "lightcoral") +
                    scale_colour_manual(values=rep("grey27", 4),) +
                    scale_x_continuous(expand = expansion(mult = .1)) 

# MCI conversion models with both proteins, metabolites and clinical covariates
MCI_top <- list(LR=MCI_top$LR, RF=MCI_top$RF, SVM=MCI_top$SVM, MLP=MCI_top$MLP)

p3 <- ggVennDiagram(MCI_top, label_alpha = 0, label_color="black", label="count", edge_size=0.1, set_size=4, label_size=4) +  
                    theme(legend.position="bottom", 
                          plot.title = element_text(size=12, hjust=0.5, vjust=5),
                          legend.key.size = unit(0.6, 'cm'),
                          legend.text = element_text(size=10)) +
                    labs(title="MCI converters models\ntop variables") + 
                    scale_fill_continuous(low = "white", high = "palegreen3") +
                    scale_colour_manual(values=rep("grey27", 4)) +
                    scale_x_continuous(expand = expansion(mult = .1))


## Arrange the plots
grid.arrange(p1, p2, p3, ncol=3)
