# SCRIPT PARA CLUSTERING



# Importing dataset ----
library(dplyr)
df <- readRDS("data/merge_moderated_dataset_completecases.rds")


# Selecao das variaveis de interesse
df_clustering <- df %>% select(AbusEmocional, AbusFisico, AbusSexual, NegligEmocional, NegligFisica)

# Transforma as variaveis em escala z
df_clustering <- scale(df_clustering)

# Carrega os packages
library(plyr) ## pacote para dar nome aos fatores ###
library(cluster) # for gower similarity and pam
library(Rtsne) # for t-SNE plot
library(ggplot2) # for visualization

library(factoextra)
library(fpc)
library(NbClust)

## calculate Gower distance ##
gower_dist <- daisy(df_clustering, metric = "gower")
summary(gower_dist)

## criar a matrix ##
gower_mat <- as.matrix(gower_dist)

# Output most similar pair
df_clustering[which(gower_mat == min(gower_mat[gower_mat != min(gower_mat)]), arr.ind = TRUE)[1, ], ]

# Output most dissimilar pair
df_clustering[ which(gower_mat == max(gower_mat[gower_mat != max(gower_mat)]), arr.ind = TRUE)[1, ], ]

# Clustering methods
# PAM seleciona automaticamente o melhor número de clusters
# Calculate silhouette width for many k using PAM
# https://discuss.analyticsvidhya.com/t/clustering-technique-for-mixed-numeric-and-categorical-variables/6753/20 

library(fpc)

pamk.best <- pamk(gower_dist, krange = 1:8, usepam = T)
cat("number of clusters estimated by optimum average silhouette width:", pamk.best$nc, "\n")

# Graficos dos resultados da clusterizacao
fviz_nbclust(df_clustering, pam, method = "silhouette") +
  labs(subtitle = "silhouette method")

fviz_nbclust(df_clustering, pam, method = "wss") + labs(subtitle = "Elbow method")
fviz_nbclust(df_clustering, pam, method = "gap_stat", nboot = 50)

#http://www.sthda.com/english/wiki/print.php?id=236
fviz_silhouette(silhouette(pam(gower_dist, pamk.best$nc)))

# Plot sihouette width (higher is better)
#sil_width <- c(2:5)
#plot(2:5, sil_width, xlab = "Number of clusters", ylab = "Silhouette Width") ## 2:11 significa numeo de coluna ##
#lines(2:5, sil_width)

# Outro método de clustering

library(NbClust) #http://www.sthda.com/english/wiki/print.php?id=239 
nb <- NbClust(df_clustering, distance = "euclidean", min.nc = 2, max.nc = 10, method = "complete", index ="all")

##### ANÁLISE DESCRITIVA DOS CLUSTERS #####
colnames(df)

pam_fit <- pam(gower_dist, diss = TRUE, k = 2)
pam_results <- df %>% dplyr::select(HAM_total, CORE_total, MINIA11, Sexo, Cor_branco, AbusEmocional, AbusFisico, AbusSexual, NegligEmocional, Irritability) %>% 
    mutate(cluster = pam_fit$clustering) %>% group_by(cluster) %>% do(the_summary = summary(.))
pam_results$the_summary

# Medoids
df_clustering[pam_fit$medoids, ]

cluster.output <- data.frame(df_clustering, "cluster" = pam_fit$clustering)
cluster.output$cluster <- as.factor(cluster.output$cluster)

### GRAFICOS ####
library(Rtsne)
library(RColorBrewer)

tsne_obj <- Rtsne(gower_dist, is_distance = TRUE)

tsne_data <- tsne_obj$Y %>% data.frame() %>% setNames(c("X", "Y")) %>% mutate(cluster = as.factor(c(trainSparse_filtered$final_decision, testSparse_filtered$final_decision)))

# Mudando o nome dos clusters
library(plyr)
#tsne_data$cluster <- revalue(tsne_data$cluster, c("1"="Cluster 1", "2"="Cluster 2")) # rename levels of factor

baseline_graph <- ggplot(aes(x = X, y = Y), data = tsne_data) + geom_point(aes(color = cluster)) + 
  labs(x="Dimension 1", y="Dimension 2") + xlim(-10, 10) + ylim(-10, 10) + theme_classic() + 
  scale_color_manual(values = c("gray", "blue"))

dev.off()
baseline_graph

# Internal Validation
# https://www.datanovia.com/en/lessons/cluster-validation-statistics-must-know-methods/

library(fpc)

# Compute cluster stats
cluster <- as.numeric(cluster.output$cluster)
clust_stats <- cluster.stats(d = dist(cluster.output), 
                             cluster, pam_fit$clustering)
# Corrected Rand index
clust_stats$corrected.rand
clust_stats$pearsongamma
clust_stats$dunn2

# Diferencas entre clusters por outras variaveis
cluster.output2 <- data.frame(df, "cluster" = cluster.output$cluster)

# variaveis numericas do banco
summ_vars_num <- c("MINIA11", "Idade", "CORE_total", "HAM_total", "ham_suicide", "AbusEmocional", "AbusFisico", "AbusSexual", "NegligEmocional", "NegligFisica")

# variaveis categoricas
summ_vars_bin <- c("Sexo", "Cor_branco", "Drogas_atual", "Estado_conjugal", "MINIC6")

media_cluster <-cluster.output2 %>% select(summ_vars_num, cluster) %>% group_by(cluster) %>% summarise_at(summ_vars_num, funs(mean(., na.rm=TRUE))) 
media_cluster

sd_cluster <-cluster.output2 %>% select(summ_vars_num, cluster) %>% group_by(cluster) %>% summarise_at(summ_vars_num, funs(sd(., na.rm=TRUE))) 
sd_cluster

# Quais sao as diferencas em ideacao suicida (bdi9) entre os clusters?
bdi9 <- as.factor(cluster.output2$BDI9)
levels(bdi9) <- c("No", "Yes", "Yes", "Yes")

tb <- table(bdi9, cluster.output2$cluster)
tb

soma <- apply(tb, 2, sum)
tb[2, ]/soma

chisq.test(tb)

table(cluster.output2$Estado_conjugal, cluster.output2$cluster)

# Quais sao as diferencas de tentativa de suicídio entre os clusters?
ts <- as.factor(cluster.output2$MINIC6)
levels(ts) <- c("No", "Yes")

tb <- table(ts, cluster.output2$cluster)
tb

soma <- apply(tb, 2, sum)
tb[2, ]/soma

chisq.test(tb)



#### FAZER CLUSTERS COM CONTROLES COLORIDOS 
#https://cran.r-project.org/web/packages/dendextend/vignettes/Cluster_Analysis.html
