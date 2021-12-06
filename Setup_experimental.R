## Script name: Análise do TCC CECDADOS
##
## Purpose of script: Aplicação de ML na classificação de crescimento de fungos em placas de petri. 
##
## Author: Edgar de Souza Vismara
##         Rafael Gomes Mantovani
##
## Date Created: 2021-12-06
##
## Copyright (c) Edgar de Souza Vismara, 2021
## Email: edgarvismara@utfpr.edu.br´
##
## ---------------------------
##
## Notes: Esta versão inclui apenas metodologia experimental. 
##        Desta forma, não inclui os códigos da analise das predições dos modelos treinados
##   
##
## ---------------------------

## load up the packages 

library(mlr3)
library(mlr3learners)
library(mlr3viz)
library(tidyverse)
library(caret)
library(corrplot)
library(agricolae)
library(scales)
library(factoextra)
library(agricolae)

## ---------------------------
## 1. Leitura dos dados
## ------------------------------------------------------------------------------------------

set.seed(123)

# Matriz de caracteríticas

features = read_csv("features_prepro.csv")

# Vetor de rótulos

labels = read_csv("rotulos.csv")

labels = labels  %>% 
  mutate(rotulo = as.factor(recode(rotulo, c1 = 1, c2 = 1, c3 = 2, c4 = 3)))

 
## 1.1 Frequência de imagens de cada rótulo
## ------------------------------------------------------------------------------------------
df = data.frame(class = factor(c("Inicial","Intermediário","Avançado"),levels=c("Inicial","Intermediário","Avançado")), freq = as.numeric(table(labels$rotulo)))

#Aqui o gráfico de barras, usando ggplot2
g = ggplot(data = df, mapping = aes(x = class, y = freq, fill = class, colour = class))
g = g + geom_bar(position = 'dodge', stat='identity') + 
  geom_text(aes(label = freq), position=position_dodge(width=0.9), vjust=-0.25) + 
  labs(x = "Classes de Crescimento", y = "Frequência de exemplos") + theme_bw() + theme(legend.position="none") 

## 2. Pré-processamento dos dados.

### 2.1 Removendo os Nas: 
## ------------------------------------------------------------------------------------------
tmp <- features[ , colSums(is.na(features)) < nrow(features)]
features <- tmp[complete.cases(tmp), ] 
dim(features)

### 2.2 Removendo features constantes (variância = 0):
## ------------------------------------------------------------------------------------------
#Criando uma função pára isso
rm0var = function(df, threshold = 0){
  df = df[, sapply(df, function(x) {var(x)}) > threshold]
  return(df)
}

## ------------------------------------------------------------------------------------------
cat("Removendo features constantes ...\n")
cat("- Antes:  ", ncol(features)-1, "features \n")

X = rm0var(features[,-1], threshold = 0)
X = cbind(features[,1],X)

cat("- Depois: ", ncol(X)-1, "features \n")


### 3 Criando os 4 datasets:
## ------------------------------------------------------------------------------------------
#dataset 1 (só features de cores)
X_cor = X %>%  select(Name_file, starts_with("cor")| starts_with("std_")|starts_with("mean")|starts_with("entropy"),-contains("hist"))
#dataset 2 (só features de histogramas)
X_esta = X %>%  select(Name_file, starts_with("skew_hist")| starts_with("kurt_hist") | starts_with("std_hist"))
#dataset 3 (features restantes)
X_rest = X %>%  select(Name_file, -names(X_cor) & -names(X_esta))
#dataset 4 (todas as features)
X_all = X

# Fazendo um plot para ver numero de features por dataset

data  = factor(c("Canais de cor", "Histogramas", "Demais características", "Completo"),levels = c("Canais de cor", "Histogramas", "Demais características", "Completo") )
value = c(ncol(X_cor)-1, ncol(X_esta)-1, ncol(X_rest)-1, ncol(X_all)-1) 
df = data.frame(data = data, value = value)
head(df)

# Aqui o gráfico de barras, usando ggplot2
g = ggplot(data = df, mapping = aes(x = data, y = value, fill = data, colour = data))
g = g + geom_bar(position = 'dodge', stat='identity') + 
  geom_text(aes(label = value), position=position_dodge(width=0.9), vjust=-0.25) + 
  labs(x = "Conjuntos de dados", y = "Número de características") + theme_bw() + theme(legend.position="none")

##2.3 Remoção das features muito correlacionadas 

#Para cada dataset quatro critérios de corte (> 0.8, 0.85, 0.9, 0.95): 

#Características Cores:
## ------------------------------------------------------------------------------------------
tmp_cor = cor(X_cor[,-1])

hc80_cor = findCorrelation(tmp_cor, cutoff=0.8,  names = T, exact = TRUE) 
hc85_cor = findCorrelation(tmp_cor, cutoff=0.85, names = T, exact = TRUE) 
hc90_cor = findCorrelation(tmp_cor, cutoff=0.9,  names = T, exact = TRUE) 
hc95_cor = findCorrelation(tmp_cor, cutoff=0.95, names = T, exact = TRUE) 

n_out_cor = lengths(list(hc80_cor,hc85_cor,hc90_cor,hc95_cor))
graf_cor = data.frame('n_out' = n_out_cor, 'cutoff' = factor(c(0.8,0.85,0.9,0.95)))
head(graf_cor)

#Características Histogramas:
## ------------------------------------------------------------------------------------------
tmp_esta = cor(X_esta[,-1])

hc80_esta = findCorrelation(tmp_esta, cutoff=0.8,  names = T, exact = TRUE) 
hc85_esta = findCorrelation(tmp_esta, cutoff=0.85, names = T, exact = TRUE) 
hc90_esta = findCorrelation(tmp_esta, cutoff=0.9,  names = T, exact = TRUE) 
hc95_esta = findCorrelation(tmp_esta, cutoff=0.95, names = T, exact = TRUE) 

n_out_esta = lengths(list(hc80_esta,hc85_esta,hc90_esta,hc95_esta))
graf_esta = data.frame('n_out' = n_out_esta, 'cutoff' = factor(c(0.8,0.85,0.9,0.95)))
head(graf_esta)

#Características restantes:
## ------------------------------------------------------------------------------------------
tmp_rest = cor(X_rest[,-1])

hc80_rest = findCorrelation(tmp_rest, cutoff=0.8, names = T, exact = TRUE) 
hc85_rest= findCorrelation(tmp_rest, cutoff=0.85, names = T, exact = TRUE) 
hc90_rest= findCorrelation(tmp_rest, cutoff=0.9, names = T, exact = TRUE) 
hc95_rest= findCorrelation(tmp_rest, cutoff=0.95, names = T, exact = TRUE) 

n_out_rest = lengths(list(hc80_rest,hc85_rest,hc90_rest,hc95_rest))
graf_rest = data.frame('n_out' = n_out_rest, 'cutoff' = factor(c(0.8,0.85,0.9,0.95)))
head(graf_rest)

#Todas as características:
## ------------------------------------------------------------------------------------------
tmp = cor(X_all[,-1])

hc80 = findCorrelation(tmp, cutoff=0.8,  names = T, exact = TRUE) 
hc85 = findCorrelation(tmp, cutoff=0.85, names = T, exact = TRUE) 
hc90 = findCorrelation(tmp, cutoff=0.9,  names = T, exact = TRUE) 
hc95 = findCorrelation(tmp, cutoff=0.95, names = T, exact = TRUE) 
n_out = lengths(list(hc80,hc85,hc90,hc95))
graf_all = data.frame('n_out' = n_out, 'cutoff' = factor(c(0.8,0.85,0.9,0.95)))
head(graf_all)

## Plot agregado com as informações dos 4 datasets
graf_cor$set  = "Canais de cor"
graf_esta$set = "Histogramas"
graf_rest$set = "Demais características" 
graf_all$set  = "Completo"

# merge nos 4 datasets 
full.df = rbind(graf_cor, graf_esta, graf_rest, graf_all)
full.df

## ------------------------------------------------------------------------------------------
# Plotar a informação conjunta, com 4 curvas
library(ggrepel)
g5 = ggplot(data=full.df, aes(x=cutoff, y=n_out, group = set, colour=set, linetype=set, shape=set, label=n_out))
g5 = g5 + geom_line() + geom_point()  
g5 = g5 + geom_label_repel( nudge_x = 0.1, vjust=+0.25, show.legend = F) + 
  labs(x = "Critério de corte (correlação)", 
       y = "Características removidas (n°)", 
       colour='Conjunto de dados', 
       linetype = 'Conjunto de dados',
       shape = 'Conjunto de dados') + 
  theme(legend.title = element_text(size=10), #change legend title font size
        legend.text = element_text(size=10)) 


## ------------------------------------------------------------------------------------------
#Optei por usar como critério 0.85 de correlação como critério de corte.
#Realizando o corte pára cada dataset
X_cor = X_cor[, ! names(X_cor) %in% c(hc85_cor)]
head(X_cor)

X_esta = X_esta[, ! names(X_esta) %in% c(hc85_esta)]
head(X_esta)

X_rest = X_rest[, ! names(X_rest) %in% c(hc85_rest)]
head(X_rest)

X_all = X_all[, ! names(X_all) %in% c(hc85)]
head(X_all)

## ------------------------------------------------------------------------------------------
# Criando o dataset agregado, unindo as features sem correlação
X_all2 = cbind(X_cor, X_esta[,-1], X_rest[,-1]) 

## Examinando as correlações existentes nos dois datasets com mais features

corrplot(cor(X_all[,-1]), method = "circle", type= "lower", diag = FALSE, order = 'hclust', tl.col = 'black', 
         cl.ratio = 0.1, tl.srt = 45)

corrplot(cor(X_all2[,-1]), method = "circle", type= "lower", diag = FALSE, order = 'hclust', tl.col = 'black', 
         cl.ratio = 0.1, tl.srt = 45)



### 2.4 Criando os datasets para treinamento (agregando rotulos)
## ------------------------------------------------------------------------------------------
df_all <- labels %>% inner_join(X_all,  by = "Name_file") %>% select(-one_of('Name_file'))
df_all2 <- labels %>% inner_join(X_all2,  by = "Name_file") %>% select(-one_of('Name_file'))
df_cor <- labels %>% inner_join(X_cor,  by = "Name_file") %>% select(-one_of('Name_file'))
df_esta <- labels %>% inner_join(X_esta,  by = "Name_file") %>% select(-one_of('Name_file'))
df_rest <- labels %>% inner_join(X_rest,  by = "Name_file") %>% select(-one_of('Name_file'))

## Normalizando os dados
df_all_resc = df_all %>% mutate_each_(list(~rescale(., to = c(-1, 1)) %>% as.vector),
                                      vars = names(df_all)[-1])
df_all2_resc = df_all2 %>% mutate_each_(list(~rescale(., to = c(-1, 1)) %>% as.vector),
                                        vars = names(df_all2)[-1])
df_cor_resc = df_cor %>% mutate_each_(list(~rescale(., to = c(-1, 1)) %>% as.vector),
                                      vars = names(df_cor)[-1])
df_esta_resc = df_esta %>% mutate_each_(list(~rescale(., to = c(-1, 1)) %>% as.vector),
                                        vars = names(df_esta)[-1])
df_rest_resc = df_rest %>% mutate_each_(list(~rescale(., to = c(-1, 1)) %>% as.vector),
                                        vars = names(df_rest)[-1])
## ------------------------------------------------------------------------------------------
## 3. Treinamento
## ------------------------------------------------------------------------------------------
# 1) criar as tarefas (classificação)

task_all  = TaskClassif$new(id = "Completo",  backend = df_all_resc,  target = "rotulo")
task_all$col_roles$stratum = task_all$target_names
task_cor  = TaskClassif$new(id = "Canais de cor",  backend = df_cor_resc,  target = "rotulo")
task_cor$col_roles$stratum = task_cor$target_names
task_esta = TaskClassif$new(id = "Histogramas", backend = df_esta_resc, target = "rotulo")
task_esta$col_roles$stratum = task_esta$target_names
task_rest = TaskClassif$new(id = "Demais características", backend = df_rest_resc, target = "rotulo")
task_rest$col_roles$stratum = task_rest$target_names
task_all2 = TaskClassif$new(id = "Agregado", backend = df_all2_resc, target = "rotulo")
task_all2$col_roles$stratum = task_all2$target_names

## ------------------------------------------------------------------------------------------
tasks <- list(task_cor, task_all, task_esta, task_rest, task_all2)
tasks

## ------------------------------------------------------------------------------------------
# 2) instanciar os algoritmos

clr_bas_1 = lrn("classif.featureless", method="mode", id = "Majoritária")  
clr_bas_2 = lrn("classif.featureless", method="sample", id = "Aleatória")  
clf_dt  = lrn("classif.rpart", id = "Árvore de decisão") 
clf_knn = lrn("classif.kknn", id = "K-nn")
clf_mn  = lrn("classif.multinom", id = "Multinomial")   
clf_rf  = lrn("classif.ranger", id = "Random Forest")         
clf_svm = lrn("classif.svm", id = "SVM")            
clf_mlp = lrn("classif.nnet", id = "MLP")          
clf_nb  = lrn("classif.naive_bayes", id = "Naive Bayes")    

#Agrupando todos algoritmos
learners = list(clr_bas_1, clr_bas_2, clf_dt, clf_knn, clf_mn, clf_nb, clf_rf, clf_svm, clf_mlp)
print(learners)

## ------------------------------------------------------------------------------------------
# 3) Escolher medidas de desempenho (classificação)
# medida que leve em consideração desbalanceamento dos dados
# BACC = Balanced Accuracy per Class 
measure = msr("classif.bacc")


## ------------------------------------------------------------------------------------------
# Definir um resampling (CV)
# Garantir a amostragem estratificada, pq o dataset é desbalanceado

cv = rsmp("repeated_cv", folds = 10, repeats = 10)
print(cv)

## ------------------------------------------------------------------------------------------
# Executar run (tasks, algorithms) -> benchmark
design = benchmark_grid(tasks = tasks, learners = learners, resamplings = cv) #definindo  
print(design)

res = benchmark(design = design)#treinando
print(res)


## ------------------------------------------------------------------------------------------
#Extraindo as performances finais
results = res$score(measure)
head(results)

## ------------------------------------------------------------------------------------------
#Agregando resultados por tarefa x algoritmo
resagg = res$aggregate(measure)

#ordenando os resultados pela performance
resagg[order(resagg$classif.bacc, decreasing = TRUE),]

##-------------------------------------------------------------------------------------------
# Visulaização de resultados
# Customizando a saida (plots proprios)
results = res$score(measure)
colnames(results) = c("unhash", "nr", "task", "task_id", "learner", "learner_id", 
                      "resampling", "resampling_id", "iteration", "prediction", "baac")
head(results)

#customizando a visualização

task_id.labs <- c("df_all_resc", "df_all2_resc", "df_cor_resc", "df_esta_resc", "df_rest_resc")
names(task_id.labs) <- c("Completo", "Agregado", "Canais de Cor", "Histograma", "Demais características")

task_id.labs <- c(
  `Completo` = "Completo",
  `Agregado` = "Agregado",
  `Canais de Cor` = "Canais de Cor",
  `Histograma` = "Histograma",
  `Demais características` = "Demais caracteristicas"
  
)

# Graficando os resultados
gf = ggplot(data = results, mapping = aes(x = learner_id, y = baac, group = learner_id))
gf = gf + geom_boxplot() + facet_grid(.~task_id, labeller = labeller(task_id.labs))
# ampliando fonte do eixo x e y
gf = gf + theme(axis.text.x = element_text(angle = 45, hjust = 1))
gf = gf + theme(axis.text = element_text(size=14), axis.title=element_text(size=14,face="bold"))
gf = gf + labs(x="Algorítimos", y="Acurácia balanceada") + theme(strip.text.x = element_text(size = 14))

##------------------------------------------------------------------------------------------
## Análise estatística 
##------------------------------------------------------------------------------------------

## Extraindo os melhores
limite = bests[16,]$classif.bacc
top = resagg[resagg$classif.bacc>=limite,]$nr

## criando o dataset para o teste
analise = results %>% select(nr, task_id, learner_id, iteration, baac)  %>%
  filter(nr %in% top)
### Teste de Kruskal wallis
library(agricolae)
(comparison<-with(analise, kruskal(baac, nr,p.adj="bonferroni", group=TRUE, main="teste de hipótese")))