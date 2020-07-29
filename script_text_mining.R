library(readxl)

# Importa o dataset com os artigos rotulados para fazer o pre processamento
random_data <- read_excel("data/random_data_primary_consensus (2).xlsx")

ndup_data <- read_excel("data/data_no_duplicate.xlsx")
ndup_data <- ndup_data[!is.na(ndup_data$Abstract), ]

head(random_data)
dim(random_data)

head(ndup_data)
dim(ndup_data)
View(ndup_data)

# Quantos artigos estao no dataset ndup_data?
length(random_data$ID...1 %in% ndup_data$ID)

# Como todos os artigos estao em ndup_data, nao sera necessario fazer um merge dos datasets.
data1 <- ndup_data[, c("Title", "Abstract")]

#coombining Abstract and Title
data1$Text <- paste(data1$Title, data1$Abstract)
head(data1)

data <- data1[, "Text"]

# Preparing the data
data <- as.data.frame(data)

#Text mining packages install.packages('tm', dependencies=TRUE, repos='http://cran.rstudio.com/')
library(tm)
library(SnowballC)
library(caret)

# Step 1 - Create a corpus text
corpus = Corpus(VectorSource(data$Text))
corpus[[1]][1]

## Step 2 - Conversion to Lowercase
corpus = tm_map(corpus, PlainTextDocument)
corpus = tm_map(corpus, tolower)
corpus[[1]][1]

#Step 3 - Removing Punctuation
corpus = tm_map(corpus, removePunctuation)
corpus[[1]][1]

#Step 4 - Removing Stopwords and other words
corpus = tm_map(corpus, removeWords, c("objective","background", "introduction", "purpose", "method", "result", "conclusion", "limitation", 
                                   stopwords("english")))
corpus[[1]][1]  

# Step 5 - Stemming: reducing the number of inflectional forms of words
corpus = tm_map(corpus, stemDocument)
corpus[[1]][1]  

corpus = tm_map(corpus, removeNumbers)
corpus[[1]][1]

# Step 6 - Create Document Term Matrix
frequencies = DocumentTermMatrix(corpus)
sparse = removeSparseTerms(frequencies, 0.995) #remove sparse terms
tSparse = as.data.frame(as.matrix(sparse)) #convert into data frame
colnames(tSparse) = make.names(colnames(tSparse)) #all the variable names R-friendly

# Colunas com palavras estranhas
colnames(tSparse)[1729:1731]

# Remocao dessas colunas
tSparse <- tSparse[, -c(1729:1731)]
colnames(tSparse)[1700:ncol(tSparse)]

# Selecao dos artigos rotulados em tSparse
labeled <- ndup_data$ID %in% random_data$ID...1 
length(labeled)

papers_attributes <- ndup_data[!labeled, ]
dim(papers_attributes)

tSparse_nolabeled <- tSparse[!labeled, ]
dim(tSparse_nolabeled)

tSparse_labeled <- tSparse[labeled, ]

ids <- random_data$ID...1
tSparse_labeled <- data.frame("final_decision" = random_data$Final_decision, tSparse_labeled)

tSparse_labeled[1:10, 1:10]

library(purrr)

prop.table(table(tSparse_labeled$final_decision))
table(tSparse_labeled$final_decision)

# Hold out
library(caret)

set.seed(123)
intrain <- createDataPartition(y = tSparse_labeled$final_decision, p= 0.8, list = FALSE)
trainSparse <-  tSparse_labeled[intrain,]
testSparse <- tSparse_labeled[-intrain,]
dim(trainSparse); dim(testSparse)

# Porcentagem do desfecho em cada banco
prop.table(table(trainSparse$final_decision))
prop.table(table(testSparse$final_decision))

table(trainSparse$final_decision)
table(testSparse$final_decision)

library(purrr)
prop_values <- map_dbl(trainSparse, function(x){prop.table(table(x))[1]})
sort(prop_values, decreasing = TRUE)

## doParallel
library(doParallel)
cl <- makeCluster(16)
registerDoParallel(cl)

#Training
trainSparse$final_decision = as.factor(trainSparse$final_decision)
testSparse$final_decision = as.factor(testSparse$final_decision)

levels(trainSparse$final_decision)
levels(trainSparse$final_decision) <- c("no", "yes")
levels(testSparse$final_decision)
levels(testSparse$final_decision) <- c("no", "yes")

############ GLM ######
library(caret)
library(glmnet)

set.seed(108)
ctrl.EN <- trainControl(method="cv", number = 10, classProbs = TRUE,savePredictions = TRUE, summaryFunction = twoClassSummary)
model_imbalance <- train(final_decision ~., data = trainSparse, method = "glmnet", trControl = ctrl.EN, tuneLength = 10, metric = "ROC")

# up sampling
set.seed(108) 
ctrl.EN_up <- trainControl(method="cv", number = 10, classProbs = TRUE,savePredictions = TRUE, summaryFunction = twoClassSummary, sampling = "up")
up_outside <- train(final_decision ~., data = trainSparse, method = "glmnet", trControl = ctrl.EN_up, tuneLength = 10, metric = "ROC")

# Smote
library(DMwR)
set.seed(108) 
ctrl.EN_smote <- trainControl(method="cv", number = 10, classProbs = TRUE,savePredictions = TRUE, summaryFunction = twoClassSummary, sampling = "smote")
smote_outside <- train(final_decision ~., data = trainSparse, method = "glmnet", trControl = ctrl.EN_smote, tuneLength = 10, metric = "ROC")

###################################
#Rose #############################
library(ROSE)
set.seed(108) 
ctrl.EN_rose <- trainControl(method="cv", number = 10, classProbs = TRUE, savePredictions = TRUE, summaryFunction = twoClassSummary, sampling = "rose")
rose_outside <- train(final_decision ~., data = trainSparse, method = "glmnet", trControl = ctrl.EN_rose, tuneLength = 10, metric = "ROC")

plot(varImp(rose_outside))
varImp(rose_outside)

plot(rose_outside)
rose_outside$bestTune

final_predictions <- predict(rose_outside, testSparse, type = "raw")
final_predictions

cm <- confusionMatrix(final_predictions, testSparse$final_decision, positive = "yes")
cm

test_probs <- predict(rose_outside, testSparse, type = "prob")
test_probs <- as.vector(test_probs[, 2])

library(pROC)
rocobj <- roc(testSparse$final_decision, test_probs, ci=TRUE, of="auc", percent = FALSE)

#rocobj_sen2 <- roc(dt3Test2$traj, predictions2, ci=TRUE, of="se", percent = FALSE)

plot(rocobj, main="Confidence intervals", percent = FALSE, ci=TRUE, print.auc=TRUE)


predictPapers <- function(model, cutoff, features_df, papers_attributes){
  test_probs <- predict(model, features_df, type = "prob")
  test_probs <- as.vector(test_probs[, 2])
  include_prediction <- ifelse(test_probs >= cutoff, "yes", "no")
  result_df <- data.frame(papers_attributes, include_prediction)
  result_df
}

dim(tSparse_nolabeled)
dim(papers_attributes)

papers_predicted <- predictPapers(rose_outside, 0.5, tSparse_nolabeled, papers_attributes)
papers_predicted

table(papers_predicted$include_prediction)




source("src/func.R")
# Atualizar code performance vs cutoff
# Ver quais variaveis tem zero variancia no dataset


# FIM ----

#Down sampling
set.seed(108)
ctrl.EN_down <- trainControl(method="repeatedcv", repeats = 1000, number = 10, classProbs = TRUE,savePredictions = TRUE, summaryFunction = twoClassSummary, sampling = "down")
down_outside <- train(final_decision ~., data = trainSparse, method = "glmnet", trControl = ctrl.EN_down, tuneLength = 10, metric = "ROC")

#escolhendo o melhor modelo 
library(mlbench) 
library(caret)
result_model_EN <- resamples(list(IMBALANCE= model_imbalance, up=up_outside, rose=rose_outside, smote=smote_outside))
summary(result_model_EN) #ROSE FOI MELHOR
bwplot(result_model_EN) 
dotplot(result_model_EN)

# confusion matrix
library(e1071)
test_pred_rose_EN <- predict(rose_outside, newdata = testSparse[,-2054])
matrixConfusao_rose_EN <- confusionMatrix(test_pred_rose_EN, testSparse$final_decision, positive = "yes")
matrixConfusao_rose_EN
# plot ROC curve
library(caret)
library(pROC)
p_rose_EN <- predict (rose_outside, testSparse[,-2054])
p_prob_rose_EN <- predict (rose_outside, testSparse[,-2054], type = "prob")
confusionMatrix(p_rose_EN, testSparse$final_decision)
print(confusionMatrix(p_rose_EN, testSparse$final_decision, positive = "yes"))
r_rose_EN <- roc (testSparse$final_decision, p_prob_rose_EN[,"yes"])
plot(r_rose_EN)
r_rose_EN$auc


#### mudando limiar do EN####
library(SDMTools)

### threshold wave 4 ####
# 0 = No, 1 = Yes
obs <- testSparse$final_decision
levels(obs)
levels(obs) <- c("0", "1")
obs <- as.numeric(as.character(obs))
obs

# model prediction with probabilities
rf.predict <- predict(rose_outside, testSparse[, -2054], type = "prob")
predictions <- as.vector(rf.predict[, 2])
predictions

confusion_df <- data.frame(obs, predictions)
threshold_seq <- seq(0, 1, by = 0.01)

confMatrix <- function(i, obs, predictions, ...){
  require(caret)
  require(SDMTools)
  
  conf_matrix <- confusion.matrix(obs, predictions, threshold = i)
  cm_table <- as.table(conf_matrix)
  cm <- confusionMatrix(cm_table, positive = "1")
  p_acc <- cm$overall[6]
  acc <- cm$overall[1]
  null_acc <- cm$overall[5]
  ppv <- cm$byClass[3]
  npv <- cm$byClass[4]
  
  result <-  c("limiar" = i, cm$byClass[1], cm$byClass[2],
               "AB" = as.numeric(c((cm$byClass[1]+cm$byClass[2]) / 2)),
               acc, p_acc, null_acc, ppv, npv)
  result
}

matrixConfusao_rose_EN$byClass
library(purrr)
library(plyr)
confMatrix(0.5, obs, predictions)
result_list <- map(threshold_seq, confMatrix, obs = obs, predictions = predictions)
result_df <- ldply(result_list)
#result with each threshold
result_df


###### Random Forest #############
#DoParalelo
library(doParallel)
gc()
cl <- makePSOCKcluster(5)
registerDoParallel(cl)

library(caret)
library(randomForest)
set.seed(108)
sqrt(ncol(trainSparse)-1)

grid.rf = expand.grid(.mtry=c(10, 41, 70))
ctrl.rf = trainControl(method = "cv", number = 10, classProbs = TRUE, savePredictions = TRUE, summaryFunction = twoClassSummary)
rf_model <- train(final_decision ~ ., data = trainSparse, method = "rf", trControl = ctrl.rf, tuneGrid = grid.rf, tuneLength = 10, metric = "ROC")

#UP
set.seed(108)
ctrl.rf_RF_up = trainControl(method="cv", number = 10, classProbs = TRUE,savePredictions = TRUE, summaryFunction = twoClassSummary, sampling = "up")
rf_model_up <- train(final_decision ~ ., data = trainSparse, method = "rf", trControl = ctrl.rf_RF_up, tuneGrid= grid.rf, tuneLength = 10, metric = "ROC")

#Smote
set.seed(108)
library(DMwR)
ctrl.rf_RF_smote = trainControl(method="cv", number = 10, classProbs = TRUE,savePredictions = TRUE, summaryFunction = twoClassSummary, sampling = "smote")
rf_model_smote <- train(final_decision ~ ., data = trainSparse, method = "rf", trControl = ctrl.rf_RF_smote, tuneGrid= grid.rf, tuneLength = 10, metric = "ROC")

#ROSE
set.seed(108)
library(ROSE)
ctrl.rf_RF_rose = trainControl(method="cv", number = 10, classProbs = TRUE,savePredictions = TRUE, summaryFunction = twoClassSummary, sampling = "rose")
rf_model_rose <- train(final_decision ~ ., data = trainSparse, method = "rf", trControl = ctrl.rf_RF_rose, tuneGrid= grid.rf, tuneLength = 10, metric = "ROC")

#Down com 1000 repetições
set.seed(108)
ctrl.rf_RF_down = trainControl(method="repeatedcv", number = 10, repeats = 1000, classProbs = TRUE,savePredictions = TRUE, summaryFunction = twoClassSummary, sampling = "down")
rf_model_down <- train(final_decision ~ ., data = trainSparse, method = "rf", trControl = ctrl.rf_RF_rose, tuneGrid= grid.rf, tuneLength = 10, metric = "ROC")

# escolhendo o melhor modelo 
library(mlbench)
library(caret)
result_model2 <- resamples(list(up=rf_model_up, rose=rf_model_rose, smote=rf_model_smote, down=rf_model_down))
summary(result_model2)
bwplot(result_model2) #the best model UP
dotplot(result_model2)

# confusion matrix
library(e1071)
test_pred_up_rf <- predict(rf_model_up, newdata = testSparse[,-2054])
matrixConfusao_up_rf <- confusionMatrix(test_pred_up_rf, testSparse$final_decision, positive = "yes")
matrixConfusao_up_rf
# plot ROC curve
library(caret)
library(pROC)
p_up_rf <- predict (rf_model_up, testSparse[,-2054])
p_prob_up_rf <- predict (rf_model_up, testSparse[,-2054], type = "prob")
confusionMatrix(p_up_rf, testSparse$final_decision)
print(confusionMatrix(p_up_rf, testSparse$final_decision, positive = "yes"))
r_up_rf <- roc (testSparse$final_decision, p_prob_up_rf[,"yes"])
plot(r_up_rf)
r_up_rf$auc

###### Mudando limiar 

library(SDMTools)

### threshold wave 4 ####
# 0 = No, 1 = Yes
obs <- testSparse$final_decision
levels(obs)
levels(obs) <- c("0", "1")
obs <- as.numeric(as.character(obs))
obs

# model prediction with probabilities
rf.predict <- predict(rf_model_up, testSparse[, -2018], type = "prob")
predictions <- as.vector(rf.predict[, 2])
predictions

confusion_df <- data.frame(obs, predictions)
threshold_seq <- seq(0, 1, by = 0.01)

confMatrix <- function(i, obs, predictions, ...){
  require(caret)
  require(SDMTools)
  
  conf_matrix <- confusion.matrix(obs, predictions, threshold = i)
  cm_table <- as.table(conf_matrix)
  cm <- confusionMatrix(cm_table, positive = "1")
  p_acc <- cm$overall[6]
  acc <- cm$overall[1]
  null_acc <- cm$overall[5]
  ppv <- cm$byClass[3]
  npv <- cm$byClass[4]
  
  result <-  c("limiar" = i, cm$byClass[1], cm$byClass[2],
               "AB" = as.numeric(c((cm$byClass[1]+cm$byClass[2]) / 2)),
               acc, p_acc, null_acc, ppv, npv)
  result
}

matrixConfusao_up_rf$byClass
library(purrr)
library(plyr)
confMatrix(0.5, obs, predictions)
result_list <- map(threshold_seq, confMatrix, obs = obs, predictions = predictions)
result_df <- ldply(result_list)
#result with each threshold
result_df



########AUTO ML#####
#DoParalelo
library(doParallel)
gc()

cl <- makePSOCKcluster(3)
registerDoParallel(cl)

library(purrr)
library(h2o)

library(h2o)
h2o.init()


# Carregue a funcao abaixo
doAutoML <- function(trainSparse, testSparse, TEXT_MINING){
  # Identify predictors and response
  dtTrain <- trainSparse
  dtTest <- testSparse
  
  # Defina o nome da variável resposta
  y <- "final_decision"
  x <- setdiff(names(dtTrain), y)
  
  
  # Run AutoML for 20 base models (limited to 1 hour max runtime by default)
  
  train_h2o <- as.h2o(dtTrain)
  test_h2o <- as.h2o(dtTest)
  
  aml <- h2o.automl(x = x, y = y,
                    training_frame = train_h2o,
                    max_models = 20,
                    nfolds = 10,
                    keep_cross_validation_fold_assignment = TRUE,
                    seed = 1, balance_classes = TRUE, project_name = TEXT_MINING)
  
  # AutoML Leaderboard
  lb <- aml@leaderboard
  
  # Optionally edd extra model information to the leaderboard
  #lb <- h2o.get_leaderboard(aml, extra_columns = "ALL")
  
  # Print all rows (instead of default 6 rows)
  print(lb, n = nrow(lb))
  
  # The leader model is stored here
  
  
  # If you need to generate predictions on a test set, you can make
  # predictions directly on the `"H2OAutoML"` object, or on the leader
  # model object directly
  
  #pred <- h2o.predict(aml, test_h2o)
  # predict(aml, test) also works
  
  # or:
  pred <- h2o.predict(aml@leader, test_h2o)
  pred
  
  perf <- h2o.performance(aml@leader, test_h2o)
  perf
  
  h2o.confusionMatrix(perf)
  
  # defina a pasta onde o primeiro modelo do leaderbord será salvo
  h2o.saveModel(aml@leader, path = "cache/cordblood_models")
  
  rm(aml)
  
  return(list("leaderboard" = lb, "predictions" = pred, "performance" = perf))
  
}

# Treinamento autoML
automl_results <- doAutoML(trainSparse, testSparse, "cordblood")

# Converte os resultados do leaderboard para formato R
automl_results$leaderboard <- as.data.frame(automl_results$leaderboard)
metrics_list <- automl_results$performance@metrics
metrics_list$AUC

automl_results$leaderboard

h2o.shutdown()





##### Neural network ####
install.packages("neuralnet")
install.packages('nnet', repos="https://cran.rstudio.com", dependencies = TRUE)
set.seed(108)
library(DMwR)
ctrl.rf_RF_smote = trainControl(method="cv", number = 10, classProbs = TRUE,savePredictions = TRUE, summaryFunction = twoClassSummary, sampling = "smote")
rf_model_smote <- train(final_decision ~ ., data = trainSparse, method = "rf", trControl = ctrl.rf_RF_smote, tuneGrid= grid.rf, tuneLength = 10, metric = "ROC")
#ROSE
set.seed(108)
library(ROSE)
library(caret)
library(nnet)
set.seed(108)
ctrl.rf_RF_rose = trainControl(method="cv", number = 10, classProbs = TRUE,savePredictions = TRUE, summaryFunction = twoClassSummary, sampling = "rose")

numFolds_up <- trainControl(method = "cv", number = 10, classProbs = TRUE, savePredictions = TRUE, summaryFunction = twoClassSummary, sampling = "up")
my.grid <- expand.grid(.decay = c(0.5, 0.1), .size = c(5, 6, 7))
NN_up <- train(final_decision ~ ., data = trainSparse, method = "nnet", preProcess = c('center', 'scale'), trControl = numFolds_up, metric = "ROC", trace = FALSE)



require(neuralnet)

# fit neural network
nn=neuralnet(final_decision~.,data=trainSparse, hidden=3,act.fct = "logistic",
             linear.output = FALSE)

library(ROSE)
set.seed(108)
numFolds_rose <- trainControl(method = "cv", number = 10, classProbs = TRUE, verboseIter = TRUE, summaryFunction = twoClassSummary, sampling = "rose", preProcOptions = list(thresh = 0.75, ICAcomp = 3, k = 5))
NN_rose <- train(final_decision ~ ., data = trainSparse, method = "nnet", trControl = numFolds_rose, tuneGrid=expand.grid(size=c(10), decay=c(0.1)))

library(DMwR)
set.seed(108)
numFolds_smote <- trainControl(method = 'cv', number = 10, classProbs = TRUE, verboseIter = TRUE, summaryFunction = twoClassSummary, sampling = "smote", preProcOptions = list(thresh = 0.75, ICAcomp = 3, k = 5))
NN_smote <- train(final_decision ~ ., data = trainSparse, method = 'nnet', preProcess = c('center', 'scale'), trControl = numFolds_smote, tuneGrid=expand.grid(size=c(10), decay=c(0.1)))



# Randomizando
set.seed(123)
data_rand = data[order(runif(5000)),]
# Processando os dados (text mining)
library(tm)
data_corpus = Corpus(VectorSource(data_rand$Text))
print(data_corpus)
inspect(data_corpus[1:3])
corpus_clean = tm_map(data_corpus, content_transformer(function(x) iconv(enc2utf8(x), sub = "byte")))
corpus_clean = tm_map(corpus_clean, content_transformer(tolower))
corpus_clean = tm_map(corpus_clean, content_transformer(removeNumbers))
corpus_clean = tm_map(corpus_clean, removeWords, stopwords("english"))
corpus_clean = tm_map(corpus_clean, removePunctuation)
corpus_clean = tm_map(corpus_clean, stripWhitespace)
inspect(corpus_clean[1:3])
data_dtm = DocumentTermMatrix(corpus_clean)



### HOLD OUT ####
library(caret)
set.seed(123)
# Separando em dataset de "treino" (75%) e de "teste" (25%)
data_raw_train = data_rand[1:4000, ]
data_raw_test = data_rand[4001:5000, ]
data_dtm_train = data_dtm[1:4000, ]
data_dtm_test = data_dtm[4001:5000, ]
data_corpus_train = corpus_clean[1:4000]
data_corpus_test = corpus_clean[4001:5000]

# Word Clouds
library(wordcloud)
# Rodando o wordcloud com um m??nimo de frequ??ncia de 20
wordcloud(data_corpus_train, min.freq = 20, random.order=F, scale = c(3, 0.5))
# Ajusta frequ??ncia m??nima se estiver dando erro.
# Criando uma nuvem para cada desfecho
Included = subset(data_raw_train, Final_decision == 1)
Excluded = subset(data_raw_train, Final_decision == 0)
# Ajustando o n??mero m??ximo de palavras em cada cloud e a escala das palavras
wordcloud(Included$Text, max.words = 60, scale = c(3, 0.5), random.order=F, colors="red")
wordcloud(Excluded$Text, max.words = 60, scale = c(3, 0.5), random.order=F, colors="blue")

# Preparando os dados
# Utilizaremos apenas as palavras que aparecem pelo menos "X" vezes. Nesse caso, X = 5.
findFreqTerms(data_dtm_train, 5)
data_dict = c(findFreqTerms(data_dtm_train, 5))
data_train = DocumentTermMatrix(data_corpus_train, list(dictionary = data_dict))
data_test = DocumentTermMatrix(data_corpus_test, list(dictionary = data_dict))

# Naives Bayes l?? apenas vari??veis categ??ricas, portanto temos que transformar os n??meros de quantas vezes uma palavra aparece em "Sim, aparece" ou "N??o aparece"
convert_counts <- function(x) {
  x = ifelse(x > 0,1,0) 
  x = factor (x, levels = c(0,1), labels = c("No", "Yes")) 
  return(x) 
} 
## doParallel
library(doParallel)
library(doParallel)
cl <- makeCluster(4)
registerDoParallel(cl)
foreach(i=1:4) %dopar% sqrt(i)
# Realizando a fun????o em cada dataset.. Usa-se MARGIN = 2, pois interessamo-nos nas colunas.
data_train = apply(data_train, MARGIN = 2, convert_counts)
data_test = apply(data_test, MARGIN = 2, convert_counts)

# Treinando o modelo
library(e1071)
set.seed(123)
data_classifier = naiveBayes(data_train, data_raw_train$Final_decision)
data_classifier

# Avaliando a performance do modelo
data_test_pred = predict(data_classifier, data_test)
data_test_pred
library(gmodels)
CrossTable(data_test_pred, data_raw_test$Final_decision, prop.chisq = FALSE, prop.t = FALSE, dnn = c('predicted Outcome', 'actual Outcome'))

# Outras medidadas de avalia????o do modelo
library(caret)
confusionMatrix(data_test_pred, data_raw_test$Final_decision, positive = "1")
# Kappa statistic adjusts accuracy by accounting for the possibility of a correct prediction by chance alone. Page 303 for values.

# Curva ROC - ver pag 312
library(ROCR)
predvec <- ifelse(data_test_pred==1, 1, 0)
realvec <- ifelse(data_raw_test$Final_decision==1, 1, 0)
pred = prediction(predictions = predvec, labels = realvec)
# Fazendo a curva
perf = performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf, main = "ROC curve", col = "blue", lwd = 3)
abline(a=0, b=1, lwd=2, lty=2)
# Medindo a AUC
perf.auc = performance(pred, measure = "auc")
str(perf.auc)
unlist(perf.auc@y.values)

# Melhorando o modelo: colocando o laplace = 1
set.seed(123)
data_classifier_2 = naiveBayes(data_train, data_raw_train$Final_decision, laplace = 1)
data_classifier_2
data_test_pred2 = predict(data_classifier_2, data_test)
library(gmodels)
CrossTable(data_test_pred2, data_raw_test$Final_decision, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, dnn = c('predicted Outcome', 'actual Outcome'))
# Outras medidadas de avaliacao do modelo
library(caret)
confusionMatrix(data_test_pred, data_raw_test$Final_decision, positive = "1")

# Acquiring the predicted probability ("raw") or class ("class") of each observed value
predicted_prob = predict(data_classifier, data_test, type = "raw")
head(predicted_prob)
predicted_class = predict(data_classifier, data_test, type = "class")
head(predicted_class)

