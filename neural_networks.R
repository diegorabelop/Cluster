##################### CARET ################
library(caret)
library(nnet)

nzv <- nearZeroVar(trainSparse, saveMetrics= TRUE)
nzv[nzv$nzv,][1:10,]

nzv_vector <- nzv$nzv

# O algoritmo considera final decision como near zero, mas e nosso desfecho
# Para ela ser incluida, mudamos o TRUE para FALSE
nzv_vector[length(nzv_vector)] <- FALSE

trainSparse_filtered <- trainSparse[, !nzv_vector]
testSparse_filtered <- testSparse[, !nzv_vector]

pp <- preProcess(trainSparse_filtered[, -ncol(trainSparse_filtered)], 
                     method = c("center", "scale"))
pp

train_transformed <- predict(pp, newdata = trainSparse_filtered[, -ncol(trainSparse_filtered)])
test_transformed <- predict(pp, newdata = testSparse_filtered[, -ncol(testSparse_filtered)])

train_transformed <- data.frame(train_transformed, final_decision = trainSparse_filtered$final_decision)
test_transformed <- data.frame(test_transformed, final_decision = testSparse_filtered$final_decision)
                                
levels(train_transformed$final_decision) <- c("No", "Yes")
levels(test_transformed$final_decision) <- c("No", "Yes")


fitControl <- trainControl(method = "repeatedcv", 
                           number = 10, 
                           repeats = 5, 
                           classProbs = TRUE, 
                           summaryFunction = twoClassSummary,
                          sampling = "up")

set.seed(1234)

#nnetGrid <-  expand.grid(.decay = c(0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7), 
 #                        .size = c(3, 5, 10, 20))

nnetGrid <-  expand.grid(.decay = c(0.99), 
                         .size = c(1, 3, 5, 10))

class(trainSparse_filtered$final_decision)

library(doParallel)
gc()

cl <- makePSOCKcluster(3)
registerDoParallel(cl)


nnetFit <- train(final_decision ~ ., 
                 data = train_transformed,
                 method = "nnet",
                 metric = "ROC",
                 trControl = fitControl,
                 tuneGrid = nnetGrid,
                 verbose = FALSE)

stopCluster(cl)
varImp(nnetFit)

plot(nnetFit)

nnetFit$bestTune


data_test_pred = predict(nnetFit, test_transformed)
data_test_pred

library(gmodels)
CrossTable(data_test_pred, test_transformed$final_decision, prop.chisq = FALSE, prop.t = FALSE, dnn = c('predicted Outcome', 'actual Outcome'))

# Outras medidadas de avalia????o do modelo
library(caret)
confusionMatrix(data_test_pred, test_transformed$final_decision, positive = "Yes")

# Kappa statistic adjusts accuracy by accounting for the possibility of a correct prediction by chance alone. Page 303 for values.

# Curva ROC - ver pag 312
library(ROCR)
predvec <- ifelse(data_test_pred == "Yes", 1, 0)
realvec <- ifelse(test_transformed$final_decision == "Yes", 1, 0)
pred = prediction(predictions = predvec, labels = realvec)

# Fazendo a curva
perf = performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf, main = "ROC curve", col = "blue", lwd = 3)
abline(a=0, b=1, lwd=2, lty=2)

# Medindo a AUC
perf.auc = performance(pred, measure = "auc")
str(perf.auc)
unlist(perf.auc@y.values)

prop.table(table(train_transformed$final_decision))
