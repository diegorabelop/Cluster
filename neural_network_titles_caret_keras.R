#Multilayer Perceptron Network with Dropout
#method = 'mlpKerasDropout'
#Type: Regression, Classification
#Tuning parameters:
 # size (#Hidden Units)
 # dropout (Dropout Rate)
 # batch_size (Batch Size)
 # lr (Learning Rate)
 # rho (Rho)
 # decay (Learning Rate Decay)
 # activation (Activation Function)
 # Required packages: keras
 # Notes: After train completes, the keras model object is serialized so that it can be used between R session. 
 # When predicting, the code will temporarily unsearalize the object. To make the predictions more efficient, 
 # the user might want to use keras::unsearlize_model(object$finalModel$object) in the current R session so that 
 # operation is only done once. Also, this model cannot be run in parallel due to the nature of how tensorflow does
 # the computations. Unlike other packages used by train, the dplyr package is fully loaded when this model is used.

load("sessions/text_mining_titulos_15-02.RData")

library(caret)


#nzv <- nearZeroVar(trainSparse, saveMetrics= TRUE)
#nzv[nzv$nzv,][1:10,]

#nzv_vector <- nzv$nzv
#nzv_vector

# O algoritmo considera final decision como near zero, mas e nosso desfecho
# Para ela ser incluida, mudamos o TRUE para FALSE
#nzv_vector[length(nzv_vector)] <- FALSE

#trainSparse_filtered <- trainSparse[, !nzv_vector]

#dim(trainSparse_filtered)

#testSparse_filtered <- testSparse[, !nzv_vector]

trainSparse_filtered <- trainSparse
testSparse_filtered <- testSparse

pp <- preProcess(trainSparse_filtered[, -ncol(trainSparse_filtered)], 
                 method = c("center", "scale"))
pp



train_transformed <- predict(pp, newdata = trainSparse_filtered[, -ncol(trainSparse_filtered)])
test_transformed <- predict(pp, newdata = testSparse_filtered[, -ncol(testSparse_filtered)])

library(purrr)
map(train_transformed, hist)

train_transformed <- data.frame(train_transformed, final_decision = trainSparse_filtered$final_decision)
test_transformed <- data.frame(test_transformed, final_decision = testSparse_filtered$final_decision)

levels(train_transformed$final_decision) <- c("No", "Yes")
levels(test_transformed$final_decision) <- c("No", "Yes")


fitControl_up <- trainControl(method = "repeatedcv", 
                           number = 10, 
                           repeats = 5, 
                           classProbs = TRUE, 
                           summaryFunction = twoClassSummary,
                           sampling = "up")

fitControl_smote <- trainControl(method = "repeatedcv", 
                              number = 10, 
                              repeats = 5, 
                              classProbs = TRUE, 
                              summaryFunction = twoClassSummary,
                              sampling = "smote")


fitControl_rose <- trainControl(method = "repeatedcv", 
                                 number = 10, 
                                 repeats = 5, 
                                 classProbs = TRUE, 
                                 summaryFunction = twoClassSummary,
                                 sampling = "rose")


#nnetGrid <-  expand.grid(.decay = c(0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7), 
#                        .size = c(3, 5, 10, 20))



kerasGrid <-  expand.grid(size = c(700),
                          dropout = c(0, 0.3, 0.75, 0.9),
                          batch_size = 1,
                          lr = 2e-06,
                          rho = 0.9,
                          decay = 0,
                          activation = "relu"
                          )
kerasGrid

class(trainSparse_filtered$final_decision)

library(doParallel)
library(keras)
library(tensorflow)
library(caret)
tf$constant("Hellow Tensorflow")


detectCores()

# smote ----
time_start <- Sys.time()

cl <- makePSOCKcluster(16)
registerDoParallel(cl)

set.seed(1234)
#kerasFit_up <- caret::train(final_decision ~ ., 
 #                data = train_transformed,
  #               method = "mlpKerasDropout",
   #              metric = "ROC",
    #             trControl = fitControl,
     #            tuneGrid = kerasGrid,
      #           verbose = FALSE)


set.seed(1234)
kerasFit_smote <- caret::train(final_decision ~ ., 
                            data = train_transformed,
                            method = "mlpKerasDropout",
                            metric = "ROC",
                            trControl = fitControl_smote,
                            tuneGrid = kerasGrid,
                            verbose = FALSE)


stopCluster(cl)
time_end <- Sys.time()
time_end - time_start

varImp(kerasFit_smote)

plot(kerasFit_smote)

kerasFit_smote$bestTune


data_test_pred = predict(kerasFit_smote, test_transformed)
data_test_pred

#library(gmodels)
#CrossTable(data_test_pred, test_transformed$final_decision, prop.chisq = FALSE, prop.t = FALSE, dnn = c('predicted Outcome', 'actual Outcome'))

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

# rose ----
time_start <- Sys.time()

cl <- makePSOCKcluster(16)
registerDoParallel(cl)

set.seed(1234)

kerasFit_rose <- caret::train(final_decision ~ ., 
                               data = train_transformed,
                               method = "mlpKerasDropout",
                               metric = "ROC",
                               trControl = fitControl_rose,
                               tuneGrid = kerasGrid,
                               verbose = FALSE)


stopCluster(cl)
time_end <- Sys.time()
time_end - time_start

varImp(kerasFit_rose)

plot(kerasFit_rose)

kerasFit_rose$bestTune


data_test_pred = predict(kerasFit_rose, test_transformed)
data_test_pred

data_test_pred_prob = predict(kerasFit_rose, test_transformed, type = "prob")
data_test_pred_prob


# Outras medidadas de avalia????o do modelo
library(caret)
confusionMatrix(data_test_pred, test_transformed$final_decision, positive = "Yes")

# Kappa statistic adjusts accuracy by accounting for the possibility of a correct prediction by chance alone. Page 303 for values.

# Curva ROC - ver pag 312

library(ROCR)
predvec <- ifelse(data_test_pred == "Yes", 1, 0)
realvec <- ifelse(test_transformed$final_decision == "Yes", 1, 0)
pred = prediction(predictions = predvec, labels = realvec)

# Medindo a AUC
perf.auc = performance(pred, measure = "auc")
unlist(perf.auc@y.values)

prop.table(table(train_transformed$final_decision))

# Performance vs. cut-off
df_rose <- data.frame("predictions" = data_test_pred_prob, "obs" = test_transformed$final_decision)

saveRDS(df_rose, file = "cache/obs_predictions_prob_titles_df_rose.rds")

install.packages("h2o")
library(h2o)
h2o.init()

Sys.getenv("JAVA_HOME")
