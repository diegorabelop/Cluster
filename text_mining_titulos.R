##### TITULOS
library(readxl)
data_titulos <- read_excel("random_data_only_titles_primary_consensus.xlsx")

data1 <- data_titulos[,c("Title", "Final_decision")]
#coombining Abstract and Title
data1$Text <- data1$Title

data <- data1[,c("Final_decision", "Text")]
print(subset(data,select =c(1, Text)))

# Preparing the data
data <- as.data.frame(data)
data$Final_decision = as.factor(data$Final_decision)
data$Text = as.character(data$Text)
print(subset(data,select =c(1, Final_decision)))

str(data)
table(data$Final_decision)

#Text mining packages install.packages('tm', dependencies=TRUE, repos='http://cran.rstudio.com/')
library(tm)
library(SnowballC)
print(data)
# Step 1 - Create a corpus text
corpus = Corpus(VectorSource(data$Text))
corpus[[1]][1]
data$Final_decision[1]
data$Text[1]
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
# Step 6 - Create Document Term Matrix
frequencies = DocumentTermMatrix(corpus)
sparse = removeSparseTerms(frequencies, 0.995) #remove sparse terms
tSparse = as.data.frame(as.matrix(sparse)) #convert into data frame
colnames(tSparse) = make.names(colnames(tSparse)) #all the variable names R-friendly
tSparse$final_decision = data$Final_decision #add the outcome

prop.table(table(tSparse$final_decision))
table(tSparse$final_decision)
tSparse1 <- tSparse
#### CONVERTENDO EM Z ####
tSparse1 <- scale(tSparse1[,-211])
tSparse1<- as.data.frame(tSparse1)
tSparse2 <- cbind(tSparse1, tSparse$final_decision)
tSparse2 <- as.data.frame(tSparse2)

names(tSparse2)[211] <- "final_decision"
# Hold out
library(caret)
set.seed(123)
intrain <- createDataPartition(y = tSparse2$final_decision, p= 0.8, list = FALSE)
trainSparse <-  tSparse[intrain,]
testSparse <- tSparse[-intrain,]
dim(trainSparse); dim(testSparse)
# Porcentagem do desfecho em cada banco
prop.table(table(tSparse$final_decision))
prop.table(table(trainSparse$final_decision))
prop.table(table(testSparse$final_decision))

table(trainSparse$final_decision)
table(testSparse$final_decision)


## doParallel
library(doParallel)
cl <- makeCluster(4)
registerDoParallel(cl)
foreach(i=1:4) %dopar% sqrt(i)

#Training
library(randomForest)
set.seed(100)
trainSparse$final_decision = as.factor(trainSparse$final_decision)
testSparse$final_decision = as.factor(testSparse$final_decision)

levels(trainSparse$final_decision)
levels(trainSparse$final_decision) <- c("no", "yes")
levels(testSparse$final_decision)
levels(testSparse$final_decision) <- c("no", "yes")

############ ELastic Net ######
library(caret)
library(glmnet)
set.seed(108)
ctrl.EN <- trainControl(method="cv", number = 10, classProbs = TRUE,savePredictions = TRUE, summaryFunction = twoClassSummary)
model_imbalance <- train(final_decision ~., data = trainSparse, method = "glmnet", trControl = ctrl.EN, tuneLength = 10, metric = "ROC")
#up regulation
set.seed(108) 
ctrl.EN_up <- trainControl(method="cv", number = 10, classProbs = TRUE,savePredictions = TRUE, summaryFunction = twoClassSummary, sampling = "up")
up_outside <- train(final_decision ~., data = trainSparse, method = "glmnet", trControl = ctrl.EN_up, tuneLength = 10, metric = "ROC")
#Smote
library(DMwR)
set.seed(108) 
ctrl.EN_smote <- trainControl(method="cv", number = 10, classProbs = TRUE,savePredictions = TRUE, summaryFunction = twoClassSummary, sampling = "smote")
smote_outside <- train(final_decision ~., data = trainSparse, method = "glmnet", trControl = ctrl.EN_smote, tuneLength = 10, metric = "ROC")
#Rose
library(ROSE)
set.seed(108) 
ctrl.EN_rose <- trainControl(method="cv", number = 10, classProbs = TRUE,savePredictions = TRUE, summaryFunction = twoClassSummary, sampling = "rose")
rose_outside <- train(final_decision ~., data = trainSparse, method = "glmnet", trControl = ctrl.EN_rose, tuneLength = 10, metric = "ROC")

#Downregulation
set.seed(108)
ctrl.EN_down <- trainControl(method="repeatedcv", repeats = 1000, number = 10, classProbs = TRUE,savePredictions = TRUE, summaryFunction = twoClassSummary, sampling = "down")
down_outside <- train(final_decision ~., data = trainSparse, method = "glmnet", trControl = ctrl.EN_down, tuneLength = 10, metric = "ROC")


#escolhendo o melhor modelo 
library(mlbench)
library(caret)
result_model_EN <- resamples(list(IMBALANCE= model_imbalance, up=up_outside, rose=rose_outside, smote=smote_outside))
summary(result_model_EN) #SMOTE FOI MELHOR
bwplot(result_model_EN) 
dotplot(result_model_EN)

# confusion matrix
library(e1071)
test_pred_rose_EN <- predict(rose_outside, newdata = testSparse[,-211])
matrixConfusao_rose_EN <- confusionMatrix(test_pred_rose_EN, testSparse$final_decision, positive = "yes")
matrixConfusao_rose_EN
# plot ROC curve
library(caret)
library(pROC)
p_rose_EN <- predict (rose_outside, testSparse[,-211])
p_prob_rose_EN <- predict (rose_outside, testSparse[,-211], type = "prob")
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
rf.predict <- predict(rose_outside, testSparse[, -211], type = "prob")
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
###### Random Forest #############
#DoParalelo
library(doParallel)
gc()
cl <- makePSOCKcluster(5)
registerDoParallel(cl)

#install.packages('corrplot', dependencies = TRUE, repos = 'http://cran.rstudio.com/')
gc()
library(caret)
library(randomForest)
set.seed(108)
grid.rf = expand.grid(.mtry=c(1,2,3,4,5,6,7,8,9,10,11,12,13,14))
ctrl.rf = trainControl(method="cv", number = 10, classProbs = TRUE,
                       savePredictions = TRUE, summaryFunction = twoClassSummary)
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
result_model2 <- resamples(list(up=rf_model_up, rose=rf_model_rose, smote=rf_model_smote))
summary(result_model2)
bwplot(result_model2) #the best model UP
dotplot(result_model2)

# confusion matrix
library(e1071)
test_pred_up_rf <- predict(rf_model_up, newdata = testSparse[,-211])
matrixConfusao_up_rf <- confusionMatrix(test_pred_up_rf, testSparse$final_decision, positive = "yes")
matrixConfusao_up_rf
# plot ROC curve
library(caret)
library(pROC)
p_up_rf <- predict (rf_model_up, testSparse[,-211])
p_prob_up_rf <- predict (rf_model_up, testSparse[,-211], type = "prob")
confusionMatrix(p_up_rf, testSparse$final_decision)
print(confusionMatrix(p_up_rf, testSparse$final_decision, positive = "yes"))
r_up_rf <- roc (testSparse$final_decision, p_prob_up_rf[,"yes"])
plot(r_up_rf)
r_up_rf$auc

###### Mudando limiar 

library(SDMTools)
# 0 = No, 1 = Yes
obs <- testSparse$final_decision
levels(obs)
levels(obs) <- c("0", "1")
obs <- as.numeric(as.character(obs))
obs

# model prediction with probabilities
rf.predict <- predict(rf_model_up, testSparse[, -211], type = "prob")
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
