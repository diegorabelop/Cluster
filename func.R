getCM <- function(best.glmnet2, dt3Test2, positive_class, ...){
  require(caret)
  final_predictions2 <- predict(best.glmnet2, data.matrix(dt3Test2[, -1]), type = "raw")
  
  cm2 <- confusionMatrix(final_predictions2, dt3Test2$traj, positive_class)
  return(cm2)
  
  
}

plotROC <- function(best.glmnet2, dt3Test2){
  require(pROC)
  
  glm.predict2 <- predict(best.glmnet2, data.matrix(dt3Test2[, -1]), type = "prob")
  predictions2 <- as.vector(glm.predict2[, 2])
  rocobj2 <- roc(dt3Test2$traj, predictions2, ci=TRUE, of="auc", percent = FALSE)
  plot(rocobj2, main="Confidence intervals", percent = FALSE, ci=TRUE, print.auc=TRUE)

}

getPerformanceVsCutoff <- function(confusion_df, threshold_seq, positive_class){
  message("positive_class = as.character(0) ou as.character(1)")
  
  require(purrr)
  require(plyr)
  
  
  confMatrix <- function(i, obs, predictions, ...){
    require(caret)
    require(SDMTools)
    
    conf_matrix <- confusion.matrix(obs, predictions, threshold = i)
    cm_table <- as.table(conf_matrix)
    cm <- confusionMatrix(cm_table, positive = positive_class)
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
  
  result_list <- map(threshold_seq, confMatrix, obs = obs, predictions = predictions)
  result_df <- ldply(result_list)
  return(result_df)
}

plotPerformancesMeasures <- function(result_df){
  require(ggplot2)
  require(reshape2)
  
  colnames(result_df)
  result_selected <- result_df[, c("limiar", "AB", "Pos Pred Value", "Neg Pred Value")]
  colnames(result_selected) <- c("Cutoff", "Balanced Accuracy", "Pos Pred Value", "Neg Pred Value")
  result_long <- melt(result_selected, id.vars = "Cutoff")
  
  p1 <-ggplot(result_long, aes(x=Cutoff, y=value, group=variable)) +
    geom_line(aes(color=variable)) +
    geom_point(aes(color=variable)) +
    theme(panel.background = element_blank(), 
          axis.line = element_line(colour = "grey"),
          legend.title = element_blank())
  p1
  
  result_selected <- result_df[, c("limiar", "AB", "Sensitivity", "Specificity")]
  colnames(result_selected) <- c("Cutoff", "Balanced Accuracy", "Sensitivity", "Specificity")
  result_long <- melt(result_selected, id.vars = "Cutoff")
  
  p2 <-ggplot(result_long, aes(x=Cutoff, y=value, group=variable)) +
    geom_line(aes(color=variable)) +
    geom_point(aes(color=variable)) +
    theme(panel.background = element_blank(), 
          axis.line = element_line(colour = "grey"),
          legend.title = element_blank())
  
  p2
  
  return(list("BA_PPV_NPV_Vs_Cutoff" = p1 , "BA_Sen_Spe_Vs_Cutoff" = p2))
  
}

plotVarImp <- function(model){
  require(caret)
  require(ggplot2)
  
  
  varimp_df <- varImp(model)
  varimp_df <- varimp_df$importance
  varimp_df
  
  Variables <- c("Age", "Familial monthly income", "Sex", "Ethnicity", "University degree", 
                 "Married", "Self-reported health", "Obesity", "Non-smoker", "SAD", "Panic Disorder",
                 "TAG", "OCD", "Heavy drinker", "Benzodiazepine", "Use of antidepressants", "Negative life events")
  
  print(data.frame(row.names(varimp_df), Variables))
  varimp_df <- data.frame(Variables, "Overall" = varimp_df$Overall)
  
  zeroimportance <- varimp_df$Variables[varimp_df$Overall == 0]
  print(zeroimportance)
  
  varimp_df <- varimp_df[varimp_df$Overall != 0, ]
  
  print(varimp_df)
  
  p <- ggplot(data = varimp_df, aes(x = reorder(varimp_df$Variable, varimp_df$Overall), y = Overall, fill = Variables)) + 
    labs(x = "Features", y = "Overall")
  
  p <- p + geom_bar(stat = "identity", color = "darkblue", fill = "steelblue") + theme_minimal() + coord_flip()
  p
}
