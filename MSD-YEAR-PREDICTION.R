library(kknn)
library(rpart)
library(MASS)
library(caret)
library(kernlab)
library(monmlp)
library(randomForest)

setwd("C:/Users/Xihan Liu/Google ÔÆ¶ËÓ²ÅÌ/PredictiveAnalytics_Project/split")

error <- matrix(nrow = 18, ncol = 7)
colnames(error) <- c("KNN.1", "KNN.2", "RT", "SVM", "MLP", "RF","LDA")
MSE <- matrix(nrow = 18, ncol = 7)
colnames(MSE) <- c("KNN.1", "KNN.2", "RT", "SVM", "MLP", "RF","LDA")

for(j in 2:18) {
	cat(j)
	train <- read.csv(paste(paste("train", j, sep = "_"), "csv", sep = "."))
	test <- read.csv(paste(paste("test", j, sep = "_"), "csv", sep = "."))
	
	## train and test data
	y <- factor(train$V1)
	Ty <- factor(test$V1)
	X <- scale(as.matrix(train[,3:ncol(train)]))
	TX <- scale(as.matrix(test[,3:ncol(test)]))
	
	learn <- data.frame(cbind(train$V1,X))
	valid <- data.frame(cbind(test$V1,TX))
	
## KNN classifier
	k <- round(sqrt(nrow(X)))
	KNN_fit.train1 <- train.kknn(factor(V1) ~ ., learn, kmax = k,
	                        kernel = c("triangular", "rectangular", "epanechnikov", "optimal"), distance = 1)
	KNN_fit.train2 <- train.kknn(factor(V1) ~ ., learn, kmax = k,
	                        kernel = c("triangular", "rectangular", "epanechnikov", "optimal"), distance = 2)
	#plot(KNN_fit.train1)
	#table(predict(KNN_fit.train1, valid), valid$V1)
	#table(predict(KNN_fit.train2, valid), valid$V1)
  KNN.eucli.pred <- as.vector(predict(KNN_fit.train1, valid))
  KNN.manha.pred <- as.vector(predict(KNN_fit.train2, valid))
  error_KNN.eucli <- mean(KNN.eucli.pred!=as.vector(valid$V1))
  error_KNN.manha <- mean(KNN.manha.pred!=as.vector(valid$V1))
  MSE_KNN.eucli <- mean((as.numeric(KNN.eucli.pred) - as.vector(valid$V1))^2)
  MSE_KNN.manha <- mean((as.numeric(KNN.manha.pred) - as.vector(valid$V1))^2)
	
	error[j,1] <- error_KNN.eucli
	error[j,2] <- error_KNN.manha

  MSE[j,1] <- MSE_KNN.eucli
  MSE[j,2] <- MSE_KNN.manha
	
## DT
	set.seed(1)
	y.indx <- createFolds(y, k = 5, returnTrain = TRUE)
	y.ctrl <- trainControl(method = "cv", number = 5, p = 0.8, savePredictions = TRUE, index = y.indx)
	y.cartTune <- train(X, y,
	                           method = "rpart",
	                           tuneLength = 25,
	                           trControl = y.ctrl)
	#y.cartTune
  y.DT.predicted <- as.vector(predict(y.cartTune, TX))
  error_DT <- mean(y.DT.predicted != as.vector(Ty))
  MSE_DT <- mean((as.numeric(y.DT.predicted) - as.numeric(as.vector(Ty)))^2)

	
	error[j,3] <- error_DT
  MSE[j,3] <- MSE_DT

## one v.s. all
	bi <- NULL
	for(i in as.numeric(levels(y))){
	  bi <- cbind(bi,ifelse(train$V1==i,1,0))
	}
	
	Tbi <- NULL
	for(i in as.numeric(levels(y))){
	  Tbi <- cbind(Tbi,ifelse(test$V1==i,1,0))
	}

# SVM
	pred_svm_OVA <- NULL
	for(i in 1:ncol(bi)){
	  svp <- ksvm(X,bi[,i],kernal="rbfdot",type="C-svc", cross = 5, 
	               prob.model = TRUE)
	  pred_svm_OVA <- cbind(pred_svm_OVA,predict(svp,TX,type="probabilities")[,2])
	}
	colnames(pred_svm_OVA) <- levels(y)
  deci_svm_OVA <- as.vector(levels(y)[max.col(pred_svm_OVA, ties.method = "random")])
  error_svm_OVA <- mean(deci_svm_OVA != as.vector(Ty))
  MSE_svm_OVA <- mean((as.numeric(deci_svm_OVA) - as.numeric(as.vector(Ty)))^2)

	error[j,4] <- error_svm_OVA
  MSE[j,4] <- MSE_svm_OVA

# MLP
	# specify number of hidden nodes, ensembels, bagging or not
	hidden <- 2 # number of hidden nodes
	n <- 5  # number of ensembles
	mlp <- monmlp.fit(X, bi, hidden1 = hidden, n.ensemble = n, bag = TRUE) 
	pred_MLP <- monmlp.predict(TX, mlp)
	colnames(pred_MLP) <- levels(y)
  deci_MLP <- as.vector(levels(y)[max.col(pred_MLP, ties.method = "random")])
  error_MLP <- mean(deci_MLP != as.vector(Ty))
  MSE_MLP <- mean((as.numeric(deci_MLP) - as.numeric(as.vector(Ty)))^2)

	error[j,5] <- error_MLP
  MSE[j,5] <- MSE_MLP
	
## Random Forest
	set.seed(300)
  rf1 <- randomForest(factor(V1) ~., learn, ntree=30, norm.votes=FALSE)
  rf2 <- randomForest(factor(V1) ~., learn, ntree=30, norm.votes=FALSE)
  rf3 <- randomForest(factor(V1) ~., learn, ntree=30, norm.votes=FALSE)
  rf <- combine(rf1, rf2, rf3)
  rf.pred <- as.vector(predict(object=rf, TX))
  error_rf <- mean(rf.pred!=as.vector(Ty))
  MSE_rf <- mean((as.numeric(rf.pred) - as.numeric(as.vector(Ty)))^2)

	error[j,6] <- error_rf
  MSE[j,6] <- MSE_rf 

## LDA
  lda <- lda(factor(V1) ~ ., learn, prior = summary(y)/nrow(X))
  lda.pred <- as.vector(predict(lda, valid)$class)
  error_lda <- mean(lda.pred!=as.vector(Ty))
  MSE_lda <- mean((as.numeric(lda.pred) - as.numeric(as.vector(Ty)))^2)

  error[j,7] <- error_lda
  MSE[j,7] <- MSE_lda 
	
write.csv(t(error[j,]), paste0("error_",j,".csv"))
write.csv(t(MSE[j,]), paste0("MSE_",j,".csv"))	

cat("Finished!\n")
}

print(error)
write.csv(error, "error.csv")
print(MSE)
write.csv(MSE, "MSE.csv")
