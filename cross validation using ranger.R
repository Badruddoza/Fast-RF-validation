## rgcv: n-fold cross validation using rf in ranger
rm(list = ls())
#install.packages("rfUtilities")
#install.packages("spm")
library("spm")

rdata<-read.csv("rdata.csv")
dim(rdata)
drop<-seq(50,282,1)
rdata<-rdata[,-c(drop)]
head(rdata[,-1])

#rgcv1 <- rgcv(rdata[,-1], rdata[,1], predacc = "ALL")

n <- 5 # number of iterations, 60 to 100 is recommended.
output<-c()
for (i in 1:n) {
  temp1<- rgcv(rdata[,-1], rdata[,1], predacc = "ALL")
  output<-c(output,temp1) 
}
output<-data.frame(matrix(c(output),nrow=n,ncol=9,byrow=T))
colnames(output)<-c("me","rme","mae","rmae","mse","rmse","rrmse","vecv","e1")
head(output)
var<-unlist(output$rmse)

plot(var~ c(1:n), xlab = "Iteration for RF", ylab = "VEcv (%)")
points(cumsum(var) / c(1:n) ~ c(1:n), col = 2)
abline(h = mean(var), col = 'blue', lwd = 2)

##mrfei becomes binary
#change mrfei as factor variable
rdata$mrfei[rdata$mrfei>0]<-1
rdata$mrfei <- as.factor(rdata$mrfei)


output2<-c()
for (i in 1:n) {
  temp2<- rgcv(rdata[,-1], rdata[,1])
  output2<-c(output2,temp2)
}
output2<-data.frame(matrix(c(output2),nrow=n,ncol=9,byrow=T))
colnames(output2)<-c("kappa","ccr","sens","spec","tss")
var<-unlist(output2$ccr)

plot(var~ c(1:n), xlab = "Iteration for RF", ylab = "Correct
     classification rate  (%)")
points(cumsum(var2) / c(1:n) ~ c(1:n), col = 2)
abline(h = mean(var2), col = 'blue', lwd = 2)
