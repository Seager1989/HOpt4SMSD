library (mlr)
library (mlbench)
library(readr)
library(mlrMBO)
library(emoa)
library(DiceKriging)
library(rgenoud)
library(mxnet)
library(PMCMR)
library(PMCMRplus)
#set current working path
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
##initial Learners
#GPR
Regression_GPR_R<- makeLearner("regr.gausspr", par.vals=list(kernel='rbfdot',sigma=0.5))        #creat gaussian process regression object

##SVM
Regression_SVM_R<-makeLearner("regr.ksvm", par.vals=list(C=1,epsilon=0.1,kernel='rbfdot',sigma=0.5))            #creat support vector machine regression object

##RFR
Regression_RFR_R<-makeLearner("regr.randomForest", par.vals=list(ntree=500,mtry=3,nodesize=5))   #creat Random Forest regression object

##mxnet
Regression_mxnet_R<-makeLearner("regr.mxff", par.vals=list(num.round=2000,layers=1,num.layer1 =10,act1='tanh',optimizer='sgd',array.batch.size=120,learning.rate=0.1))#,,layers=1,eval.data=list(data=test.x,label=test.y),

#############################################from the optimal tunning to be here######################################################
##Optimal Learners
Regression_GPR_tube<-makeLearner("regr.gausspr",par.vals=list(kernel="polydot",degree=3,scale=7.673433,offset=9.301306))
Regression_SVM_tube<- makeLearner("regr.ksvm",par.vals=list(kernel="polydot",C=9.200679,epsilon=0.04904031,degree=2,scale=9.25414,offset=2.963397))
Regression_RFR_tube<-setHyperPars(Regression_RFR,par.vals=optimal_RFR_tube$x[[1]])
Regression_mxnet_tube<-setHyperPars(Regression_mxnet, par.vals=optimal_mxnet_tube$x[[1]])

Regression_GPR_sbeam<- makeLearner("regr.gausspr",par.vals=list(kernel="polydot",degree=3,scale=2.298314,offset=1.079224))
Regression_SVM_sbeam<-makeLearner("regr.ksvm",par.vals=list(kernel="laplacedot",C=9.769786,epsilon=0.1286712,sigma=0.3712581))
Regression_RFR_sbeam<-setHyperPars(Regression_RFR,par.vals=optimal_RFR_sbeam$x[[1]])
Regression_mxnet_sbeam<-makeLearner("regr.mxff",par.vals=list(num.round=2000,array.layout="rowmajor",layers=1,num.layer1=8,act1="tanh",optimizer="adagrad",array.batch.size=85,learning.rate=0.2980667))

Regression_GPR_tenbars<-makeLearner("regr.gausspr",par.vals=list(kernel="polydot",degree=3,scale=7.216975,offset=-2.236809))
Regression_SVM_tenbars<-makeLearner("regr.ksvm",par.vals=list(kernel="polydot",C=0.8110992,epsilon=0.07477254,degree=7,scale=2.725278,offset=5.866074))
Regression_RFR_tenbars=setHyperPars(Regression_RFR,par.vals=optimal_RFR_tenbars$x[[1]])
Regression_mxnet_tenbars=setHyperPars(Regression_mxnet, par.vals=optimal_mxnet_tenbars$x[[1]])

Regression_GPR_torsionb<-makeLearner("regr.gausspr",par.vals=list(kernel="polydot",degree=2,scale=1.726261,offset=1.069620))
Regression_SVM_torsionb<-makeLearner("regr.ksvm",par.vals=list(kernel="polydot",C=2.805555,epsilon=0.4067082,degree=1,scale=4.026725,offset=-2.089295))
Regression_RFR_torsionb=setHyperPars(Regression_RFR,par.vals=optimal_RFR_torsionb$x[[3]])
Regression_mxnet_torsionb<-makeLearner("regr.mxff", par.vals=list(num.round=2000,array.layout="rowmajor",layers=1,num.layer1=19,act1="relu",optimizer="adagrad",array.batch.size=108,learning.rate=0.5665241))

##banchmark problem for origin##################################bmr_Re with ANN 2000 iterations and bmr with 100 iterations
LMLA=list(Regression_GPR_R,Regression_SVM_R,Regression_RFR_R,Regression_mxnet_R)
#bmr=benchmark(LMLA,tasks=list(tasktube,tasksbeam,tasktenbars,tasktorsionb),descALL,measures = list(rmse,my.mxae,timetrain))#tasktube,tasksbeam,tasktenbars,tasktorsionb
bmr_Re=benchmark(LMLA,tasks=list(tasktube,tasksbeam,tasktenbars,tasktorsionb),descALL,measures = list(rmse,my.mxae,timetrain))#tasktube,tasksbeam,tasktenbars,tasktorsionb

##for specific learners tunned
LMLA_tube=list(Regression_GPR_tube,Regression_SVM_tube,Regression_RFR_tube,Regression_mxnet_tube)
bmr_tube=benchmark(LMLA_tube,tasks=tasktube,descALL,measures = list(rmse,my.mxae,timetrain))

LMLA_sbeam=list(Regression_GPR_sbeam,Regression_SVM_sbeam,Regression_RFR_sbeam,Regression_mxnet_sbeam)
bmr_sbeam=benchmark(LMLA_sbeam,tasks=tasksbeam,descALL,measures = list(rmse,my.mxae,timetrain))

LMLA_tenbars=list(Regression_GPR_tenbars,Regression_SVM_tenbars,Regression_RFR_tenbars,Regression_mxnet_tenbars)
bmr_tenbars=benchmark(LMLA_tenbars,tasks=tasktenbars,descALL,measures = list(rmse,my.mxae,timetrain))

LMLA_torsionb=list(Regression_GPR_torsionb,Regression_SVM_torsionb,Regression_RFR_torsionb,Regression_mxnet_torsionb)
bmr_torsionb=benchmark(LMLA_torsionb,tasks=tasktorsionb,descALL,measures = list(rmse,my.mxae,timetrain))

#For RMSE before tunning
bench_RMSE_Sbeam_before<-as.matrix(c(bmr_Re$results$NDSbeamCAD$regr.gausspr$measures.test$rmse,
                                    bmr_Re$results$NDSbeamCAD$regr.ksvm$measures.test$rmse,
                                    bmr_Re$results$NDSbeamCAD$regr.randomForest$measures.test$rmse,
                                    bmr_Re$results$NDSbeamCAD$regr.mxff$measures.test$rmse),nrow=5,ncol=4) #pltboxT$data$rmse[1:20]
bench_RMSE_Tube_before<-as.matrix(c(bmr_Re$results$NDTube$regr.gausspr$measures.test$rmse,
                                   bmr_Re$results$NDTube$regr.ksvm$measures.test$rmse,
                                   bmr_Re$results$NDTube$regr.randomForest$measures.test$rmse,
                                   bmr_Re$results$NDTube$regr.mxff$measures.test$rmse),nrow=5,ncol=4)
bench_RMSE_Tenbars_before<-as.matrix(c(bmr_Re$results$NDTenbars$regr.gausspr$measures.test$rmse,
                                      bmr_Re$results$NDTenbars$regr.ksvm$measures.test$rmse,
                                      bmr_Re$results$NDTenbars$regr.randomForest$measures.test$rmse,
                                      bmr_Re$results$NDTenbars$regr.mxff$measures.test$rmse),nrow=5,ncol=4)
bench_RMSE_TorsionB_before<-as.matrix(c(bmr_Re$results$NDTorsionB$regr.gausspr$measures.test$rmse,
                                       bmr_Re$results$NDTorsionB$regr.ksvm$measures.test$rmse,
                                       bmr_Re$results$NDTorsionB$regr.randomForest$measures.test$rmse,
                                       bmr_Re$results$NDTorsionB$regr.mxff$measures.test$rmse),nrow=5,ncol=4) 

colnames(bench_RMSE_Sbeam_before)<-c('GPR','SVM','RFR','ANN')
colnames(bench_RMSE_Tube_before)<-c('GPR','SVM','RFR','ANN')
colnames(bench_RMSE_Tenbars_before)<-c('GPR','SVM','RFR','ANN')
colnames(bench_RMSE_TorsionB_before)<-c('GPR','SVM','RFR','ANN')
##tune ANN
bench_RMSE_Sbeam_before[,4]<-bmr_Re$results$NDSbeamCAD$regr.mxff$measures.test$rmse
bench_RMSE_Tube_before[,4]<-bmr_Re$results$NDTube$regr.mxff$measures.test$rmse
bench_RMSE_Tenbars_before[,4]<-bmr_Re$results$NDTenbars$regr.mxff$measures.test$rmse
bench_RMSE_TorsionB_before[,4]<-bmr_Re$results$NDTorsionB$regr.mxff$measures.test$rmse

#For RMSE after tunning
bench_RMSE_Sbeam_after<-as.matrix(c(bmr_sbeam$results$NDSbeamCAD$regr.gausspr$measures.test$rmse,
                                    bmr_sbeam$results$NDSbeamCAD$regr.ksvm$measures.test$rmse,
                                    bmr_sbeam$results$NDSbeamCAD$regr.randomForest$measures.test$rmse,
                                    bmr_sbeam$results$NDSbeamCAD$regr.mxff$measures.test$rmse),nrow=5,ncol=4) #pltboxT$data$rmse[1:20]
bench_RMSE_Tube_after<-as.matrix(c(bmr_tube$results$NDTube$regr.gausspr$measures.test$rmse,
                                   bmr_tube$results$NDTube$regr.ksvm$measures.test$rmse,
                                   bmr_tube$results$NDTube$regr.randomForest$measures.test$rmse,
                                   bmr_tube$results$NDTube$regr.mxff$measures.test$rmse),nrow=5,ncol=4)
bench_RMSE_Tenbars_after<-as.matrix(c(bmr_tenbars$results$NDTenbars$regr.gausspr$measures.test$rmse,
                                      bmr_tenbars$results$NDTenbars$regr.ksvm$measures.test$rmse,
                                      bmr_tenbars$results$NDTenbars$regr.randomForest$measures.test$rmse,
                                      bmr_tenbars$results$NDTenbars$regr.mxff$measures.test$rmse),nrow=5,ncol=4)
bench_RMSE_TorsionB_after<-as.matrix(c(bmr_torsionb$results$NDTorsionB$regr.gausspr$measures.test$rmse,
                                       bmr_torsionb$results$NDTorsionB$regr.ksvm$measures.test$rmse,
                                       bmr_torsionb$results$NDTorsionB$regr.randomForest$measures.test$rmse,
                                       bmr_torsionb$results$NDTorsionB$regr.mxff$measures.test$rmse),nrow=5,ncol=4) 
#####
colnames(bench_RMSE_Sbeam_after)<-c('GPR','SVM','RFR','ANN')
colnames(bench_RMSE_Tube_after)<-c('GPR','SVM','RFR','ANN')
colnames(bench_RMSE_Tenbars_after)<-c('GPR','SVM','RFR','ANN')
colnames(bench_RMSE_TorsionB_after)<-c('GPR','SVM','RFR','ANN')

datagroup_before<-c('bench_RMSE_Sbeam_before','bench_RMSE_Tube_before','bench_RMSE_Tenbars_before','bench_RMSE_TorsionB_before')
datagroup_after<-c('bench_RMSE_Sbeam_after','bench_RMSE_Tube_after','bench_RMSE_Tenbars_after','bench_RMSE_TorsionB_after')
title<-c('(a) S-beam','(b) Tube','(c) Ten bars','(d) Torsion bar')
par(mfrow=c(2,2),mai=c(0.4,0.6,0.4,0.3),mgp=c(2.4, 0.8, 0),font=1,cex.lab=1.4,cex.axis=1.2,pch=13,cex.main=1.4)
for (i in 1:4) {
  boxplot(get(datagroup_before[i]),boxwex = 0.3,at = 1:4 - 0.18,col = "blue",xlim = c(0.5, 4.5),main=title[i], ylim = c(0.02, 0.16), ylab = "Loss (RMSE)",names=NA )
  points(1:4 - 0.18,ValidationInitial[(4*(i-1)+1):(4*i),1],type="p", pch=23, col="black",bg="red",cex=1.8)
  boxplot(get(datagroup_after[i]),add=TRUE,boxwex = 0.3,at = 1:4 + 0.18,col = "green",names=NA )
  points(1:4 + 0.18,ValidationFinal[(4*(i-1)+1):(4*i),1],type="p", pch=23, col="black",bg="red",cex=1.8)
  axis(side=1,at=c(1,2,3,4),labels = c('GPR','SVM','RFR','ANN'))
  #legend(0.5, 0.15, c("Before optimization", "After optimization"),fill = c("blue", "green"),cex = 1.4)
  legend(2.4, 0.16, pch=c(NA,NA,23),c("Before Optimization", "After Optimization",'New-data validation'),
         fill=c("blue", "green",NA),border = c('Black','Black',NA),pt.bg = c(NA, NA,"red"),cex = 1.4,pt.cex=2)
}

#For MXAE before tunning
bench_MXAE_Sbeam_before<-as.matrix(c(bmr_Re$results$NDSbeamCAD$regr.gausspr$measures.test$my.mxae,
                                     bmr_Re$results$NDSbeamCAD$regr.ksvm$measures.test$my.mxae,
                                     bmr_Re$results$NDSbeamCAD$regr.randomForest$measures.test$my.mxae,
                                     bmr_Re$results$NDSbeamCAD$regr.mxff$measures.test$my.mxae),nrow=5,ncol=4) #pltboxT$data$rmse[1:20]
bench_MXAE_Tube_before<-as.matrix(c(bmr_Re$results$NDTube$regr.gausspr$measures.test$my.mxae,
                                    bmr_Re$results$NDTube$regr.ksvm$measures.test$my.mxae,
                                    bmr_Re$results$NDTube$regr.randomForest$measures.test$my.mxae,
                                    bmr_Re$results$NDTube$regr.mxff$measures.test$my.mxae),nrow=5,ncol=4)
bench_MXAE_Tenbars_before<-as.matrix(c(bmr_Re$results$NDTenbars$regr.gausspr$measures.test$my.mxae,
                                       bmr_Re$results$NDTenbars$regr.ksvm$measures.test$my.mxae,
                                       bmr_Re$results$NDTenbars$regr.randomForest$measures.test$my.mxae,
                                       bmr_Re$results$NDTenbars$regr.mxff$measures.test$my.mxae),nrow=5,ncol=4)
bench_MXAE_TorsionB_before<-as.matrix(c(bmr_Re$results$NDTorsionB$regr.gausspr$measures.test$my.mxae,
                                        bmr_Re$results$NDTorsionB$regr.ksvm$measures.test$my.mxae,
                                        bmr_Re$results$NDTorsionB$regr.randomForest$measures.test$my.mxae,
                                        bmr_Re$results$NDTorsionB$regr.mxff$measures.test$my.mxae),nrow=5,ncol=4) 
#changing ANN
##tune ANN
bench_MXAE_Sbeam_before[,4]<-bmr_Re$results$NDSbeamCAD$regr.mxff$measures.test$my.mxae
bench_MXAE_Tube_before[,4]<-bmr_Re$results$NDTube$regr.mxff$measures.test$my.mxae
bench_MXAE_Tenbars_before[,4]<-bmr_Re$results$NDTenbars$regr.mxff$measures.test$my.mxae
bench_MXAE_TorsionB_before[,4]<-bmr_Re$results$NDTorsionB$regr.mxff$measures.test$my.mxae
##

colnames(bench_MXAE_Sbeam_before)<-c('GPR','SVM','RFR','ANN')
colnames(bench_MXAE_Tube_before)<-c('GPR','SVM','RFR','ANN')
colnames(bench_MXAE_Tenbars_before)<-c('GPR','SVM','RFR','ANN')
colnames(bench_MXAE_TorsionB_before)<-c('GPR','SVM','RFR','ANN')

###################

#For MXAE after tunning
bench_MXAE_Sbeam_after<-as.matrix(c(bmr_sbeam$results$NDSbeamCAD$regr.gausspr$measures.test$my.mxae,
                                    bmr_sbeam$results$NDSbeamCAD$regr.ksvm$measures.test$my.mxae,
                                    bmr_sbeam$results$NDSbeamCAD$regr.randomForest$measures.test$my.mxae,
                                    bmr_sbeam$results$NDSbeamCAD$regr.mxff$measures.test$my.mxae),nrow=5,ncol=4) #pltboxT$data$rmse[1:20]
bench_MXAE_Tube_after<-as.matrix(c(bmr_tube$results$NDTube$regr.gausspr$measures.test$my.mxae,
                                   bmr_tube$results$NDTube$regr.ksvm$measures.test$my.mxae,
                                   bmr_tube$results$NDTube$regr.randomForest$measures.test$my.mxae,
                                   bmr_tube$results$NDTube$regr.mxff$measures.test$my.mxae),nrow=5,ncol=4)
bench_MXAE_Tenbars_after<-as.matrix(c(bmr_tenbars$results$NDTenbars$regr.gausspr$measures.test$my.mxae,
                                      bmr_tenbars$results$NDTenbars$regr.ksvm$measures.test$my.mxae,
                                      bmr_tenbars$results$NDTenbars$regr.randomForest$measures.test$my.mxae,
                                      bmr_tenbars$results$NDTenbars$regr.mxff$measures.test$my.mxae),nrow=5,ncol=4)
bench_MXAE_TorsionB_after<-as.matrix(c(bmr_torsionb$results$NDTorsionB$regr.gausspr$measures.test$my.mxae,
                                       bmr_torsionb$results$NDTorsionB$regr.ksvm$measures.test$my.mxae,
                                       bmr_torsionb$results$NDTorsionB$regr.randomForest$measures.test$my.mxae,
                                       bmr_torsionb$results$NDTorsionB$regr.mxff$measures.test$my.mxae),nrow=5,ncol=4) 
##

colnames(bench_MXAE_Sbeam_after)<-c('GPR','SVM','RFR','ANN')
colnames(bench_MXAE_Tube_after)<-c('GPR','SVM','RFR','ANN')
colnames(bench_MXAE_Tenbars_after)<-c('GPR','SVM','RFR','ANN')
colnames(bench_MXAE_TorsionB_after)<-c('GPR','SVM','RFR','ANN')

datagroup_before<-c('bench_MXAE_Sbeam_before','bench_MXAE_Tube_before','bench_MXAE_Tenbars_before','bench_MXAE_TorsionB_before')
datagroup_after<-c('bench_MXAE_Sbeam_after','bench_MXAE_Tube_after','bench_MXAE_Tenbars_after','bench_MXAE_TorsionB_after')
title<-c('(a) S-beam','(b) Tube','(c) Ten bars','(d) Torsion bar')
par(mfrow=c(2,2),mai=c(0.4,0.6,0.4,0.3),mgp=c(2.4, 0.8, 0),font=1,cex.lab=1.4,cex.axis=1.2,pch=13,cex.main=1.4)
for (i in 1:4) {
  boxplot(get(datagroup_before[i]),boxwex = 0.3,at = 1:4 - 0.18,col = "blue",xlim = c(0.5, 4.5),main=title[i], ylim = c(0.1, 1.2), ylab = "Loss (MXAE)",names=NA )
  points(1:4 - 0.18,ValidationInitial[(4*(i-1)+1):(4*i),2],type="p", pch=23, col="black",bg="red",cex=1.8)
  boxplot(get(datagroup_after[i]),add=TRUE,boxwex = 0.3,at = 1:4 + 0.18,col = "green",names=NA )
  points(1:4 + 0.18,ValidationFinal[(4*(i-1)+1):(4*i),2],type="p", pch=23, col="black",bg="red",cex=1.8)
  axis(side=1,at=c(1,2,3,4),labels = c('GPR','SVM','RFR','ANN'))
  legend(2.4, 1.22, pch=c(NA,NA,23),c("Before Optimization", "After Optimization",'New-data validation'),
         fill=c("blue", "green",NA),border = c('Black','Black',NA),pt.bg = c(NA, NA,"red"),cex = 1.4,pt.cex=2)
}
## for the model t=robustness analysis with the sd of measures
#For TrainT before tunning pltboxTm/pltboxTmT

bench_TrainT_Sbeam_before<-as.matrix(c(bmr_Re$results$NDSbeamCAD$regr.gausspr$measures.test$timetrain,
                                     bmr_Re$results$NDSbeamCAD$regr.ksvm$measures.test$timetrain,
                                     bmr_Re$results$NDSbeamCAD$regr.randomForest$measures.test$timetrain,
                                     bmr_Re$results$NDSbeamCAD$regr.mxff$measures.test$timetrain),nrow=5,ncol=4) #pltboxT$data$rmse[1:20]
bench_TrainT_Tube_before<-as.matrix(c(bmr_Re$results$NDTube$regr.gausspr$measures.test$timetrain,
                                    bmr_Re$results$NDTube$regr.ksvm$measures.test$timetrain,
                                    bmr_Re$results$NDTube$regr.randomForest$measures.test$timetrain,
                                    bmr_Re$results$NDTube$regr.mxff$measures.test$timetrain),nrow=5,ncol=4)
bench_TrainT_Tenbars_before<-as.matrix(c(bmr_Re$results$NDTenbars$regr.gausspr$measures.test$timetrain,
                                       bmr_Re$results$NDTenbars$regr.ksvm$measures.test$timetrain,
                                       bmr_Re$results$NDTenbars$regr.randomForest$measures.test$timetrain,
                                       bmr_Re$results$NDTenbars$regr.mxff$measures.test$timetrain),nrow=5,ncol=4)
bench_TrainT_TorsionB_before<-as.matrix(c(bmr_Re$results$NDTorsionB$regr.gausspr$measures.test$timetrain,
                                        bmr_Re$results$NDTorsionB$regr.ksvm$measures.test$timetrain,
                                        bmr_Re$results$NDTorsionB$regr.randomForest$measures.test$timetrain,
                                        bmr_Re$results$NDTorsionB$regr.mxff$measures.test$timetrain),nrow=5,ncol=4) 
#tune ANN
bench_TrainT_Sbeam_before[,4]<-bmr_Re$results$NDSbeamCAD$regr.mxff$measures.test$timetrain
bench_TrainT_Tube_before[,4]<-bmr_Re$results$NDTube$regr.mxff$measures.test$timetrain
bench_TrainT_Tenbars_before[,4]<-bmr_Re$results$NDTenbars$regr.mxff$measures.test$timetrain
bench_TrainT_TorsionB_before[,4]<-bmr_Re$results$NDTorsionB$regr.mxff$measures.test$timetrain
##

colnames(bench_TrainT_Sbeam_before)<-c('GPR','SVM','RFR','ANN')
colnames(bench_TrainT_Tube_before)<-c('GPR','SVM','RFR','ANN')
colnames(bench_TrainT_Tenbars_before)<-c('GPR','SVM','RFR','ANN')
colnames(bench_TrainT_TorsionB_before)<-c('GPR','SVM','RFR','ANN')

#For TrainT after tunning (training time)
bench_TrainT_Sbeam_after<-pltboxTmT$data$timetrain[1:20]
bench_TrainT_Tube_after<-pltboxTmT$data$timetrain[61:80]
bench_TrainT_Tenbars_after<-pltboxTmT$data$timetrain[21:40]
bench_TrainT_TorsionB_after<-pltboxTmT$data$timetrain[41:60]

bench_TrainT_Sbeam_after<-as.matrix(c(bmr_sbeam$results$NDSbeamCAD$regr.gausspr$measures.test$timetrain,
                                    bmr_sbeam$results$NDSbeamCAD$regr.ksvm$measures.test$timetrain,
                                    bmr_sbeam$results$NDSbeamCAD$regr.randomForest$measures.test$timetrain,
                                    bmr_sbeam$results$NDSbeamCAD$regr.mxff$measures.test$timetrain),nrow=5,ncol=4) #pltboxT$data$rmse[1:20]
bench_TrainT_Tube_after<-as.matrix(c(bmr_tube$results$NDTube$regr.gausspr$measures.test$timetrain,
                                   bmr_tube$results$NDTube$regr.ksvm$measures.test$timetrain,
                                   bmr_tube$results$NDTube$regr.randomForest$measures.test$timetrain,
                                   bmr_tube$results$NDTube$regr.mxff$measures.test$timetrain),nrow=5,ncol=4)
bench_TrainT_Tenbars_after<-as.matrix(c(bmr_tenbars$results$NDTenbars$regr.gausspr$measures.test$timetrain,
                                      bmr_tenbars$results$NDTenbars$regr.ksvm$measures.test$timetrain,
                                      bmr_tenbars$results$NDTenbars$regr.randomForest$measures.test$timetrain,
                                      bmr_tenbars$results$NDTenbars$regr.mxff$measures.test$timetrain),nrow=5,ncol=4)
bench_TrainT_TorsionB_after<-as.matrix(c(bmr_torsionb$results$NDTorsionB$regr.gausspr$measures.test$timetrain,
                                       bmr_torsionb$results$NDTorsionB$regr.ksvm$measures.test$timetrain,
                                       bmr_torsionb$results$NDTorsionB$regr.randomForest$measures.test$timetrain,
                                       bmr_torsionb$results$NDTorsionB$regr.mxff$measures.test$timetrain),nrow=5,ncol=4) 

colnames(bench_TrainT_Sbeam_after)<-c('GPR','SVM','RFR','ANN')
colnames(bench_TrainT_Tube_after)<-c('GPR','SVM','RFR','ANN')
colnames(bench_TrainT_Tenbars_after)<-c('GPR','SVM','RFR','ANN')
colnames(bench_TrainT_TorsionB_after)<-c('GPR','SVM','RFR','ANN')

datagroup_before<-c('bench_TrainT_Sbeam_before','bench_TrainT_Tube_before','bench_TrainT_Tenbars_before','bench_TrainT_TorsionB_before')
datagroup_after<-c('bench_TrainT_Sbeam_after','bench_TrainT_Tube_after','bench_TrainT_Tenbars_after','bench_TrainT_TorsionB_after')
title<-c('(a) S-beam','(b) Tube','(c) Ten bars','(d) Torsion bar')
par(mfrow=c(2,2),mai=c(0.4,0.6,0.4,0.3),mgp=c(2.4, 0.8, 0),font=1,cex.lab=1.4,cex.axis=1.2,pch=13,cex.main=1.4)
rangees=c(100,100,100,100)
for (i in 1:4) {
  boxplot(get(datagroup_before[i]),boxwex = 0.3,at = 1:4 - 0.18,col = "blue",xlim = c(0.5, 4.5),main=title[i], ylim = c(0.1, rangees[i]), ylab = "Training time (s)",names=NA,border=c("blue","blue","blue","blue") )
  boxplot(get(datagroup_after[i]),add=TRUE,boxwex = 0.3,at = 1:4 + 0.18,col = "green",names=NA,border=c("green", "green","green","green") )
  axis(side=1,at=c(1,2,3,4),labels = c('GPR','SVM','RFR','ANN'))
  legend(2.35, rangees[i], c("Before Optimization", "After Optimization"),fill = c("blue", "green"),cex = 1.4,border=c("blue", "green"))
  transite1=matrix(c(apply(get(datagroup_before[i]),2,mean),apply(get(datagroup_after[i]),2,mean)),nrow = 2,byrow = TRUE)
  print(c('Mean of',datagroup_before[i]))
  print(transite1)
  }
