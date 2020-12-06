###############################################################################################################################
# The main code for the HOpt. The trianing and validation of the initial and final modles using the new data.
#Wrte out the final validation errors also
#To do: add the code of define initial models in this code
################################################################################################################################
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
library(openxlsx)
#set current working path
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
#import data
DTube<- read.xlsx("./Data/TrainData_1000_tube.xlsx")
DSbeamCAD<- read.xlsx("./Data/TrainData_1000_sbeamcad.xlsx")
DTenbars<- read.xlsx("./Data/TrainData_1000_tenbars.xlsx")
DTorsionB<- read.xlsx("./Data/TrainData_1000_torsionb.xlsx")

###Normalization
maxvalue<-apply(DTube,2,max)
minvalue<-apply(DTube,2,min)
NDTube<-as.data.frame(scale(DTube,center = minvalue,scale = maxvalue-minvalue))

maxvalue<-apply(DSbeamCAD,2,max)
minvalue<-apply(DSbeamCAD,2,min)
NDSbeamCAD<-as.data.frame(scale(DSbeamCAD,center = minvalue,scale = maxvalue-minvalue))

maxvalue<-apply(DTenbars,2,max)
minvalue<-apply(DTenbars,2,min)
NDTenbars<-as.data.frame(scale(DTenbars,center = minvalue,scale = maxvalue-minvalue))

maxvalue<-apply(DTorsionB,2,max)
minvalue<-apply(DTorsionB,2,min)
NDTorsionB<-as.data.frame(scale(DTorsionB,center = minvalue,scale = maxvalue-minvalue))

##Task
dm=dim(NDTube)
objtube=colnames(NDTube[dm[2]])

dm=dim(NDSbeamCAD)
objsbeam=colnames(NDSbeamCAD[dm[2]])

dm=dim(NDTenbars)
objtenbars=colnames(NDTenbars[dm[2]])

dm=dim(NDTorsionB)
objtorsionb=colnames(NDTorsionB[dm[2]])

tasktube = makeRegrTask(data=NDTube,target=objtube)  #define a regression task
tasksbeam = makeRegrTask(data=NDSbeamCAD,target=objsbeam)  #define a regression task
tasktenbars = makeRegrTask(data=NDTenbars,target=objtenbars)  #define a regression task
tasktorsionb = makeRegrTask(data=NDTorsionB,target=objtorsionb)  #define a regression task

#control object for the task
controlALL=makeTuneMultiCritControlMBO(n.objectives=2L,budget = 100) #budget=20L

##strategies for the task
descALL=makeResampleDesc(method="CV",predict="both",iters=5L) ##5L 5-folds CV

##Learner
Regression_GPR<- makeLearner("regr.gausspr")        #creat gaussian process regression object
ps_GPR = makeParamSet (makeDiscreteParam(id="kernel",values=c("rbfdot","polydot","tanhdot","laplacedot")),
                       makeNumericParam (id="sigma",lower=0,upper=10,requires=quote(kernel=="rbfdot"|kernel=="laplacedot")),
                       makeIntegerParam(id="degree",lower=1L,upper=10L,requires=quote(kernel=="polydot")),
                       makeNumericParam(id="scale",lower=0,upper=10,requires=quote(kernel=="polydot"|kernel=="tanhdot")),
                       makeNumericParam(id="offset",lower=-10,upper=10,requires=quote(kernel=="polydot"|kernel=="tanhdot")))  ##trafo=function (x)2^x
optimal_GPR_tube=tuneParamsMultiCrit(Regression_GPR,tasktube,descALL,par.set=ps_GPR,control=controlALL,measures = list(rmse,my.mxae));
optimal_GPR_sbeam=tuneParamsMultiCrit(Regression_GPR,tasksbeam,descALL,par.set=ps_GPR,control=controlALL,measures = list(rmse,my.mxae));
optimal_GPR_tenbars=tuneParamsMultiCrit(Regression_GPR,tasktenbars,descALL,par.set=ps_GPR,control=controlALL,measures = list(rmse,my.mxae));
optimal_GPR_torsionb=tuneParamsMultiCrit(Regression_GPR,tasktorsionb,descALL,par.set=ps_GPR,control=controlALL,measures = list(rmse,my.mxae));

##SVM
Regression_SVM<-makeLearner("regr.ksvm")            #creat support vector machine regression object
ps_SVM = makeParamSet (makeDiscreteParam(id="kernel",values=c("rbfdot","polydot","tanhdot","laplacedot")),
                       makeNumericParam (id="C",lower=0,upper=10),
                       makeNumericParam (id="epsilon",lower=0,upper=1),
                       makeNumericParam (id="sigma",lower=0,upper=10,requires=quote(kernel=="rbfdot"|kernel=="laplacedot")),
                       makeIntegerParam(id="degree",lower=1L,upper=10L,requires=quote(kernel=="polydot")),
                       makeNumericParam(id="scale",lower=0,upper=10,requires=quote(kernel=="polydot"|kernel=="tanhdot")),
                       makeNumericParam(id="offset",lower=-10,upper=10,requires=quote(kernel=="polydot"|kernel=="tanhdot")))  ##trafo=function (x)2^x

optimal_SVM_tube=tuneParamsMultiCrit(Regression_SVM,tasktube,descALL,par.set=ps_SVM,control=controlALL,measures=list(rmse,my.mxae))
optimal_SVM_sbeam=tuneParamsMultiCrit(Regression_SVM,tasksbeam,descALL,par.set=ps_SVM,control=controlALL,measures=list(rmse,my.mxae))
optimal_SVM_tenbars=tuneParamsMultiCrit(Regression_SVM,tasktenbars,descALL,par.set=ps_SVM,control=controlALL,measures=list(rmse,my.mxae))
optimal_SVM_torsionb=tuneParamsMultiCrit(Regression_SVM,tasktorsionb,descALL,par.set=ps_SVM,control=controlALL,measures=list(rmse,my.mxae))

##RFR
Regression_RFR<-makeLearner("regr.randomForest")   #creat Random Forest regression object
ps_RFR =makeParamSet(makeIntegerParam(id="ntree",lower=1L,upper=1000L),
                     makeIntegerParam(id="mtry",lower=1L,upper=100L),
                     makeIntegerParam(id="nodesize",lower=1L,upper=50L),
                     makeIntegerParam(id="maxnodes",lower=1L,upper=1000L))  ##trafo=function (x)2^x

optimal_RFR_tube=tuneParamsMultiCrit(Regression_RFR,tasktube,descALL,par.set=ps_RFR,control=controlALL,measures=list(rmse,my.mxae))
optimal_RFR_sbeam=tuneParamsMultiCrit(Regression_RFR,tasksbeam,descALL,par.set=ps_RFR,control=controlALL,measures=list(rmse,my.mxae))
optimal_RFR_tenbars=tuneParamsMultiCrit(Regression_RFR,tasktenbars,descALL,par.set=ps_RFR,control=controlALL,measures=list(rmse,my.mxae))
optimal_RFR_torsionb=tuneParamsMultiCrit(Regression_RFR,tasktorsionb,descALL,par.set=ps_RFR,control=controlALL,measures=list(rmse,my.mxae))

##mxnet
Regression_mxnet<-makeLearner("regr.mxff",par.vals=list(num.round=2000,array.layout="rowmajor",layers=1))#,,layers=1,eval.data=list(data=test.x,label=test.y),
                                                        #layers=1,eval.metric=mx.metric.mse,
                                                        #epoch.end.callback = mx.callback.log.train.metric(period=2, log)))   #creat a mxnet FF BP neural network object
ps_mxnet = makeParamSet(makeIntegerParam(id="num.layer1",lower=1L,upper=100L),
                       makeDiscreteParam(id="act1",values=c("tanh","relu","sigmoid","softrelu")),
                       makeDiscreteParam(id="optimizer",values=c("sgd","rmsprop","adam","adagrad")), ##
                       makeIntegerParam(id="array.batch.size",default=100,lower=50L,upper=200L),
                       makeNumericParam(id="learning.rate",default=0.1,lower=0.01,upper=1),  ##,requires=quote(optimizer!=="adagrad"
                       makeNumericParam(id ="momentum",default=0.9,lower=0.5,upper=0.99,requires=quote(optimizer=="sgd")))

optimal_mxnet_tube=tuneParamsMultiCrit(Regression_mxnet,tasktube,descALL,par.set=ps_mxnet,control=controlALL,measures=list(rmse,my.mxae))
optimal_mxnet_sbeam=tuneParamsMultiCrit(Regression_mxnet,tasksbeam,descALL,par.set=ps_mxnet,control=controlALL,measures=list(rmse,my.mxae))
optimal_mxnet_tenbars=tuneParamsMultiCrit(Regression_mxnet,tasktenbars,descALL,par.set=ps_mxnet,control=controlALL,measures=list(rmse,my.mxae))
optimal_mxnet_torsionb=tuneParamsMultiCrit(Regression_mxnet,tasktorsionb,descALL,par.set=ps_mxnet,control=controlALL,measures=list(rmse,my.mxae))

#renew learners based on the optimal result

##Learners
Regression_GPR_tube<-makeLearner("regr.gausspr",par.vals=list(kernel="polydot",degree=3,scale=7.673433,offset=9.301306))
Regression_SVM_tube<- makeLearner("regr.ksvm",par.vals=list(kernel="polydot",C=9.200679,epsilon=0.04904031,degree=2,scale=9.25414,offset=2.963397))
Regression_RFR_tube<-setHyperPars(Regression_RFR,par.vals=optimal_RFR_tube$x[[1]])
Regression_mxnet_tube<-setHyperPars(Regression_mxnet, par.vals=optimal_mxnet_tube$x[[1]])

Regression_GPR_sbeam<- makeLearner("regr.gausspr",par.vals=list(kernel="polydot",degree=3,scale=2.298314,offset=1.079224))
Regression_SVM_sbeam<-makeLearner("regr.ksvm",par.vals=list(kernel="laplacedot",C=9.769786,epsilon=0.1286712,sigma=0.3712581))
Regression_RFR_sbeam<-setHyperPars(Regression_RFR,par.vals=optimal_RFR_sbeam$x[[1]])
Regression_mxnet_sbeam<-makeLearner("regr.mxff",par.vals=list(num.round=2000,array.layout="rowmajor",layers=1,num.layer1=8,
                                                              act1="tanh",optimizer="adagrad",array.batch.size=85,learning.rate=0.2980667))

Regression_GPR_tenbars<-makeLearner("regr.gausspr",par.vals=list(kernel="polydot",degree=3,scale=7.216975,offset=-2.236809))
Regression_SVM_tenbars<-makeLearner("regr.ksvm",par.vals=list(kernel="polydot",C=0.8110992,epsilon=0.07477254,degree=7,scale=2.725278,offset=5.866074))
Regression_RFR_tenbars=setHyperPars(Regression_RFR,par.vals=optimal_RFR_tenbars$x[[1]])
Regression_mxnet_tenbars=setHyperPars(Regression_mxnet, par.vals=optimal_mxnet_tenbars$x[[1]])

Regression_GPR_torsionb<-makeLearner("regr.gausspr",par.vals=list(kernel="polydot",degree=2,scale=1.726261,offset=1.069620))
Regression_SVM_torsionb<-makeLearner("regr.ksvm",par.vals=list(kernel="polydot",C=2.805555,epsilon=0.4067082,degree=1,
                                                               scale=4.026725,offset=-2.089295))
Regression_RFR_torsionb=setHyperPars(Regression_RFR,par.vals=optimal_RFR_torsionb$x[[3]])
Regression_mxnet_torsionb<-makeLearner("regr.mxff", par.vals=list(num.round=2000,array.layout="rowmajor",layers=1,num.layer1=19,act1="relu",
                                                                  optimizer="adagrad",array.batch.size=108,learning.rate=0.5665241))
#Train the models
##Sbeam train for four Regression algorithms
FinalSbeamCAD_model_GPR <- train(Regression_GPR_sbeam, tasksbeam)
FinalSbeamCAD_model_SVM <- train(Regression_SVM_sbeam, tasksbeam)
FinalSbeamCAD_model_RFR <- train(Regression_RFR_sbeam, tasksbeam)
FinalSbeamCAD_model_mxnet <- train(Regression_mxnet_sbeam, tasksbeam)
##tube train for four Regression algorithms
FinalTube_model_GPR <- train(Regression_GPR_tube, tasktube)
FinalTube_model_SVM <- train(Regression_SVM_tube, tasktube)
FinalTube_model_RFR <- train(Regression_RFR_tube, tasktube)
FinalTube_model_mxnet <- train(Regression_mxnet_tube, tasktube)
##ten bars train for four Regression algorithms
FinalTenbars_model_GPR <- train(Regression_GPR_tenbars, tasktenbars)
FinalTenbars_model_SVM <- train(Regression_SVM_tenbars, tasktenbars)
FinalTenbars_model_RFR <- train(Regression_RFR_tenbars, tasktenbars)
FinalTenbars_model_mxnet <- train(Regression_mxnet_tenbars, tasktenbars)
##torsion bar train for four Regression algorithms
FinalTorsionB_model_GPR <- train(Regression_GPR_torsionb, tasktorsionb)
FinalTorsionB_model_SVM <- train(Regression_SVM_torsionb, tasktorsionb)
FinalTorsionB_model_RFR <- train(Regression_RFR_torsionb, tasktorsionb)
FinalTorsionB_model_mxnet <- train(Regression_mxnet_torsionb, tasktorsionb)

############################Test the model by the new data#########################################
library(ModelMetrics) 
TeTube<- read.xlsx("./Data/Testdata_180_tube.xlsx")
TeTube<-TeTube[,2:dim(TeTube)[2]]
TeSbeamCAD<- read.xlsx("./Data/Testdata_180_sbeamcad.xlsx")
TeSbeamCAD<-TeSbeamCAD[,2:dim(TeSbeamCAD)[2]]
TeTenbars<- read.xlsx("./Data/Testdata_180_tenbars.xlsx")
TeTenbars<-TeTenbars[,2:dim(TeTenbars)[2]]
TeTorsionB<- read.xlsx("./Data/Testdata_180_torsionb.xlsx")
TeTorsionB<-TeTorsionB[,2:dim(TeTorsionB)[2]]
#normalization
###Normalization
maxvalue<-apply(DTenbars,2,max)
minvalue<-apply(DTenbars,2,min)
NTeTenbars<- as.data.frame(scale(TeTenbars,center = minvalue,scale = maxvalue-minvalue))

maxvalue<-apply(DTorsionB,2,max)
minvalue<-apply(DTorsionB,2,min)
NTeTorsionB<- as.data.frame(scale(TeTorsionB,center = minvalue,scale = maxvalue-minvalue))

maxvalue<-apply(DSbeamCAD,2,max)
minvalue<-apply(DSbeamCAD,2,min)
NTeSbeamCAD<- as.data.frame(scale(TeSbeamCAD,center = minvalue,scale = maxvalue-minvalue))

maxvalue<-apply(DTube,2,max)
minvalue<-apply(DTube,2,min)
NTeTube<- as.data.frame(scale(TeTube,center = minvalue,scale = maxvalue-minvalue))

##predict the response
#update the task
utasktenbars = makeRegrTask(data=NTeTenbars,target=objtenbars)  #define a regression task
utasktorsionb = makeRegrTask(data=NTeTorsionB,target=objtorsionb)  #define a regression task
utasksbeam = makeRegrTask(data=NTeSbeamCAD,target=objsbeam)  #define a regression task
utasktube = makeRegrTask(data=NTeTube,target=objtube)  #define a regression task
##datasets for saving
ValidationFinal<-as.data.frame(matrix(NA,nrow = 16,ncol = 2))
##for SbeamCAD
FinalSbeamCAD_pre_GPR <- predict(FinalSbeamCAD_model_GPR, utasksbeam)
FinalSbeamCAD_pre_GPR<-as.data.frame(FinalSbeamCAD_pre_GPR$data)
ValidationFinal[1,1]<-rmse(FinalSbeamCAD_pre_GPR$truth,FinalSbeamCAD_pre_GPR$response)
ValidationFinal[1,2]<-max(abs(FinalSbeamCAD_pre_GPR$truth-FinalSbeamCAD_pre_GPR$response))

FinalSbeamCAD_pre_SVM <- predict(FinalSbeamCAD_model_SVM, utasksbeam)
FinalSbeamCAD_pre_SVM<-as.data.frame(FinalSbeamCAD_pre_SVM$data)
ValidationFinal[2,1]<-rmse(FinalSbeamCAD_pre_SVM$truth,FinalSbeamCAD_pre_SVM$response)
ValidationFinal[2,2]<-max(abs(FinalSbeamCAD_pre_SVM$truth-FinalSbeamCAD_pre_SVM$response))

FinalSbeamCAD_pre_RFR <- predict(FinalSbeamCAD_model_RFR, utasksbeam)
FinalSbeamCAD_pre_RFR<-as.data.frame(FinalSbeamCAD_pre_RFR$data)
ValidationFinal[3,1]<-rmse(FinalSbeamCAD_pre_RFR$truth,FinalSbeamCAD_pre_RFR$response)
ValidationFinal[3,2]<-max(abs(FinalSbeamCAD_pre_RFR$truth-FinalSbeamCAD_pre_RFR$response))

FinalSbeamCAD_pre_mxnet <- predict(FinalSbeamCAD_model_mxnet, utasksbeam)
FinalSbeamCAD_pre_mxnet<-as.data.frame(FinalSbeamCAD_pre_mxnet$data)
ValidationFinal[4,1]<-rmse(FinalSbeamCAD_pre_mxnet$truth,FinalSbeamCAD_pre_mxnet$response)
ValidationFinal[4,2]<-max(abs(FinalSbeamCAD_pre_mxnet$truth-FinalSbeamCAD_pre_mxnet$response))
##
FinalTube_pre_GPR <- predict(FinalTube_model_GPR, utasktube)
FinalTube_pre_GPR<-as.data.frame(FinalTube_pre_GPR$data)
ValidationFinal[5,1]<-rmse(FinalTube_pre_GPR$truth,FinalTube_pre_GPR$response)
ValidationFinal[5,2]<-max(abs(FinalTube_pre_GPR$truth-FinalTube_pre_GPR$response))

FinalTube_pre_SVM <- predict(FinalTube_model_SVM, utasktube)
FinalTube_pre_SVM<-as.data.frame(FinalTube_pre_SVM$data)
ValidationFinal[6,1]<-rmse(FinalTube_pre_SVM$truth,FinalTube_pre_SVM$response)
ValidationFinal[6,2]<-max(abs(FinalTube_pre_SVM$truth-FinalTube_pre_SVM$response))

FinalTube_pre_RFR <- predict(FinalTube_model_RFR, utasktube)
FinalTube_pre_RFR<-as.data.frame(FinalTube_pre_RFR$data)
ValidationFinal[7,1]<-rmse(FinalTube_pre_RFR$truth,FinalTube_pre_RFR$response)
ValidationFinal[7,2]<-max(abs(FinalTube_pre_RFR$truth-FinalTube_pre_RFR$response))

FinalTube_pre_mxnet <- predict(FinalTube_model_mxnet, utasktube)
FinalTube_pre_mxnet<-as.data.frame(FinalTube_pre_mxnet$data)
ValidationFinal[8,1]<-rmse(FinalTube_pre_mxnet$truth,FinalTube_pre_mxnet$response)
ValidationFinal[8,2]<-max(abs(FinalTube_pre_mxnet$truth-FinalTube_pre_mxnet$response))

##
FinalTenbars_pre_GPR <- predict(FinalTenbars_model_GPR, utasktenbars)
FinalTenbars_pre_GPR<-as.data.frame(FinalTenbars_pre_GPR$data)
ValidationFinal[9,1]<-rmse(FinalTenbars_pre_GPR$truth,FinalTenbars_pre_GPR$response)
ValidationFinal[9,2]<-max(abs(FinalTenbars_pre_GPR$truth-FinalTenbars_pre_GPR$response))

FinalTenbars_pre_SVM <- predict(FinalTenbars_model_SVM, utasktenbars)
FinalTenbars_pre_SVM<-as.data.frame(FinalTenbars_pre_SVM$data)
ValidationFinal[10,1]<-rmse(FinalTenbars_pre_SVM$truth,FinalTenbars_pre_SVM$response)
ValidationFinal[10,2]<-max(abs(FinalTenbars_pre_SVM$truth-FinalTenbars_pre_SVM$response))

FinalTenbars_pre_RFR <- predict(FinalTenbars_model_RFR, utasktenbars)
FinalTenbars_pre_RFR<-as.data.frame(FinalTenbars_pre_RFR$data)
ValidationFinal[11,1]<-rmse(FinalTenbars_pre_RFR$truth,FinalTenbars_pre_RFR$response)
ValidationFinal[11,2]<-max(abs(FinalTenbars_pre_RFR$truth-FinalTenbars_pre_RFR$response))

FinalTenbars_pre_mxnet <- predict(FinalTenbars_model_mxnet, utasktenbars)
FinalTenbars_pre_mxnet<-as.data.frame(FinalTenbars_pre_mxnet$data)
ValidationFinal[12,1]<-rmse(FinalTenbars_pre_mxnet$truth,FinalTenbars_pre_mxnet$response)
ValidationFinal[12,2]<-max(abs(FinalTenbars_pre_mxnet$truth-FinalTenbars_pre_mxnet$response))

##
FinalTorsionB_pre_GPR <- predict(FinalTorsionB_model_GPR, utasktorsionb)
FinalTorsionB_pre_GPR<-as.data.frame(FinalTorsionB_pre_GPR$data)
ValidationFinal[13,1]<-rmse(FinalTorsionB_pre_GPR$truth,FinalTorsionB_pre_GPR$response)
ValidationFinal[13,2]<-max(abs(FinalTorsionB_pre_GPR$truth-FinalTorsionB_pre_GPR$response))

FinalTorsionB_pre_SVM <- predict(FinalTorsionB_model_SVM, utasktorsionb)
FinalTorsionB_pre_SVM<-as.data.frame(FinalTorsionB_pre_SVM$data)
ValidationFinal[14,1]<-rmse(FinalTorsionB_pre_SVM$truth,FinalTorsionB_pre_SVM$response)
ValidationFinal[14,2]<-max(abs(FinalTorsionB_pre_SVM$truth-FinalTorsionB_pre_SVM$response))

FinalTorsionB_pre_RFR <- predict(FinalTorsionB_model_RFR, utasktorsionb)
FinalTorsionB_pre_RFR<-as.data.frame(FinalTorsionB_pre_RFR$data)
ValidationFinal[15,1]<-rmse(FinalTorsionB_pre_RFR$truth,FinalTorsionB_pre_RFR$response)
ValidationFinal[15,2]<-max(abs(FinalTorsionB_pre_RFR$truth-FinalTorsionB_pre_RFR$response))

FinalTorsionB_pre_mxnet <- predict(FinalTorsionB_model_mxnet, utasktorsionb)
FinalTorsionB_pre_mxnet<-as.data.frame(FinalTorsionB_pre_mxnet$data)
ValidationFinal[16,1]<-rmse(FinalTorsionB_pre_mxnet$truth,FinalTorsionB_pre_mxnet$response)
ValidationFinal[16,2]<-max(abs(FinalTorsionB_pre_mxnet$truth-FinalTorsionB_pre_mxnet$response))

colnames(ValidationFinal)<-c('RMSE','MXAE')
rownames(ValidationFinal)<-c('SBeam-GPR','SBeam-SVM','SBeam-RFR','SBeam-ANN','Tube-GPR','Tube-SVM',
                             'Tube-RFR','Tube-ANN','Tenbars-GPR','Tenbars-SVM','Tenbars-RFR','Tenbars-ANN',
                             'TorsionB-GPR','TorsionB-SVM','TorsionB-RFR','TorsionB-ANN')

#####train the initial models
#Train the models
##Sbeam train for four Regression algorithms
InitialSbeamCAD_model_GPR <- train(Regression_GPR_R, tasksbeam)
InitialSbeamCAD_model_SVM <- train(Regression_SVM_R, tasksbeam)
InitialSbeamCAD_model_RFR <- train(Regression_RFR_R, tasksbeam)
InitialSbeamCAD_model_mxnet <- train(Regression_mxnet_R, tasksbeam)
##tube train for four Regression algorithms
InitialTube_model_GPR <- train(Regression_GPR_R, tasktube)
InitialTube_model_SVM <- train(Regression_SVM_R, tasktube)
InitialTube_model_RFR <- train(Regression_RFR_R, tasktube)
InitialTube_model_mxnet <- train(Regression_mxnet_R, tasktube)
##ten bars train for four Regression algorithms
InitialTenbars_model_GPR <- train(Regression_GPR_R, tasktenbars)
InitialTenbars_model_SVM <- train(Regression_SVM_R, tasktenbars)
InitialTenbars_model_RFR <- train(Regression_RFR_R, tasktenbars)
InitialTenbars_model_mxnet <- train(Regression_mxnet_R, tasktenbars)
##torsion bar train for four Regression algorithms
InitialTorsionB_model_GPR <- train(Regression_GPR_R, tasktorsionb)
InitialTorsionB_model_SVM <- train(Regression_SVM_R, tasktorsionb)
InitialTorsionB_model_RFR <- train(Regression_RFR_R, tasktorsionb)
InitialTorsionB_model_mxnet <- train(Regression_mxnet_R, tasktorsionb)

##predict using the initial models
##datasets for saving
ValidationInitial<-as.data.frame(matrix(NA,nrow = 16,ncol = 2))
##for SbeamCAD
InitialSbeamCAD_pre_GPR <- predict(InitialSbeamCAD_model_GPR, utasksbeam)
InitialSbeamCAD_pre_GPR<-as.data.frame(InitialSbeamCAD_pre_GPR$data)
ValidationInitial[1,1]<-rmse(InitialSbeamCAD_pre_GPR$truth,InitialSbeamCAD_pre_GPR$response)
ValidationInitial[1,2]<-max(abs(InitialSbeamCAD_pre_GPR$truth-InitialSbeamCAD_pre_GPR$response))

InitialSbeamCAD_pre_SVM <- predict(InitialSbeamCAD_model_SVM, utasksbeam)
InitialSbeamCAD_pre_SVM<-as.data.frame(InitialSbeamCAD_pre_SVM$data)
ValidationInitial[2,1]<-rmse(InitialSbeamCAD_pre_SVM$truth,InitialSbeamCAD_pre_SVM$response)
ValidationInitial[2,2]<-max(abs(InitialSbeamCAD_pre_SVM$truth-InitialSbeamCAD_pre_SVM$response))

InitialSbeamCAD_pre_RFR <- predict(InitialSbeamCAD_model_RFR, utasksbeam)
InitialSbeamCAD_pre_RFR<-as.data.frame(InitialSbeamCAD_pre_RFR$data)
ValidationInitial[3,1]<-rmse(InitialSbeamCAD_pre_RFR$truth,InitialSbeamCAD_pre_RFR$response)
ValidationInitial[3,2]<-max(abs(InitialSbeamCAD_pre_RFR$truth-InitialSbeamCAD_pre_RFR$response))

InitialSbeamCAD_pre_mxnet <- predict(InitialSbeamCAD_model_mxnet, utasksbeam)
InitialSbeamCAD_pre_mxnet<-as.data.frame(InitialSbeamCAD_pre_mxnet$data)
ValidationInitial[4,1]<-rmse(InitialSbeamCAD_pre_mxnet$truth,InitialSbeamCAD_pre_mxnet$response)
ValidationInitial[4,2]<-max(abs(InitialSbeamCAD_pre_mxnet$truth-InitialSbeamCAD_pre_mxnet$response))
##
InitialTube_pre_GPR <- predict(InitialTube_model_GPR, utasktube)
InitialTube_pre_GPR<-as.data.frame(InitialTube_pre_GPR$data)
ValidationInitial[5,1]<-rmse(InitialTube_pre_GPR$truth,InitialTube_pre_GPR$response)
ValidationInitial[5,2]<-max(abs(InitialTube_pre_GPR$truth-InitialTube_pre_GPR$response))

InitialTube_pre_SVM <- predict(InitialTube_model_SVM, utasktube)
InitialTube_pre_SVM<-as.data.frame(InitialTube_pre_SVM$data)
ValidationInitial[6,1]<-rmse(InitialTube_pre_SVM$truth,InitialTube_pre_SVM$response)
ValidationInitial[6,2]<-max(abs(InitialTube_pre_SVM$truth-InitialTube_pre_SVM$response))

InitialTube_pre_RFR <- predict(InitialTube_model_RFR, utasktube)
InitialTube_pre_RFR<-as.data.frame(InitialTube_pre_RFR$data)
ValidationInitial[7,1]<-rmse(InitialTube_pre_RFR$truth,InitialTube_pre_RFR$response)
ValidationInitial[7,2]<-max(abs(InitialTube_pre_RFR$truth-InitialTube_pre_RFR$response))

InitialTube_pre_mxnet <- predict(InitialTube_model_mxnet, utasktube)
InitialTube_pre_mxnet<-as.data.frame(InitialTube_pre_mxnet$data)
ValidationInitial[8,1]<-rmse(InitialTube_pre_mxnet$truth,InitialTube_pre_mxnet$response)
ValidationInitial[8,2]<-max(abs(InitialTube_pre_mxnet$truth-InitialTube_pre_mxnet$response))

##
InitialTenbars_pre_GPR <- predict(InitialTenbars_model_GPR, utasktenbars)
InitialTenbars_pre_GPR<-as.data.frame(InitialTenbars_pre_GPR$data)
ValidationInitial[9,1]<-rmse(InitialTenbars_pre_GPR$truth,InitialTenbars_pre_GPR$response)
ValidationInitial[9,2]<-max(abs(InitialTenbars_pre_GPR$truth-InitialTenbars_pre_GPR$response))

InitialTenbars_pre_SVM <- predict(InitialTenbars_model_SVM, utasktenbars)
InitialTenbars_pre_SVM<-as.data.frame(InitialTenbars_pre_SVM$data)
ValidationInitial[10,1]<-rmse(InitialTenbars_pre_SVM$truth,InitialTenbars_pre_SVM$response)
ValidationInitial[10,2]<-max(abs(InitialTenbars_pre_SVM$truth-InitialTenbars_pre_SVM$response))

InitialTenbars_pre_RFR <- predict(InitialTenbars_model_RFR, utasktenbars)
InitialTenbars_pre_RFR<-as.data.frame(InitialTenbars_pre_RFR$data)
ValidationInitial[11,1]<-rmse(InitialTenbars_pre_RFR$truth,InitialTenbars_pre_RFR$response)
ValidationInitial[11,2]<-max(abs(InitialTenbars_pre_RFR$truth-InitialTenbars_pre_RFR$response))

InitialTenbars_pre_mxnet <- predict(InitialTenbars_model_mxnet, utasktenbars)
InitialTenbars_pre_mxnet<-as.data.frame(InitialTenbars_pre_mxnet$data)
ValidationInitial[12,1]<-rmse(InitialTenbars_pre_mxnet$truth,InitialTenbars_pre_mxnet$response)
ValidationInitial[12,2]<-max(abs(InitialTenbars_pre_mxnet$truth-InitialTenbars_pre_mxnet$response))

##
InitialTorsionB_pre_GPR <- predict(InitialTorsionB_model_GPR, utasktorsionb)
InitialTorsionB_pre_GPR<-as.data.frame(InitialTorsionB_pre_GPR$data)
ValidationInitial[13,1]<-rmse(InitialTorsionB_pre_GPR$truth,InitialTorsionB_pre_GPR$response)
ValidationInitial[13,2]<-max(abs(InitialTorsionB_pre_GPR$truth-InitialTorsionB_pre_GPR$response))

InitialTorsionB_pre_SVM <- predict(InitialTorsionB_model_SVM, utasktorsionb)
InitialTorsionB_pre_SVM<-as.data.frame(InitialTorsionB_pre_SVM$data)
ValidationInitial[14,1]<-rmse(InitialTorsionB_pre_SVM$truth,InitialTorsionB_pre_SVM$response)
ValidationInitial[14,2]<-max(abs(InitialTorsionB_pre_SVM$truth-InitialTorsionB_pre_SVM$response))

InitialTorsionB_pre_RFR <- predict(InitialTorsionB_model_RFR, utasktorsionb)
InitialTorsionB_pre_RFR<-as.data.frame(InitialTorsionB_pre_RFR$data)
ValidationInitial[15,1]<-rmse(InitialTorsionB_pre_RFR$truth,InitialTorsionB_pre_RFR$response)
ValidationInitial[15,2]<-max(abs(InitialTorsionB_pre_RFR$truth-InitialTorsionB_pre_RFR$response))

InitialTorsionB_pre_mxnet <- predict(InitialTorsionB_model_mxnet, utasktorsionb)
InitialTorsionB_pre_mxnet<-as.data.frame(InitialTorsionB_pre_mxnet$data)
ValidationInitial[16,1]<-rmse(InitialTorsionB_pre_mxnet$truth,InitialTorsionB_pre_mxnet$response)
ValidationInitial[16,2]<-max(abs(InitialTorsionB_pre_mxnet$truth-InitialTorsionB_pre_mxnet$response))

colnames(ValidationInitial)<-c('RMSE','MXAE')
rownames(ValidationInitial)<-c('SBeam-GPR','SBeam-SVM','SBeam-RFR','SBeam-ANN','Tube-GPR','Tube-SVM',
                               'Tube-RFR','Tube-ANN','Tenbars-GPR','Tenbars-SVM','Tenbars-RFR','Tenbars-ANN',
                               'TorsionB-GPR','TorsionB-SVM','TorsionB-RFR','TorsionB-ANN')

FinalValudationAll<-as.data.frame(matrix(NA,nrow = 8,ncol = 8))
for (i in 1:4) { #for each two rows
  for (j in 1:4) { #for each two cols
    FinalValudationAll[2*(i-1)+1,2*(j-1)+1] <- ValidationInitial[4*(i-1)+j,1]
    FinalValudationAll[2*(i-1)+1,2*j] <- ValidationFinal[4*(i-1)+j,1]
    FinalValudationAll[2*i,2*(j-1)+1] <- ValidationInitial[4*(i-1)+j,2]
    FinalValudationAll[2*i,2*j] <- ValidationFinal[4*(i-1)+j,2]
  }
}
colnames(FinalValudationAll)<-c('GPR-Before HOpt','GPR-After HOpt','SVM-Before HOpt','SVM-After HOpt',
                                'RFR-Before HOpt','RFR-After HOpt','ANN-Before HOpt','ANN-After HOpt')
rownames(FinalValudationAll)<-c('Sbeam-RMSE','Sbeam-MXAE','Tube-RMSE','Tube-MXAE','Tenbars-RMSE',
                                'Tenbars-MXAE','TorsionB-RMSE','TorsionB-MXAE')
write.xlsx(FinalValudationAll, file = "./Data/TestedbyNewData_Initial-Finalmodels.xlsx",
           sheetName="ValidationAccuracy", col.names=TRUE, row.names=TRUE, showNA=TRUE, password=NULL)
