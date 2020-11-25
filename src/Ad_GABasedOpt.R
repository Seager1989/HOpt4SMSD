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
library(metaheuristicOpt)

#######################################################################################################################
## Optimizing the established models
#construct the learners list for Tube
Tube_learners=c('Tube_model_GPR','Tube_model_SVM','Tube_model_RFR','Tube_model_mxnet')
#the circulation for the tube optimization
Tube_optimum<-matrix(nrow = 4,ncol = 10)

for (i in 1:4) {
Tubef <- function(X){
  names(X)<-c("T1","T2","T3","T4","T5","T6","T7","T8","T9")
  X=as.data.frame(X)
  X=t(X)
  X=as.data.frame(X)
  trans=predict(get(Tube_learners[i]),newdata=X)
  re<-trans$data$response
  return(re)}
## Define parameters of GA algorithms
Pm <- 0.1
Pc <- 0.8
numVar <- 9
rangeVar <- matrix(c(0,1), nrow=2)
## calculate the optimum solution using GA 
Tube_optimum[i,1:9] <- GA(Tubef, optimType="MAX", numVar, numPopulation=20, maxIter=100, rangeVar, Pm, Pc)
## calculate the optimum value using trained learner
Tube_optimum[i,10] <- Tubef(Tube_optimum[i,1:9])
}
####################################################################################################################
########construct the learners list for Sbeam
SbeamCAD_learners=c('SbeamCAD_model_GPR','SbeamCAD_model_SVM','SbeamCAD_model_RFR','SbeamCAD_model_mxnet')
#Define dataset to save optimum
SbeamCAD_optimum<-matrix(nrow = 4,ncol = 8)
#the circulation for the Sbeam optimization
for (i in 1:4) {
  SbeamCADf <- function(X){
    names(X)<-c("AN",  "T",   "DR",  "UL",  "UR",  "H",   "CL")
    X=as.data.frame(X)
    X=t(X)
    X=as.data.frame(X)
    trans=predict(get(SbeamCAD_learners[i]),newdata=X)
    re<-trans$data$response
    return(re)}
  
  ## Define parameters of GA algorithms
  Pm <- 0.1
  Pc <- 0.8
  numVar <- 7
  rangeVar <- matrix(c(0,1), nrow=2)
  ## calculate the optimum solution using GA 
  SbeamCAD_optimum[i,1:7] <- GA(SbeamCADf, optimType="MAX", numVar, numPopulation=20, maxIter=100, rangeVar, Pm, Pc)
  ## calculate the optimum value using trained learner
  SbeamCAD_optimum[i,8] <- SbeamCADf(SbeamCAD_optimum[i,1:7])
}

#########################################################################################################################
#construct the learners list for Ten bars
Tenbars_learners=c('Tenbars_model_GPR','Tenbars_model_SVM','Tenbars_model_RFR','Tenbars_model_mxnet')
#Define dataset to save optimum
Tenbars_optimum<-matrix(nrow = 4,ncol = 11)
#the circulation for the Tenbars optimization
for (i in 1:4) {
Tenbarsf <- function(X){
    names(X)<-c("A1" ,  "A2" ,  "A3" ,  "A4"  , "A5"  , "A6"  , "A7"  , "A8"  , "A9"  , "A10")
    X=as.data.frame(X)
    X=t(X)
    X=as.data.frame(X)
    trans=predict(get(Tenbars_learners[i]),newdata=X)
    dis4<-trans$data$response
    if ((555.58*dis4+36.6836) < 100){
      mass=9144*(22520*(X[,1]+X[,2]+X[,3]+X[,4]+X[,5]+X[,6])+360)+12931.6*(22520*(X[,7]+X[,8]+X[,9]+X[,10])+240)
      } else {
        mass=1e30
        }
    return(mass)
}
  
## Define parameters of GA algorithms
  Pm <- 0.1
  Pc <- 0.8
  numVar <- 10
  rangeVar <- matrix(c(0,1), nrow=2)
  ## calculate the optimum solution using GA 
  Tenbars_optimum[i,1:10] <- GA(Tenbarsf, optimType="MIN", numVar, numPopulation=20, maxIter=100, rangeVar, Pm, Pc)
  ## calculate the optimum value using trained learner
  Tenbars_optimum[i,11] <- Tenbarsf(Tenbars_optimum[i,1:10])
}
#############################################################################################################################
#construct the learners list for Ten bars
TorsionB_learners=c('TorsionB_model_GPR','TorsionB_model_SVM','TorsionB_model_RFR','TorsionB_model_mxnet')
#Define dataset to save optimum
TorsionB_optimum<-matrix(nrow = 4,ncol = 15)
#the circulation for the TorsionB optimization
for (i in 1:4) {
  TorsionBf <- function(X){
    names(X)<-c("Y1","Y2","Y3","Y4","Y5","Y6","Y7","Y8","Y9","Y10","r1","r2","X1","X2")
    X=as.data.frame(X)
    X=t(X)
    X=as.data.frame(X)
    trans=predict(get(TorsionB_learners[i]),newdata=X)
    X=data.matrix(X)
    stress=predict(Torsiob_ANN_stress,X)
    stress=354.21+stress*78018.99
    if (stress<800){
      re<-trans$data$response
    } else {
      re<-100
    }
    return(re)}
  
  ## Define parameters of GA algorithms
  Pm <- 0.1
  Pc <- 0.8
  numVar <- 14
  rangeVar <- matrix(c(0,1), nrow=2)
  ## calculate the optimum solution using GA 
  TorsionB_optimum[i,1:14] <- GA(TorsionBf, optimType="MIN", numVar, numPopulation=20, maxIter=100, rangeVar, Pm, Pc)
  ## calculate the optimum value using trained learner
  TorsionB_optimum[i,15] <- TorsionBf(TorsionB_optimum[i,1:14])
}

##############################################################################################################################
#tube real optimum
Tube_optimum_R<-Tube_optimum[,dim(Tube_optimum)[2]]*Est_tube_scale+Est_tube_min

Tube_var<-matrix(data=NA,nrow=dim(Tube_optimum)[1],ncol=dim(Tube_optimum)[2])

Tube_var_max<-apply(DTube,2,max)
Tube_var_min<-apply(DTube,2,min)
Tube_var_range<-Tube_var_max-Tube_var_min
for (i in 1:dim(Tube_optimum)[1]){
  Tube_var[i,]<-Tube_optimum[i,]*Tube_var_range+Tube_var_min
}

#SbeamCAD real optimum
SbeamCAD_optimum_R<-SbeamCAD_optimum[,dim(SbeamCAD_optimum)[2]]*Est_sbeam_scale+Est_sbeam_min
#construct a null matrix
SbeamCAD_var<-matrix(data = NA,nrow = dim(SbeamCAD_optimum)[1],ncol = dim(SbeamCAD_optimum)[2])

SbeamCAD_var_max<-apply(DSbeamCAD,2,max)
SbeamCAD_var_min<-apply(DSbeamCAD,2,min)
SbeamCAD_var_range<-SbeamCAD_var_max-SbeamCAD_var_min
for (i in 1:dim(SbeamCAD_optimum)[1]){
  SbeamCAD_var[i,]<-SbeamCAD_optimum[i,]*SbeamCAD_var_range+SbeamCAD_var_min
}

#Ten bars real optimum

Tenbars_var<-matrix(data = NA,nrow = dim(Tenbars_optimum)[1],ncol = dim(Tenbars_optimum)[2])
Tenbars_var_max<-apply(DTenbars,2,max)
Tenbars_var_min<-apply(DTenbars,2,min)
Tenbars_var_range<-Tenbars_var_max[-dim(Tenbars_optimum)[2]]-Tenbars_var_min[-dim(Tenbars_optimum)[2]]
for (i in 1:dim(Tenbars_optimum)[1]){
  Tenbars_var[i,-dim(Tenbars_optimum)[2]]<-Tenbars_optimum[i,-dim(Tenbars_optimum)[2]]*Tenbars_var_range+Tenbars_var_min[-dim(Tenbars_optimum)[2]]
}
Tenbars_var[,dim(Tenbars_optimum)[2]]<-Tenbars_optimum[,dim(Tenbars_optimum)[2]]

#Torsion Bars real optimum
TorsionB_optimum_R<-TorsionB_optimum[,dim(TorsionB_optimum)[2]]*Est_torsionb_scale+Est_torsionb_min
TorsionB_var<-matrix(data = NA,nrow = dim(TorsionB_optimum)[1],ncol = dim(TorsionB_optimum)[2])
TorsionB_var_max<-apply(DTorsionB,2,max)
TorsionB_var_min<-apply(DTorsionB,2,min)
TorsionB_var_range<-TorsionB_var_max-TorsionB_var_min
for (i in 1:dim(TorsionB_optimum)[1]){
  TorsionB_var[i,]<-TorsionB_optimum[i,]*TorsionB_var_range+TorsionB_var_min
}
##present the optimium data
print(Tube_var)
print(SbeamCAD_var)
print(Tenbars_var)
print(TorsionB_var)
colnames(Tube_var)<-colnames(DTube)
colnames(SbeamCAD_var)<-colnames(DSbeamCAD)
colnames(Tenbars_var)<-colnames(DTenbars)
colnames(TorsionB_var)<-colnames(DTorsionB )
rownames(Tube_var)<-c('GPR','SVM','RFR','mxnet')
rownames(SbeamCAD_var)<-c('GPR','SVM','RFR','mxnet')
rownames(Tenbars_var)<-c('GPR','SVM','RFR','mxnet')
rownames(TorsionB_var)<-c('GPR','SVM','RFR','mxnet')
##write out the data
library(xlsx)
write.xlsx(Tube_var, file = "C:/Users/DUX1/Desktop/PHD Project/3rd Deep learning/3.1 Three other machine learning R/Hypereffectstudy/Fourexample_4MLs_optimum.xlsx",
           sheetName="Tube_optimum_4MLs", col.names=TRUE, row.names=TRUE, append=FALSE, showNA=TRUE, password=NULL)
write.xlsx(SbeamCAD_var, file = "C:/Users/DUX1/Desktop/PHD Project/3rd Deep learning/3.1 Three other machine learning R/Hypereffectstudy/Fourexample_4MLs_optimum.xlsx",
           sheetName="SbeamCAD_optimum_4MLs", col.names=TRUE, row.names=TRUE, append=TRUE, showNA=TRUE, password=NULL)
write.xlsx(Tenbars_var, file = "C:/Users/DUX1/Desktop/PHD Project/3rd Deep learning/3.1 Three other machine learning R/Hypereffectstudy/Fourexample_4MLs_optimum.xlsx",
           sheetName="Tenbars_optimum_4MLs", col.names=TRUE, row.names=TRUE, append=TRUE, showNA=TRUE, password=NULL)
write.xlsx(TorsionB_var, file = "C:/Users/DUX1/Desktop/PHD Project/3rd Deep learning/3.1 Three other machine learning R/Hypereffectstudy/Fourexample_4MLs_optimum.xlsx",
           sheetName="TorsionB_optimum_4MLs", col.names=TRUE, row.names=TRUE, append=TRUE, showNA=TRUE, password=NULL)
