####for this part, the structures and hyperparamters of ANN model are explored on their
##working performance on the four examples
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

##random sampling the int for train set
ind<-1:1000
train.set<-sample.int(1000,800)        # No. of samples in traning set
test.set<-ind[-train.set] # No. of samples in test set
##Make tasks
tasktube = makeRegrTask(data=NDTube,target=objtube)  #define a regression task
tasksbeam = makeRegrTask(data=NDSbeamCAD,target=objsbeam)  #define a regression task
tasktenbars = makeRegrTask(data=NDTenbars,target=objtenbars)  #define a regression task
tasktorsionb = makeRegrTask(data=NDTorsionB,target=objtorsionb)  #define a regression task
##make learners
#ps_GPR = makeParamSet (makeDiscreteParam(id="kernel",values=c("rbfdot","polydot","tanhdot","laplacedot")),
                       #makeNumericParam (id="sigma",lower=0,upper=10,requires=quote(kernel=="rbfdot"|kernel=="laplacedot")),
                       #makeIntegerParam(id="degree",lower=1L,upper=10L,requires=quote(kernel=="polydot")),
                       #makeNumericParam(id="scale",lower=0,upper=10,requires=quote(kernel=="polydot"|kernel=="tanhdot")),
                       #makeNumericParam(id="offset",lower=-10,upper=10,requires=quote(kernel=="polydot"|kernel=="tanhdot")))  ##trafo=function (x)2^x
#GPR_Regression_Par<-makeLearner("regr.gausspr",par.vals=list(kernel="polydot",degree=3,scale=7.673433,offset=9.301306))

knels<-c("rbfdot","polydot","tanhdot","laplacedot")
#for the four kernels
#for rbfdot
GPR_SbeamCAD_ParRMSE_rbfdotsg<-array(data=NA,dim = 11)
GPR_Tube_ParRMSE_rbfdotsg<-array(data=NA,dim = 11)
GPR_Tenbars_ParRMSE_rbfdotsg<-array(data=NA,dim = 11)
GPR_TorsionB_ParRMSE_rbfdotsg<-array(data=NA,dim = 11)
#for polynomial 21rows+11cols+10tables
GPR_SbeamCAD_ParRMSE_poly3<-array(dim=c(21,11,10))
GPR_Tube_ParRMSE_poly3<-array(dim=c(21,11,10))
GPR_Tenbars_ParRMSE_poly3<-array(dim=c(21,11,10))
GPR_TorsionB_ParRMSE_poly3<-array(dim=c(21,11,10))
#for tanh 21rows+11cols
GPR_SbeamCAD_ParRMSE_tanh2<-matrix(nrow = 21,ncol = 11)
GPR_Tube_ParRMSE_tanh2<-matrix(nrow = 21,ncol = 11)
GPR_Tenbars_ParRMSE_tanh2<-matrix(nrow = 21,ncol = 11)
GPR_TorsionB_ParRMSE_tanh2<-matrix(nrow = 21,ncol = 11)
#for the laplace
GPR_SbeamCAD_ParRMSE_lapsg<-array(data=NA,dim = 11)
GPR_Tube_ParRMSE_lapsg<-array(data=NA,dim = 11)
GPR_Tenbars_ParRMSE_lapsg<-array(data=NA,dim = 11)
GPR_TorsionB_ParRMSE_lapsg<-array(data=NA,dim = 11)

for (i in 1:4) {
  GPR_Regression_Par<-makeLearner("regr.gausspr",par.vals=list(kernel=knels[i]))
  if (knels[i]=="rbfdot") {
    for (j in 1:11) {
      ##add hyperparameters
      GPR_Regression_Par<-setHyperPars(GPR_Regression_Par,par.vals=list(sigma=j-1))
      ##model training
      GPR_SbeamCAD_Par <- train(GPR_Regression_Par, tasksbeam, subset = train.set)
      GPR_SbeamCAD_Parpre <- predict(GPR_SbeamCAD_Par, tasksbeam, subset = test.set)
      GPR_SbeamCAD_Parpre<-as.data.frame(GPR_SbeamCAD_Parpre)
      GPR_SbeamCAD_ParRMSE_rbfdotsg[j]<-sqrt(mean((GPR_SbeamCAD_Parpre[,2] - GPR_SbeamCAD_Parpre[,3])^2))
      
      GPR_Tube_Par <- train(GPR_Regression_Par, tasktube, subset = train.set)
      GPR_Tube_Parpre <- predict(GPR_Tube_Par, tasktube, subset = test.set)
      GPR_Tube_Parpre<-as.data.frame(GPR_Tube_Parpre)
      GPR_Tube_ParRMSE_rbfdotsg[j]<-sqrt(mean((GPR_Tube_Parpre[,2] - GPR_Tube_Parpre[,3])^2))
      
      GPR_Tenbars_Par <- train(GPR_Regression_Par, tasktenbars, subset = train.set)
      GPR_Tenbars_Parpre <- predict(GPR_Tenbars_Par, tasktenbars, subset = test.set)
      GPR_Tenbars_Parpre<-as.data.frame(GPR_Tenbars_Parpre)
      GPR_Tenbars_ParRMSE_rbfdotsg[j]<-sqrt(mean((GPR_Tenbars_Parpre[,2] - GPR_Tenbars_Parpre[,3])^2))
      
      GPR_TorsionB_Par <- train(GPR_Regression_Par, tasktorsionb, subset = train.set)
      GPR_TorsionB_Parpre <- predict(GPR_TorsionB_Par, tasktorsionb, subset = test.set)
      GPR_TorsionB_Parpre<-as.data.frame(GPR_TorsionB_Parpre)
      GPR_TorsionB_ParRMSE_rbfdotsg[j]<-sqrt(mean((GPR_TorsionB_Parpre[,2] - GPR_TorsionB_Parpre[,3])^2))
                                       
    }
    
  }
  else if (knels[i]=="polydot"){
    for (deg in 1:10) {
      for (scl in 0:10) {
        for (ofset in -10:10) {
          ##add hyperparameters
          GPR_Regression_Par<-setHyperPars(GPR_Regression_Par,par.vals=list(degree=deg,scale=scl,offset=ofset))
          ##model training
          GPR_SbeamCAD_Par <- train(GPR_Regression_Par, tasksbeam, subset = train.set)
          GPR_SbeamCAD_Parpre <- predict(GPR_SbeamCAD_Par, tasksbeam, subset = test.set)
          GPR_SbeamCAD_Parpre<-as.data.frame(GPR_SbeamCAD_Parpre)
          GPR_SbeamCAD_ParRMSE_poly3[ofset+11,scl+1,deg]<-sqrt(mean((GPR_SbeamCAD_Parpre[,2] - GPR_SbeamCAD_Parpre[,3])^2))
          
          GPR_Tube_Par <- train(GPR_Regression_Par, tasktube, subset = train.set)
          GPR_Tube_Parpre <- predict(GPR_Tube_Par, tasktube, subset = test.set)
          GPR_Tube_Parpre<-as.data.frame(GPR_Tube_Parpre)
          GPR_Tube_ParRMSE_poly3[ofset+11,scl+1,deg]<-sqrt(mean((GPR_Tube_Parpre[,2] - GPR_Tube_Parpre[,3])^2))
          
          GPR_Tenbars_Par <- train(GPR_Regression_Par, tasktenbars, subset = train.set)
          GPR_Tenbars_Parpre <- predict(GPR_Tenbars_Par, tasktenbars, subset = test.set)
          GPR_Tenbars_Parpre<-as.data.frame(GPR_Tenbars_Parpre)
          GPR_Tenbars_ParRMSE_poly3[ofset+11,scl+1,deg]<-sqrt(mean((GPR_Tenbars_Parpre[,2] - GPR_Tenbars_Parpre[,3])^2))
          
          GPR_TorsionB_Par <- train(GPR_Regression_Par, tasktorsionb, subset = train.set)
          GPR_TorsionB_Parpre <- predict(GPR_TorsionB_Par, tasktorsionb, subset = test.set)
          GPR_TorsionB_Parpre<-as.data.frame(GPR_TorsionB_Parpre)
          GPR_TorsionB_ParRMSE_poly3[ofset+11,scl+1,deg]<-sqrt(mean((GPR_TorsionB_Parpre[,2] - GPR_TorsionB_Parpre[,3])^2))
        }
        
      }
      
    }
  }
  else if (knels[i]=="tanhdot") {
    for (scl in 0:10) {
      for (ofset in -10:10) {
        ##add hyperparameters
        GPR_Regression_Par<-setHyperPars(GPR_Regression_Par,par.vals=list(scale=scl,offset=ofset))
        ##model training
        GPR_SbeamCAD_Par <- train(GPR_Regression_Par, tasksbeam, subset = train.set)
        GPR_SbeamCAD_Parpre <- predict(GPR_SbeamCAD_Par, tasksbeam, subset = test.set)
        GPR_SbeamCAD_Parpre<-as.data.frame(GPR_SbeamCAD_Parpre)
        GPR_SbeamCAD_ParRMSE_tanh2[ofset+11,scl+1]<-sqrt(mean((GPR_SbeamCAD_Parpre[,2] - GPR_SbeamCAD_Parpre[,3])^2))
        
        GPR_Tube_Par <- train(GPR_Regression_Par, tasktube, subset = train.set)
        GPR_Tube_Parpre <- predict(GPR_Tube_Par, tasktube, subset = test.set)
        GPR_Tube_Parpre<-as.data.frame(GPR_Tube_Parpre)
        GPR_Tube_ParRMSE_tanh2[ofset+11,scl+1]<-sqrt(mean((GPR_Tube_Parpre[,2] - GPR_Tube_Parpre[,3])^2))
        
        GPR_Tenbars_Par <- train(GPR_Regression_Par, tasktenbars, subset = train.set)
        GPR_Tenbars_Parpre <- predict(GPR_Tenbars_Par, tasktenbars, subset = test.set)
        GPR_Tenbars_Parpre<-as.data.frame(GPR_Tenbars_Parpre)
        GPR_Tenbars_ParRMSE_tanh2[ofset+11,scl+1]<-sqrt(mean((GPR_Tenbars_Parpre[,2] - GPR_Tenbars_Parpre[,3])^2))
        
        GPR_TorsionB_Par <- train(GPR_Regression_Par, tasktorsionb, subset = train.set)
        GPR_TorsionB_Parpre <- predict(GPR_TorsionB_Par, tasktorsionb, subset = test.set)
        GPR_TorsionB_Parpre<-as.data.frame(GPR_TorsionB_Parpre)
        GPR_TorsionB_ParRMSE_tanh2[ofset+11,scl+1]<-sqrt(mean((GPR_TorsionB_Parpre[,2] - GPR_TorsionB_Parpre[,3])^2))
        
      }
      
    }
    
  }
  else if (knels[i]=="laplacedot") {
    for (j in 0:10){
      ##add hyperparameters
      GPR_Regression_Par<-setHyperPars(GPR_Regression_Par,par.vals=list(sigma=j))
      ##model training
      GPR_SbeamCAD_Par <- train(GPR_Regression_Par, tasksbeam, subset = train.set)
      GPR_SbeamCAD_Parpre <- predict(GPR_SbeamCAD_Par, tasksbeam, subset = test.set)
      GPR_SbeamCAD_Parpre<-as.data.frame(GPR_SbeamCAD_Parpre)
      GPR_SbeamCAD_ParRMSE_lapsg[j]<-sqrt(mean((GPR_SbeamCAD_Parpre[,2] - GPR_SbeamCAD_Parpre[,3])^2))
      
      GPR_Tube_Par <- train(GPR_Regression_Par, tasktube, subset = train.set)
      GPR_Tube_Parpre <- predict(GPR_Tube_Par, tasktube, subset = test.set)
      GPR_Tube_Parpre<-as.data.frame(GPR_Tube_Parpre)
      GPR_Tube_ParRMSE_lapsg[j]<-sqrt(mean((GPR_Tube_Parpre[,2] - GPR_Tube_Parpre[,3])^2))
      
      GPR_Tenbars_Par <- train(GPR_Regression_Par, tasktenbars, subset = train.set)
      GPR_Tenbars_Parpre <- predict(GPR_Tenbars_Par, tasktenbars, subset = test.set)
      GPR_Tenbars_Parpre<-as.data.frame(GPR_Tenbars_Parpre)
      GPR_Tenbars_ParRMSE_lapsg[j]<-sqrt(mean((GPR_Tenbars_Parpre[,2] - GPR_Tenbars_Parpre[,3])^2))
      
      GPR_TorsionB_Par <- train(GPR_Regression_Par, tasktorsionb, subset = train.set)
      GPR_TorsionB_Parpre <- predict(GPR_TorsionB_Par, tasktorsionb, subset = test.set)
      GPR_TorsionB_Parpre<-as.data.frame(GPR_TorsionB_Parpre)
      GPR_TorsionB_ParRMSE_lapsg[j]<-sqrt(mean((GPR_TorsionB_Parpre[,2] - GPR_TorsionB_Parpre[,3])^2))
    }
  }
}

####For the rbfdot and laplace
par(mfrow=c(1,2))
plot(0:10,GPR_Tenbars_ParRMSE_rbfdotsg,"o",lty="solid",col="black",main="(a) RBF dot kernel",xlim=c(-1,11),ylim=c(0,0.3),axes=TRUE, ann=TRUE,xlab="Sigma",ylab="Loss (RMSE)",cex.lab=1.2,lwd=2)
lines(0:10,GPR_TorsionB_ParRMSE_rbfdotsg,"o",lty="solid", col="blue",lwd=2)
lines(0:10,GPR_SbeamCAD_ParRMSE_rbfdotsg,"o",lty="solid", col="red",lwd=2)
lines(0:10,GPR_Tube_ParRMSE_rbfdotsg,"o",lty="solid", col="green",lwd=2)
legend(7, 0.085, legend=c('TbPT','TqA','ShB','OMcT'), cex=1.1, col=c("black","blue","red","green"),lty=1,lwd=2)

plot(0:10,GPR_Tenbars_ParRMSE_lapsg,"o",lty="solid",col="black",main="(b) Laplacian kernel",xlim=c(-1,11),ylim=c(0,0.3),axes=TRUE, ann=TRUE,xlab="Sigma",ylab="Loss (RMSE)",cex.lab=1.2,lwd=2)
lines(0:10,GPR_TorsionB_ParRMSE_lapsg,"o",lty="solid", col="blue",lwd=2)
lines(0:10,GPR_SbeamCAD_ParRMSE_lapsg,"o",lty="solid", col="red",lwd=2)
lines(0:10,GPR_Tube_ParRMSE_lapsg,"o",lty="solid", col="green",lwd=2)
legend(7,0.085, legend=c('TbPT','TqA','ShB','OMcT'),cex=1.1, col=c("black","blue","red","green"),lty=1,lwd=2)

#For the Polynomial
GPR_SbeamCAD_ParRMSE_poly3_degM<-array(dim=10)
GPR_SbeamCAD_ParRMSE_poly3_degsd<-array(dim=10)
for (i in 1:10) {
  GPR_SbeamCAD_ParRMSE_poly3_degM[i]<-mean(GPR_SbeamCAD_ParRMSE_poly3[,,i])
  GPR_SbeamCAD_ParRMSE_poly3_degsd[i]<-sd(GPR_SbeamCAD_ParRMSE_poly3[,,i])
}

GPR_Tube_ParRMSE_poly3_degM<-array(dim=10)
GPR_Tube_ParRMSE_poly3_degsd<-array(dim=10)
for (i in 1:10) {
  GPR_Tube_ParRMSE_poly3_degM[i]<-mean(GPR_Tube_ParRMSE_poly3[,,i])
  GPR_Tube_ParRMSE_poly3_degsd[i]<-sd(GPR_Tube_ParRMSE_poly3[,,i])
}

GPR_Tenbars_ParRMSE_poly3_degM<-array(dim=10)
GPR_Tenbars_ParRMSE_poly3_degsd<-array(dim=10)
for (i in 1:10) {
  GPR_Tenbars_ParRMSE_poly3_degM[i]<-mean(GPR_Tenbars_ParRMSE_poly3[,,i])
  GPR_Tenbars_ParRMSE_poly3_degsd[i]<-sd(GPR_Tenbars_ParRMSE_poly3[,,i])
}

GPR_TorsionB_ParRMSE_poly3_degM<-array(dim=10)
GPR_TorsionB_ParRMSE_poly3_degsd<-array(dim=10)
for (i in 1:10) {
  GPR_TorsionB_ParRMSE_poly3_degM[i]<-mean(GPR_TorsionB_ParRMSE_poly3[,,i])
  GPR_TorsionB_ParRMSE_poly3_degsd[i]<-sd(GPR_TorsionB_ParRMSE_poly3[,,i])
}

library(Hmisc)
par(mfrow=c(1,2))
plot(1:10,GPR_Tenbars_ParRMSE_poly3_degM,"o",lty="solid",col="black",xlim=c(1,10),ylim=c(-1000,2000),axes=TRUE, ann=TRUE,xlab="Degree",ylab="Loss (RMSE)",mgp=c(1.8,0.6,0),cex.lab=1.2,lwd=2)
title("(a) Polynomial degree effect",line=1)
lines(1:10,GPR_TorsionB_ParRMSE_poly3_degM,"o",lty="solid", col="blue",lwd=2)
lines(1:10,GPR_SbeamCAD_ParRMSE_poly3_degM,"o",lty="solid", col="red",lwd=2)
lines(1:10,GPR_Tube_ParRMSE_poly3_degM,"o",lty="solid", col="green",lwd=2)
errbar( 1:10, GPR_SbeamCAD_ParRMSE_poly3_degM, errbar.col="red",lwd=1.8,add=TRUE,(GPR_SbeamCAD_ParRMSE_poly3_degM + GPR_SbeamCAD_ParRMSE_poly3_degsd), (GPR_SbeamCAD_ParRMSE_poly3_degM - GPR_SbeamCAD_ParRMSE_poly3_degsd))
errbar( 1:10, GPR_Tube_ParRMSE_poly3_degM, errbar.col="green",lwd=1.8,add=TRUE,GPR_Tube_ParRMSE_poly3_degM + GPR_Tube_ParRMSE_poly3_degsd, GPR_Tube_ParRMSE_poly3_degM - GPR_Tube_ParRMSE_poly3_degsd)
errbar( 1:10, GPR_Tenbars_ParRMSE_poly3_degM,errbar.col="black",lwd=1.8,add=TRUE, GPR_Tenbars_ParRMSE_poly3_degM + GPR_Tenbars_ParRMSE_poly3_degsd, GPR_Tenbars_ParRMSE_poly3_degM - GPR_Tenbars_ParRMSE_poly3_degsd)
errbar( 1:10, GPR_TorsionB_ParRMSE_poly3_degM, errbar.col="blue",lwd=1.8,add=TRUE,GPR_TorsionB_ParRMSE_poly3_degM + GPR_TorsionB_ParRMSE_poly3_degsd, GPR_TorsionB_ParRMSE_poly3_degM - GPR_TorsionB_ParRMSE_poly3_degsd)
legend(1, 2000, legend=c("TbPT","TqA","ShB","OMcT"), cex=1, col=c("black","blue","red","green"),lty=1,lwd=2)
##detailed
plot(1:4,GPR_SbeamCAD_ParRMSE_poly3_degM[1:4],"o",lty="solid",col="red",xlim=c(1,4),ylim=c(0,0.3),axes=TRUE, ann=TRUE,xlab="Degree",ylab="Loss (RMSE)",mgp=c(1.8,0.6,0),cex.lab=1.2,lwd=2)
title("(b) Local detailed view",line=1)
errbar( 1:4, GPR_SbeamCAD_ParRMSE_poly3_degM[1:4],errbar.col="red",lwd=1.8,add=TRUE,yplus=GPR_SbeamCAD_ParRMSE_poly3_degM[1:4] + GPR_SbeamCAD_ParRMSE_poly3_degsd[1:4], yminus=GPR_SbeamCAD_ParRMSE_poly3_degM[1:4] - GPR_SbeamCAD_ParRMSE_poly3_degsd[1:4])
lines(1:4,GPR_Tube_ParRMSE_poly3_degM[1:4],"o",lty="solid", col="green",lwd=2)
errbar( 1:4, GPR_Tube_ParRMSE_poly3_degM[1:4], errbar.col="green",lwd=1.8,add=TRUE,yplus=GPR_Tube_ParRMSE_poly3_degM[1:4] + GPR_Tube_ParRMSE_poly3_degsd[1:4], yminus=GPR_Tube_ParRMSE_poly3_degM[1:4] - GPR_Tube_ParRMSE_poly3_degsd[1:4])
lines(1:4,GPR_Tenbars_ParRMSE_poly3_degM[1:4],"o",lty="solid", col="black",lwd=2)
errbar( 1:3, GPR_Tenbars_ParRMSE_poly3_degM[1:3], errbar.col="black",lwd=1.8,add=TRUE,yplus=GPR_Tenbars_ParRMSE_poly3_degM[1:3] + GPR_Tenbars_ParRMSE_poly3_degsd[1:3], yminus=GPR_Tenbars_ParRMSE_poly3_degM[1:3] - GPR_Tenbars_ParRMSE_poly3_degsd[1:3])
lines(1:4,GPR_TorsionB_ParRMSE_poly3_degM[1:4],"o",lty="solid", col="blue",lwd=2)
errbar( 1:3, GPR_TorsionB_ParRMSE_poly3_degM[1:3], errbar.col="blue",lwd=1.8,add=TRUE,yplus=GPR_TorsionB_ParRMSE_poly3_degM[1:3] + GPR_TorsionB_ParRMSE_poly3_degsd[1:3], yminus=GPR_TorsionB_ParRMSE_poly3_degM[1:3] - GPR_TorsionB_ParRMSE_poly3_degsd[1:3])
#mtext("Local deteil view",par(ylbias=0.2))
#legend(0, 0.4, legend=c("S-beam","Tube","Ten bars","Torsion bar"), cex=1, col=c("black","blue","red","green"),lty=1,lwd=2)
library(plotly)
library(listviewer)

scale=0:10
offset=-10:10
plot_ly(x=~scale ,y=~offset,z=GPR_SbeamCAD_ParRMSE_poly3[,,3],colorscale='Jet',type = "contour",contours = list(start = 0.04,end = 0.08,size = 0.005))%>%
  colorbar(title="Loss",len=1,color=I('black'),colorscale='red',thickness=20,tickfont=list(size=16),titlefont=list(size=18))%>%layout(title="(c) ShB",titlefont=list(size=18),xaxis =list(title="Scale",titlefont=list(size=18),tickfont=list(size=16)), yaxis = list(title="Offset",titlefont=list(size=18),tickfont=list(size=16)))
plot_ly(x=~scale ,y=~offset,z=GPR_Tube_ParRMSE_poly3[,,3],colorscale='Jet',type = "contour",contours = list(start = 0.04,end = 0.08,size = 0.005))%>%
  colorbar(title="Loss",len=1,color=I('black'),colorscale='red',thickness=20,tickfont=list(size=16),titlefont=list(size=18))%>%layout(title="(d) OMcT",titlefont=list(size=18),xaxis =list(title="Scale",titlefont=list(size=18),tickfont=list(size=16)), yaxis = list(title="Offset",titlefont=list(size=18),tickfont=list(size=16)))
plot_ly(x=~scale ,y=~offset,z=GPR_Tenbars_ParRMSE_poly3[,,2],colorscale='Jet',type = "contour",contours = list(start = 0.04,end = 0.08,size = 0.005))%>%
  colorbar(title="Loss",len=1,color=I('black'),colorscale='red',thickness=20,tickfont=list(size=16),titlefont=list(size=18))%>%layout(title="(a) TbPT",titlefont=list(size=18),xaxis =list(title="Scale",titlefont=list(size=18),tickfont=list(size=16)), yaxis = list(title="Offset",titlefont=list(size=18),tickfont=list(size=16)))
plot_ly(x=~scale ,y=~offset,z=GPR_TorsionB_ParRMSE_poly3[,,2],colorscale='Jet',type = "contour",contours = list(start = 0.04,end = 0.08,size = 0.005))%>%
  colorbar(title="Loss",len=1,color=I('black'),colorscale='red',thickness=20,tickfont=list(size=16),titlefont=list(size=18))%>%layout(title="(b) TqA",titlefont=list(size=18),xaxis =list(title="Scale",titlefont=list(size=18),tickfont=list(size=16)), yaxis = list(title="Offset",titlefont=list(size=18),tickfont=list(size=16)))
##for tanh
scale=0:5
offset=0:10
plot_ly(x=~scale ,y=~offset,z=GPR_SbeamCAD_ParRMSE_tanh2[11:21,1:6],colorscale='Jet',type = "contour",contours = list(start = 0.04,end = 0.2,size = 0.01))%>%
  colorbar(title="Loss",len=1,color=I('black'),colorscale='red',thickness=20,tickfont=list(size=16),titlefont=list(size=18))%>%layout(title="(c) ShB",titlefont=list(size=18),xaxis =list(title="Scale",titlefont=list(size=18),tickfont=list(size=16)), yaxis = list(title="Offset",titlefont=list(size=18),tickfont=list(size=16)))
plot_ly(x=~scale ,y=~offset,z=GPR_Tube_ParRMSE_tanh2[11:21,1:6],colorscale='Jet',type = "contour",contours = list(start = 0.04,end = 0.2,size = 0.01))%>%
  colorbar(title="Loss",len=1,color=I('black'),colorscale='red',thickness=20,tickfont=list(size=16),titlefont=list(size=18))%>%layout(title="(d) OMcT",titlefont=list(size=18),xaxis =list(title="Scale",titlefont=list(size=18),tickfont=list(size=16)), yaxis = list(title="Offset",titlefont=list(size=18),tickfont=list(size=16)))
plot_ly(x=~scale ,y=~offset,z=GPR_Tenbars_ParRMSE_tanh2[11:21,1:6],colorscale='Jet',type = "contour",contours = list(start = 0.04,end = 0.2,size = 0.01))%>%
  colorbar(title="Loss",len=1,color=I('black'),colorscale='red',thickness=20,tickfont=list(size=16),titlefont=list(size=18))%>%layout(title="(a) TbPT",titlefont=list(size=18),xaxis =list(title="Scale",titlefont=list(size=18),tickfont=list(size=16)), yaxis = list(title="Offset",titlefont=list(size=18),tickfont=list(size=16)))
plot_ly(x=~scale ,y=~offset,z=GPR_TorsionB_ParRMSE_tanh2[11:21,1:6],colorscale='Jet',type = "contour",contours = list(start = 0.04,end = 0.2,size = 0.01))%>%
  colorbar(title="Loss",len=1,color=I('black'),colorscale='red',thickness=20,tickfont=list(size=16),titlefont=list(size=18))%>%layout(title="(b) TqA",titlefont=list(size=18),xaxis =list(title="Scale",titlefont=list(size=18),tickfont=list(size=16)), yaxis = list(title="Offset",titlefont=list(size=18),tickfont=list(size=16)))
##data
GPR_SbeamCAD_ParRMSE_rbfdotsg
GPR_Tube_ParRMSE_rbfdotsg
GPR_Tenbars_ParRMSE_rbfdotsg
GPR_TorsionB_ParRMSE_rbfdotsg
#for polynomial 21rows+11cols+10tables
GPR_SbeamCAD_ParRMSE_poly3
GPR_Tube_ParRMSE_poly3
GPR_Tenbars_ParRMSE_poly3
GPR_TorsionB_ParRMSE_poly3
#for tanh 21rows+11cols
GPR_SbeamCAD_ParRMSE_tanh2
GPR_Tube_ParRMSE_tanh2
GPR_Tenbars_ParRMSE_tanh2
GPR_TorsionB_ParRMSE_tanh2
#for the laplace
GPR_SbeamCAD_ParRMSE_lapsg
GPR_Tube_ParRMSE_lapsg
GPR_Tenbars_ParRMSE_lapsg
GPR_TorsionB_ParRMSE_lapsg

