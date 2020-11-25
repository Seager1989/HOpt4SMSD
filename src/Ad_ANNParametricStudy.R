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
sampling<-sample.int(1000,800)

##Dataset construction
NDTube_NN_train.x=data.matrix(NDTube[sampling,-dim(NDTube)[2]])
NDTube_NN_train.y=NDTube[sampling,dim(NDTube)[2]]
NDTube_NN_test.x=data.matrix(NDTube[-sampling,-dim(NDTube)[2]])
NDTube_NN_test.y=NDTube[-sampling,dim(NDTube)[2]]

NDSbeamCAD_NN_train.x=data.matrix(NDSbeamCAD[sampling,-dim(NDSbeamCAD)[2]])
NDSbeamCAD_NN_train.y=NDSbeamCAD[sampling,dim(NDSbeamCAD)[2]]
NDSbeamCAD_NN_test.x=data.matrix(NDSbeamCAD[-sampling,-dim(NDSbeamCAD)[2]])
NDSbeamCAD_NN_test.y=NDSbeamCAD[-sampling,dim(NDSbeamCAD)[2]]

NDTenbars_NN_train.x=data.matrix(NDTenbars[sampling,-dim(NDTenbars)[2]])
NDTenbars_NN_train.y=NDTenbars[sampling,dim(NDTenbars)[2]]
NDTenbars_NN_test.x=data.matrix(NDTenbars[-sampling,-dim(NDTenbars)[2]])
NDTenbars_NN_test.y=NDTenbars[-sampling,dim(NDTenbars)[2]]

NDTorsionB_NN_train.x=data.matrix(NDTorsionB[sampling,-dim(NDTorsionB)[2]])
NDTorsionB_NN_train.y=NDTorsionB[sampling,dim(NDTorsionB)[2]]
NDTorsionB_NN_test.x=data.matrix(NDTorsionB[-sampling,-dim(NDTorsionB)[2]])
NDTorsionB_NN_test.y=NDTorsionB[-sampling,dim(NDTorsionB)[2]]

###for the nerons and iteration study
#Tube_log_train<-matrix(nrow = 20000,ncol = 5)
#SbeamCAD_log_train<-matrix(nrow = 20000,ncol = 5)
#Tenbars_log_train<-matrix(nrow = 20000,ncol = 5)
#TorsionB_log_train<-matrix(nrow = 20000,ncol = 5)

#Tube_log_test<-matrix(nrow = 20000,ncol = 5)
#SbeamCAD_log_test<-matrix(nrow = 20000,ncol = 5)
#Tenbars_log_test<-matrix(nrow = 20000,ncol = 5)
#TorsionB_log_test<-matrix(nrow = 20000,ncol = 5)
j=1##
for (i in c(5)){
#####for the task of single layer neurons
# Define the structures:5-20-50-100-200-500
data = mx.symbol.Variable(name = 'data')
label = mx.symbol.Variable(name = 'label')
fc_layer_1 = mx.symbol.FullyConnected(data = data, num.hidden = i, name = 'fc_layer_1')
acti = mx.symbol.Activation(data=fc_layer_1,act.type='tanh',name='acti')
fc_layer_2  = mx.symbol.FullyConnected(data = acti, num.hidden = 1, name = 'fc_layer_2')
out_layer = mx.symbol.LinearRegressionOutput(data = fc_layer_2, label = label,name = 'out_layer')

#FFP
#rm(Tube_log)
Tube_log <- mx.metric.logger$new()  # error vector
Tube_FFN <- mx.model.FeedForward.create(out_layer, X=NDTube_NN_train.x, y=NDTube_NN_train.y,eval.data=list(data=NDTube_NN_test.x,label=NDTube_NN_test.y),
                                          ctx=mx.cpu(),num.round=20000,optimizer="sgd",array.batch.size=100,
                                          learning.rate=1e-1, momentum=0.9,eval.metric=mx.metric.rmse,
                                          epoch.end.callback = mx.callback.log.train.metric(period=8, Tube_log))
Tube_log_train<-cbind(as.matrix(Tube_log$train),Tube_log_train) #Tube_log_train[,j]<-as.matrix(Tube_log$train)
Tube_log_test<-cbind(as.matrix(Tube_log$eval),Tube_log_test) #Tube_log_test[,j]<-as.matrix(Tube_log$eval)
SbeamCAD_log <- mx.metric.logger$new()  # error vector
SbeamCAD_FFN <- mx.model.FeedForward.create(out_layer, X=NDSbeamCAD_NN_train.x, y=NDSbeamCAD_NN_train.y,eval.data=list(data=NDSbeamCAD_NN_test.x,label=NDSbeamCAD_NN_test.y),
                                        ctx=mx.cpu(),num.round=20000,optimizer="sgd",array.batch.size=100,
                                        learning.rate=1e-1, momentum=0.9,eval.metric=mx.metric.rmse,
                                        epoch.end.callback = mx.callback.log.train.metric(period=8, SbeamCAD_log))
SbeamCAD_log_train<-cbind(as.matrix(SbeamCAD_log$train),SbeamCAD_log_train) #SbeamCAD_log_train[,j]<-as.matrix(SbeamCAD_log$train)
SbeamCAD_log_test<-cbind(as.matrix(SbeamCAD_log$eval),SbeamCAD_log_test) #SbeamCAD_log_test[,j]<-as.matrix(SbeamCAD_log$eval)

Tenbars_log <- mx.metric.logger$new()  # error vector
Tenbars_FFN <- mx.model.FeedForward.create(out_layer, X=NDTenbars_NN_train.x, y=NDTenbars_NN_train.y,eval.data=list(data=NDTenbars_NN_test.x,label=NDTenbars_NN_test.y),
                                        ctx=mx.cpu(),num.round=20000,optimizer="sgd",array.batch.size=100,
                                        learning.rate=1e-1, momentum=0.9,eval.metric=mx.metric.rmse,
                                        epoch.end.callback = mx.callback.log.train.metric(period=8, Tenbars_log))
Tenbars_log_train<-cbind(as.matrix(Tenbars_log$train),Tenbars_log_train) ##Tenbars_log_train[,j]<-as.matrix(Tenbars_log$train)
Tenbars_log_test<-cbind(as.matrix(Tenbars_log$eval),Tenbars_log_test)   ##Tenbars_log_test[,j]<-as.matrix(Tenbars_log$eval)

TorsionB_log <- mx.metric.logger$new()  # error vector
TorsionB_FFN <- mx.model.FeedForward.create(out_layer, X=NDTorsionB_NN_train.x, y=NDTorsionB_NN_train.y,eval.data=list(data=NDTorsionB_NN_test.x,label=NDTorsionB_NN_test.y),
                                        ctx=mx.cpu(),num.round=20000,optimizer="sgd",array.batch.size=100,
                                        learning.rate=1e-1, momentum=0.9,eval.metric=mx.metric.rmse,
                                        epoch.end.callback = mx.callback.log.train.metric(period=8, TorsionB_log))
TorsionB_log_train<-cbind(as.matrix(TorsionB_log$train),TorsionB_log_train)  ##TorsionB_log_train[,j]<-as.matrix(TorsionB_log$train)
TorsionB_log_test<-cbind(as.matrix(TorsionB_log$eval),TorsionB_log_test)   ##TorsionB_log_test[,j]<-as.matrix(TorsionB_log$eval)
#j=j+1
}

##plot the loss curve for the train and test dataset
logrecord<-c("Tube_log","SbeamCAD_log","Tenbars_log","TorsionB_log")
par(mfrow=c(2,2))
#tiff
train_dataset<-c("SbeamCAD_log_train","Tube_log_train","Tenbars_log_train","TorsionB_log_train")
test_dataset<-c("SbeamCAD_log_test","Tube_log_test","Tenbars_log_test","TorsionB_log_test")
title<-c("(a) S-beam","(b) Tube","(c) Ten bars","(d) Torsion bar")
for (i in 1:4) {
plot(1:20000,get(test_dataset[i])[1:20000,1],"l",lty="solid",col="black",main=title[i],xlim=c(0,10000),ylim=c(0.02,0.12),axes=TRUE, ann=TRUE,xlab="Epoch",ylab="Loss (RMSE)",cex.lab=1.2,lwd=2)
#lines(1:20000,get(test_dataset[i])[1:20000,1],"l",lty="dashed", col="black",xlim=c(0,20000),ylim=c(0,0.15),lwd=2)
#lines(1:20000,get(train_dataset[i])[1:20000,2],"l",lty="solid",col="blue",main=title[i],xlim=c(0,20000),ylim=c(0.01,0.13),axes=TRUE, ann=TRUE,xlab="Epoch",ylab="Loss (RMSE)",cex.lab=1.2,lwd=2)
lines(1:20000,get(test_dataset[i])[1:20000,2],"l",lty="solid", col="blue",lwd=2)
#lines(1:20000,get(train_dataset[i])[1:20000,3],"l",lty="solid",col="red",xlim=c(0,20000),ylim=c(0,0.15),axes=TRUE, ann=TRUE,xlab="Epoch",ylab="Loss (RMSE)",cex.lab=1.2,lwd=2)
lines(1:20000,get(test_dataset[i])[1:20000,3],"l",lty="solid", col="red",lwd=2)
#lines(1:20000,get(train_dataset[i])[1:20000,4],"l",lty="solid",col="green",xlim=c(0,20000),ylim=c(0,0.15),axes=TRUE, ann=TRUE,xlab="Epoch",ylab="Loss (RMSE)",cex.lab=1.2,lwd=2)
lines(1:20000,get(test_dataset[i])[1:20000,4],"l",lty="solid", col="green",lwd=2)
#lines(1:20000,get(train_dataset[i])[1:20000,5],"l",lty="solid",col="orange",xlim=c(0,20000),ylim=c(0,0.15),axes=TRUE, ann=TRUE,xlab="Epoch",ylab="Loss (RMSE)",cex.lab=1.2,lwd=2)
lines(1:20000,get(test_dataset[i])[1:20000,5],"l",lty="solid", col="orange",lwd=2)
legend(6000, 0.12, c("5 hidden neurons","20 hidden neurons","50 hidden neurons","100 hidden neurons","500 hidden neurons"), 
       cex=1.2, col=c("black","blue","red","green","orange"),lty=1,lwd=2);
}

##for mini batch and learning rate
Sbeam_LR_error<-matrix(nrow = 5, ncol = 5)
Tube_LR_error<-matrix(nrow = 5, ncol = 5)
Tenbars_LR_error<-matrix(nrow = 5, ncol = 5)
TorsionB_LR_error<-matrix(nrow = 5, ncol = 5)
jj=1
ii=1
for (i in c(0.01,0.05,0.1,0.5,1)){
  jj=1
  for (j in c(10,20,50,100,200)) {
##10,20,50,100,200
  #####for the task of single layer neurons
  # Define the structures
  data = mx.symbol.Variable(name = 'data')
  label = mx.symbol.Variable(name = 'label')
  fc_layer_1 = mx.symbol.FullyConnected(data = data, num.hidden = 20, name = 'fc_layer_1')
  acti = mx.symbol.Activation(data=fc_layer_1,act.type='tanh',name='acti')
  fc_layer_2  = mx.symbol.FullyConnected(data = acti, num.hidden = 1, name = 'fc_layer_2')
  out_layer = mx.symbol.LinearRegressionOutput(data = fc_layer_2, label = label,name = 'out_layer')
  
  #FFP
  #rm(Tube_log)
  SbeamCAD_log <- mx.metric.logger$new()  # error vector
  SbeamCAD_FFN <- mx.model.FeedForward.create(out_layer, X=NDSbeamCAD_NN_train.x, y=NDSbeamCAD_NN_train.y,eval.data=list(data=NDSbeamCAD_NN_test.x,label=NDSbeamCAD_NN_test.y),
                                              ctx=mx.cpu(),num.round=10000,optimizer="sgd",array.batch.size=j,
                                              learning.rate=i, momentum=0.9,eval.metric=mx.metric.rmse,
                                              epoch.end.callback = mx.callback.log.train.metric(period=800/j, SbeamCAD_log))
  Sbeam_LR_error[jj,ii]=SbeamCAD_log$eval[10000]
  
  Tube_log <- mx.metric.logger$new()  # error vector
  Tube_FFN <- mx.model.FeedForward.create(out_layer, X=NDTube_NN_train.x, y=NDTube_NN_train.y,eval.data=list(data=NDTube_NN_test.x,label=NDTube_NN_test.y),
                                          ctx=mx.cpu(),num.round=10000,optimizer="sgd",array.batch.size=j,
                                          learning.rate=0.01, momentum=0.9,eval.metric=mx.metric.rmse,
                                          epoch.end.callback = mx.callback.log.train.metric(period=800/j, Tube_log))
  Tube_LR_error[jj,ii]=Tube_log$eval[10000]
  
  Tenbars_log <- mx.metric.logger$new()  # error vector
  Tenbars_FFN <- mx.model.FeedForward.create(out_layer, X=NDTenbars_NN_train.x, y=NDTenbars_NN_train.y,eval.data=list(data=NDTenbars_NN_test.x,label=NDTenbars_NN_test.y),
                                             ctx=mx.cpu(),num.round=10000,optimizer="sgd",array.batch.size=j,
                                             learning.rate=i, momentum=0.9,eval.metric=mx.metric.rmse,
                                             epoch.end.callback = mx.callback.log.train.metric(period=800/j, Tenbars_log))
  Tenbars_LR_error[jj,ii]=Tenbars_log$eval[10000]
  
  TorsionB_log <- mx.metric.logger$new()  # error vector
  TorsionB_FFN <- mx.model.FeedForward.create(out_layer, X=NDTorsionB_NN_train.x, y=NDTorsionB_NN_train.y,eval.data=list(data=NDTorsionB_NN_test.x,label=NDTorsionB_NN_test.y),
                                              ctx=mx.cpu(),num.round=10000,optimizer="sgd",array.batch.size=j,
                                              learning.rate=1, momentum=0.9,eval.metric=mx.metric.rmse,
                                              epoch.end.callback = mx.callback.log.train.metric(period=800/j, TorsionB_log))
  TorsionB_LR_error[jj,ii]=TorsionB_log$eval[10000]
  jj=jj+1
  }
  ii=ii+1
}
Sbeam_LR_error[,]
Tube_LR_error[,]
Tenbars_LR_error[,]
TorsionB_LR_error[,]
Learning_rate<-(c(0.01,0.05,0.1,0.5,1))
Batch_size<-c(10,20,50,100,200)
x= c(0.01,0.05,0.1,0.5,1)
y=c(10,20,50,100,200)
z=Sbeam_LR_error
library(plotly)
library(listviewer)
##single Sbeam ploty
plot_ly(x= ~Learning_rate,y=~Batch_size,z=Sbeam_LR_error,color=I('black'),colorscale='Jet',type = "contour",colorbar=list(title='Loss',titlefont=list(size=18)),contours = list(main='Loss',start = 0.04,end = 0.1,size = 0.005))%>%
  layout(title="S-beam",titlefont=list(size=18),xaxis =list(title="Learning rate",titlefont=list(size=18),tickfont=list(size=16)), yaxis = list(title="Batch size",type="log",titlefont=list(size=18),tickfont=list(size=16)))

##
plot_ly(x= ~Learning_rate,y=~Batch_size,z=Sbeam_LR_error,color=I('black'),colorscale='Jet',type = "contour",colorbar=list(title='Loss',titlefont=list(size=18)),contours = list(main='Loss',start = 0.04,end = 0.1,size = 0.005))%>%
  layout(title="(a) S-beam",titlefont=list(size=18),xaxis =list(title="Learning rate",type="log",titlefont=list(size=18),tickfont=list(size=16)), yaxis = list(title="Batch size",type="log",titlefont=list(size=18),tickfont=list(size=16)))
plot_ly(x= ~Learning_rate,y=~Batch_size,z=Tube_LR_error,color=I('black'),colorscale='Jet',type = "contour",colorbar=list(title='Loss',titlefont=list(size=18)),contours = list(main='Loss',start = 0.04,end = 0.1,size = 0.005))%>%
  layout(title="(b) Tube",titlefont=list(size=18),xaxis =list(title="Learning rate",type="log",titlefont=list(size=18),tickfont=list(size=16)), yaxis = list(title="Batch size",type="log",titlefont=list(size=18),tickfont=list(size=16)))
plot_ly(x= ~Learning_rate,y=~Batch_size,z=Tenbars_LR_error,color=I('black'),colorscale='Jet',type = "contour",colorbar=list(title='Loss',titlefont=list(size=18)),contours = list(main='Loss',start = 0.04,end = 0.1,size = 0.005))%>%
  layout(title="(c) Ten bars",titlefont=list(size=18),xaxis =list(title="Learning rate",type="log",titlefont=list(size=18),tickfont=list(size=16)), yaxis = list(title="Batch size",type="log",titlefont=list(size=18),tickfont=list(size=16)))
plot_ly(x= ~Learning_rate,y=~Batch_size,z=TorsionB_LR_error,color=I('black'),colorscale='Jet',type = "contour",colorbar=list(title='Loss',titlefont=list(size=18)),contours = list(main='Loss',start = 0.04,end = 0.1,size = 0.005))%>%
  layout(title="(d) Torsion bar",titlefont=list(size=18),xaxis =list(title="Learning rate",type="log",titlefont=list(size=18),tickfont=list(size=16)), yaxis = list(title="Batch size",type="log",titlefont=list(size=18),tickfont=list(size=16)))

plot_ly(x= ~Learning_rate,y=~Batch_size,z=Tube_LR_error,color=I('black'),colorscale='Jet',type = "contour",contours = list(start = 0.04,end = 0.1,size = 0.005))
plot_ly(x= ~Learning_rate,y=~Batch_size,z=Tenbars_LR_error,color=I('black'),colorscale='Jet',type = "contour",contours = list(start = 0.04,end = 0.1,size = 0.005))
plot_ly(x= ~Learning_rate,y=~Batch_size,z=TorsionB_LR_error,color=I('black'),colorscale='Jet',type = "contour",contours = list(start = 0.04,end = 0.1,size = 0.005))

###FOR the optimizer study with epoch-10000 and learning rate=0.1,minibatch=100 sufficient to accurate within 10000
optimizers=c("sgd","adagrad","adadelta","rmsprop","adam")
SbeamCADopt_log_train=matrix(nrow = 10000,ncol = 5)
SbeamCADopt_log_test=matrix(nrow = 10000,ncol = 5)
j=1
for (j in 2:5){
data = mx.symbol.Variable(name = 'data')
label = mx.symbol.Variable(name = 'label')
fc_layer_1 = mx.symbol.FullyConnected(data = data, num.hidden = 20, name = 'fc_layer_1')
acti = mx.symbol.Activation(data=fc_layer_1,act.type='tanh',name='acti')
fc_layer_2  = mx.symbol.FullyConnected(data = acti, num.hidden = 1, name = 'fc_layer_2')
out_layer = mx.symbol.LinearRegressionOutput(data = fc_layer_2, label = label,name = 'out_layer')

#FFP
rm(SbeamCAD_log)
SbeamCAD_log <- mx.metric.logger$new()  # error vector
SbeamCAD_FFN_optrSdy <- mx.model.FeedForward.create(out_layer, X=NDSbeamCAD_NN_train.x, y=NDSbeamCAD_NN_train.y,eval.data=list(data=NDSbeamCAD_NN_test.x,label=NDSbeamCAD_NN_test.y),
                                        ctx=mx.cpu(),num.round=10000,optimizer=optimizers[j],array.batch.size=100,
                                        eval.metric=mx.metric.rmse,
                                        epoch.end.callback = mx.callback.log.train.metric(period=8, SbeamCAD_log))#learning.rate=1e-1,momentum=0.9,
SbeamCADopt_log_train[,j]<-as.matrix(SbeamCAD_log$train)
SbeamCADopt_log_test[,j]<-as.matrix(SbeamCAD_log$eval)
}

#tiff
plot(1:10000,SbeamCADopt_log_test[,1],"l",lty="solid",col="black",xlim=c(0,10000),ylim=c(0.02,0.1),axes=TRUE, ann=TRUE,xlab="Epoch",ylab="Loss (RMSE)",cex.lab=1.2,lwd=2)
lines(1:10000,SbeamCADopt_log_test[,2],"l",lty="solid", col="blue",lwd=2)
lines(1:10000,SbeamCADopt_log_test[,3],"l",lty="solid",col="red",lwd=2)
lines(1:10000,SbeamCADopt_log_test[,4],"l",lty="solid", col="green",lwd=2)
lines(1:10000,SbeamCADopt_log_test[,5],"l",lty="solid",col="orange",lwd=2)
legend(8000, 0.1, c("SGD","Adagrad","Adadelta","Rmsprop","Adam"), 
         cex=1, col=c("black","blue","red","green","orange"),lty=1,lwd=2)

##For the activation function study:'relu','sigmoid','softrelu','tanh','
Activators=c("tanh","sigmoid","relu","softrelu")
SbeamCADacti_log_train=matrix(nrow = 10000,ncol = 4)
SbeamCADacti_log_test=matrix(nrow = 10000,ncol = 4)
j=3
for (j in 1:4){
  data = mx.symbol.Variable(name = 'data')
  label = mx.symbol.Variable(name = 'label')
  fc_layer_1 = mx.symbol.FullyConnected(data = data, num.hidden = 20, name = 'fc_layer_1')
  acti = mx.symbol.Activation(data=fc_layer_1,act.type=Activators[j],name='acti')
  fc_layer_2  = mx.symbol.FullyConnected(data = acti, num.hidden = 1, name = 'fc_layer_2')
  out_layer = mx.symbol.LinearRegressionOutput(data = fc_layer_2, label = label,name = 'out_layer')
  
  # only sgd with momentum (0.9 used) and adadelta no learning rate
  rm(SbeamCAD_log)
  SbeamCAD_log <- mx.metric.logger$new()  # error vector
  SbeamCAD_FFN_optrSdy <- mx.model.FeedForward.create(out_layer, X=NDSbeamCAD_NN_train.x, y=NDSbeamCAD_NN_train.y,eval.data=list(data=NDSbeamCAD_NN_test.x,label=NDSbeamCAD_NN_test.y),
                                                  ctx=mx.cpu(),num.round=10000,optimizer="adagrad",array.batch.size=100,
                                                  eval.metric=mx.metric.rmse,
                                                  epoch.end.callback = mx.callback.log.train.metric(period=8, SbeamCAD_log))#
  SbeamCADacti_log_train[,j]<-as.matrix(SbeamCAD_log$train)
  SbeamCADacti_log_test[,j]<-as.matrix(SbeamCAD_log$eval)
}

#plot
plot(1:10000,SbeamCADacti_log_test[1:10000,1],"l",lty="solid",col="black",xlim=c(0,10000),ylim=c(0.02,0.1),axes=TRUE, ann=TRUE,xlab="Epoch",ylab="Loss (RMSE)",cex.lab=1.2,lwd=2)
lines(1:10000,SbeamCADacti_log_test[1:10000,2],"l",lty="solid", col="blue",lwd=2)
lines(1:10000,SbeamCADacti_log_test[1:10000,3],"l",lty="solid",col="red",lwd=2)
lines(1:10000,SbeamCADacti_log_test[1:10000,4],"l",lty="solid", col="green",lwd=2)
legend(7000, 0.1, legend = c("Tanh","Sigmoid","Relu","Softrelu"), 
       cex=1.2, col=c("black","blue","red","green"),lty=1,lwd=2)

##structure study
##1:9-20-1
  data = mx.symbol.Variable(name = 'data')
  label = mx.symbol.Variable(name = 'label')
  fc_layer_11 = mx.symbol.FullyConnected(data = data, num.hidden = 20, name = 'fc_layer_11')
  acti11 = mx.symbol.Activation(data=fc_layer_11,act.type='tanh',name='acti11')
  fc_layer_21  = mx.symbol.FullyConnected(data = acti11, num.hidden = 1, name = 'fc_layer_21')
  out_layer11 = mx.symbol.LinearRegressionOutput(data = fc_layer_21, label = label,name = 'out_layer11')
##2:9-20-20-1
  data = mx.symbol.Variable(name = 'data')
  label = mx.symbol.Variable(name = 'label')
  fc_layer_12 = mx.symbol.FullyConnected(data = data, num.hidden = 20, name = 'fc_layer_12')
  acti12 = mx.symbol.Activation(data=fc_layer_12,act.type='tanh',name='acti12')
  fc_layer_22 = mx.symbol.FullyConnected(data = acti12, num.hidden = 20, name = 'fc_layer_22')
  acti22 = mx.symbol.Activation(data=fc_layer_22,act.type='tanh',name='acti22')
  fc_layer_32  = mx.symbol.FullyConnected(data = acti22, num.hidden = 1, name = 'fc_layer_32')
  out_layer22 = mx.symbol.LinearRegressionOutput(data = fc_layer_32, label = label,name = 'out_layer22')
##3:9-20-20-20-1
  data = mx.symbol.Variable(name = 'data')
  label = mx.symbol.Variable(name = 'label')
  fc_layer_13 = mx.symbol.FullyConnected(data = data, num.hidden = 20, name = 'fc_layer_13')
  acti13 = mx.symbol.Activation(data=fc_layer_13,act.type='tanh',name='acti13')
  fc_layer_23 = mx.symbol.FullyConnected(data = acti13, num.hidden = 20, name = 'fc_layer_23')
  acti23 = mx.symbol.Activation(data=fc_layer_23,act.type='tanh',name='acti23')
  fc_layer_33 = mx.symbol.FullyConnected(data = acti23, num.hidden = 20, name = 'fc_layer_33')
  acti33 = mx.symbol.Activation(data=fc_layer_33,act.type='tanh',name='acti33')
  fc_layer_43  = mx.symbol.FullyConnected(data = acti33, num.hidden = 1, name = 'fc_layer_43')
  out_layer33 = mx.symbol.LinearRegressionOutput(data = fc_layer_43, label = label,name = 'out_layer33')
##4:9-10-5-5-1
  data = mx.symbol.Variable(name = 'data')
  label = mx.symbol.Variable(name = 'label')
  fc_layer_14 = mx.symbol.FullyConnected(data = data, num.hidden = 10, name = 'fc_layer_14')
  acti14 = mx.symbol.Activation(data=fc_layer_14,act.type='tanh',name='acti14')
  fc_layer_24 = mx.symbol.FullyConnected(data = acti14, num.hidden = 5, name = 'fc_layer_24')
  acti24 = mx.symbol.Activation(data=fc_layer_24,act.type='tanh',name='acti24')
  fc_layer_34 = mx.symbol.FullyConnected(data = acti24, num.hidden = 5, name = 'fc_layer_34')
  acti34 = mx.symbol.Activation(data=fc_layer_34,act.type='tanh',name='acti34')
  fc_layer_44  = mx.symbol.FullyConnected(data = acti34, num.hidden = 1, name = 'fc_layer_44')
  out_layer44 = mx.symbol.LinearRegressionOutput(data = fc_layer_44, label = label,name = 'out_layer44')
##5:9-10-5-1
  data = mx.symbol.Variable(name = 'data')
  label = mx.symbol.Variable(name = 'label')
  fc_layer_15 = mx.symbol.FullyConnected(data = data, num.hidden = 10, name = 'fc_layer_15')
  acti15 = mx.symbol.Activation(data=fc_layer_15,act.type='tanh',name='acti15')
  fc_layer_25 = mx.symbol.FullyConnected(data = acti1, num.hidden = 5, name = 'fc_layer_25')
  acti25 = mx.symbol.Activation(data=fc_layer_25,act.type='tanh',name='acti25')
  fc_layer_35  = mx.symbol.FullyConnected(data = acti25, num.hidden = 1, name = 'fc_layer_35')
  out_layer55 = mx.symbol.LinearRegressionOutput(data = fc_layer_35, label = label,name = 'out_layer55')
  
  # only sgd with momentum (0.9 used) and adadelta no learning rate
  structures=c('out_layer11','out_layer22','out_layer33','out_layer44','out_layer55')
  SbeamCADstrt_log_train<-matrix(nrow = 10000,ncol = 5)
  SbeamCADstrt_log_test<-matrix(nrow = 10000,ncol = 5)
  SbeamCADstrt_trainRMSE<-matrix(nrow = 5,ncol = 5)
  SbeamCADstrt_testRMSE<-matrix(nrow = 5,ncol = 5)
  #i=1
for (i in 1:5) {
  for (j in 1:5) {
  rm(SbeamCAD_log)
  SbeamCAD_log <- mx.metric.logger$new()  # error vector
  SbeamCAD_FFN_strtSdy <- mx.model.FeedForward.create(get(structures[i]), X=NDSbeamCAD_NN_train.x, y=NDSbeamCAD_NN_train.y,eval.data=list(data=NDSbeamCAD_NN_test.x,label=NDSbeamCAD_NN_test.y),
                                                      ctx=mx.cpu(),num.round=10000,optimizer="adagrad",array.batch.size=100,
                                                      eval.metric=mx.metric.rmse,
                                                      epoch.end.callback = mx.callback.log.train.metric(period=8, SbeamCAD_log))#
  SbeamCADstrt_trainRMSE[j,i]<-SbeamCAD_log$train[10000]
  SbeamCADstrt_testRMSE[j,i]<-SbeamCAD_log$eval[10000]
  
  }
  SbeamCADstrt_log_train[,i]<-as.matrix(SbeamCAD_log$train)
  SbeamCADstrt_log_test[,i]<-as.matrix(SbeamCAD_log$eval)
}
colnames(SbeamCADstrt_trainRMSE)<-c("7-20-1","7-20-20-1","7-20-20-20-1","7-10-5-5-1","7-10-5-1")
colnames(SbeamCADstrt_testRMSE)<-c("7-20-1","7-20-20-1","7-20-20-20-1","7-10-5-5-1","7-10-5-1")
#plot
plot(1:10000,SbeamCADstrt_log_test[1:10000,1],"l",lty="solid",col="black",xlim=c(0,10000),ylim=c(0.02,0.08),axes=TRUE, ann=TRUE,xlab="Epoch",ylab="Loss (RMSE)",cex.lab=1.2,lwd=2)
lines(1:10000,SbeamCADstrt_log_test[1:10000,2],"l",lty="solid", col="blue",lwd=2)
lines(1:10000,SbeamCADstrt_log_test[1:10000,3],"l",lty="solid",col="red",lwd=2)
lines(1:10000,SbeamCADstrt_log_test[1:10000,4],"l",lty="solid", col="green",lwd=2)
lines(1:10000,SbeamCADstrt_log_test[1:10000,5],"l",lty="solid", col="orange",lwd=2)
legend(7000, 0.08, c("7-20-1","7-20-20-1","7-20-20-20-1","7-10-5-5-1","7-10-5-1"), 
       cex=1.2, col=c("black","blue","red","green","orange"),lty=1,lwd=2)
#### for the boxplot
par(font=1,cex.lab=1.4,cex.axis=1.2,pch=13,cex.main=1.4)
boxplot(SbeamCADstrt_trainRMSE,boxwex = 0.3,at = 1:5 - 0.18,col = "blue",border="blue",xlim = c(0.5, 5.5), ylim = c(0.02, 0.06), xlab = "Layer structures", ylab = "Loss (RMSE)",names=NA )
boxplot(SbeamCADstrt_testRMSE,add=TRUE,boxwex = 0.3,at = 1:5 + 0.18,col = "green",border="green", names=NA )
axis(side=1,at=c(1,2,3,4,5),labels = c('7-20-1','7-20-20-1','7-20-20-20-1','7-10-5-5-1','7-10-5-1'))
#par(font=1,cex=1.8,pch=13)
legend(0.5, 0.06, c("Training", "Validation"),fill = c("blue", "green"),border=c("blue", "green"),cex = 1.4)



##data size effect on specific test set with 400 test samples#######################
Sbeam<- read_csv("C:/Users/DUX1/Desktop/PHD Project/3rd Deep learning/3.1 Three other machine learning R/ANN structures/12500examples for CAD parameters model.csv")
ind<-sample.int(dim(Sbeam)[1],10400)  #sampling 1000 observations from the original data
Depth_Sbeam<-Sbeam[ind,2:dim(Sbeam)[2]]
maxvalue<-apply(Depth_Sbeam,2,max)
minvalue<-apply(Depth_Sbeam,2,min)
Depth_SbeamN<-as.data.frame(scale(Depth_Sbeam,center = minvalue,scale = maxvalue-minvalue))
#test_ind<-sample.int(dim(Depth_Sbeam)[1],400)

Sbeam_de_train<-Depth_SbeamN[-test_ind,]
Sbeam_de_test<-Depth_SbeamN[test_ind,]

UNSbeam_de_train=matrix(nrow=dim(Sbeam_de_train)[1],ncol=8)
UNSbeam_de_test=matrix(nrow=dim(Sbeam_de_test)[1],ncol=8)
for (i in 1:dim(Sbeam_de_train)[2]) {
  UNSbeam_de_train[,i]<-as.matrix(Sbeam_de_train[,i]*(maxvalue[i]-minvalue[i])+minvalue[i])
  UNSbeam_de_test[,i]<-as.matrix(Sbeam_de_test[,i]*(maxvalue[i]-minvalue[i])+minvalue[i])
}
colnames(UNSbeam_de_train)<-names(Sbeam_de_train)
colnames(UNSbeam_de_test)<-names(Sbeam_de_train)
library(xlsx)
write.xlsx(UNSbeam_de_train, file = "C:/Users/DUX1/Desktop/PHD Project/3rd Deep learning/3.1 Three other machine learning R/ANN structures/10400 S-beam data used.xlsx",
           sheetName="Original_Tr", col.names=TRUE, row.names=TRUE, append=FALSE, showNA=TRUE, password=NULL)
write.xlsx(UNSbeam_de_test, file = "C:/Users/DUX1/Desktop/PHD Project/3rd Deep learning/3.1 Three other machine learning R/ANN structures/10400 S-beam data used.xlsx",
           sheetName="Original_Te", col.names=TRUE, row.names=TRUE, append=TRUE, showNA=TRUE, password=NULL)
write.xlsx(Sbeam_de_train, file = "C:/Users/DUX1/Desktop/PHD Project/3rd Deep learning/3.1 Three other machine learning R/ANN structures/10400 S-beam data used.xlsx",
           sheetName="Normalized_Tr", col.names=TRUE, row.names=TRUE, append=TRUE, showNA=TRUE, password=NULL)
write.xlsx(Sbeam_de_test, file = "C:/Users/DUX1/Desktop/PHD Project/3rd Deep learning/3.1 Three other machine learning R/ANN structures/10400 S-beam data used.xlsx",
           sheetName="Normalized_Te", col.names=TRUE, row.names=TRUE, append=TRUE, showNA=TRUE, password=NULL)

##Dataset construction

Sbeam_de_test.x=data.matrix(Sbeam_de_train[,-dim(Depth_Sbeam)[2]])
Sbeam_de_test.y=Sbeam_de_train[,dim(Depth_Sbeam)[2]]

Sbeam_de_log_train<-matrix(nrow = 10000,ncol = 8)
Sbeam_de_log_test<-matrix(nrow = 10000,ncol = 8)

Sbeam_de_train5E<-matrix(nrow = 5,ncol = 8)
Sbeam_de_test5E<-matrix(nrow = 5,ncol = 8)
j=1
#c(50,100,500,800,1000,2000,5000,10000)
for (i in c(50,100,500,800,1000,2000,5000,10000)) {
  for (ii in 1:5) {
de_ind<-sample.int(dim(Sbeam_de_train)[1],i)
Sbeam_de_train.x=data.matrix(Sbeam_de_train[de_ind,-dim(Sbeam_de_train)[2]])
Sbeam_de_train.y=Sbeam_de_train[de_ind,dim(Sbeam_de_train)[2]]
##
data = mx.symbol.Variable(name = 'data')
label = mx.symbol.Variable(name = 'label')
fc_layer_1 = mx.symbol.FullyConnected(data = data, num.hidden = 20, name = 'fc_layer_1')
acti = mx.symbol.Activation(data=fc_layer_1,act.type='tanh',name='acti')
fc_layer_2  = mx.symbol.FullyConnected(data = acti, num.hidden = 1, name = 'fc_layer_2')
out_layer = mx.symbol.LinearRegressionOutput(data = fc_layer_2, label = label,name = 'out_layer')
#FFP
rm(Sbeam_de_log)
Sbeam_de_log <- mx.metric.logger$new()  # error vector
Sbeam_de_FFN_stdy <- mx.model.FeedForward.create(out_layer, X=Sbeam_de_train.x, y=Sbeam_de_train.y,eval.data=list(data=Sbeam_de_test.x,label=Sbeam_de_test.y),
                                                    ctx=mx.cpu(),num.round=10000,optimizer="adagrad",array.batch.size=100,
                                                    eval.metric=mx.metric.rmse,
                                                    epoch.end.callback = mx.callback.log.train.metric(period=i/100, Sbeam_de_log))#learning.rate=1e-1,momentum=0.9,

Sbeam_de_train5E[ii,j]<-Sbeam_de_log$train[10000]
Sbeam_de_test5E[ii,j]<-Sbeam_de_log$eval[10000]
#Sbeam_de_log_train[,j]<-Sbeam_de_log$train[10000]
#Sbeam_de_log_test[,j]<-Sbeam_de_log$eval[10000]
  }
  Sbeam_de_log_train[,j]<-as.matrix(Sbeam_de_log$train)
  Sbeam_de_log_test[,j]<-as.matrix(Sbeam_de_log$eval)
  save.image("C:/Users/DUX1/Desktop/PHD Project/3rd Deep learning/3.1 Three other machine learning R/ANN structures/Data.RData")
  j=j+1
}
par(mfrow=c(2,1),mai=c(0.6,1,0.5,0.5),mgp=c(2, 0.8, 0))
plot(1:10000,Sbeam_de_log_train[1:10000,1],"l",main="(a) Less than 1000 training samples", lty="solid",col="black",xlim=c(0,10000),ylim=c(0.02,0.09),axes=TRUE, ann=TRUE,xlab="Epoch",ylab="Loss (RMSE)",cex.lab=1.2,lwd=2)
lines(1:10000,Sbeam_de_log_test[1:10000,1],"l",lty="dashed", col="black",lwd=2)
lines(1:10000,Sbeam_de_log_train[1:10000,2],"l",lty="solid",col="blue",lwd=2)
lines(1:10000,Sbeam_de_log_test[1:10000,2],"l",lty="dashed", col="blue",lwd=2)
lines(1:10000,Sbeam_de_log_train[1:10000,3],"l",lty="solid",col="red",lwd=2)
lines(1:10000,Sbeam_de_log_test[1:10000,3],"l",lty="dashed", col="red",lwd=2)
lines(1:10000,Sbeam_de_log_train[1:10000,4],"l",lty="solid",col="green",lwd=2)
lines(1:10000,Sbeam_de_log_test[1:10000,4],"l",lty="dashed", col="green",lwd=2)
legend(5500, 0.09, c("50 training samples","100 training samples","500 training samples","800 training samples"), 
       cex=1.2, col=c("black","blue","red","green"),lty=1,lwd=2)
plot(1:10000,Sbeam_de_log_train[1:10000,5],"l",main="(b) No less than 1000 training samples", lty="solid",col="black",ylim=c(0.02,0.06),axes=TRUE, ann=TRUE,xlab="Epoch",ylab="Loss (RMSE)",cex.lab=1.2,lwd=2)
lines(1:10000,Sbeam_de_log_test[1:10000,5],"l",lty="dashed", col="black",lwd=2)
lines(1:10000,Sbeam_de_log_train[1:10000,6],"l",lty="solid",col="blue",lwd=2)
lines(1:10000,Sbeam_de_log_test[1:10000,6],"l",lty="dashed", col="blue",lwd=2)
lines(1:10000,Sbeam_de_log_train[1:10000,7],"l",lty="solid",col="red",lwd=2)
lines(1:10000,Sbeam_de_log_test[1:10000,7],"l",lty="dashed", col="red",lwd=2)
lines(1:10000,Sbeam_de_log_train[1:10000,8],"l",lty="solid",col="green",lwd=2)
lines(1:10000,Sbeam_de_log_test[1:10000,8],"l",lty="dashed", col="green",lwd=2)
legend(5500, 0.06, c("1000 training samples","2000 training samples","5000 training samples","10000 training samples"), 
       cex=1.2, col=c("black","blue","red","green"),lty=1,lwd=2)

##for error box plot
par(mfrow=c(2,1))
boxplot(Sbeam_de_train5E)
boxplot(Sbeam_de_test5E)

par(font=1,cex.lab=1.4,cex.axis=1.2,pch=13,cex.main=1.4)
boxplot(Sbeam_de_train5E,boxwex = 0.3,at = 1:8 - 0.18,col = "blue",xlim = c(0.5, 8.5), ylim = c(0.02, 0.065), xlab = "Training dataset size", ylab = "Loss (RMSE)",names=NA )
boxplot(Sbeam_de_test5E,add=TRUE,boxwex = 0.3,at = 1:8 + 0.18,col = "green",names=NA ) #main='(a) Specific validation dataset',
axis(side=1,at=c(1,2,3,4,5,6,7,8),labels = c('50','100','500','800','1,000','2,000','5,000','10,000'))
#par(font=1,cex=1.8,pch=13)
mtext('(a)',side = 3,line = 1.2,cex = 1.5,at=c(-0.7))
legend(6, 0.065, c("Training", "Validation"),fill = c("blue", "green"),cex = 1.5)


##data size effect on specific test set with 400 test samples#######################
#matrix for saving training and validation data
Sbeam_de_log_train_trainratio0.8<-matrix(nrow = 10000,ncol = 8)
Sbeam_de_log_test_trainratio0.8<-matrix(nrow = 10000,ncol = 8)

Sbeam_de_train5E_trainratio0.8<-matrix(nrow = 5,ncol = 8)
Sbeam_de_test5E_trainratio0.8<-matrix(nrow = 5,ncol = 8)
j=1
#c(50,100,500,800,1000,2000,5000,10000) 65,65,125,625,
for (i in c(65,125,625,1000,1250,2500,6250,10000)) {
  for (ii in 1:5) {
    de_ind<-sample.int(10000,i)
    Tdata_set<-data.matrix(Sbeam_de_train[de_ind,])
    ind0.8<-sample.int(i,0.8*i)
    Sbeam_de_train.x=as.matrix(Tdata_set[ind0.8,-dim(Tdata_set)[2]])
    Sbeam_de_train.y=Tdata_set[ind0.8,dim(Tdata_set)[2]]
    Sbeam_de_test.x=as.matrix(Tdata_set[-ind0.8,-dim(Tdata_set)[2]])
    Sbeam_de_test.y=Tdata_set[-ind0.8,dim(Tdata_set)[2]]
    ##ANN model
    data = mx.symbol.Variable(name = 'data')
    label = mx.symbol.Variable(name = 'label')
    fc_layer_1 = mx.symbol.FullyConnected(data = data, num.hidden = 20, name = 'fc_layer_1')
    acti = mx.symbol.Activation(data=fc_layer_1,act.type='tanh',name='acti')
    fc_layer_2  = mx.symbol.FullyConnected(data = acti, num.hidden = 1, name = 'fc_layer_2')
    out_layer = mx.symbol.LinearRegressionOutput(data = fc_layer_2, label = label,name = 'out_layer')
    #FFP
    rm(Sbeam_de_log)
    Sbeam_de_log <- mx.metric.logger$new()  # error vector
    Sbeam_de_FFN_stdy <- mx.model.FeedForward.create(out_layer, X=Sbeam_de_train.x, y=Sbeam_de_train.y,eval.data=list(data=Sbeam_de_test.x,label=Sbeam_de_test.y),
                                                     ctx=mx.cpu(),num.round=10000,optimizer="adagrad",array.batch.size=floor(0.2*i/4),
                                                     eval.metric=mx.metric.rmse,
                                                     epoch.end.callback = mx.callback.log.train.metric(period=4, Sbeam_de_log))#learning.rate=1e-1,momentum=0.9,
    
    Sbeam_de_train5E_trainratio0.8[ii,j]<-Sbeam_de_log$train[10000]
    Sbeam_de_test5E_trainratio0.8[ii,j]<-Sbeam_de_log$eval[10000]
    #Sbeam_de_log_train[,j]<-Sbeam_de_log$train[10000]
    #Sbeam_de_log_test[,j]<-Sbeam_de_log$eval[10000]
  }
  Sbeam_de_log_train_trainratio0.8[,j]<-as.matrix(Sbeam_de_log$train)
  Sbeam_de_log_test_trainratio0.8[,j]<-as.matrix(Sbeam_de_log$eval)
  save.image("C:/Users/DUX1/Desktop/PHD Project/3rd Deep learning/3.1 Three other machine learning R/ANN structures/Data.RData")
  j=j+1
}
par(mfrow=c(2,1),mai=c(0.6,1,0.5,0.5),mgp=c(2, 0.8, 0))
plot(1:10000,Sbeam_de_log_train_trainratio0.8[1:10000,1],"l",main="(a) Less than 1000 training samples", lty="solid",col="black",xlim=c(0,10000),ylim=c(0.02,0.09),axes=TRUE, ann=TRUE,xlab="Epoch",ylab="Loss (RMSE)",cex.lab=1.2,lwd=2)
lines(1:10000,Sbeam_de_log_test_trainratio0.8[1:10000,1],"l",lty="dashed", col="black",lwd=2)
lines(1:10000,Sbeam_de_log_train_trainratio0.8[1:10000,2],"l",lty="solid",col="blue",lwd=2)
lines(1:10000,Sbeam_de_log_test_trainratio0.8[1:10000,2],"l",lty="dashed", col="blue",lwd=2)
lines(1:10000,Sbeam_de_log_train_trainratio0.8[1:10000,3],"l",lty="solid",col="red",lwd=2)
lines(1:10000,Sbeam_de_log_test_trainratio0.8[1:10000,3],"l",lty="dashed", col="red",lwd=2)
lines(1:10000,Sbeam_de_log_train_trainratio0.8[1:10000,4],"l",lty="solid",col="green",lwd=2)
lines(1:10000,Sbeam_de_log_test_trainratio0.8[1:10000,4],"l",lty="dashed", col="green",lwd=2)
legend(5500, 0.09, c("50 training samples","100 training samples","500 training samples","800 training samples"), 
       cex=1.2, col=c("black","blue","red","green"),lty=1,lwd=2)
plot(1:10000,Sbeam_de_log_train_trainratio0.8[1:10000,5],"l",main="(b) No less than 1000 training samples", lty="solid",col="black",ylim=c(0.02,0.06),axes=TRUE, ann=TRUE,xlab="Epoch",ylab="Loss (RMSE)",cex.lab=1.2,lwd=2)
lines(1:10000,Sbeam_de_log_test_trainratio0.8[1:10000,5],"l",lty="dashed", col="black",lwd=2)
lines(1:10000,Sbeam_de_log_train_trainratio0.8[1:10000,6],"l",lty="solid",col="blue",lwd=2)
lines(1:10000,Sbeam_de_log_test_trainratio0.8[1:10000,6],"l",lty="dashed", col="blue",lwd=2)
lines(1:10000,Sbeam_de_log_train_trainratio0.8[1:10000,7],"l",lty="solid",col="red",lwd=2)
lines(1:10000,Sbeam_de_log_test_trainratio0.8[1:10000,7],"l",lty="dashed", col="red",lwd=2)
lines(1:10000,Sbeam_de_log_train_trainratio0.8[1:10000,8],"l",lty="solid",col="green",lwd=2)
lines(1:10000,Sbeam_de_log_test_trainratio0.8[1:10000,8],"l",lty="dashed", col="green",lwd=2)
legend(5500, 0.06, c("1000 training samples","2000 training samples","5000 training samples","10000 training samples"), 
       cex=1.2, col=c("black","blue","red","green"),lty=1,lwd=2)

##for error box plot
par(font=1,cex.lab=1.4,cex.axis=1.2,pch=13,cex.main=1.4)
boxplot(Sbeam_de_train5E_trainratio0.8,boxwex = 0.3,at = 1:8 - 0.18,col = "blue",xlim = c(0.5, 8.5), ylim = c(0.01, 0.08), xlab = "Training dataset size", ylab = "Loss (RMSE)",names=NA )
boxplot(Sbeam_de_test5E_trainratio0.8,add=TRUE,boxwex = 0.3,at = 1:8 + 0.18,col = "green",names=NA ) # main='(b) Fixed validation data ratio (20%)',
axis(side=1,at=c(1,2,3,4,5,6,7,8),labels = c('50','100','500','800','1,000','2,000','5,000','8,000'))
#par(font=1,cex=1.8,pch=13)
mtext('(b)',side = 3,line = 1.2,cex = 1.5,at=c(-0.7))
legend(6, 0.08, c("Training", "Validation"),fill = c("blue", "green"),cex = 1.5)

