library (mlr)
#Regression_GPR_tube;Regression_SVM_tube;Regression_RFR_tube;Regression_mxnet_tube;
#Regression_GPR_sbeam;Regression_SVM_sbeam;Regression_RFR_sbeam;Regression_mxnet_sbeam;
#Regression_GPR_tenbars;Regression_SVM_tenbars;Regression_RFR_tenbars;Regression_mxnet_tenbars;
#Regression_GPR_torsionb;Regression_SVM_torsionb;Regression_RFR_torsionb;Regression_mxnet_torsionb
#set current working path
ind<-1:1000
train.set<-sample.int(1000,800)        # No. of samples in traning set
test.set<-ind[-train.set] # No. of samples in test set

tasksbeam = makeRegrTask(data=NDSbeamCAD,target=objsbeam)  #define a regression task
tasktube = makeRegrTask(data=NDTube,target=objtube)  #define a regression task
tasktenbars = makeRegrTask(data=NDTenbars,target=objtenbars)  #define a regression task
tasktorsionb = makeRegrTask(data=NDTorsionB,target=objtorsionb)  #define a regression task

##Sbeam train for four Regression algorithms
SbeamCAD_model_GPR <- train(Regression_GPR_sbeam, tasksbeam, subset = train.set)
SbeamCAD_model_SVM <- train(Regression_SVM_sbeam, tasksbeam, subset = train.set)
SbeamCAD_model_RFR <- train(Regression_RFR_sbeam, tasksbeam, subset = train.set)
SbeamCAD_model_mxnet <- train(Regression_mxnet_sbeam, tasksbeam, subset = train.set)
##tube train for four Regression algorithms
Tube_model_GPR <- train(Regression_GPR_tube, tasktube, subset = train.set)
Tube_model_SVM <- train(Regression_SVM_tube, tasktube, subset = train.set)
Tube_model_RFR <- train(Regression_RFR_tube, tasktube, subset = train.set)
Tube_model_mxnet <- train(Regression_mxnet_tube, tasktube, subset = train.set)
##ten bars train for four Regression algorithms
Tenbars_model_GPR <- train(Regression_GPR_tenbars, tasktenbars, subset = train.set)
Tenbars_model_SVM <- train(Regression_SVM_tenbars, tasktenbars, subset = train.set)
Tenbars_model_RFR <- train(Regression_RFR_tenbars, tasktenbars, subset = train.set)
Tenbars_model_mxnet <- train(Regression_mxnet_tenbars, tasktenbars, subset = train.set)
##torsion bar train for four Regression algorithms
TorsionB_model_GPR <- train(Regression_GPR_torsionb, tasktorsionb, subset = train.set)
TorsionB_model_SVM <- train(Regression_SVM_torsionb, tasktorsionb, subset = train.set)
TorsionB_model_RFR <- train(Regression_RFR_torsionb, tasktorsionb, subset = train.set)
TorsionB_model_mxnet <- train(Regression_mxnet_torsionb, tasktorsionb, subset = train.set)

##predict the response
##
SbeamCAD_pre_GPR <- predict(SbeamCAD_model_GPR, tasksbeam, subset = test.set)
SbeamCAD_pre_GPR<-as.data.frame(SbeamCAD_pre_GPR)
SbeamCAD_pre_SVM <- predict(SbeamCAD_model_SVM, tasksbeam, subset = test.set)
SbeamCAD_pre_SVM<-as.data.frame(SbeamCAD_pre_SVM)
SbeamCAD_pre_RFR <- predict(SbeamCAD_model_RFR, tasksbeam, subset = test.set)
SbeamCAD_pre_RFR<-as.data.frame(SbeamCAD_pre_RFR)
SbeamCAD_pre_mxnet <- predict(SbeamCAD_model_mxnet, tasksbeam, subset = test.set)
SbeamCAD_pre_mxnet<-as.data.frame(SbeamCAD_pre_mxnet)
##
Tube_pre_GPR <- predict(Tube_model_GPR, tasktube, subset = test.set)
Tube_pre_GPR<-as.data.frame(Tube_pre_GPR)
Tube_pre_SVM <- predict(Tube_model_SVM, tasktube, subset = test.set)
Tube_pre_SVM<-as.data.frame(Tube_pre_SVM)
Tube_pre_RFR <- predict(Tube_model_RFR, tasktube, subset = test.set)
Tube_pre_RFR<-as.data.frame(Tube_pre_RFR)
Tube_pre_mxnet <- predict(Tube_model_mxnet, tasktube, subset = test.set)
Tube_pre_mxnet<-as.data.frame(Tube_pre_mxnet)
##
Tenbars_pre_GPR <- predict(Tenbars_model_GPR, tasktenbars, subset = test.set)
Tenbars_pre_GPR<-as.data.frame(Tenbars_pre_GPR)
Tenbars_pre_SVM <- predict(Tenbars_model_SVM, tasktenbars, subset = test.set)
Tenbars_pre_SVM<-as.data.frame(Tenbars_pre_SVM)
Tenbars_pre_RFR <- predict(Tenbars_model_RFR, tasktenbars, subset = test.set)
Tenbars_pre_RFR<-as.data.frame(Tenbars_pre_RFR)
Tenbars_pre_mxnet <- predict(Tenbars_model_mxnet, tasktenbars, subset = test.set)
Tenbars_pre_mxnet<-as.data.frame(Tenbars_pre_mxnet)
##
TorsionB_pre_GPR <- predict(TorsionB_model_GPR, tasktorsionb, subset = test.set)
TorsionB_pre_GPR<-as.data.frame(TorsionB_pre_GPR)
TorsionB_pre_SVM <- predict(TorsionB_model_SVM, tasktorsionb, subset = test.set)
TorsionB_pre_SVM<-as.data.frame(TorsionB_pre_SVM)
TorsionB_pre_RFR <- predict(TorsionB_model_RFR, tasktorsionb, subset = test.set)
TorsionB_pre_RFR<-as.data.frame(TorsionB_pre_RFR)
TorsionB_pre_mxnet <- predict(TorsionB_model_mxnet, tasktorsionb, subset = test.set)
TorsionB_pre_mxnet<-as.data.frame(TorsionB_pre_mxnet)
####################################################################################################

##generate Sbeam prediction
Est_sbeam_max<-apply(DSbeamCAD[dim(DSbeamCAD)[2]],2,max)
Est_sbeam_min<-apply(DSbeamCAD[dim(DSbeamCAD)[2]],2,min)
Est_sbeam_scale<-Est_sbeam_max-Est_sbeam_min
#sbeam GPR
RSbeamCAD_pre_GPR<-SbeamCAD_pre_GPR[,2:3]*Est_sbeam_scale+Est_sbeam_min
#sbeam SVM
RSbeamCAD_pre_SVM<-SbeamCAD_pre_SVM[,2:3]*Est_sbeam_scale+Est_sbeam_min
#sbeam RFR
RSbeamCAD_pre_RFR<-SbeamCAD_pre_RFR[,2:3]*Est_sbeam_scale+Est_sbeam_min
#sbeam mxnet
RSbeamCAD_pre_mxnet<-SbeamCAD_pre_mxnet[,2:3]*Est_sbeam_scale+Est_sbeam_min

## generate Tube prediction
Est_tube_max<-apply(DTube[dim(DTube)[2]],2,max)
Est_tube_min<-apply(DTube[dim(DTube)[2]],2,min)
Est_tube_scale<-Est_tube_max-Est_tube_min
#tube GPR
RTube_pre_GPR<-Tube_pre_GPR[,2:3]*Est_tube_scale+Est_tube_min
#tube SVM
RTube_pre_SVM<-Tube_pre_SVM[,2:3]*Est_tube_scale+Est_tube_min
#tube RFR
RTube_pre_RFR<-Tube_pre_RFR[,2:3]*Est_tube_scale+Est_tube_min
#tube mxnet
RTube_pre_mxnet<-Tube_pre_mxnet[,2:3]*Est_tube_scale+Est_tube_min

## generate Ten bars prediction
Est_tenbars_max<-apply(DTenbars[dim(DTenbars)[2]],2,max)
Est_tenbars_min<-apply(DTenbars[dim(DTenbars)[2]],2,min)
Est_tenbars_scale<-Est_tenbars_max-Est_tenbars_min
#ten bars GPR
RTenbars_pre_GPR<-Tenbars_pre_GPR[,2:3]*Est_tenbars_scale+Est_tenbars_min
#tenbars SVM
RTenbars_pre_SVM<-Tenbars_pre_SVM[,2:3]*Est_tenbars_scale+Est_tenbars_min
#tenbars RFR
RTenbars_pre_RFR<-Tenbars_pre_RFR[,2:3]*Est_tenbars_scale+Est_tenbars_min
#tenbars mxnet
RTenbars_pre_mxnet<-Tenbars_pre_mxnet[,2:3]*Est_tenbars_scale+Est_tenbars_min

#generate TorsionB prediction
Est_torsionb_max<-apply(DTorsionB[dim(DTorsionB)[2]],2,max)
Est_torsionb_min<-apply(DTorsionB[dim(DTorsionB)[2]],2,min)
Est_torsionb_scale<-Est_torsionb_max-Est_torsionb_min
#torsionb GPR
RTorsionB_pre_GPR<-TorsionB_pre_GPR[,2:3]*Est_torsionb_scale+Est_torsionb_min
#torsionb SVM
RTorsionB_pre_SVM<-TorsionB_pre_SVM[,2:3]*Est_torsionb_scale+Est_torsionb_min
#torsionb RFR
RTorsionB_pre_RFR<-TorsionB_pre_RFR[,2:3]*Est_torsionb_scale+Est_torsionb_min
#torsionb mxnet
RTorsionB_pre_mxnet<-TorsionB_pre_mxnet[,2:3]*Est_torsionb_scale+Est_torsionb_min


library(graphics)
##plot the match effect of surrogates models prediction
par(mfrow=c(2,2),mai=c(0.6,0.6,0.6,0.3),mgp=c(2.4, 0.8, 0),font=1,cex.lab=1.4,cex.axis=1.2,pch=13,cex.main=1.4)
##Sbeam
plot(RSbeamCAD_pre_GPR[,1],RSbeamCAD_pre_GPR[,2],pch=15,main="(c) ShB",col="black",xlim=c(800,2200),ylim=c(800,2200),axes=TRUE, lty = 1,ann=TRUE,xlab="Simulated SEA (J/kg)",ylab="Predicted SEA (J/kg)")
points(RSbeamCAD_pre_SVM[,1],RSbeamCAD_pre_SVM[,2],pch=16,col="blue")
points(RSbeamCAD_pre_RFR[,1],RSbeamCAD_pre_RFR[,2],pch=17,col="red")
points(RSbeamCAD_pre_mxnet[,1],RSbeamCAD_pre_mxnet[,2],pch=18,col="green")
Bench_x<-c(0,2800)
Bench_y<-c(0,2800)
lines(Bench_x,Bench_y,col="black",lwd=1.2)
legend(800, 2200, c("GPR","SVM","RFR","ANN","Perfect match line"),
       cex=1,col=c("black","blue","red","green","black"),pch=c(15,16,17,18,NA),lty = c(NA,NA,NA,NA,1));

##tube
plot(RTube_pre_GPR[,1],RTube_pre_GPR[,2],pch=15,main="(d) OMcT",col="black",xlim=c(0,0.9),ylim=c(0,0.9),axes=TRUE, lty = 1,ann=TRUE,xlab="Simulated CFE",ylab="Predicted CFE")
points(RTube_pre_SVM[,1],RTube_pre_SVM[,2],pch=16,col="blue")
points(RTube_pre_RFR[,1],RTube_pre_RFR[,2],pch=17,col="red")
points(RTube_pre_mxnet[,1],RTube_pre_mxnet[,2],pch=18,col="green")
Bench_x<-c(0,2500)
Bench_y<-c(0,2500)
lines(Bench_x,Bench_y,col="black",lwd=1.2)
legend(0, 0.9, c("GPR","SVM","RFR","ANN","Perfect match line"),
       cex=1,col=c("black","blue","red","green","black"),pch=c(15,16,17,18,NA),lty = c(NA,NA,NA,NA,1));

##Ten bars plot
plot(RTenbars_pre_GPR[,1],RTenbars_pre_GPR[,2],pch=15,main="(a) TbPT",col="black",xlim=c(0,600),ylim=c(0,600),axes=TRUE, lty = 1,ann=TRUE,xlab="Simulated displacement (mm)",ylab="Predicted displacement (mm)")
points(RTenbars_pre_SVM[,1],RTenbars_pre_SVM[,2],pch=16,col="blue")
points(RTenbars_pre_RFR[,1],RTenbars_pre_RFR[,2],pch=17,col="red")
points(RTenbars_pre_mxnet[,1],RTenbars_pre_mxnet[,2],pch=18,col="green")
Bench_x<-c(0,2500)
Bench_y<-c(0,2500)
lines(Bench_x,Bench_y,col="black",lwd=1.2)
legend(0, 600, c("GPR","SVM","RFR","ANN","Perfect match line"),
       cex=1,col=c("black","blue","red","green","black"),pch=c(15,16,17,18,NA),lty = c(NA,NA,NA,NA,1));

##Torsion Bar plot
plot(RTorsionB_pre_GPR[,1],RTorsionB_pre_GPR[,2],pch=15,main="(b) TqA",col="black",xlim=c(0.4,1),ylim=c(0.4,1),axes=TRUE, lty = 1,ann=TRUE,xlab="Simulated mass (kg)",ylab="Predicted mass (kg)")
points(RTorsionB_pre_SVM[,1],RTorsionB_pre_SVM[,2],pch=16,col="blue")
points(RTorsionB_pre_RFR[,1],RTorsionB_pre_RFR[,2],pch=17,col="red")
points(RTorsionB_pre_mxnet[,1],RTorsionB_pre_mxnet[,2],pch=18,col="green")
Bench_x<-c(0,2)
Bench_y<-c(0,2)
lines(Bench_x,Bench_y,col="black",lwd=1.2)
legend(0.4, 1.0, c("GPR","SVM","RFR","ANN","Perfect match line"),
       cex=1,col=c("black","blue","red","green","black"),pch=c(15,16,17,18,NA),lty = c(NA,NA,NA,NA,1));

####the relative error histogram
##calculation of relative error:(prediction-real)/real
Re_tube_GPR=(RTube_pre_GPR[,2]-RTube_pre_GPR[,1])/RTube_pre_GPR[,1]
Re_tube_SVM=(RTube_pre_SVM[,2]-RTube_pre_SVM[,1])/RTube_pre_SVM[,1]
Re_tube_RFR=(RTube_pre_RFR[,2]-RTube_pre_RFR[,1])/RTube_pre_RFR[,1]
Re_tube_mxnet=(RTube_pre_mxnet[,2]-RTube_pre_mxnet[,1])/RTube_pre_mxnet[,1]

Re_sbeam_GPR=(RSbeamCAD_pre_GPR[,2]-RSbeamCAD_pre_GPR[,1])/RSbeamCAD_pre_GPR[,1]
Re_sbeam_SVM=(RSbeamCAD_pre_SVM[,2]-RSbeamCAD_pre_SVM[,1])/RSbeamCAD_pre_SVM[,1]
Re_sbeam_RFR=(RSbeamCAD_pre_RFR[,2]-RSbeamCAD_pre_RFR[,1])/RSbeamCAD_pre_RFR[,1]
Re_sbeam_mxnet=(RSbeamCAD_pre_mxnet[,2]-RSbeamCAD_pre_mxnet[,1])/RSbeamCAD_pre_mxnet[,1]

Re_tenbars_GPR=(RTenbars_pre_GPR[,2]-RTenbars_pre_GPR[,1])/RTenbars_pre_GPR[,1]
Re_tenbars_SVM=(RTenbars_pre_SVM[,2]-RTenbars_pre_SVM[,1])/RTenbars_pre_SVM[,1]
Re_tenbars_RFR=(RTenbars_pre_RFR[,2]-RTenbars_pre_RFR[,1])/RTenbars_pre_RFR[,1]
Re_tenbars_mxnet=(RTenbars_pre_mxnet[,2]-RTenbars_pre_mxnet[,1])/RTenbars_pre_mxnet[,1]

Re_torsionb_GPR=(RTorsionB_pre_GPR[,2]-RTorsionB_pre_GPR[,1])/RTorsionB_pre_GPR[,1]
Re_torsionb_SVM=(RTorsionB_pre_SVM[,2]-RTorsionB_pre_SVM[,1])/RTorsionB_pre_SVM[,1]
Re_torsionb_RFR=(RTorsionB_pre_RFR[,2]-RTorsionB_pre_RFR[,1])/RTorsionB_pre_RFR[,1]
Re_torsionb_mxnet=(RTorsionB_pre_mxnet[,2]-RTorsionB_pre_mxnet[,1])/RTorsionB_pre_mxnet[,1]

##plot of histgrams
par(mfrow=c(4,4),mar=c(3, 3, 0.5, 0)+1)
hist(Re_sbeam_GPR,main="S-beam GPR",freq=TRUE,xlim=c(-1,1),ylim=c(0,120),xlab=NA,ylab="Frequency",cex.lab=1.2,breaks=c(-10,-1,-0.95-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,10))
hist(Re_sbeam_SVM,main="S-beam SVM",freq=TRUE,xlim=c(-1,1),ylim=c(0,120),xlab=NA,ylab=NA,breaks=c(-10,-1,-0.95-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,10))
hist(Re_sbeam_RFR,main="S-beam RFR",freq=TRUE,xlim=c(-1,1),ylim=c(0,120),xlab=NA,ylab=NA,breaks=c(-10,-1,-0.95-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,10))
hist(Re_sbeam_mxnet,main="S-beam ANN",freq=TRUE,xlim=c(-1,1),ylim=c(0,120),xlab=NA,ylab=NA,breaks=c(-10,-1,-0.95-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,10))

hist(Re_tube_GPR,main="Tube GPR",freq=TRUE,xlim=c(-1,1),ylim=c(0,120),xlab=NA,ylab="Frequency",cex.lab=1.2,breaks=c(-10,-1,-0.95-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,10))
hist(Re_tube_SVM,main="Tube SVM",freq=TRUE,xlim=c(-1,1),ylim=c(0,120),xlab=NA,ylab=NA,breaks=c(-10,-1,-0.95-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,10))
hist(Re_tube_RFR,main="Tube RFR",freq=TRUE,xlim=c(-1,1),ylim=c(0,120),xlab=NA,ylab=NA,breaks=c(-10,-1,-0.95-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,10))
hist(Re_tube_mxnet,main="Tube ANN",freq=TRUE,xlim=c(-1,1),ylim=c(0,120),xlab=NA,ylab=NA,breaks=c(-10,-1,-0.95-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,10))

hist(Re_tenbars_GPR,main="Ten bars GPR",freq=TRUE,xlim=c(-1,1),ylim=c(0,120),xlab=NA,ylab="Frequency",cex.lab=1.2,breaks=c(-10,-1,-0.95-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,10))
hist(Re_tenbars_SVM,main="Ten bars SVM",freq=TRUE,xlim=c(-1,1),ylim=c(0,120),xlab=NA,ylab=NA,breaks=c(-10,-1,-0.95-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,10))
hist(Re_tenbars_RFR,main="Ten bars RFR",freq=TRUE,xlim=c(-1,1),ylim=c(0,120),xlab=NA,ylab=NA,breaks=c(-10,-1,-0.95-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,10))
hist(Re_tenbars_mxnet,main="Ten bars ANN",freq=TRUE,xlim=c(-1,1),ylim=c(0,120),xlab=NA,ylab=NA,breaks=c(-10,-1,-0.95-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,10))

hist(Re_torsionb_GPR,main="Torsion bar GPR",freq=TRUE,xlim=c(-1,1),ylim=c(0,120),xlab="RE",cex.lab=1.2,ylab="Frequency",cex.lab=1.2,breaks=c(-10,-1,-0.95-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,10))
hist(Re_torsionb_SVM,main="Torsion bar SVM",freq=TRUE,xlim=c(-1,1),ylim=c(0,120),xlab="RE",cex.lab=1.2,ylab=NA,breaks=c(-10,-1,-0.95-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,10))
hist(Re_torsionb_RFR,main="Torsion bar RFR",freq=TRUE,xlim=c(-1,1),ylim=c(0,120),xlab="RE",cex.lab=1.2,ylab=NA,breaks=c(-10,-1,-0.95-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,10))
hist(Re_torsionb_mxnet,main="Torsion bar ANN",freq=TRUE,xlim=c(-1,1),ylim=c(0,120),xlab="RE",cex.lab=1.2,ylab=NA,breaks=c(-10,-1,-0.95-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,10))

