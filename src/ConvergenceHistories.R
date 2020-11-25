library(openxlsx)
#The 16 HOpt models for our examples' four MLAs
optimal_GPR_tenbars;optimal_GPR_torsionb;optimal_GPR_sbeam;optimal_GPR_tube;
optimal_SVM_tenbars;optimal_SVM_torsionb;optimal_SVM_sbeam;optimal_SVM_tube;
optimal_RFR_tenbars;optimal_RFR_torsionb;optimal_RFR_sbeam;optimal_RFR_tube;
optimal_mxnet_tenbars;optimal_mxnet_torsionb;optimal_mxnet_sbeam;optimal_mxnet_tube;
#Using the optimal_GPR_tenbars$mbo.result$opt.path$env$path to call the HOpt path
MLAs<-c('GPR','SVM','RFR','mxnet')
Examples<-c('tenbars','torsionb','sbeam','tube')
TuningNt<-as.data.frame(matrix(0,nrow=100,ncol = 32))
for (i in 1:4) { #for each example
  for (j in 1:4) { #for each MLA
    modelstr=paste('optimal_',MLAs[j],'_',Examples[i],sep='')
    Tunemodel=get(modelstr)
    TuneMtrans=Tunemodel$mbo.result$opt.path$env$path
    TuningNt[,((2*j-1+8*(i-1)):(2*j+8*(i-1)))]=TuneMtrans[,(dim(TuneMtrans)[2]-1):dim(TuneMtrans)[2]]
    colnames(TuningNt)[((2*j-1+8*(i-1)):(2*j+8*(i-1)))]<-c(paste(MLAs[j],'_',Examples[i],'_RMSE',sep=''),
                                                           paste(MLAs[j],'_',Examples[i],'_MXAE',sep=''))
  }
}

write.xlsx(TuningNt, file = "./Data/TuningNtHoPTs_AllModels.xlsx",
           sheetName="HOptHistory", col.names=TRUE, row.names=TRUE, showNA=TRUE, password=NULL)
# the convergency history of MLA
#dataprocessing
AllMLAsNt<-TuningNt
for (ii in 1:32) {
  for (jj in 1:100) {
    AllMLAsNt[jj,ii]<-min(TuningNt[1:jj,ii])
  }
}
#plot
LabelMLAs<-c('GPR','SVM','RFR','ANN')
Convergetitle<-c('(a) tenbars','(b) torsionb','(c) sbeam','(d) tube') 
par(mfrow=c(4,2),mai=c(0.4,0.6,0.2,0.3),mgp=c(2.0, 0.8, 0),font=1,cex.lab=1.2,cex.axis=1.3,pch=13,cex.main=1.2)
ylims=c(0,0.2,0.2,0.8,0,0.2,0.3,0.5,0,0.1,0.1,0.6,0,0.2,0.1,0.7)
for (i in 1:4) { #for each model
  plot(1:100,AllMLAsNt[,8*(i-1)+1],type='l',col = "black",pch=0,lty=1,xlim = c(0, 100),lwd=2,
       ylim=c(ylims[4*(i-1)+1], ylims[4*(i-1)+2]),xlab = "No. of Evaluations", ylab = "RMSE",names=NA) #main=paste(Convergetitle[i],'RMSE'),
  lines(1:100,AllMLAsNt[,8*(i-1)+3],type='l',col = "blue",pch=1,lty=2,lwd=2,names=NA)
  lines(1:100,AllMLAsNt[,8*(i-1)+5],type='l',col = "red",pch=2,lty=3,lwd=2,names=NA)
  lines(1:100,AllMLAsNt[,8*(i-1)+7],type='l',col = "orange",pch=3,lty=4,lwd=2,names=NA)
  if (i==1) {
    legend(70, 0.2, LabelMLAs,cex = 1.1,lty=c(1,2,3,4), lwd=2,col=c('black','blue','red','orange'))
  }
  plot(1:100,AllMLAsNt[,8*(i-1)+2],type='l',col = "black",pch=0,lty=1,lwd=2,xlim = c(0, 100),
       ylim=c(ylims[4*(i-1)+3], ylims[4*(i-1)+4]),xlab = "No. of Evaluations", ylab = "MXAE",names=NA) #main=paste(Convergetitle[i],'MXAE'),
  lines(1:100,AllMLAsNt[,8*(i-1)+4],type='l',col = "blue",pch=1,lty=2,lwd=2,names=NA)
  lines(1:100,AllMLAsNt[,8*(i-1)+6],type='l',col = "red",pch=2,lty=3,lwd=2,names=NA)
  lines(1:100,AllMLAsNt[,8*(i-1)+8],type='l',col = "orange",pch=3,lty=4,lwd=2,names=NA)
  if (i==1) {
    legend(70, 0.8, LabelMLAs,cex = 1.1,lty=c(1,2,3,4), lwd=2,col=c('black','blue','red','orange'))
  }
}


##No/Location of optimum
OptimumLoc<-as.data.frame(matrix(NA,nrow=100,ncol = 32))
for (i in 1:4) { #for each example
  for (j in 1:4) { #for each MLA
    modelstr=paste('optimal_',MLAs[j],'_',Examples[i],sep='')
    Tunemodel=get(modelstr)
    TuneMtrans=optimal_GPR_tenbars$y
    count=1
    for (jj in as.numeric(rownames(TuneMtrans))) {
      OptimumLoc[jj,((2*j-1+8*(i-1)):(2*j+8*(i-1)))]<-TuneMtrans[count,]
      count=count+1
    }
    colnames(OptimumLoc)[((2*j-1+8*(i-1)):(2*j+8*(i-1)))]<-c(paste(MLAs[j],'_',Examples[i],'_RMSE_LOC',sep=''),
                                                             paste(MLAs[j],'_',Examples[i],'_MXAE_LOC',sep=''))
  }
}
write.xlsx(OptimumLoc, file = "./Data/TuningNtHoPTs_AllModels_OPtimumLocation.xlsx",
           sheetName="HOptHistory_Optimal_LOC", col.names=TRUE, row.names=TRUE, showNA=TRUE, password=NULL)
