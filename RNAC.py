import numpy
from xgboost import XGBClassifier
import scipy.io
import pandas as pd
from CALC_FEAT import feat_ext
from gtfparse import read_gtf
import sys


class preprocess:
    def multiclass_problem(self,RNA_types,dfex):
        classes=['protein_coding','Housekeeping','sncRNA','lncRNA']
        new_dfex=pd.DataFrame(columns=dfex.columns)
        new_dfex=new_dfex.append(dfex.loc[dfex['transcript_type'].isin(RNA_types)],ignore_index=True)
        
        HK=['tRNA','rRNA']
        new_dfex=new_dfex.replace({'transcript_type':HK},{'transcript_type':'Housekeeping'},regex=True)
        
        LRNA=['lincRNA','antisense_RNA','antisense','sense_intronic','sense_overlapping']
        new_dfex=new_dfex.replace({'transcript_type':LRNA},{'transcript_type':'lncRNA'},regex=True)
        
        dc=dfex.index[dfex['transcript_type']=='pre_miRNA']
        if(len(dc)>0):
            SRNA=['snRNA','snoRNA','pre_miRNA']
        else:
            SRNA=['snRNA','snoRNA','miRNA']
        new_dfex=new_dfex.replace({'transcript_type':SRNA},{'transcript_type':'sncRNA'},regex=True)
        return classes,new_dfex

    def remove_nans(self,test_feat):        
        check_nan=numpy.argwhere(numpy.isnan(test_feat))
        for i in range(0,len(check_nan)):
            test_feat[check_nan[i,0],check_nan[i,1]]=0
        
        check_inf=numpy.argwhere(numpy.isinf(test_feat))
        for i in range(0,len(check_inf)):
            test_feat[check_inf[i,0],check_inf[i,1]]=0
        
        check_largevals=numpy.where(test_feat>1e5)
        for i in range(0,len(check_largevals[0])):
            test_feat[check_largevals[0][i],check_largevals[1][i]]=1e5
        
        check_largevals=numpy.where(test_feat<-1e5)
        for i in range(0,len(check_largevals[0])):
            test_feat[check_largevals[0][i],check_largevals[1][i]]=-1e5
        return test_feat
    
    ##Data Normalization
    def MinMax(self,data,mn,mx):
        n_data=numpy.zeros(data.shape)
        for i in range(0,data.shape[1]):
            n_data[:,i]=(data[:,i]-mn[i])/(numpy.finfo(numpy.float).eps+mx[i]-mn[i])
        return n_data

def get_binary(tslabel):
    idx=numpy.where(tslabel==2)[0]
    tslabel[idx]=1
    idx=numpy.where(tslabel==3)[0]
    tslabel[idx]=1
    return tslabel

def get_feat(gtf_file,species):
    nclass=['protein_coding','Housekeeping','sncRNA','lncRNA']
    RNA_types=['protein_coding','tRNA','rRNA','lincRNA','antisense_RNA','antisense','sense_intronic','sense_overlapping','miRNA','pre_miRNA','snRNA','snoRNA']
    gfile=read_gtf(gtf_file)
    if(species not in ['human','mouse']):
        cols=['seqname','feature','start','end','strand','transcript_id','source']
        df=gfile[cols]
        df.columns=['seqname','feature','start','end','strand','transcript_id','transcript_type']
        mark=1
    else:
        cols=['seqname','feature','start','end','strand','transcript_id','transcript_type']
        df=gfile[cols]
        mark=0
    R=numpy.unique(gfile['transcript_id'])
    dc=df.index[df['feature']=='exon']
    dfex=pd.DataFrame(columns=df.columns)
    dfex=dfex.append(df.loc[dc],ignore_index=True)
    P=preprocess()
    classes,ex_data=P.multiclass_problem(RNA_types,dfex)
    Y=feat_ext()
    BDATA,LDATA,BD=Y.get_data(species)
    TR=[]
    TRP=[]
    for i in range(0,len(R)):
        sel_exon=ex_data.loc[ex_data['transcript_id']==R[i]].reset_index()
        sel_exon=Y.correct_order(sel_exon)
        indexes=Y.bins(sel_exon)
        if(len(indexes)>0):
            IDX=[indexes]
            STRAND=[sel_exon['strand'][0]]
        else:
            print('Skipping Transcript '+R[i])
            continue
        for x in range(0,len(nclass)):
            if(sel_exon.loc[0,'transcript_type']==nclass[x]):
                TMPL=[x]
                break
        if(mark==1):
            ff,ll=Y.calc_BIN(BDATA,LDATA,IDX,STRAND,sel_exon.loc[0,'seqname'],TMPL,BD)
        else:
            ff,ll=Y.calc_BIN(BDATA,LDATA,IDX,STRAND,sel_exon.loc[0,'seqname'][3:],TMPL,BD)
        TR.append(R[i])
        TRP.append(sel_exon.loc[0,'transcript_type'])
        if(i==0):
            feat=ff
            label=ll
        else:
            feat=numpy.concatenate((feat,ff),axis=0)
            label=numpy.concatenate((label,ll),axis=0)
    
    TRD = pd.DataFrame(TR,columns=['Transcript_id'])
    TRD.insert(1,'Transcript_type',TRP)
    return feat,label,TRD

def Main_Multi(gtf_file,species):
    P=preprocess()
    feat,label,TRD=get_feat(gtf_file,species)
    
    BDN=scipy.io.loadmat('./UTILS/norm.mat')['bd']
    feat=P.remove_nans(feat)
    
    model = XGBClassifier()
    if(species=='human'):
        nfeat=P.MinMax(feat,BDN[0,:],BDN[1,:])
        model.load_model('./MODELS/human.Multiclass')
        h=scipy.io.loadmat('./UTILS/fs_human.mat')['fs_list'].reshape(-1)
        K=784
    elif(species=='mouse'):
        nfeat=P.MinMax(feat,BDN[2,:],BDN[3,:])
        model.load_model('./MODELS/mouse.Multiclass')
        h=scipy.io.loadmat('./UTILS/fs_mouse.mat')['fs_list'].reshape(-1)
        K=484
    elif(species=='worm'):
        nfeat=P.MinMax(feat,BDN[4,:],BDN[5,:])
        model.load_model('./MODELS/Celegans.Multiclass')
        h=scipy.io.loadmat('./UTILS/fs_worm.mat')['fs_list'].reshape(-1)
        K=34
    elif(species=='plant'):
        nfeat=P.MinMax(feat,BDN[6,:],BDN[7,:])
        model.load_model('./MODELS/Arabidopsisthaliana.Multiclass')
        h=scipy.io.loadmat('./UTILS/fs_plant.mat')['fs_list'].reshape(-1)
        K=139
    xtest=nfeat[:,h[:K]]
    pred = model.predict(xtest)
    pred_l=[]
    for i in range(0,len(pred)):
        if(pred[i]==0):
            pred_l.append('CRNA')
        elif(pred[i]==1):
            pred_l.append('HKRNA')
        elif(pred[i]==2):
            pred_l.append('SNCRNA')
        elif(pred[i]==3):
            pred_l.append('LNCRNA')
    
    TRD.insert(2,'Prediction',pred_l)
    TRD.to_csv('./TEST_OUTPUT/'+species+'_Multiclaspred',index=False)

def Main_Binary(gtf_file,species):
    P=preprocess()
    feat,label,TRD=get_feat(gtf_file,species)
    
    BDN=scipy.io.loadmat('./UTILS/norm.mat')['bd']
    feat=P.remove_nans(feat)
    label=get_binary(label)
    model = XGBClassifier()
    if(species=='human'):
        nfeat=P.MinMax(feat,BDN[0,:],BDN[1,:])
        model.load_model('./MODELS/human.Binary')
        h=scipy.io.loadmat('./UTILS/fs_human.mat')['fs_list'].reshape(-1)
        K=784
    elif(species=='mouse'):
        nfeat=P.MinMax(feat,BDN[2,:],BDN[3,:])
        model.load_model('./MODELS/mouse.Binary')
        h=scipy.io.loadmat('./UTILS/fs_mouse.mat')['fs_list'].reshape(-1)
        K=484
    elif(species=='worm'):
        nfeat=P.MinMax(feat,BDN[4,:],BDN[5,:])
        model.load_model('./MODELS/Celegans.Binary')
        h=scipy.io.loadmat('./UTILS/fs_worm.mat')['fs_list'].reshape(-1)
        K=34
    elif(species=='plant'):
        nfeat=P.MinMax(feat,BDN[6,:],BDN[7,:])
        model.load_model('./MODELS/Arabidopsisthaliana.Binary')
        h=scipy.io.loadmat('./UTILS/fs_plant.mat')['fs_list'].reshape(-1)
        K=139
    
    xtest=nfeat[:,h[:K]]
    pred = model.predict_proba(xtest)[:,0]
    
    pred_l=[]
    for i in range(0,len(pred)):
        if(pred[i]>=0.5):
            pred_l.append('CRNA')
        else:
            pred_l.append('NCRNA')
    
    TRD.insert(2,'Probabilty',pred)
    TRD.insert(3,'Prediction',pred_l)
    TRD.to_csv('./TEST_OUTPUT/'+species+'_Binarypred',index=False)

if __name__ == "__main__":
    if(len(sys.argv) == 1 or len(sys.argv) == 3 or len(sys.argv) > 4):
        print('Incorrect number of input parameters. Type python RNAC.py -h or --help for help \n\n')
        sys.exit()
        
    if(len(sys.argv) == 2 and sys.argv[1] == '-h' or sys.argv[1] == '--help'):
        print("Usage: python RNAC.py [--classification-type] [--species] [-gtf_file] \n\n" +
              "classification-type: \t {Multiclass, Binary} \n" +
              "species: \t\t {Human, Mouse, Caenorhabditis_elegans, Arabidopsis_thaliana} \n" +
              "gtf_file: \t\t GTF file location \n\n"+
              "Example: python RNAC.py --Binary --Caenorhabditis_elegans ./TEST_SAMPLES/Caenorhabditis_elegans.gtf \n\n"+
              "Note: RNAC supports Multiclass (cRNA, hkRNA, sncRNA and lncRNA) and Binary (coding vs non-coding) classification of these four species only.\n\n")
        sys.exit()
        
    if(sys.argv[1]=='--Multiclass' and sys.argv[2] == '--Human'):
        Main_Multi(sys.argv[3],'human')
    elif(sys.argv[1]=='--Binary' and sys.argv[2] == '--Human'):
        Main_Binary(sys.argv[3],'human')
    elif(sys.argv[1]=='--Multiclass' and sys.argv[2] == '--Mouse'):
        Main_Multi(sys.argv[3],'mouse')
    elif(sys.argv[1]=='--Binary' and sys.argv[2] == '--Mouse'):
        Main_Binary(sys.argv[3],'mouse')
    elif(sys.argv[1]=='--Multiclass' and sys.argv[2] == '--Caenorhabditis_elegans'):
        Main_Multi(sys.argv[3],'worm')
    elif(sys.argv[1]=='--Binary' and sys.argv[2] == '--Caenorhabditis_elegans'):
        Main_Binary(sys.argv[3],'worm')
    elif(sys.argv[1]=='--Multiclass' and sys.argv[2] == '--Arabidopsis_thaliana'):
        Main_Multi(sys.argv[3],'plant')
    elif(sys.argv[1]=='--Binary' and sys.argv[2] == '--Arabidopsis_thaliana'):
        Main_Binary(sys.argv[3],'plant')
    else:
        print('Unknown classification category or species. Type python RNAC.py -h or --help for help \n\n')