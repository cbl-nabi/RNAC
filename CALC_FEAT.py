import numpy
import h5py
import pandas as pd
from os import listdir
from scipy.stats import kurtosis,moment,skew,entropy,gmean,median_absolute_deviation,gstd
from re import search
from warnings import filterwarnings
filterwarnings('ignore')
numpy.seterr(divide='ignore',invalid='ignore')


class feat_extract:        
    def compute(self,idxs,data,ldata,bounds,lbp_idx):
        idxs=idxs-1
        chk=numpy.where(idxs<1)
        if(len(chk[0])>0):
            idxs=numpy.delete(idxs,chk[0])
        chk=numpy.where(idxs>=data.shape[0])
        if(len(chk[0])>0):
            idxs=numpy.delete(idxs,chk[0])
        feat=numpy.zeros((1,135))
        if(len(idxs)==1):
            val=data[idxs[0]:idxs[0]+1]
            lbpval=ldata[idxs[0]:idxs[0]+1]
        else:
            val=data[idxs]
            lbpval=ldata[idxs]
        if(len(val)==0):
            return feat
        feat[0,0:17]=self.stat_feats(val)
        feat[0,17:75]=self.global_lbp(lbpval,lbp_idx)
        feat[0,75:135]=numpy.histogram(val,range=bounds,bins=60)[0]
        return feat

    def stat_feats(self,val):
        feat=numpy.zeros((1,17))
        feat[0,0]=numpy.mean(val)
        feat[0,1]=numpy.max(val)
        feat[0,2]=numpy.var(val,ddof=1)
        feat[0,3]=numpy.min(val)
        feat[0,4]=numpy.median(val)
        feat[0,5]=numpy.quantile(val,0.9)
        feat[0,6]=numpy.quantile(val,0.7)
        feat[0,7]=numpy.var(val,ddof=1)/numpy.mean(val)
        feat[0,8]=skew(val)
        feat[0,9]=kurtosis(val)
        feat[0,10]=moment(val,moment=2)
        feat[0,11]=moment(val,moment=3)
        feat[0,12]=moment(val,moment=4)
        feat[0,13]=entropy(val)
        feat[0,14]=gmean(val)
        feat[0,15]=median_absolute_deviation(val)
        if(len(val)>1):
            feat[0,16]=gstd(val)
        else:
            feat[0,16]=numpy.nan
        return feat
    
    def global_lbp(self,lbp_val,x_idx):
        lbp_feat=numpy.histogram(lbp_val,range=(0,256),bins=256)[0]
        lbp_feat=lbp_feat[x_idx]
        return lbp_feat
    
    def get_uni_idx(self):
        ls=numpy.zeros(58).astype('int')
        cont=0
        for i in range(256):
            x=numpy.array(list(format(i,"08b"))).astype('int')
            xd=numpy.roll(x,-1)
            S=sum(numpy.logical_xor(x,xd).astype('int'))
            if(S<=2):
                ls[cont]=i
                cont=cont+1
        return ls

class feat_ext:
    def bins(self,dfx):
        OR=0.50     #Overlapping ratio, 50%
        RI=50       #Half bin size, 100/2=50    
        Over=OR*RI  #Overlapped length
        FRANG=[]
        for y in dfx.index:
            S=dfx.loc[y,'start']
            E=dfx.loc[y,'end']
            N1=int((S-1)/RI)+1
            N2=int((E-1)/RI)+0
            n1=(S-1)%RI
            n2=(E-1)%RI
            if(n1>=Over):
                N1=N1+1
            if(n2>=Over):
                N2=N2+1
            if(N2>=N1):
                FRANG.extend(list(range(N1,N2+1)))
            else:
                FRANG.extend(list(range(0,1)))
        return FRANG

    def correct_order(self,sel_exon):
        total_exons=sel_exon.shape[0]
        str_idx=sel_exon.loc[0,'start']
        end_idx=sel_exon.loc[total_exons-1,'start']
        if(str_idx>end_idx):
            sel_exon1=pd.DataFrame(columns=sel_exon.columns)
            sel_exon1=sel_exon1.append(sel_exon.iloc[::-1],ignore_index=True)
            return sel_exon1
        else:
            return sel_exon

    def calc_BIN(self,BDATA,LDATA,IDX,STRAND,chromo,TMPL,BD):
        feat=numpy.zeros((len(STRAND),1215),dtype='float32')
        label=numpy.zeros(len(STRAND))
        MARK=[]
        step=135
        cont=0
        FE=feat_extract()
        LBP_idx=FE.get_uni_idx()
        chromo='chr'+chromo
        for i in range(0,len(BDATA)):
            conti=0
            dn = BDATA[i][chromo+'.-'][:]
            dp = BDATA[i][chromo+'.+'][:]
            ldn = LDATA[i][chromo+'.-'][:]
            ldp = LDATA[i][chromo+'.+'][:]
            len_data = len(dn)
            for j in range(0,len(STRAND)):
                indexes=numpy.array(IDX[j])
                end_idx=indexes[-1]
                if( end_idx >= len_data):
                    if(j not in MARK):
                        MARK.append(j)
                        feat=numpy.delete(feat,-1,0)
                        label=numpy.delete(label,-1,0)
                    continue
                if(STRAND[j] == '-'):
                    bds = BD[(BD['chromo']==chromo) & (BD['feat']==i) & (BD['strand']=='-')]
                    bounds = (bds.loc[bds.index[0],'min'],bds.loc[bds.index[0],'max'])
                    feat[conti,cont:cont+step] = FE.compute(indexes,dn,ldn,bounds,LBP_idx)
                elif(STRAND[j] == '+'):
                    bds = BD[(BD['chromo']==chromo) & (BD['feat']==i) & (BD['strand']=='+')]
                    bounds = (bds.loc[bds.index[0],'min'],bds.loc[bds.index[0],'max'])
                    feat[conti,cont:cont+step] = FE.compute(indexes,dp,ldp,bounds,LBP_idx)
                label[conti]=TMPL[j]
                conti=conti+1
            cont=cont+step
        return feat,label
    
    def get_data(self,species):
        #Read original and LBP genomic descriptors
        BDATA=[]
        LDATA=[]
        bio_data_files='./DESC/'
        bio_feat=sorted(listdir(bio_data_files))
        for hfile_name in bio_feat:
            if(search(species,hfile_name)):
                BDATA.append(h5py.File('./DESC/'+'/'+hfile_name,'r'))
                LDATA.append(h5py.File('./LBPDESC/'+hfile_name,'r'))
        BD=pd.read_csv('./UTILS/bounds_'+species+'.info')
        return BDATA,LDATA,BD   