import numpy
import h5py
from os import listdir
from re import search

class LBP_feat:
    def get_lbpchr(self,val):
        ln=val.shape[0]
        lbp_val=numpy.zeros(ln)
        pad_val=numpy.zeros(ln+8)
        idx=4
        step=9
        pad_val[idx:idx+ln]=val
        for i in range(0,ln):
            str_idx=i
            end_idx=i+step
            tmp_val=pad_val[str_idx:end_idx]
            bin_values=(tmp_val>=tmp_val[4]).astype('int')
            bin_values=numpy.delete(bin_values,4)
            x=2**numpy.arange(8)[::-1]
            lbp_val[i]=x@bin_values
        return lbp_val
    
    def comp_LBP(self,species):
        if(species=='human'):
            classeschr=numpy.array(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y'])
        elif(species=='mouse'):
            classeschr=numpy.array(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','X','Y'])
        elif(species=='worm'):
            classeschr=numpy.array(['I','II','III','IV','V','X'])
        elif(species=='plant'):
            classeschr=['1','2','3','4','5']
        classeschr=['chr'+s for s in classeschr]
        
        bio_data_files='./DESC/'
        bio_feat=sorted(listdir(bio_data_files))
        for hfile_name in bio_feat:
            if(search(species,hfile_name)):
                re_F = h5py.File('./DESC/'+'/'+hfile_name,'r')
                hf = h5py.File('./LBPDESC/'+hfile_name,'w')
                for j in range(0,len(classeschr)):
                    dn=re_F[classeschr[j]+'.-'][:]
                    dp=re_F[classeschr[j]+'.+'][:]
                    LBPP=self.get_lbpchr(dp)
                    LBPN=self.get_lbpchr(dn)
                    hf.create_dataset(classeschr[j]+'.-', data=LBPN,compression="gzip")
                    hf.create_dataset(classeschr[j]+'.+', data=LBPP,compression="gzip")
                hf.close()

if __name__ == "__main__":
    LB=LBP_feat()
    LB.comp_LBP('plant')
    LB.comp_LBP('worm')
    LB.comp_LBP('mouse')
    LB.comp_LBP('human')