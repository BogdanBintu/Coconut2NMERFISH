#conda activate cellpose2&&python D:\Carlos\NMERFISH\Worker_D106_RNA.py

master_analysis_folder = r'C:\Scripts\CarlosRNAScripts'



from multiprocessing import Pool, TimeoutError
import time,sys
import os,sys,numpy as np

sys.path.append(master_analysis_folder)
from ioMicroN import *

def get_dic_driftZ(X,X_ref,delta=2,Nmin=100):
    zs = X[:,0]
    zs_ref = X_ref[:,0]
    tzxy_plus,N_plus = get_best_translation_points(X,X_ref,resc=5,return_counts=True)
    zdif = tzxy_plus[0]
    zref=10
    dic_driftZ = {}
    for zref in np.arange(int(np.max(zs_ref))):
        X_ = X[np.abs(zs-zref-zdif)<delta]
        X_ref_ = X_ref[np.abs(zs_ref-zref)<delta]
        if (len(X_)>Nmin) and (len(X_ref_)>Nmin):
            tzxy_plus,N_plus = get_best_translation_points(X_,X_ref_,resc=5,return_counts=True)
            #print(tzxy_plus,N_plus,zref)
            if N_plus>Nmin:
                dic_driftZ[zref]=[tzxy_plus,N_plus]
    return dic_driftZ
def main_f_t(fl_drift):
    fl_drift_ = fl_drift.replace('driftNew','dicdrift')
    save_folder = r'S:\20231024_D106Luo_RNA_analysis'
    if not os.path.exists(fl_drift_):
        drifts,flds,fov_,fl_ref = np.load(fl_drift,allow_pickle=True)
        dic_driftZ_minusS,dic_driftZ_plusS = [],[]
        for fld in tqdm(flds):
            fl = fld+os.sep+fov_
            obj = get_dapi_features(fl,save_folder,'')
            obj_ref = get_dapi_features(fl_ref,save_folder,'')
            dic_driftZ_minus = get_dic_driftZ(obj.Xh_minus[:,:3],obj_ref.Xh_minus[:,:3],delta=2)
            dic_driftZ_plus = get_dic_driftZ(obj.Xh_plus[:,:3],obj_ref.Xh_plus[:,:3],delta=2)
            dic_driftZ_minusS.append(dic_driftZ_minus)
            dic_driftZ_plusS.append(dic_driftZ_plus)
        pickle.dump([[dic_driftZ_plusS,dic_driftZ_minusS],flds,fov_,fl_ref],open(fl_drift_,'wb'))
def main_f(fl_drift,try_mode=True):
    if try_mode:
        try:
            main_f_t(fl_drift)
        except:
            print("Failed:",fl_drift)
    else:
        main_f_t(fl_drift)
if __name__ == '__main__':
   
    #fls_ = np.sort(glob.glob(r'S:\20231024_D106Luo_RNA_analysis\drift*'))
    fls_ = np.load(r'S:\20231024_D106Luo_RNA_analysis\fls_drift.npy')
    main_f(fls_[10],try_mode=False)
    if True:
        with Pool(processes=15) as pool:
            print('starting pool')
            result = pool.map(main_f, fls_)
#conda activate cellpose&&python C:\Scripts\NMERFISHNEW\driftMultiplex.py
