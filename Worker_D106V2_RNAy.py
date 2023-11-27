#conda activate cellpose2&&python D:\Carlos\NMERFISH\Worker_D106_RNA.py

master_analysis_folder = r'/home/cfg001/Desktop/Coconut2NMERFISH'
lib_fl = master_analysis_folder+r'/codebook_code_color2__LouBBBrain_10_21V2_blank.csv'
### Did you compute PSF and median flat field images?
psf_file = master_analysis_folder+r'/psfs/psf_647_Kiwi.npy'
flat_field_tag = master_analysis_folder+r'/flat_field/D106_RNA__'
master_data_folder = r'/mnt/merfish10/20231107_D106LuoRMER/RNA'
save_folder =r'/mnt/merfish10/20231107_D106LuoRMER_analysis'
iHm=1
iHM=16


from multiprocessing import Pool, TimeoutError
import time,sys
import os,sys,numpy as np

sys.path.append(master_analysis_folder)
from ioMicroN import *

def compute_drift_features(save_folder,fov,all_flds,set_,redo=False,gpu=True):
    fls = [fld+os.sep+fov for fld in all_flds]
    for fl in fls:
        get_dapi_features(fl,save_folder,set_,gpu=gpu,im_med_fl = flat_field_tag+r'med_col_raw3.npz',
                    psf_fl = psf_file)
    
def main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method):
    im_ = read_im(fld+os.sep+fov)
    im__ = np.array(im_[icol],dtype=np.float32)
    
    if old_method:
        ### previous method
        im_n = norm_slice(im__,s=30)
        #Xh = get_local_max(im_n,500,im_raw=im__,dic_psf=None,delta=1,delta_fit=3,dbscan=True,
        #      return_centers=False,mins=None,sigmaZ=1,sigmaXY=1.5)
        Xh = get_local_maxfast_tensor(im_n,th_fit=500,im_raw=im__,dic_psf=None,delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5,gpu=False)
    else:
        ### new method
        fl_med = flat_field_tag+'med_col_raw'+str(icol)+'.npz'
        if os.path.exists(fl_med):
            im_med = np.array(np.load(fl_med)['im'],dtype=np.float32)
            im_med = cv2.blur(im_med,(20,20))
            im__ = im__/im_med*np.median(im_med)
        else:
            print("Did not find flat field")
        try:
            Xh = get_local_max_tile(im__,th=3600,s_ = 500,pad=100,psf=psf,plt_val=None,snorm=30,gpu=True,
                                    deconv={'method':'wiener','beta':0.0001},
                                    delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5)
        except:
            Xh = get_local_max_tile(im__,th=3600,s_ = 500,pad=100,psf=psf,plt_val=None,snorm=30,gpu=False,
                                    deconv={'method':'wiener','beta':0.0001},
                                    delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5)
    np.savez_compressed(save_fl,Xh=Xh)
def compute_fits(save_folder,fov,all_flds,redo=False,ncols=4,
                psf_file = psf_file,try_mode=True,old_method=False):
    psf = np.load(psf_file)
    redid = False   
    for fld in tqdm(all_flds):
        for icol in range(ncols-1):
            tag = os.path.basename(fld)
            save_fl = save_folder+os.sep+fov.split('.')[0]+'--'+tag+'--col'+str(icol)+'__Xhfits.npz'
            try:
                np.load(save_fl)['Xh']
                redo2=False
            except:
                redo2 = True
            if not os.path.exists(save_fl) or redo or redo2:
                if try_mode:
                    try:
                        main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method)
                    except:
                        print("Failed",fld,fov,icol)
                else:
                    main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method) 
                redid = True        
    return redid
def get_XHV2(dec,ncols=3,nbits=16,th_h=5000,filter_tag = ''):
    set_ = dec.set_
    drift_fl = dec.drift_fl
    drifts,all_flds,fov,fl_ref = pickle.load(open(drift_fl,'rb'))
    all_flds = [os.path.dirname(fld) for fld in all_flds if os.path.basename(fld)=='']
    dec.drifts,dec.all_flds,dec.fov,dec.fl_ref = drifts,all_flds,fov,fl_ref
    
    XH = []
    for iH in tqdm(np.arange(len(all_flds))):
        fld = all_flds[iH]
        ih = get_iH(fld) # get bit
        if ih>=1 and ih<=nbits:
            if filter_tag in os.path.basename(fld):
                for icol in range(ncols):
                    tag = os.path.basename(fld)
                    save_fl = save_folder+os.sep+fov.split('.')[0]+'--'+tag+'--col'+str(icol)+'__Xhfits.npy.npz'
                    if not os.path.exists(save_fl):save_fl = save_fl.replace('.npy','')
                    Xh = np.load(save_fl,allow_pickle=True)['Xh']
                    print(Xh.shape)
                    if len(Xh.shape):
                        Xh = Xh[Xh[:,-1]>th_h]
                        if len(Xh):
                            tzxy = drifts[iH][0]
                            Xh[:,:3]+=tzxy# drift correction
                            bit = ((ih-1)%nbits)*ncols+icol
                            icolR = np.array([[icol,bit]]*len(Xh))
                            XH_ = np.concatenate([Xh,icolR],axis=-1)
                            XH.extend(XH_)
    dec.XH = np.array(XH)
def compute_decoding(save_folder,fov,set_,redo=False):
    redid_decoding_tag=False
    dec = decoder_simple(save_folder,fov,set_)
    complete = dec.check_is_complete()
    try:
        np.load(dec.decoded_fl)['XH_pruned']
    except:
        complete=0
    if complete==0 or redo:
        #compute_drift(save_folder,fov,all_flds,set_,redo=False,gpu=False)
        dec = decoder_simple(save_folder,fov=fov,set_=set_)
        #dec.get_XH(fov,set_,ncols=3,nbits=16,th_h=10000,filter_tag = '')#number of colors match 
        get_XHV2(dec,ncols=3,nbits=16,th_h=7500,filter_tag = '')
        dec.XH = dec.XH[dec.XH[:,-4]>0.25] ### keep the spots that are correlated with the expected PSF for 60X
        dec.load_library(lib_fl,nblanks=-1)
        
        dec.ncols = 3
        if False:
            dec.XH_save = dec.XH.copy()
            def keep_best_N_for_each_Readout(dec,Nkeep = 15000,iH=-3):
                iRs = dec.XH[:,-1]
                iRsu = np.unique(iRs)
                H = dec.XH[:,iH]
                keep = []
                for iR in iRsu:
                    keep_ = np.where(iRs==iR)[0]
                    keep.extend(keep_[np.argsort(H[keep_])[::-1]][:Nkeep])
                dec.XH = dec.XH[keep]


            dec.XH = dec.XH_save.copy()
            for Nkeep,dist_th in [(15000,8),(15000,7),(15000,6),(30000,5),(60000,3)]:
                dec.XH_save_ = dec.XH.copy()
                keep_best_N_for_each_Readout(dec,Nkeep = Nkeep,iH=-3)
                dec.get_inters(dinstance_th=dist_th,enforce_color=True)# enforce_color=False
                dec.get_icodes(nmin_bits=4,method = 'top4',norm_brightness=-1)
                dec.XH_prunedT = dec.XH_pruned.copy()
                apply_fine_drift(dec,plt_val=False)
                #plt.show()
                dec.XH = dec.XH_save_.copy()
                R = dec.XH[:,-1].astype(int)
                dec.XH[:,:3] -= dec.drift_arr[R]
                
            dif = dec.XH.copy()
            dif[...,:3]=dec.XH_save[...,:3]-dec.XH[...,:3]
            iRs = dif[...,-1]
            iRsu,ctsR = np.unique(iRs,return_counts=True)
            iRsu = iRsu.astype(int)
            drift_arr = np.zeros([np.max(iRsu)+1,3])
            for iR in iRsu:
                keep = dif[...,-1]==iR
                drift_arr[iR]=np.mean(dif[keep][...,:3],axis=0)
            drift_arr_ = drift_arr.copy()
            drift_arr_ = np.median(drift_arr_.reshape([-1,dec.ncols,3]),axis=1)
            drift_arr_ = drift_arr_[np.repeat(np.arange(len(drift_arr_)),dec.ncols)]
            #drift_arr_
            print("Drift based on spots:")
            print(np.round(drift_arr,1))

            dec.XH = dec.XH_save.copy()
            R = dec.XH[:,-1].astype(int)
            dec.XH[:,:3] -= drift_arr_[R]
        #dec.get_inters(dinstance_th=2,enforce_color=True)# enforce_color=False
        dec.get_inters(dinstance_th=2,nmin_bits=4,enforce_color=True,redo=True)
        #dec.get_icodes(nmin_bits=4,method = 'top4',norm_brightness=None,nbits=24)#,is_unique=False)
        get_icodesV2(dec,nmin_bits=4,delta_bits=None,iH=-3,redo=False,norm_brightness=False,nbits=48,is_unique=True)
        redid_decoding_tag = True
    return redid_decoding_tag
def get_iH(fld): 
    try:
        return int(os.path.basename(fld).split('_')[0][1:])
    except:
        return np.inf

def get_files(set_ifov,iHm=iHm,iHM=iHM):
    master_folder = master_data_folder
    
    if not os.path.exists(save_folder): os.makedirs(save_folder)
        
    all_flds = [fld for fld in glob.glob(master_folder+r'/H*') if os.path.isdir(fld)]
    #all_flds += glob.glob(master_folder+r'\P*')
    
    ### reorder based on hybe
    all_flds = np.array(all_flds)[np.argsort([get_iH(fld)for fld in all_flds])] 
    
    set_,ifov = set_ifov
    all_flds = [fld for fld in all_flds if set_ in os.path.basename(fld)]
    
    all_flds = [fld for fld in all_flds if ((get_iH(fld)>=iHm) and (get_iH(fld)<=iHM))]
    
    
    fovs_fl = save_folder+os.sep+'fovsRNA__'+set_+'.npy'
    if not os.path.exists(fovs_fl):
        folder_map_fovs = all_flds[0]#[fld for fld in all_flds if 'low' not in os.path.basename(fld)][0]
        fls = glob.glob(folder_map_fovs+os.sep+'*.zarr')
        fovs = np.sort([os.path.basename(fl) for fl in fls])
        np.save(fovs_fl,fovs)
    else:
        fovs = np.sort(np.load(fovs_fl))
        #fovs = np.load(r'R:\20230908_D106Luo\DNA\AnalysisDeconvolveCG\fovs__.npy')
    fov=None
    if ifov<len(fovs):
        fov = fovs[ifov]
        all_flds = [fld for fld in all_flds if os.path.exists(fld+os.sep+fov)]
    return save_folder,all_flds,fov
        




############### New Code inserted here!!! ##################
### First copy ioMicro to the other computer from Scope1A1
### To analyze only D9 change items to ['_D9']
### Move the decodedNew files and the driftNew files to another folder #############!!!!!!
def compute_drift_features(save_folder,fov,all_flds,set_,redo=False,gpu=True):
    fls = [fld+os.sep+fov for fld in all_flds]
    for fl in fls:
        get_dapi_features(fl,save_folder,set_,gpu=gpu,im_med_fl = flat_field_tag+r'med_col_raw3.npz',
                    psf_fl = psf_file,redo=redo)
                    
def get_best_translation_pointsV2(fl,fl_ref,save_folder,set_,resc=5):
    
    obj = get_dapi_features(fl,save_folder,set_)
    obj_ref = get_dapi_features(fl_ref,save_folder,set_)
    tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus = np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),0,0
    if (len(obj.Xh_plus)>0) and (len(obj_ref.Xh_plus)>0):
        X = obj.Xh_plus[:,:3]
        X_ref = obj_ref.Xh_plus[:,:3]
        tzxy_plus,N_plus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)
    if (len(obj.Xh_minus)>0) and (len(obj_ref.Xh_minus)>0):
        X = obj.Xh_minus[:,:3]
        X_ref = obj_ref.Xh_minus[:,:3]
        tzxy_minus,N_minus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)
    if np.max(np.abs(tzxy_minus-tzxy_plus))<=2:
        tzxyf = -(tzxy_plus*N_plus+tzxy_minus*N_minus)/(N_plus+N_minus)
    else:
        tzxyf = -[tzxy_plus,tzxy_minus][np.argmax([N_plus,N_minus])]
    

    return [tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus]
def compute_drift_V2(save_folder,fov,all_flds,set_,redo=False,gpu=True):
    drift_fl = save_folder+os.sep+'driftNew_'+fov.split('.')[0]+'--'+set_+'.pkl'
    if not os.path.exists(drift_fl) or redo:
        fls = [fld+os.sep+fov for fld in all_flds]
        #fl_ref = fls[len(fls)//2]
        fl_ref = fls[0]
        newdrifts = []
        for fl in fls:
            drft = get_best_translation_pointsV2(fl,fl_ref,save_folder,set_,resc=5)
            print(drft)
            newdrifts.append(drft)
        pickle.dump([newdrifts,all_flds,fov,fl_ref],open(drift_fl,'wb'))
def compute_main_f(save_folder,all_flds,fov,set_,ifov,redo_fits,redo_drift,redo_decoding,try_mode,old_method):
    print("Computing fitting on: "+str(fov))
    print(len(all_flds),all_flds)
    redid_fit = compute_fits(save_folder,fov,all_flds,redo=redo_fits,try_mode=try_mode,old_method=old_method)
    print("Computing drift on: "+str(fov))
    #compute_drift(save_folder,fov,all_flds,set_,redo=redo_drift)
    compute_drift_features(save_folder,fov,all_flds,set_,redo=redo_drift,gpu=True)
    compute_drift_V2(save_folder,fov,all_flds,set_,redo=redo_drift,gpu=True)
    redid_decoding_tag = compute_decoding(save_folder,fov,set_,redo=(redo_decoding or redid_fit))
    return redid_decoding_tag
    
############### End Code inserted here!!! ##################

def main_f(set_ifov,redo_fits = False,redo_drift=False,redo_decoding=False,try_mode=True,old_method=False):
    set_,ifov = set_ifov
    
    save_folder,all_flds,fov = get_files(set_ifov)
    
    fl = save_folder+os.sep+'coconut2__'+str(ifov)+'.txt'
    
    if True:#not os.path.exists(fl):
        if try_mode:
            try:
                redid_decoding_tag = compute_main_f(save_folder,all_flds,fov,set_,ifov,redo_fits,redo_drift,redo_decoding,try_mode,old_method)
                if redid_decoding_tag:
                    fid = open(fl,'w')
                    fid.close()
            except:
                print("Failed within the main analysis:")
        else:
            redid_decoding_tag = compute_main_f(save_folder,all_flds,fov,set_,ifov,redo_fits,redo_drift,redo_decoding,try_mode,old_method)
            if redid_decoding_tag:
                    fid = open(fl,'w')
                    fid.close()
    
    
    return set_ifov
    

    
if __name__ == '__main__':
    # start 4 worker processes
    items = [('',ifov) for ifov in np.arange(1300)]
    #items = [('',ifov) for ifov in [94, 99, 129, 149, 154, 155, 156, 157, 158, 159, 171, 175, 180, 181, 193, 194, 195, 211, 212]]
    #items = [('',ifov) for ifov in [17, 37, 38, 39, 40, 44, 45, 46, 47, 48, 50, 51, 55, 56, 58, 60, 63, 68, 70, 71, 77, 84, 85, 86, 87, 88, 91, 92, 107, 108, 110, 116, 120, 126, 131, 133, 159, 183, 208, 215, 229, 243, 246, 248, 265, 267, 269]]
    #print("Found decode fls:",len(glob.glob(save_folder+os.sep+'driftNew*')))
    #main_f(['_set2',81],redo_decoding=False,try_mode=False)#_010--_set2
    ifovs_ = [ 1000]
    items_ = [('',ifov) for ifov in ifovs_]
    for item in items_:
        main_f(item,redo_drift=False,redo_decoding=False,try_mode=False)
    if False:
        with Pool(processes=4) as pool:
            print('starting pool')
            result = pool.map(main_f, items)
#conda activate cellpose&&python /home/cfg001/Desktop/Coconut2NMERFISH/Worker_D106V2_RNAy.py
