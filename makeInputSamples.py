import copy
import pandas as pd
import numpy as np
from ROOT import TVector3
from root_pandas import read_root
from root_pandas import to_root 
from sklearn.model_selection import train_test_split

from utils.selections import preselection, preselection_mc, pass_id
def add_cartesian_vars(df):
  pv_sv_dist = []
  pv_sv_eta = []
  pv_sv_phi = []
  bpx = []
  bpy = []
  bpz = []
  mu1px = []
  mu1py = []
  mu1pz = []
  mu2px = []
  mu2py = []
  mu2pz = []
  kpx = []
  kpy = []
  kpz = []

  isMC =  "mu1_gen_vx" in df

  for index, entry in df.iterrows():
    pv_p3 = TVector3(entry.beamspot_x, entry.beamspot_y, entry.pv_z)
    sv_p3 = TVector3(entry.bvtx_vtx_x, entry.bvtx_vtx_y, entry.bvtx_vtx_z)
    '''
    if isMC:
      pv_p3.SetXYZ(entry.mu1_grandmother_vx, entry.mu1_grandmother_vy, entry.mu1_grandmother_vz)
      sv_p3.SetXYZ(entry.mu1_mother_vx, entry.mu1_mother_vy, entry.mu1_mother_vz)
    '''
    pv_sv_dist.append((sv_p3 - pv_p3).Mag())
    pv_sv_eta.append((sv_p3 - pv_p3).Eta())
    pv_sv_phi.append((sv_p3 - pv_p3).Phi())
    

    b_p3 = TVector3()
    k_p3 = TVector3()
    mu1_p3 = TVector3()
    mu2_p3 = TVector3()

    b_p3.SetPtEtaPhi(entry.Bpt, entry.Beta, entry.Bphi)
    k_p3.SetPtEtaPhi(entry.kpt, entry.keta, entry.kphi)
    mu1_p3.SetPtEtaPhi(entry.mu1pt, entry.mu1eta, entry.mu1phi)
    mu2_p3.SetPtEtaPhi(entry.mu2pt, entry.mu2eta, entry.mu2phi)

    bpx.append(b_p3.Px())
    bpy.append(b_p3.Py())
    bpz.append(b_p3.Pz())

    kpx.append(k_p3.Px())
    kpy.append(k_p3.Py())
    kpz.append(k_p3.Pz())

    mu1px.append(mu1_p3.Px())
    mu1py.append(mu1_p3.Py())
    mu1pz.append(mu1_p3.Pz())

    mu2px.append(mu2_p3.Px())
    mu2py.append(mu2_p3.Py())
    mu2pz.append(mu2_p3.Pz())

  df['pv_sv_dist'] = pv_sv_dist
  df['pv_sv_eta'] = pv_sv_eta
  df['pv_sv_phi'] = pv_sv_phi

  df['Bpx'] = bpx
  df['Bpy'] = bpy
  df['Bpz'] = bpz
  
  df['kpx'] = kpx
  df['kpy'] = kpy
  df['kpz'] = kpz

  df['mu1px'] = mu1px
  df['mu1py'] = mu1py
  df['mu1pz'] = mu1pz

  df['mu2px'] = mu2px
  df['mu2py'] = mu2py
  df['mu2pz'] = mu2pz
  
  df['ip3d_s'] = df.ip3d.div(df.ip3d_e)
  return df



def main():
  trigger = "mu1_isFromMuT & mu2_isFromMuT & k_isFromMuT"
  selection_mc = " & ".join([ preselection_mc, pass_id, trigger])
  selection_data = " & ".join([ preselection, pass_id, trigger])
  selection_minimal = ' & '.join([
    'Bmass < 6.2',
    'abs(jpsi_mass - 3.0969)<0.1',
    'mu1_mediumID>0.5',
    'mu2_mediumID>0.5',
    trigger,
    pass_id])

  #BcToJPsiMuMu_is_chic0_mu_merged.root     BcToJPsiMuMu_is_jpsi_3pi_merged.root
  #BcToJPsiMuMu_is_jpsi_tau_merged.root     data_ptmax_merged.root
  #BcToJPsiMuMu_is_chic1_mu_merged.root     BcToJPsiMuMu_is_jpsi_hc_merged.root
  #BcToJPsiMuMu_is_psi2s_mu_merged.root     HbToJPsiMuMu3MuFilter_ptmax_merged.root
  #BcToJPsiMuMu_is_chic2_mu_merged.root     BcToJPsiMuMu_is_jpsi_mu_merged.root
  #BcToJPsiMuMu_is_psi2s_tau_merged.root    HbToJPsiMuMu_ptmax_merged.root
  #BcToJPsiMuMu_is_hc_mu_merged.root        BcToJPsiMuMu_is_jpsi_pi_merged.root
  #datalowmass_ptmax_merged.root
  #Open original files
  all_files_names = ['BcToJPsiMuMu_is_chic0_mu_merged.root',
                     'BcToJPsiMuMu_is_jpsi_tau_merged.root',
                     'BcToJPsiMuMu_is_chic1_mu_merged.root',
                     'BcToJPsiMuMu_is_psi2s_mu_merged.root',
                     'BcToJPsiMuMu_is_chic2_mu_merged.root',
                     #'BcToJPsiMuMu_is_psi2s_tau_merged.root',
                     'BcToJPsiMuMu_is_hc_mu_merged.root',
                     #'BcToJPsiMuMu_is_jpsi_3pi_merged.root',
                     'BcToJPsiMuMu_is_jpsi_hc_merged.root',
                     'HbToJPsiMuMu3MuFilter_ptmax_merged.root',
                     'BcToJPsiMuMu_is_jpsi_mu_merged.root',
                     'HbToJPsiMuMu_ptmax_merged.root',
                     #'BcToJPsiMuMu_is_jpsi_pi_merged.root'
                    ]
  
  all_files_names1 = [
                     'BcToJPsiMuMu_is_chic0_mu_merged.root',
                     'BcToJPsiMuMu_is_jpsi_tau_merged.root',
                     'BcToJPsiMuMu_is_chic1_mu_merged.root',
                     'BcToJPsiMuMu_is_psi2s_mu_merged.root',
                     'BcToJPsiMuMu_is_chic2_mu_merged.root',
                     #'BcToJPsiMuMu_is_psi2s_tau_merged.root',
                     'BcToJPsiMuMu_is_hc_mu_merged.root',
                     #'BcToJPsiMuMu_is_jpsi_3pi_merged.root',
                     'BcToJPsiMuMu_is_jpsi_hc_merged.root',
                     'BcToJPsiMuMu_is_jpsi_mu_merged.root',
                     #'BcToJPsiMuMu_is_jpsi_pi_merged.root'
                    ]
  all_files_names2 = ['HbToJPsiMuMu_ptmax_merged.root']
  all_files_names3 = ['HbToJPsiMuMu3MuFilter_ptmax_merged.root']

  out_dir = 'data'
  #input_dir = "/gpfs/ddn/srm/cms/store/user/garamire/ntuples/2021Mar23/"
  input_dir = '/scratch/parolia/2021Mar23/'
  data_file = '/gpfs/ddn/srm/cms/store/user/garamire/ntuples/2021Mar23/data_ptmax_merged.root'
  norm_file = "/gpfs/ddn/srm/cms/store/user/garamire/ntuples/2021Mar23/BcToJPsiMuMu_is_jpsi_mu_merged.root"
  signal_file = "/gpfs/ddn/srm/cms/store/user/garamire/ntuples/2021Mar23/BcToJPsiMuMu_is_jpsi_tau_merged.root"
  input_tree = 'BTo3Mu'

  '''
  print("DATA...")
  print("----------------------------------------------------------------------------------")
  print("Reading data ntuple")
  #data_df = read_root(data_file, input_tree).query(selection_data).copy()
  data_df = read_root(data_file, input_tree).query(selection_minimal).copy()
  print("Adding momentum in square coordinates for data")
  data_df = add_cartesian_vars(data_df)
  print("")
  print("Saving %s/alldata_selected.root"%(out_dir))
  data_df.to_root("%s/alldata_selected.root"%(out_dir), key=input_tree)
  '''

  print("")
  print("")
  print("")


  print("MC Bkg...")
  print("----------------------------------------------------------------------------------")
  print("Reading all mc files")
  all_mix_dfs_1 = [read_root(input_dir + sample, input_tree).query(selection_minimal) for sample in all_files_names1]
  f1 = 0.52/8.5
  f2 = 6.7/8.5
  all_mix_dfs_1_accepted = [] 
  all_mix_dfs_1_rejected = []
  for df in all_mix_dfs_1:
    all_mix_dfs_1_a, all_mix_dfs_1_r= train_test_split( df,
      test_size = f1,
      random_state = 0,
      shuffle = True) 
    all_mix_dfs_1_accepted.append(all_mix_dfs_1_a)
    all_mix_dfs_1_rejected.append(all_mix_dfs_1_r)

  all_mix_dfs_2 = read_root(input_dir + all_files_names2[0], input_tree).query(selection_minimal).copy() 
  all_mix_dfs_2_accepted, all_mix_dfs_2_rejected= train_test_split( all_mix_dfs_2,
      test_size = f2,
      random_state = 0,
      shuffle = True) 
  all_mix_dfs_3 = read_root(input_dir + all_files_names3[0], input_tree).query(selection_minimal)

  #all_mix_df = pd.concat([read_root(input_dir + sample, input_tree) for sample in all_files_names]).query(selection_data).copy()
  all_mix_dfs_1_accepted_concat = pd.concat(all_mix_dfs_1_accepted)
  all_mix_df = pd.concat([all_mix_dfs_1_accepted_concat, all_mix_dfs_2_accepted, all_mix_dfs_3])
  print("Adding momentum in square coordinates for all MC")
  all_mix_df = add_cartesian_vars(all_mix_df)
  print("Saving %s/mc_all.root"%(out_dir))
  all_mix_df.to_root("%s/mc_all.root"%(out_dir), key=input_tree)

  print("")
  print("")
  print("")

  print("MC Signal...")
  print("----------------------------------------------------------------------------------")
  print("Reading mu channel")
  norm_df = read_root(norm_file, input_tree).query(selection_mc).copy()
  print("Reading tau channel")
  signal_df = read_root(signal_file, input_tree).query(selection_mc).copy()

  print("Adding momentum in square coordinates for normalisation")
  norm_df = add_cartesian_vars(norm_df)
  print("Adding momentum in square coordinates for signal")
  signal_df = add_cartesian_vars(norm_df)

  print("Writing is_signal flag")
  signal_df['is_signal_channel'] = 1
  norm_df['is_signal_channel'] = 0
  
  #Split samples for training, validation and test.
  msk = np.random.rand(len(signal_df)) < 0.33
  
  print("")
  print("Spliting signal sample to get the correct RJpsi ratio")
  signal_selected_df = signal_df[msk]
  signal_discarted_df = signal_df[~msk]
  
  mix_df = pd.concat([norm_df, signal_selected_df])
  print("Spliting mixel sample into train and test sub-samples")
  mix_train_df, mix_test_df = train_test_split( mix_df,
      test_size = 0.3,
      random_state = 0,
      shuffle = True,
      stratify= mix_df['is_signal_channel']) 

  print("Saving %s/BcToJPsiMuMu_is_jpsi_tau_selected.root"%(out_dir))
  signal_selected_df.to_root("%s/BcToJPsiMuMu_is_jpsi_tau_selected.root"%(out_dir), key=input_tree)

  print("Saving %s/BcToJPsiMuMu_is_jpsi_tau_discarted.root"%(out_dir))
  signal_discarted_df.to_root("%s/BcToJPsiMuMu_is_jpsi_tau_discarted.root"%(out_dir), key=input_tree)

  print("Saving %s/BcToJPsiMuMu_is_jpsi_lepton_test.root"%(out_dir))
  mix_test_df.to_root("%s/BcToJPsiMuMu_is_jpsi_lepton_test.root"%(out_dir), key=input_tree)

  print("Saving %s/BcToJPsiMuMu_is_jpsi_lepton_train.root"%(out_dir))
  mix_train_df.to_root("%s/BcToJPsiMuMu_is_jpsi_lepton_train.root"%(out_dir), key=input_tree)

  print("Saving %s/BcToJPsiMuMu_is_jpsi_lepton.root"%(out_dir))
  mix_df.to_root("%s/BcToJPsiMuMu_is_jpsi_lepton.root"%(out_dir), key=input_tree)
  
  


  

if __name__ == '__main__':
  main()