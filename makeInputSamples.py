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
  selection_minimal_data= ' & '.join([
    'Bmass < 6.2',
    'abs(jpsi_mass - 3.0969)<0.1',
    'mu1_mediumID>0.5',
    'mu2_mediumID>0.5',
    trigger,
    pass_id])
  selection_minimal_mc= ' & '.join([selection_minimal_data,'abs(k_genpdgId)==13'])

  #Open original files
  all_files_names = ['BcToJPsiMuMu_is_chic0_mu_merged.root',
                     'BcToJPsiMuMu_is_jpsi_tau_merged.root',
                     'BcToJPsiMuMu_is_chic1_mu_merged.root',
                     'BcToJPsiMuMu_is_psi2s_mu_merged.root',
                     'BcToJPsiMuMu_is_chic2_mu_merged.root',
                     #'BcToJPsiMuMu_is_psi2s_tau_merged.root',
                     'BcToJPsiMuMu_is_hc_mu_merged.root',
                     'BcToJPsiMuMu_is_jpsi_hc_merged.root',
                     'HbToJPsiMuMu3MuFilter_ptmax_merged.root',
                     'BcToJPsiMuMu_is_jpsi_mu_merged.root',
                     'HbToJPsiMuMu_ptmax_merged.root',
                    ]
  
  all_files_names1 = [
                     'BcToJPsiMuMu_is_chic0_mu_merged.root',
                     #'BcToJPsiMuMu_is_jpsi_tau_merged.root',
                     'BcToJPsiMuMu_is_chic1_mu_merged.root',
                     'BcToJPsiMuMu_is_psi2s_mu_merged.root',
                     'BcToJPsiMuMu_is_chic2_mu_merged.root',
                     #'BcToJPsiMuMu_is_psi2s_tau_merged.root',
                     'BcToJPsiMuMu_is_hc_mu_merged.root',
                     'BcToJPsiMuMu_is_jpsi_hc_merged.root',
                     #'BcToJPsiMuMu_is_jpsi_mu_merged.root',
                    ]
  all_files_names2 = ['HbToJPsiMuMu_ptmax_merged.root']
  all_files_names3 = ['HbToJPsiMuMu3MuFilter_ptmax_merged.root']

  out_dir = 'data'
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
  data_df = read_root(data_file, input_tree).query(selection_minimal_data).copy()
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
  #hybrid_bkg_dfs_1 = [read_root(input_dir + sample, input_tree).query(selection_minimal_mc).copy() for sample in all_files_names1]
  hybrid_bkg_dfs_1 = [read_root(input_dir + sample, input_tree) for sample in all_files_names1]
  f1 = 0.52/8.5
  f2 = 6.7/8.5
  hybrid_bkg_dfs_1_accepted = [] 
  hybrid_bkg_dfs_1_rejected = []
  for df in hybrid_bkg_dfs_1:
    hybrid_bkg_dfs_1_a, hybrid_bkg_dfs_1_r= train_test_split( df,
      train_size = f1,
      #test_size = 1-f1,
      random_state = 0,
      shuffle = True) 
    print(hybrid_bkg_dfs_1_a.size)
    print(hybrid_bkg_dfs_1_r.size)
    hybrid_bkg_dfs_1_accepted.append(hybrid_bkg_dfs_1_a)
    hybrid_bkg_dfs_1_rejected.append(hybrid_bkg_dfs_1_r)

  #hybrid_bkg_dfs_2 = read_root(input_dir + all_files_names2[0], input_tree).query(selection_minimal_mc).copy() 
  hybrid_bkg_dfs_2 = read_root(input_dir + all_files_names2[0], input_tree).copy() 
  hybrid_bkg_dfs_2_accepted, hybrid_bkg_dfs_2_rejected= train_test_split( hybrid_bkg_dfs_2,
      train_size = f2,
      #test_size = 1-f2,
      random_state = 0,
      shuffle = True) 
  print(hybrid_bkg_dfs_2_accepted.size)
  #hybrid_bkg_dfs_3 = read_root(input_dir + all_files_names3[0], input_tree).query(selection_minimal_mc)
  hybrid_bkg_dfs_3 = read_root(input_dir + all_files_names3[0], input_tree)

  hybrid_bkg_dfs_1_accepted_concat = pd.concat(hybrid_bkg_dfs_1_accepted)
  hybrid_bkg_df = pd.concat([hybrid_bkg_dfs_1_accepted_concat, hybrid_bkg_dfs_2_accepted, hybrid_bkg_dfs_3])
  '''
  print("Adding momentum in square coordinates for all MC")
  hybrid_bkg_df = add_cartesian_vars(hybrid_bkg_df)
  '''

  print("Saving %s/mc_bkg_all.root"%(out_dir))
  hybrid_bkg_df.to_root("%s/mc_bkg_all.root"%(out_dir), key=input_tree)

  print("")
  print("")
  print("")

  print("MC Signal...")
  print("----------------------------------------------------------------------------------")
  print("Reading mu channel")
  #norm_df = read_root(norm_file, input_tree).query(selection_minimal_mc).copy()
  norm_df = read_root(norm_file, input_tree).copy()
  print("Reading tau channel")
  #signal_df = read_root(signal_file, input_tree).query(selection_minimal_mc).copy()
  signal_df = read_root(signal_file, input_tree).copy()

  '''
  print("Adding momentum in square coordinates for normalisation")
  norm_df = add_cartesian_vars(norm_df)
  print("Adding momentum in square coordinates for signal")
  signal_df = add_cartesian_vars(norm_df)
  '''

  print("Writing is_signal flag")
  signal_df['is_signal_channel'] = 1
  norm_df['is_signal_channel'] = 0
  
  #Split samples for training, validation and test.
  msk = np.random.rand(len(signal_df)) < 0.25
  
  print("")
  print("Spliting signal sample to get the correct RJpsi ratio")
  signal_selected_df = signal_df[msk]
  signal_discarted_df = signal_df[~msk]
  
  signal_mix_df = pd.concat([norm_df, signal_selected_df]).copy()
  print("Spliting signal_mixel sample into train and test sub-samples")
  signal_mix_train_df, signal_mix_test_df = train_test_split( signal_mix_df,
      test_size = 0.3,
      random_state = 0,
      shuffle = True,
      stratify= signal_mix_df['is_signal_channel']) 

  signal_mix_accepted_df, signal_mix_rejected_df = train_test_split( signal_mix_df,
      train_size = f1,
      test_size = 1-f1,
      random_state = 0,
      shuffle = True,
      stratify= signal_mix_df['is_signal_channel']) 

  print("Saving %s/BcToJPsiMuMu_is_jpsi_tau_selected.root"%(out_dir))
  signal_selected_df.to_root("%s/BcToJPsiMuMu_is_jpsi_tau_selected.root"%(out_dir), key=input_tree)

  print("Saving %s/BcToJPsiMuMu_is_jpsi_tau_discarted.root"%(out_dir))
  signal_discarted_df.to_root("%s/BcToJPsiMuMu_is_jpsi_tau_discarted.root"%(out_dir), key=input_tree)

  print("Saving %s/BcToJPsiMuMu_is_jpsi_lepton_test.root"%(out_dir))
  signal_mix_test_df.to_root("%s/BcToJPsiMuMu_is_jpsi_lepton_test.root"%(out_dir), key=input_tree)

  print("Saving %s/BcToJPsiMuMu_is_jpsi_lepton_train.root"%(out_dir))
  signal_mix_train_df.to_root("%s/BcToJPsiMuMu_is_jpsi_lepton_train.root"%(out_dir), key=input_tree)

  print("saving %s/BcToJPsiMuMu_is_jpsi_lepton.root"%(out_dir))
  signal_mix_df.to_root("%s/BcToJPsiMuMu_is_jpsi_lepton.root"%(out_dir), key=input_tree)
  
  print("saving %s/BcToJPsiMuMu_is_jpsi_lepton_weight.root"%(out_dir))
  signal_mix_accepted_df.to_root("%s/BcToJPsiMuMu_is_jpsi_lepton_weight.root"%(out_dir), key=input_tree)
  

  all_mc_df = pd.concat([hybrid_bkg_df, signal_mix_accepted_df])
  print("saving %s/all_mc_weights.root"%(out_dir))
  all_mc_df.to_root("%s/all_mc_weights.root"%(out_dir), key=input_tree)

  

if __name__ == '__main__':
  main()
