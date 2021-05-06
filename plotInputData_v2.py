import math

from utils.variable import variable
from utils.category import category
from utils.selections import preselection, pass_id
from utils.plotting import *

import ROOT
from ROOT import gSystem, gROOT, gStyle

from root_pandas import read_root
import numpy as np
import pandas as pd
gStyle.SetOptStat(0)
dir_plots = 'plots/input_regression'

def main():
  '''
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
  '''
  input_dir = "/gpfs/ddn/srm/cms/store/user/garamire/ntuples/2021Mar23/"
  #inputFile = '/gpfs/ddn/srm/cms/store/user/garamire/ntuples/2021Mar23/data_ptmax_merged.root'
  inputFile = 'data/alldata_selected.root'
  inputFile_mc = 'data/mc_all.root'
  inputTree = 'BTo3Mu'
  common_cuts = " & ".join([ preselection, pass_id])
  #data_df = read_root(inputFile, inputTree).query(common_cuts).copy()
  data_df = read_root(inputFile, inputTree)
  mc_df = read_root(inputFile_mc, inputTree)
  #mc_df = pd.concat([read_root(input_dir + sample, inputTree) for sample in all_files_names])
  cat_data = category("data", data_df, "trigger_dimuon0", [2.95,3.25], "Data", ROOT.kBlack, ROOT.kFullCircle, 1.0)
  cat_mc = category("mc", mc_df, "trigger_dimuon0", [2.95,3.25], "MC", ROOT.kRed, ROOT.kFullSquare, 1.0)

  var_list = [
    #variable("Q_sq", "Q^{2}", "Q^{2}", "GeV^{2}", 50, 0., 14.),
    #variable("m_miss_sq", "m_{miss}^{2}", "m_{miss}^{2}", "GeV^{2}", 60, 0., 10.),
    #variable("pt_var", "pt_var", "pt_var", "GeV", 50, 0., 50.),
    #variable("E_mu_star", "E^{*}", "E^{*}", "GeV", 60, 0., 3.),
    variable("pv_sv_dist", "Dist(pv,sv)", "Dist(pv,sv)", "cm", 60, 0., 0.02),
    variable("Bpt", "p_{T}(#mu^{+}#mu^{-}#mu)", "p{T}(#mu^{+}#mu^{-}#mu)", "GeV", 50, 0., 80.),

    variable("Bpx", "p_{x}(#mu^{+}#mu^{-}#mu)", "p{x}(#mu^{+}#mu^{-}#mu)", "GeV", 50, 0., 80.),
    variable("Bpy", "p_{y}(#mu^{+}#mu^{-}#mu)", "p{y}(#mu^{+}#mu^{-}#mu)", "GeV", 50, 0., 80.),
    variable("Bpz", "p_{z}(#mu^{+}#mu^{-}#mu)", "p{z}(#mu^{+}#mu^{-}#mu)", "GeV", 50, 0., 80.),

    variable("kpx", "p_{x}(#mu_{3})", "p{x}(#mu_{3})", "GeV", 50, 0., 30.),
    variable("kpy", "p_{y}(#mu_{3})", "p{y}(#mu_{3})", "GeV", 50, 0., 30.),
    variable("kpz", "p_{z}(#mu_{3})", "p{z}(#mu_{3})", "GeV", 50, 0., 30.),

    variable("mu1px", "p_{x}(#mu_{1})", "p{x}(#mu_{1})", "GeV", 50, 0., 30.),
    variable("mu1py", "p_{y}(#mu_{1})", "p{y}(#mu_{1})", "GeV", 50, 0., 30.),
    variable("mu1pz", "p_{z}(#mu_{1})", "p{z}(#mu_{1})", "GeV", 50, 0., 30.),

    variable("mu2px", "p_{x}(#mu_{2})", "p{x}(#mu_{2})", "GeV", 50, 0., 30.),
    variable("mu2py", "p_{y}(#mu_{2})", "p{y}(#mu_{2})", "GeV", 50, 0., 30.),
    variable("mu2pz", "p_{z}(#mu_{2})", "p{z}(#mu_{2})", "GeV", 50, 0., 30.),
    #variable("Bmass", "m(#mu^{+}#mu^{-}#mu)", "m(#mu^{+}#mu^{-}#mu)", "GeV", 50, 3., 10.),
    #variable("Bmass", "m(#mu^{+}#mu^{-}#mu)", "m(#mu^{+}#mu^{-}#mu)", "GeV", 50, 3., 10.),
    variable("Beta", "#eta(#mu^{+}#mu^{-}#mu)", "#eta(#mu^{+}#mu^{-}#mu)", "", 50, 2.5, 2.5),
    variable("Bphi", "#phi(#mu^{+}#mu^{-}#mu)", "#phi(#mu^{+}#mu^{-}#mu)", "", 50, -3.2, 3.2),
    variable("jpsi_mass", "m(#mu^{+}#mu^{-})", "m(#mu^{+}#mu^{-})", "GeV", 50, 2.95, 3.25),
    variable("mu1pt", "p_{T}(#mu_{1})", "p_{T}(#mu_{1})", "GeV", 50, 0., 30.),
    variable("mu1eta", "#eta(#mu_{1})", "#eta(#mu_{1})", "GeV", 50, -2.5, 2.5),
    variable("mu1phi", "#phi(#mu_{1})", "#phi(#mu_{1})", "GeV", 50, -3.2, 3.2),
    variable("mu2pt", "p_{T}(#mu_{2})", "p_{T}(#mu_{2})", "GeV", 50, 0., 30.),
    variable("mu2eta", "#eta(#mu_{2})", "#eta(#mu_{2})", "GeV", 50, -2.5, 2.5),
    variable("mu2phi", "#phi(#mu_{2})", "#phi(#mu_{2})", "GeV", 50, -3.2, 3.2),
    variable("kpt", "p_{T}(#mu_{3})", "p_{T}(#mu_{3})", "GeV", 50, 0., 30.),
    variable("keta", "#eta(#mu_{3})", "#eta(#mu_{3})", "GeV", 50, -2.5, 2.5),
    variable("kphi", "#phi(#mu_{3})", "#phi(#mu_{3})", "GeV", 50, -3.2, 3.2),
    #variable("pv_x", "Primary vertex x", "Primary vertex x", "cm", 50, -0.5, 0.5 ),
    #variable("pv_y", "Primary vertex y", "Primary vertex y", "cm", 50, -0.5, 0.5 ),
    variable("pv_z", "Primary vertex z", "Primary vertex z", "cm", 50, -12., 12. ),
    #variable("jpsivtx_vtx_x", "Vtx_x(#mu^{+}#mu^{-})", "Vtx_x(#mu^{+}#mu^{-})", "cm", 50, -0.5, 0.5 ),
    #variable("jpsivtx_vtx_y", "Vtx_y(#mu^{+}#mu^{-})", "Vtx_y(#mu^{+}#mu^{-})", "cm", 50, -0.5, 0.5 ),
    #variable("jpsivtx_vtx_z", "Vtx_z(#mu^{+}#mu^{-})", "Vtx_z(#mu^{+}#mu^{-})", "cm", 50, -12., 12. ),
    #variable("jpsivtx_vtx_ex", "Vtx_ex(#mu^{+}#mu^{-})", "Vtx_ex(#mu^{+}#mu^{-})", "cm", 50, 0., 0.03 ),
    #variable("jpsivtx_vtx_ey", "Vtx_ey(#mu^{+}#mu^{-})", "Vtx_ey(#mu^{+}#mu^{-})", "cm", 50, 0., 0.03 ),
    #variable("jpsivtx_vtx_ez", "Vtx_ez(#mu^{+}#mu^{-})", "Vtx_ez(#mu^{+}#mu^{-})", "cm", 50, 0., 0.1 ),
    ]

  plotComparisonByCats([cat_data,cat_mc], var_list, "jpsiregion_dimuon0", True)

if __name__ == '__main__':
  gROOT.SetBatch()
  gStyle.SetOptStat(0)
  main()
