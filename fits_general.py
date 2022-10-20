import dadi
from dadi import Numerics, PhiManip, Integration, Spectrum
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from multiprocessing import Pool
import random
# import tqdm

np.random.seed(12345)
i = 12 # number of dipl inds
n = 2*i # number of genomes
ss = {'T':i, 'D':i}
dd = dadi.Misc.make_data_dict_vcf("Euphrasia_gbs_080520_fSNPs50_fINDs75_conserved_uni_scaff.vcf",
                                    "popfile.txt", subsample = ss)
# generate 999 down samplings
fss = [dadi.Spectrum.from_data_dict(dd, ['T', 'D'], projections = [n, n], polarized = False) for _ in range(999)]

pts_l = [n, n+10, n+20]

def twoPopGf(params, ns, pts):
    nuT, nuD, mTD, mDT, T0 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, T0, nuT, nuD, m12 = mTD, m21 = mDT)
#    phi = Integration.two_pops(phi, xx, T1, nuT, nuD, m12 = mTD, m21 = mDT)
    fs = Spectrum.from_phi_inbreeding(phi, ns, (xx, xx), (0.75, 0.81), (2,2))
    return fs


def twoPopOld(params, ns, pts):
    nuT, nuD, mTD, mDT, T0, T1 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, T0, nuT, nuD, m12 = mTD, m21 = mDT)
    phi = Integration.two_pops(phi, xx, T1, nuT, nuD, m12 = 0, m21 = 0)
    fs = Spectrum.from_phi_inbreeding(phi, ns, (xx, xx), (0.75, 0.81), (2,2))
    return fs


def twoPop2nd(params, ns, pts): # secondary contact
    nuT, nuD, mTD, mDT, T0, T1 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, T0, nuT, nuD, m12 = 0, m21 = 0)
    phi = Integration.two_pops(phi, xx, T1, nuT, nuD, m12 = mTD, m21 = mTD)
    fs = Spectrum.from_phi_inbreeding(phi, ns, (xx, xx), (0.75, 0.81), (2,2))
    return fs

def twoPopSplit(params, ns, pts):
    nuT, nuD, T0 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, T0, nuT, nuD, m12 = 0, m21 = 0)
    fs = Spectrum.from_phi_inbreeding(phi, ns, (xx, xx), (0.75, 0.81), (2,2))
    return fs


upperList=[[   5,    5,   10,   10,   50],
           [   5,    5,   10,   10,   50,   20],
           [   5,    5,   10,   10,   50,   20],
           [   5,    5,   50]]
lowerList=[[1e-1, 1e-1, 1e-2, 1e-2, 0.1],
           [1e-1, 1e-1, 1e-2, 1e-2, 0.1, 0.01],
           [1e-1, 1e-1, 1e-2, 1e-2, 0.1, 0.01],
           [1e-1, 1e-1, 0.01]]
fixedList=[[None, None, None, None, None],
           [None, None, None, None, None, None],
           [None, None, None, None, None, None],
           [None, None, None]]
p0List=[[   1,    1,    1,    1,   10],
        [   1,    1,    1,    1,   10,    4],
        [   1,    1,    1,    1,   10,    4],
        [   1,    1,    4]]

extrapList = [dadi.Numerics.make_extrap_func(twoPopGf),
              dadi.Numerics.make_extrap_func(twoPopOld),
              dadi.Numerics.make_extrap_func(twoPop2nd),
              dadi.Numerics.make_extrap_func(twoPopSplit)
              ]





def run_fun(moNo, sfsNo, sd): # number of model (0-2), number of down sampling (0-998), random seed
    np.random.seed(sd)
    pPert = dadi.Misc.perturb_params(p0List[moNo], fold=2, lower_bound=lowerList[moNo], upper_bound=upperList[moNo])
    print(pPert)
    #print(fss[sfsNo])
    t0 = time.time()
    fit= dadi.Inference.optimize_log(pPert, fss[sfsNo], extrapList[moNo], pts_l,
                                   lower_bound=lowerList[moNo],
                                   upper_bound=upperList[moNo],
                                   fixed_params=fixedList[moNo],
                                   verbose=0, maxiter=3,full_output=True)
    t = time.time() - t0
    print("Done.")
    return moNo, sfsNo, sd, fit, t

p = Pool(40)
#p = Pool(3)

params = zip([0,1,2,3]* 99 * 99,
             [item for sublist in [[i]*4 for i in range(99)] * 99 for item in sublist],
             random.sample(range(1000000000),99*99*4)

)

fits = p.starmap(run_fun, params)
# pall = [i for i in params]
# fits = p.starmap(run_fun, [pall[i] for i in [3,7,11,15]])

with open("FITS_general_4mod.dump", "wb") as f:
    pickle.dump(fits, f)
