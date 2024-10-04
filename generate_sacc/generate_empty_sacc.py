# script to create sacc file needed for TJPCov
# inputs: ell binning set-up, n(z), n_eff, sigma_e, mask
# outputs for TJPCov: NaMaster workspace, sacc file containing n(z) and model spectra for all tomographic auto and cross-spectra
# the cosmic shear power spectra saved in this file IS NOT the mock spectra used in analysis, this is only used for making the covariance with TJPCov


import sys
import numpy as np
import pyccl as ccl
import sacc
import healpy as hp
import pymaster as nmt
import os
import h5py

nz_file = sys.argv[1]
output_file = sys.argv[2]

# Select survey
survey = "Y1"

# The namaster workspace depends only on the mask which we take to be the same for each tomographic bin,
# this is the slow step so only do once for a given mask, if the healpix mask changes this will need to be recomputed,
# the workspace is also a large file so use $PSCRATCH on NERSC.
compute_workspace = True

workspace_directory = "./workspace"
mask_name = f"lsst_{survey}_binary_mask"


with h5py.File(nz_file, "r") as f:
    z = f["Z"][:]
    nz = f["DATA"][:]
    num_z_bins = nz.shape[0]

if survey == "Y1":
    n_eff_tot = 9.78  # in per sqarcmin
elif survey == "Y10":
    n_eff_tot = 24.4  # in per sqarcmin
else:
    raise ValueError("survey not recognised")

# WARNING, TJPCov is expecting the neff in units of per sq radian
n_eff_tot /= (np.pi / (180 * 60)) ** 2.0  # into radians

# intrinsic ellipticity disperson
sigma_e = 0.26

# set-up ell binning, use logarithmic bins
ell_min = 20
ell_max = 5000
num_ell_bins = 24


# read in mask
mask = hp.read_map(
    f"./lsst_{survey}_binary_mask.fits", verbose=False
)
nside = hp.npix2nside(mask.size)
f_sky = mask[mask == 1.0].size / mask.size

n_eff = np.array([n_eff_tot / num_z_bins] * num_z_bins)  # n_eff per tomographic bin
n_ell = sigma_e**2.0 / n_eff


# set-up for sacc file and CCL
s = sacc.Sacc()
cosmo = ccl.Cosmology(
    Omega_c=0.255,
    Omega_b=0.044,
    h=0.715,
    n_s=0.97,
    sigma8=0.8,
    transfer_function="boltzmann_camb",
)



wl = []  # add tracer for ccl
for i in range(num_z_bins):
    s.add_tracer("NZ", f"source_{i}", z, nz[i, :])
    wl.append(
        ccl.tracers.WeakLensingTracer(
            cosmo, dndz=[z, nz[i, :]], has_shear=True, ia_bias=None, use_A_ia=True
        )
    )
    tr = s.tracers[f"source_{i}"]
    # Note this differs from A9 in Prat and Zuntz et al., but is the appropriate value for TJPCov - Alonso private communication
    tr.metadata["n_ell_coupled"] = f_sky * n_ell[i]


ell = np.arange(3 * nside, dtype="int32")
cl = np.zeros((num_z_bins, num_z_bins, ell.size))
CEE = sacc.standard_types.galaxy_shear_cl_ee


# NaMaster friendly ell binning for TJPCov to work
bin_edges = np.logspace(np.log10(ell_min), np.log10(ell_max), num=num_ell_bins + 1)
bpws = np.zeros(ell.size) - 1.0
weights = np.ones(ell.size)
w = np.zeros((ell.size, num_ell_bins))
for nb in range(num_ell_bins):
    inbin = (ell <= bin_edges[nb + 1]) & (ell > bin_edges[nb])
    bpws[inbin] = nb
    norm_denom = np.sum(weights[inbin])
    weights[inbin] /= norm_denom
    w[inbin, nb] = 1.0
b = nmt.NmtBin(bpws=bpws, ells=ell, weights=weights)
bin_ell = b.get_effective_ells()
win = sacc.windows.BandpowerWindow(ell, w)

# loop over tomographic bins bins
for j in range(num_z_bins):
    for k in range(num_z_bins):
        print(f"j={j}, k={k}")
        cl[j, k, :] = ccl.angular_cl(cosmo, wl[j], wl[k], ell)
        cl_bin = b.bin_cell(np.array([cl[j, k, :]]))[
            0
        ]  # arrays are shape (1,num_ell_bins) so need [0]
        nl_bin = b.bin_cell(np.array([np.ones(3 * nside) * n_ell[j]]))[0]

        if compute_workspace:
            if (j == k) & (j == 0):
                f = nmt.NmtField(mask, [mask * 0.0, mask * 0.0])
                w = nmt.NmtWorkspace()
                w.compute_coupling_matrix(f, f, b)
                w.write_to(
                    f"{workspace_directory}/workspace_lsst_{survey}_binary_mask.fits"
                )

        for n in range(num_ell_bins):
            s.add_data_point(
                CEE,
                (f"source_{j}", f"source_{k}"),
                cl_bin[n],
                ell=bin_ell[n],
                window=win,
                i=j,
                j=k,
                n_ell=nl_bin[n],
                window_ind=n,
            )
s.save_fits(output_file, overwrite=True)
