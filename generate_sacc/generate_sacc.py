# script to create sacc file needed for TJPCov
# inputs: ell binning set-up, n(z), n_eff, sigma_e, mask
# outputs for TJPCov: NaMaster workspace, sacc file containing n(z) and model spectra for all tomographic auto and cross-spectra
# the cosmic shear power spectra saved in this file IS NOT the mock spectra used in analysis, this is only used for making the covariance with TJPCov


import numpy as np
import pyccl as ccl
import sacc
import h5py


# Select survey
output_file = "mock_3x2pt_data_vector_without_cov.sacc"


nz_source_file = "SOURCE.HDF5"
nz_lens_file = "LENS.HDF5"

def load_nz(nz_file):
    with h5py.File(nz_file, "r") as f:
        z = f["Z"][:]
        nz = f["DATA"][:]
        num_z_bins = nz.shape[0]
    return z, nz, num_z_bins

z_source, nz_source, num_z_bins_source = load_nz(nz_source_file)
z_lens, nz_lens, num_z_bins_lens = load_nz(nz_lens_file)

z_means_source = []
for i in range(num_z_bins_source):
    z_mean_source = np.sum(z_source * nz_source[i, :]) / np.sum(nz_source[i, :])
    z_means_source.append(z_mean_source)

z_means_lens = []
for i in range(num_z_bins_lens):
    z_mean_lens = np.sum(z_lens * nz_lens[i, :]) / np.sum(nz_lens[i, :])
    z_means_lens.append(z_mean_lens)

print(z_means_source)
print(z_means_lens)

galaxy_biases = [1.2** (1+z_means_lens[i]) for i in range(num_z_bins_lens)]
print(galaxy_biases)



n_eff_source_total = 9.78  # in per sqarcmin
n_eff_lens = np.array([2.25, 3.098, 3.071, 2.595, 1.998])

assert len(n_eff_lens) == num_z_bins_lens


# WARNING, TJPCov is expecting the neff in units of per sq radian
n_eff_source_total /= (np.pi / (180 * 60)) ** 2.0  # into radians
n_eff_lens = n_eff_lens / (np.pi / (180 * 60)) ** 2.0  # into radians

# intrinsic ellipticity disperson
sigma_e = 0.26

# set-up ell binning, use logarithmic bins
ell_min = 20
ell_max = 3000
num_ell_bins = 24

n_eff_source = np.array([n_eff_source_total / num_z_bins_source] * num_z_bins_source)  # n_eff per tomographic bin
n_ell_source = sigma_e**2.0 / n_eff_source
n_ell_lens = 1 / n_eff_lens


# set-up for sacc file and CCL
s = sacc.Sacc()
cosmo = ccl.Cosmology(
    Omega_c=0.255,
    Omega_b=0.044,
    h=0.715,
    n_s=0.97,
    sigma8=0.8,
    transfer_function="boltzmann_camb",
    matter_power_spectrum="camb",
    extra_parameters={
        "camb": {
            "halofit_version": "mead2020_feedback", 
             "HMCode_logT_AGN": 8.0
        }
                      
    }
)
print(cosmo)


source_tracers = []  # add tracer for ccl
for i in range(num_z_bins_source):
    s.add_tracer("NZ", f"source_{i}", z_source, nz_source[i, :])
    source_tracers.append(
        ccl.tracers.WeakLensingTracer(
            cosmo, dndz=[z_source, nz_source[i, :]], ia_bias=None, use_A_ia=False
        )
    )


lens_tracers = []
for i in range(num_z_bins_lens):
    s.add_tracer("NZ", f"lens_{i}", z_lens, nz_lens[i, :])
    lens_tracers.append(
        ccl.NumberCountsTracer(
            cosmo, dndz=[z_lens, nz_lens[i, :]], bias=(z_lens, np.repeat(galaxy_biases[i], z_lens.size)), has_rsd=False)
    )



# dense ell - unbinned, just all integers up to ell_max
ell = np.arange(ell_max, dtype="int32")

# type codes to tell sacc what kinds of data we are using
CEE = sacc.standard_types.galaxy_shear_cl_ee
CPP = sacc.standard_types.galaxy_density_cl
CPE = sacc.standard_types.galaxy_shearDensity_cl_e


# NaMaster friendly ell binning for TJPCov to work
bin_edges = np.logspace(np.log10(ell_min), np.log10(ell_max), num=num_ell_bins + 1)
print(bin_edges)

# window function for each bin c^ij_b = sum_ell w^ij_ell c_ell
# our windows are just top hat functions
w = np.zeros((ell.size, num_ell_bins))
for b in range(num_ell_bins):
    in_bin = (ell > bin_edges[b]) & (ell <= bin_edges[b + 1])
    w[in_bin, b] = 1.0
    w[:, b] /= w[:, b].sum()

bin_ell = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# window function to save in sacc
win = sacc.windows.BandpowerWindow(ell, w)

# Shear-shear
for j in range(num_z_bins_source):
    for k in range(num_z_bins_source):
        # generate dense c_ell (on all ell values)
        cl = cosmo.angular_cl(source_tracers[j], source_tracers[k], ell)

        # convert to binned c_ell
        cl_bin = win.weight.T @ cl

        # add all data points to sacc file with metadata
        for n in range(num_ell_bins):
            s.add_data_point(
                CEE,
                (f"source_{j}", f"source_{k}"),
                cl_bin[n],
                ell=bin_ell[n],
                window=win,
                i=j,
                j=k,
                window_ind=n,
            )


# Density
for j in range(num_z_bins_lens):
    # THINK ABOUT THIS! Do we want to do this?
    # only do auto-correlations for j
    k = j
    cl = cosmo.angular_cl(lens_tracers[j], lens_tracers[k], ell)
    cl_bin = win.weight.T @ cl

    for n in range(num_ell_bins):
        s.add_data_point(
            CPP,
            (f"lens_{j}", f"lens_{k}"),
            cl_bin[n],
            ell=bin_ell[n],
            window=win,
            i=j,
            j=k,
            window_ind=n,
        )

# GGL
for j in range(num_z_bins_source):
    for k in range(num_z_bins_lens):
        # decide whether to skip this bin or not
        if z_means_source[j] < z_means_lens[k]:
            print("SKIPPING", j, k)
            continue

        cl = cosmo.angular_cl(source_tracers[j], lens_tracers[k], ell)
        cl_bin = win.weight.T @ cl

        for n in range(num_ell_bins):
            s.add_data_point(
                CPE,
                (f"source_{j}", f"lens_{k}"),
                cl_bin[n],
                ell=bin_ell[n],
                window=win,
                i=j,
                j=k,
                window_ind=n,
            )




s.save_fits(output_file, overwrite=True)
