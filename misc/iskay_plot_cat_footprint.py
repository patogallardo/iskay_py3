#!/nfs/user/pag227/miniconda/bin/python
from iskay import paramTools
from iskay import catalogTools
from iskay import tiled_JK
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

show = False
param_fname = sys.argv[1]
NSIDE = 64
params = paramTools.params(param_fname)

print("Approximate resolution at NSIDE {} is {:.2} deg".format(
      NSIDE, hp.nside2resol(NSIDE, arcmin=True) / 60))
res_el = hp.nside2resol(NSIDE, arcmin=True) / 60

df = catalogTools.preProcessedCat(howMany=params.N_OBJ,
                                  query=params.CAT_QUERY).df

df = tiled_JK.classify_grid(df, Nside=NSIDE)
hist = tiled_JK.healpix_histogram_catalog(df, NSIDE)
number_of_nonzero_pixs = np.sum(hist > 0)

area_el = res_el**2
cat_area = area_el * number_of_nonzero_pixs

print("number of non-zero pixs: %i" % number_of_nonzero_pixs)
print("covered_area is: %1.2f sq deg" % (cat_area))

DIRNAME = './cat_plots'
if not os.path.exists(DIRNAME):
    os.mkdir(DIRNAME)
plot_fname = os.path.join(DIRNAME, params.NAME)

hp.mollview(hist)
plt.title("Area: %1.2f sq. deg" % cat_area)

if show:
    plt.show()
else:
    plt.savefig(plot_fname + ".pdf")
    plt.savefig(plot_fname + ".png", dpi=120)
    plt.close()
