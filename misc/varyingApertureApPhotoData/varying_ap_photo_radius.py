import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def mk_ap_photo_interp(fname='Da_vs_z.dat'):
    df_Da = pd.read_csv(fname, delim_whitespace=True)[['z', 'Da']]
    f_interp = interp1d(df_Da.z.values, df_Da.Da.values,
                        fill_value='extrapolate')
    return f_interp


f_interp = mk_ap_photo_interp()
z_smooth = np.linspace(0.01, 1.5, 1000)
y_smooth = f_interp(0.5)/f_interp(z_smooth) * 2.1

plt.figure(figsize=[8, 4.5])
plt.plot(z_smooth, y_smooth)
plt.grid()
plt.xlabel('z')
plt.ylabel('$Da(0.5)/Da(z)\\times 2.1$')
plt.ylim([0, 10])
plt.savefig('apertures.png', dpi=150)
plt.savefig('apertures.pdf')

plt.close()
