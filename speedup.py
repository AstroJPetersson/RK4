import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from imgcat import imgcat

def speed_up(n, p):
    S = 1 / ((1 - p) + p/n)
    return S

nproc = np.arange(1, 100) 
S_theo = speed_up(nproc, p=0.88)
S_ideal = speed_up(nproc, p=1)

plt.style.use('style.mplstyle')

#---------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# Ideal scaling:
ax.plot(nproc, S_ideal, c='k', ls=':', label='Ideal ($p=1.00$)')

# Theoretical speed-up:
ax.plot(nproc, S_theo, c='b', label='Theoretical ($p=0.88$)')

# Actual strong scaling:
T = [27.74, 15.81, 8.956, 6.354, 7.371, 8.670, 9.485, 26.56, 31.40]
ntasks = [1, 2, 4, 8, 16, 24, 32, 48, 64]
S_actual = T[0]/np.array(T)

ax.plot(ntasks, S_actual, c='r', ls='-', marker='o', label='Actual')

ax.axvline(x=32, c='k', ls='--', alpha=0.5, zorder=0)

# Make plot look nice:
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Number of processors')
ax.set_ylabel('Speed-up $S(n)$')
f = ScalarFormatter()
f.set_scientific(False)
ax.xaxis.set_major_formatter(f)
ax.yaxis.set_major_formatter(f)
ax.legend(frameon=False)
ax.set_title('Strong Scaling')

fig.savefig('strong_scaling_S.pdf')
imgcat(fig)

#---------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(nproc, S_ideal/nproc, c='k', ls=':', label='Ideal ($p=1.00$)')
ax.plot(nproc, S_theo/nproc, c='b', label='Theoretical ($p=0.78$)')
ax.plot(ntasks, S_actual/ntasks, c='r', ls='-', marker='o', label='Actual')
ax.axvline(x=32, c='k', ls='--', alpha=0.5)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Number of processors')
ax.set_ylabel('Efficiency $E(n)$')
f = ScalarFormatter()
f.set_scientific(False)
ax.xaxis.set_major_formatter(f)
ax.yaxis.set_major_formatter(f)
ax.legend(frameon=False)
ax.set_title('Strong Scaling')

fig.savefig('strong_scaling_E.pdf')
imgcat(fig)


