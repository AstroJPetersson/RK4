import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from imgcat import imgcat

def speed_up(n, p):
    S = 1 / ((1 - p) + p/n)
    return S

nproc = np.arange(1, 100) 
S = speed_up(nproc, p=0.78)
S_ideal = speed_up(nproc, p=1)

plt.style.use('style.mplstyle')

#---------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# Ideal scaling:
ax.plot(nproc, S_ideal, c='k', ls=':', label='Ideal ($p=1.00$)')

# Theoretical speed-up:
ax.plot(nproc, S, c='b', label='Theoretical ($p=0.78$)')

# Actual strong scaling:
ax.plot([], [], c='r', label='Actual')

# Make plot look nice:
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Number of processors')
ax.set_ylabel('Speed-up: $S(n)$')
f = ScalarFormatter()
f.set_scientific(False)
ax.xaxis.set_major_formatter(f)
ax.yaxis.set_major_formatter(f)
ax.legend(frameon=False)
ax.set_title('Strong Scaling')

imgcat(fig)


#---------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(nproc, S_ideal/nproc, c='k', ls=':', label='Ideal ($p=1.00$)')
ax.plot(nproc, S/nproc, c='b', label='Theoretical ($p=0.78$)')
ax.plot([], [], c='r', label='Actual')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Number of processors')
ax.set_ylabel('$E(n) = S(n)/n$')
f = ScalarFormatter()
f.set_scientific(False)
ax.xaxis.set_major_formatter(f)
ax.yaxis.set_major_formatter(f)
ax.legend(frameon=False)
ax.set_title('Weak Scaling')

imgcat(fig)


