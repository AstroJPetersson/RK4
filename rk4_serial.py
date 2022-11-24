#!/Users/jonathanpetersson/opt/anaconda3/bin/python

#--------------------------------------------
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel

import time
#--------------------------------------------
G = 4*np.pi**2  # AU^3 yr^-2 Msol


#--------------------------------------------
def g(W, mj):
    # Caclculate dv/dt:
    xi = W[:,0,np.newaxis]
    xj = W[:,0]
    deltax = xi-xj
    deno = np.linalg.norm(deltax, axis=2)**3
    deno = deno[:,:,np.newaxis]
    with np.errstate(divide='ignore', invalid='ignore'):
        bsum = G*mj[:,np.newaxis]*deltax/deno
        bsum = np.where(np.isfinite(bsum), bsum, 0)
        dvdt = -np.sum(bsum, axis=1)
    # Calculate dx/dt:
    dxdt = W[:,1]
    # Create dw/dt:
    dwdt = np.zeros(shape=(len(W), 2, 3))
    dwdt[:,0] = dxdt
    dwdt[:,1] = dvdt
    
    return dwdt

def RungeKutta4th(Wn, mj, t0, tn, h):
    # Time:
    t = np.arange(t0, tn+h, h)
    # Create W_array for position and velocity:
    W_array = np.zeros(shape=(len(t), len(Wn), 2, 3))
    W_array[0] = Wn
    # Runge-Kutta:
    for i in range(1, len(t)):
        fa = g(Wn, mj)
        Wb = Wn + h/2*fa
        fb = g(Wb, mj)
        Wc = Wn + h/2*fb
        fc = g(Wc, mj)
        Wd = Wn + h*fc
        fd = g(Wd, mj)
        Wn1 = Wn + 1/6*h*fa + 1/3*h*fb + 1/3*h*fc + 1/6*h*fd
        W_array[i] = Wn1 - Wn1[0]
        Wn = Wn1
    
    return t, W_array


#--------------------------------------------
# Time at which positions and velocities of the planets in the Solar system will be obtained from:
T = Time("2022-11-18 00:00")

# What bodies to include in the solar system:
bodies = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
mj_solarsystem = np.array([1.989e30, 3.30e23, 4.87e24, 5.97e24, 6.42e23, 1.90e27, 5.69e26, 8.66e25, 1.03e26])/(1.989e30)

# Find position and velocity for each body:
vector6N_solarsystem = np.zeros(shape=(len(bodies), 2, 3))
for n in range(len(bodies)):
    p, v = get_body_barycentric_posvel(bodies[n], T)
    p, v = p.get_xyz().value, v.get_xyz().to(u.AU/u.yr).value
    w = np.array([p, v])
    vector6N_solarsystem[n] = w

# N-body integration:
start = time.time()
time_solarsystem, phase_solarsystem = RungeKutta4th(vector6N_solarsystem, mj_solarsystem, 0, 100, 1/(365.25))
print('Integration time: %s s' % (time.time() - start))


#--------------------------------------------
# Animation:
movie = True

if movie == True:
    plt.style.use('dark_background')

    fig, ax = plt.subplots(1, figsize=(9, 9))
    l, b, h, w = .75, .75, .2, .2
    ax_zoom = fig.add_axes([l, b, w, h])

    label  = ['The Sun'] + bodies[1:]
    marker = ['*', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
    markersize = [10, 5, 5, 5, 5, 10, 10, 10, 10]
    markersize_zoom = [20, 10, 10, 10, 10] + markersize[5:]
    color  = ['yellow', 'darkgrey', 'wheat', 'deepskyblue', 'red', 'orange', 'sienna', 'cyan', 'mediumslateblue']
    line_list = []
    traj_list = []
    traj_zoom_list = []
    zoom_list = []
    for i in range(0, len(bodies)):
        line, = ax.plot([], [], marker=marker[i], markersize=markersize[i], color=color[i], ls='none', label=label[i])
        line_list.append(line)
        zoom, = ax_zoom.plot([], [], marker=marker[i], markersize=markersize[i], color=color[i], ls='none', label=label[i])
        zoom_list.append(zoom)
        traj, = ax.plot([], [], color=color[i], ls='-', lw=2, alpha=0.4)
        traj_list.append(traj)
        traj_zoom, = ax_zoom.plot([], [], color=color[i], ls='-', lw=2, alpha=0.4)
        traj_zoom_list.append(traj_zoom)
    txt = ax.text(0.05, 0.95, '', fontsize=14, ha='left', va='top', transform=ax.transAxes)

    # Make the plot look nice:
    ax.set_xlim(-32, 32)
    ax.set_ylim(-32, 32)
    ax_zoom.set_xlim(-2, 2)
    ax_zoom.set_ylim(-2, 2)
    ax_zoom.set_xlabel('$x$ [AU]', fontsize=12)
    ax_zoom.set_ylabel('$y$ [AU]', fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().set_aspect('equal')
    ax.legend(fontsize=12, frameon=False, loc='lower right')
    ax.plot([-30, -20], [-29, -29], lw=0.9, c='w')
    ax.plot([-2, 2, 2, -2, -2], [-2, -2, 2, 2, -2], lw=0.9, c='w')
    ax.text(-25, -29, '10 AU', fontsize=14, ha='center', va='bottom')
    fig.tight_layout()

    # Function for animation:
    def animate(frame):
        for i in range(0, len(line_list)):
            line_list[i].set_data(phase_solarsystem[:,i,0,0][frame], phase_solarsystem[:,i,0,1][frame])
            zoom_list[i].set_data(phase_solarsystem[:,i,0,0][frame], phase_solarsystem[:,i,0,1][frame])
            traj_list[i].set_data(phase_solarsystem[:,i,0,0][:frame], phase_solarsystem[:,i,0,1][:frame])
            traj_zoom_list[i].set_data(phase_solarsystem[:,i,0,0][:frame], phase_solarsystem[:,i,0,1][:frame])
        txt.set_text('{:0.2f} yr'.format(time_solarsystem[frame]))
        if frame%100 == 0:
            print(f'Frame: {frame}')
        return line, txt

    # Make and save animation:
    which_frames = np.arange(0, 1+len(time_solarsystem), 10)
    fps = 30
    ani = animation.FuncAnimation(fig, animate, frames=which_frames, interval=fps, blit=True)
    ani.save("solarsystem.gif")


#--------------------------------------------


