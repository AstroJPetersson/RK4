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
# bodies to include in the N-body integration:
planets = ['Sun', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
m_planets = np.array([1.989e30, 1.90e27, 5.69e26, 8.66e25, 1.03e26])/(1.989e30)
N_asteroids = 0

vector6N_solarsystem = np.zeros(shape=(len(planets)+N_asteroids, 2, 3))
m_solarsystem = np.append(m_planets, np.zeros(N_asteroids))

# positions and velocities for the Solar system at a given time T:
Time = Time("2022-12-24 00:00")
for n in range(len(planets)):
    p, v = get_body_barycentric_posvel(planets[n], Time)
    p, v = p.get_xyz().value, v.get_xyz().to(u.AU/u.yr).value
    w = np.array([p, v])
    vector6N_solarsystem[n] = w

# generate some asteroids in the Solar system:
for i in range(len(planets), len(planets)+N_asteroids):
    x, y = 100 * np.random.rand(2) - 50
    p = np.array([x, y, 0])
    vv = np.cross(np.array([0, 0, 1]), p)
    vc = np.sqrt(G*np.sum(m_planets)/np.linalg.norm(p))
    v = vc * vv / np.linalg.norm(vv)
    w = np.array([p, v]) + vector6N_solarsystem[0]
    vector6N_solarsystem[i] = w


#--------------------------------------------
def g(W, m):
    # create dw/dt:
    dwdt = np.zeros(shape=(len(W), 2, 3))
    
    # calculate dx/dt:
    dwdt[:,0] = W[:,1]
    
    # caclculate dv/dt:
    xi = W[:,0,np.newaxis]
    xj = W[:,0]
    deltax = xi-xj
    deltax_norm = np.linalg.norm(deltax, axis=2, keepdims=True)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        before_sum = G * m[:,np.newaxis] * deltax / deltax_norm**3
        before_sum = np.where(np.isfinite(before_sum), before_sum, 0)
        dwdt[:,1] = -np.sum(before_sum, axis=1)
    
    return dwdt

def RungeKutta4th(W, m, t0, tf, h):
    # time:
    t = np.arange(t0, tf+h, h)
    
    # create W_save:
    W_save = np.zeros(shape=(len(t), len(W), 2, 3))
    W_save[0] = W
    
    # Runge-Kutta:
    for i in range(1, len(t)):
        fa = g(W, m)
        Wb = W + h/2*fa
        fb = g(Wb, m)
        Wc = W + h/2*fb
        fc = g(Wc, m)
        Wd = W + h*fc
        fd = g(Wd, m)
        W = W + 1/6*h*fa + 1/3*h*fb + 1/3*h*fc + 1/6*h*fd
        W_save[i] = W - W[0]
    
    return t, W_save


#--------------------------------------------
# N-body integration:
time0 = time.time()
time_solarsystem, phase_solarsystem = RungeKutta4th(vector6N_solarsystem, m_solarsystem, 0, 165, 1/(365.25))
print('Integration time: %s s' % (time.time() - time0))


#--------------------------------------------
# Animation:
make_animation = True

if make_animation == True:
    fig, ax = plt.subplots(1, figsize=(9, 9))

    label  = ['The Sun'] + planets[1:]
    marker = ['*', 'o', 'o', 'o', 'o']
    markersize = [40, 40, 40, 40, 40]
    color  = ['yellow', 'orangered', 'darkorange', 'darkturquoise', 'darkblue']
    line_list = []
    traj_list = []
    for i in range(0, len(planets)):
        line, = ax.plot([], [], marker=marker[i], markersize=markersize[i], color=color[i], markeredgecolor='k', ls='none', label=label[i])
        line_list.append(line)
        traj, = ax.plot([], [], color=color[i], ls='-', lw=2, alpha=0.6)
        traj_list.append(traj)

    # make the plot look nice:
    ax.set_xlim(-32, 32)
    ax.set_ylim(-32, 32)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().set_aspect('equal')
    ax.set_axis_off()
    fig.tight_layout()

    
    for i in range(0, len(line_list)):
        traj_list[i].set_data(phase_solarsystem[:,i,0,0], phase_solarsystem[:,i,0,1])

    # Function for animation:
    def animate(frame):
        for i in range(0, len(line_list)):
            line_list[i].set_data(phase_solarsystem[:,i,0,0][frame], phase_solarsystem[:,i,0,1][frame])
        
        return line,

    # Make and save animation:
    which_frames = np.arange(0, len(time_solarsystem), 100)
    fps = 30
    ani = animation.FuncAnimation(fig, animate, frames=which_frames, interval=fps, blit=True)
    ani.save('solarsystem_mod.gif')


#--------------------------------------------


