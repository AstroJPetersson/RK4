#!/Users/jonathanpetersson/opt/anaconda3/bin/python


#--------------------------------------------
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel

import time
from mpi4py import MPI


#--------------------------------------------
G = 4*np.pi**2  # AU^3 yr^-2 Msol

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


#--------------------------------------------
# bodies to include in the N-body integration:
planets = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
m_planets = np.array([1.989e30, 3.30e23, 4.87e24, 5.97e24, 6.42e23, 1.90e27, 5.69e26, 8.66e25, 1.03e26])/(1.989e30)
N_asteroids = 100

vector6N_solarsystem = np.zeros(shape=(len(planets)+N_asteroids, 2, 3))
m_solarsystem = np.append(m_planets, np.zeros(N_asteroids))

if rank == 0:
    # positions and velocities for the Solar system at a given time T:
    Time = Time("2022-12-02 00:00")
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

comm.Bcast(vector6N_solarsystem, root=0)

#--------------------------------------------
N = len(vector6N_solarsystem)  # total number of bodies
start = rank * (N // size)     
end = start + (N // size)
if (N % size) != 0 and (rank == (size-1)):
    end += (N % size)

print(f'Process {rank+1} out of {size} takes care of bodies {start} - {end}')


#--------------------------------------------
def g(W, m):
    # create dw/dt:
    dwdt = np.zeros(shape=(len(W), 2, 3))
    
    # calculate dx/dt:
    dwdt[:,0] = W[:,1]

    # calculate dv/dt:
    xi = W[start:end,0,np.newaxis]  # select bodies between [start, end)
    xj = W[:,0]                     # all bodies
    deltax = xi - xj                # distances between bodies in xi and all bodies in xj
    deltax_norm = np.linalg.norm(deltax, axis=2, keepdims=True)

    with np.errstate(divide='ignore', invalid='ignore'):
        before_sum = G * m[:,np.newaxis] * deltax / deltax_norm**3
        before_sum = np.where(np.isfinite(before_sum), before_sum, 0)
        dvidt = -np.sum(before_sum, axis=1)

    # MPI Allgatherv():
    sendbuf = dvidt.flatten()
    recvbuf = np.empty(N*3)
    recvcounts = np.array(comm.allgather(len(sendbuf)))
    displs = np.array(comm.allgather(3*rank*(N // size)))
    comm.Allgatherv(sendbuf=sendbuf, recvbuf=[recvbuf, recvcounts, displs, MPI.DOUBLE]) 
    dvdt = recvbuf.reshape((N, 3))
    dwdt[:,1] = dvdt

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
Wtime0 = MPI.Wtime()
time_solarsystem, phase_solarsystem = RungeKutta4th(vector6N_solarsystem, m_solarsystem, 0, 100, 1/(365.25))
print('Integration time: %s s' % (MPI.Wtime() - Wtime0))


#--------------------------------------------
# Animation:
make_animation = False

if make_animation == True and rank == 0:
    plt.style.use('dark_background')

    fig, ax = plt.subplots(1, figsize=(9, 9))
    l, b, h, w = .75, .75, .2, .2
    ax_zoom = fig.add_axes([l, b, w, h])

    label  = ['The Sun'] + planets[1:]
    marker = ['*', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
    markersize = [10, 5, 5, 5, 5, 10, 10, 10, 10]
    markersize_zoom = [20, 10, 10, 10, 10] + markersize[5:]
    color  = ['yellow', 'darkgrey', 'wheat', 'deepskyblue', 'red', 'orange', 'sienna', 'cyan', 'mediumslateblue']
    line_list = []
    traj_list = []
    traj_zoom_list = []
    zoom_list = []
    for i in range(0, len(planets)):
        line, = ax.plot([], [], marker=marker[i], markersize=markersize[i], color=color[i], ls='none', label=label[i])
        line_list.append(line)
        zoom, = ax_zoom.plot([], [], marker=marker[i], markersize=markersize[i], color=color[i], ls='none', label=label[i])
        zoom_list.append(zoom)
        traj, = ax.plot([], [], color=color[i], ls='-', lw=2, alpha=0.4)
        traj_list.append(traj)
        traj_zoom, = ax_zoom.plot([], [], color=color[i], ls='-', lw=2, alpha=0.4)
        traj_zoom_list.append(traj_zoom)
    txt = ax.text(0.05, 0.95, '', fontsize=14, ha='left', va='top', transform=ax.transAxes)

    line_ast, = ax.plot([], [], marker='.', markersize=5, color='grey', ls='none', label='Asteroids')

    # make the plot look nice:
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

    # function for animation:
    def animate(frame):
        for i in range(0, len(line_list)):
            line_list[i].set_data(phase_solarsystem[:,i,0,0][frame], phase_solarsystem[:,i,0,1][frame])
            zoom_list[i].set_data(phase_solarsystem[:,i,0,0][frame], phase_solarsystem[:,i,0,1][frame])
            traj_list[i].set_data(phase_solarsystem[:,i,0,0][:frame], phase_solarsystem[:,i,0,1][:frame])
            traj_zoom_list[i].set_data(phase_solarsystem[:,i,0,0][:frame], phase_solarsystem[:,i,0,1][:frame])
            
        line_ast.set_data(phase_solarsystem[:,len(line_list):,0,0][frame], phase_solarsystem[:,len(line_list):,0,1][frame])
        
        txt.set_text('{:0.2f} yr'.format(time_solarsystem[frame]))
        
        return line, txt

    # make and save animation:
    which_frames = np.arange(0, len(time_solarsystem), 100)
    fps = 30
    ani = animation.FuncAnimation(fig, animate, frames=which_frames, interval=fps, blit=True)
    ani.save("solarsystem.mp4")


#--------------------------------------------
MPI.Finalize()


#--------------------------------------------


