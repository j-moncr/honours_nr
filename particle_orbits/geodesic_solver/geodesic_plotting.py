import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from IPython.display import HTML
import os.path


# 3D trajectory plotting
def plot_traj(x, y, z, rs_1, rs_2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x,y,z, label="Particle")
    ax.set_title("Particle Trajectory near BBH using 1.5PN Approximation")
    ax.set_xlabel("x / (c^2 / GM)")
    ax.set_ylabel("y / (c^2 / GM)")
    ax.set_zlabel("z / (c^2 / GM)")
    

    ax.plot(rs_1[:,0], rs_1[:,1], rs_1[:,2], label='BH1', color="blue")
    ax.plot(rs_2[:,0], rs_2[:,1], rs_2[:,2], label='BH2', color="red")

    ax.legend()
    # ax.set_xlim(-200,200)
    # ax.set_ylim(-200, 200)
    # ax.set_zlim(-200, 200)
    
    plt.show()

def animate_trajectories(x,y,z,rs_1,rs_2, a=None, save_fig=False):

    # Create the figure and axes
    fig, ax = plt.subplots()
    
    if a:
        ax.set_xlim(-a, a)
        ax.set_ylim(-a, a)

    # Plot two primary masses (initial position)
    mass1, = ax.plot(rs_1[0,0], rs_1[0,1], 'o', color='blue', markersize=15, label="mass 1")
    mass2, = ax.plot(rs_2[0,0], rs_2[0,1], 'o', color='black', markersize=20, label="mass 2")

    # Plot initial position of test particle
    particle, = ax.plot(x[0], y[0], 'o', color='red', markersize=5, label="Particle")
    particle_trail, = ax.plot(x[0], y[0], '-', color='red', markersize=1)

    # Function to update the positions
    def update(i):
        mass1.set_data(rs_1[i, 0], rs_1[i, 1])
        mass2.set_data(rs_2[i, 0], rs_2[i, 1])
        particle.set_data(x[i], y[i])
        particle_trail.set_data(x[:i], y[:i])
    

    # Create the animation
    ani = FuncAnimation(fig, update, frames=range(0, len(x), len(x)//100), interval=200)

    plt.legend()
    
    fig.suptitle("1.5PN binary")
    if save_fig:
        # saving to m4 using ffmpeg writer            
        writervideo = FFMpegWriter(fps=60)
        save_fig = f"{save_fig}x.mp4" if os.path.isfile(f"./{save_fig}.mp4") else f"{save_fig}.mp4"
        ani.save(save_fig, writer=writervideo)
        plt.close()

