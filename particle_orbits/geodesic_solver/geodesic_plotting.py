import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from IPython.display import HTML
import os

def name_file(filename, iteration=0, no_parent=False):
    """If filename already exists, add a number for each time same file name is written.
        For example if test.mp4, test1.mp4, and test2.mp4 exist, return test3.mp4"""
    # no_parent=True when running from py file in main, no_parent=False when running from notebook

    cwd = os.getcwd()
    parent_dir = cwd if no_parent else os.path.dirname(cwd)
    new_filename = os.path.join(parent_dir, "animations", f"{filename}_{iteration}.mp4")
    if os.path.isfile(new_filename):
        new_filename = name_file(f"{filename}", iteration=iteration+1)
    
    return new_filename
    

# 3D trajectory plotting
def plot_traj(x, y, z, rs_1, rs_2, proj="2D"):
    if proj == "3D":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x,y,z, label="Particle")
        plt.plot(rs_1[:,0], rs_1[:,1], rs_1[:,2], label='BH1', color="blue")
        plt.plot(rs_2[:,0], rs_2[:,1], rs_2[:,2], label='BH2', color="red")
        ax.set_title("Particle Trajectory near BBH using 1.5PN Approximation")
        ax.set_xlabel("x / L_0")
        ax.set_ylabel("y / L_0")
        ax.set_zlabel("z / L_0")
    elif proj == "2D":
        fig, ax = plt.subplots()
        ax.plot(x, y, label="Particle")
        ax.set_title("Particle Trajectory near BBH using 1.5PN Approximation")
        ax.set_xlabel("x / L_0")
        ax.set_ylabel("y / L_0")    
    # M1, M2, a1, a2, b = kwargs.get("m1", 1), kwargs["m2"], kwargs["a1"], kwargs["a2"], kwargs["b"]
    # M1, M2, a1, a2, b = kwargs.get("m1", 1), kwargs.get("m2", 1), kwargs.get("a1", 1), kwargs.get("a2", 1), kwargs.get("b", 1)
    
    # M1 = M1/M2 * b / 10
    # M2 = M2/M1 * b / 10
    
    # a1 = a1 * M1
    # a2 = a2 * M2
    
    # Reh = M1 + np.sqrt(M1**2 - a1**2)
    # u, v = np.mgrid[0:2*np.pi:200j, 0:np.pi:100j]
    # xb = Reh * np.cos(u)*np.sin(v)
    # yb = Reh * np.sin(u)*np.sin(v)
    # zb = Reh * np.cos(v)
    # ax.plot_wireframe(xb+b/2, yb, zb, color="r")
    
    # Reh2 = M2 + np.sqrt(M2**2 - a2**2)
    # u2, v2 = np.mgrid[0:2*np.pi:200j, 0:np.pi:100j]
    # xb2 = Reh2 * np.cos(u2)*np.sin(v2)
    # yb2 = Reh2 * np.sin(u2)*np.sin(v2)
    # zb2 = Reh2 * np.cos(v2)
    # ax.plot_wireframe(xb2-b/2, yb2, zb2, color="b")

    # ax.plot(rs_1[:,0], rs_1[:,1], rs_1[:,2], label='BH1', color="blue")
    # ax.plot(rs_2[:,0], rs_2[:,1], rs_2[:,2], label='BH2', color="red")

    ax.legend()
    # ax.set_xlim(-200,200)
    # ax.set_ylim(-200, 200)
    # ax.set_zlim(-200, 200)
    
    plt.show()

def animate_trajectories(x, y, z, rs_1, rs_2, a=None,save_fig=False,no_parent=False,title="1.5PN", proj="2D"):
    # Create the figure and axes
    fig = plt.figure()
    
    if proj=="3D":
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel("z / L_0")
        
    else:
        ax = fig.add_subplot(111)
    
    ax.set_xlabel("x / L_0")
    ax.set_ylabel("y / L_0")

    if a:
        a_x, a_y, a_z = a[0], a[1], a[2]
        ax.set_xlim(-a_x, a_x)
        ax.set_ylim(-a_y, a_y)
        if proj=="3D":
            ax.set_zlim(-a_z, a_z)

    # Plot two primary masses (initial position)
    if proj=="3D":
        # mass1, = ax.plot(rs_1[0:1, 0], rs_1[0:1, 1], rs_1[0:1, 2], 'o', color='blue', markersize=15, label="mass 1")
        # mass2, = ax.plot(rs_2[0:1, 0], rs_2[0:1, 1], rs_2[0:1, 2], 'o', color='black', markersize=20, label="mass 2")
        # particle, = ax.plot(x[0:1], y[0:1], z[0:1], 'o', color='red', markersize=5, label="Particle")
        mass1 = ax.scatter(rs_1[0,0], rs_1[0,1], c='blue', s=150, label="mass 1", depthshade=False)
        mass2 = ax.scatter(rs_2[0,0], rs_2[0,1], c='black', s=200, label="mass 2", depthshade=False)
        particle = ax.scatter(x[0], y[0], z[0], c='red', s=50, label="Particle", depthshade=False)
        particle_trail, = ax.plot(x[0:1], y[0:1], z[0:1], '-', color='red', markersize=1)

    else:
        mass1, = ax.plot(rs_1[0, 0], rs_1[0, 1], 'o', color='blue', markersize=15, label="mass 1")
        mass2, = ax.plot(rs_2[0, 0], rs_2[0, 1], 'o', color='black', markersize=20, label="mass 2")
        particle, = ax.plot(x[0], y[0], 'o', color='red', markersize=5, label="Particle")
        particle_trail, = ax.plot(x[0], y[0], '-', color='red', markersize=1)

    # Function to update the positions
    def update(i):
        if proj=="3D":
            # mass1.set_data(rs_1[i, 0], rs_1[i, 1])
            # mass1.set_3d_properties([rs_1[i, 2]])
            
            # mass2.set_data(rs_2[i, 0], rs_2[i, 1])
            # mass2.set_3d_properties([rs_2[i, 2]])
            
            # particle.set_data(x[i], y[i])
            # particle.set_3d_properties([z[i]])
            
            particle_trail.set_data(x[:i], y[:i])
            particle_trail.set_3d_properties(z[:i])
            mass1._offsets3d = ([rs_1[i, 0]], [rs_1[i, 1]], [rs_1[i, 2]])
            mass2._offsets3d = ([rs_2[i, 0]], [rs_2[i, 1]], [rs_2[i, 2]])
            particle._offsets3d = ([x[i]], [y[i]], [z[i]])

        else:
            mass1.set_data(rs_1[i, 0], rs_1[i, 1])
            mass2.set_data(rs_2[i, 0], rs_2[i, 1])
            particle.set_data(x[i], y[i])
            particle_trail.set_data(x[:i], y[:i])

    # Create the animation
    ani = FuncAnimation(fig, update, frames=range(0, len(x), len(x) // min(len(x), 300)), interval=200)

    plt.legend()
    
    fig.suptitle(title)
    if save_fig:
        # saving to mp4 using ffmpeg writer            
        writervideo = FFMpegWriter(fps=24)
        save_fig = name_file(title, no_parent=no_parent)
        ani.save(save_fig, writer=writervideo)
    plt.show()


# if __name__ == "__main__":
#     x = np.zeros(100)
#     y = np.zeros(100)
#     z = np.zeros(100)
#     rs_1 = np.zeros((100, 3))
#     rs_2 = np.zeros((100, 3))
#     animate_trajectories(x, y, z, rs_1, rs_2, save_fig="test", proj="3D", no_parent=True)

# def animate_trajectories(x,y,z,rs_1,rs_2, a=None, save_fig=False, no_parent=False):

#     # Create the figure and axes
#     fig, ax = plt.subplots()
    
#     if a:
#         ax.set_xlim(-a, a)
#         ax.set_ylim(-a, a)

#     # Plot two primary masses (initial position)
#     mass1, = ax.plot(rs_1[0,0], rs_1[0,1], 'o', color='blue', markersize=15, label="mass 1")
#     mass2, = ax.plot(rs_2[0,0], rs_2[0,1], 'o', color='black', markersize=20, label="mass 2")

#     # Plot initial position of test particle
#     particle, = ax.plot(x[0], y[0], 'o', color='red', markersize=5, label="Particle")
#     particle_trail, = ax.plot(x[0], y[0], '-', color='red', markersize=1)

#     # Function to update the positions
#     def update(i):
#         mass1.set_data(rs_1[i, 0], rs_1[i, 1])
#         mass2.set_data(rs_2[i, 0], rs_2[i, 1])
#         particle.set_data(x[i], y[i])
#         particle_trail.set_data(x[:i], y[:i])
    

#     # Create the animation
#     ani = FuncAnimation(fig, update, frames=range(0, len(x), len(x)//min(len(x), 300)), interval=200)

#     plt.legend()
    
#     fig.suptitle("1.5PN binary")
#     if save_fig:
#         # saving to m4 using ffmpeg writer            
#         writervideo = FFMpegWriter(fps=24)
#         save_fig = name_file(save_fig, no_parent=no_parent)
#         ani.save(save_fig, writer=writervideo)
#         plt.close()

