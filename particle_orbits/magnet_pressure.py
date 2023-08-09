import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the magnetic field function
def magnetic_field(x, y, z):
    # Define the magnetic field vector B = (Bx, By, Bz)
    Bx = 0
    By = 0
    Bz = 1 + 0.09*x  # Uniform magnetic field in the z-direction

    # Return the magnetic field vector as a tuple
    return Bx, By, Bz

# Define the electric field function
def electric_field(x, y, z):
    # Define the electric field vector E = (Ex, Ey, Ez)
    Ex = 0
    Ey = 0
    Ez = 0  # No electric field

    # Return the electric field vector as a tuple
    return Ex, Ey, Ez

# Define the Lorentz force function
def lorentz_force(q, v, B, E):
    # Define the Lorentz force vector F = (Fx, Fy, Fz)
    Fx = q * (E[0] + np.cross(v, B)[0])
    Fy = q * (E[1] + np.cross(v, B)[1])
    Fz = q * (E[2] + np.cross(v, B)[2])

    # Return the Lorentz force vector as a tuple
    return Fx, Fy, Fz

# Define the particle trajectory function
def particle_trajectory(q, m, x0, v0, dt, t_max):
    # Initialize the particle position and velocity
    x = np.array(x0, dtype=float)
    v = np.array(v0, dtype=float)

    # Initialize the time array
    t = np.arange(0, t_max, dt)

    # Initialize the arrays to store the particle position and velocity at each time step
    x_array = np.zeros((len(t), 3))
    v_array = np.zeros((len(t), 3))

    # Run the simulation
    for i in range(len(t)):
        # Compute the magnetic and electric fields at the current position
        B = np.array(magnetic_field(*x))
        E = np.array(electric_field(*x))

        # Compute the Lorentz force on the particle
        F = np.array(lorentz_force(q, v, B, E))

        # Update the particle position and velocity using the Euler method
        x += v * dt
        v += F / m * dt

        # Store the particle position and velocity in the arrays
        x_array[i] = x
        v_array[i] = v

    # Return the particle position and velocity arrays
    return x_array, v_array

# Define the animation function
def animate(i, x_array=None):
    i += 1
    # Clear the axes
    ax.clear()

    # Set the axis limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Set the axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the title
    ax.set_title('Charged Particle Spiraling Around Magnetic Field')

    # # Compute the particle trajectory
    x_array, v_array = particle_trajectory(q, m, x0, v0, dt, i * dt)

    # Plot the particle trajectory
    ax.plot(x_array[:i,0], x_array[:i,1], x_array[:i,2], color='red')

    # Plot the magnetic field lines
    x = np.linspace(-1, 1, 60)
    y = np.linspace(-1, 1, 60)
    z = np.linspace(-1, 1, 60)
    X, Y, Z = np.meshgrid(x, y, z)
    Bx, By, Bz = magnetic_field(X, Y, Z)
    B_norm = np.sqrt(Bx**2 + By**2 + Bz**2)
    # ax.quiver(X, Y, Z, Bx/B_norm, By/B_norm, Bz/B_norm, length=0.2, color='blue')


    # Plot the particle
    ax.scatter(x_array[i,0], x_array[i,1], x_array[i,2], color='green', s=50)

    # Adjust the viewing angle
    ax.view_init(elev=20, azim=i*4)

    # Return the plot
    return ax



q = 1 # Charge of the particle (in Coulombs)
m = 1 # Mass of the particle (in kg)
x0 = [0, m*np.sqrt(101)/(q*1), 0] # Initial position of the particle (in meters)
v0 = [1.0, 0.0, 10.0] # Initial velocity of the particle (in meters per second)
dt = 0.01 # Time step of the simulation (in seconds)
t_max = 100 # Maximum time of the simulation (in seconds)

x_array, v_array = particle_trajectory(q, m, x0, v0, dt, t_max)

# 3D plot of the particle trajectory
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_array[:,0], x_array[:,1], x_array[:,2], color='red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Charged Particle Spiraling Around Magnetic Field')

ax.set_xlim([-10, 10])
ax.set_ylim([0, 10])

# 3D plot of the magnetic field lines - showing field strength increasing along x direction
# Make it a stream plot



x = np.linspace(-10, 10, 60)
y = np.linspace(-0, 20, 60)
z = np.linspace(x_array[0, 2], x_array[-1, 2], 60)
X, Y, Z = np.meshgrid(x, y, z)
Bx, By, Bz = magnetic_field(X, Y, Z)
B_norm = np.sqrt(Bx**2 + By**2 + Bz**2)
# ax.quiver(X, Y, Z, Bx/B_norm, By/B_norm, Bz/B_norm, length=0.2, color='blue')
ax.stream_plot(X, Y, Z, Bx/B_norm, By/B_norm, Bz/B_norm, color='blue', linewidth=1, density=2, arrowstyle='->', arrowsize=1.5)

plt.show()


# print(x_array)
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# num_steps = 10
# init = lambda: animate(0)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
#
# anim = animation.FuncAnimation(fig, animate, fargs=(x_array,), init_func=init, frames=num_steps, interval=20, blit=True)
#
# anim.save('charged_particle_animation.mp4', writer=animation.FFMpegWriter(fps=30))