# import numpy as np
# import matplotlib.pyplot as plt
#
# # Define the coordinates of the points where we want to calculate the electric field
# x = np.linspace(-5, 5, 50)
# y = np.linspace(-5, 5, 50)
# X, Y = np.meshgrid(x, y)
#
# # Define the position, strength, and orientation of the two electric dipoles
# d1_pos = np.array([-1, 0])
# d1_strength = 1
# d1_orientation = np.array([1, 0])
#
# d2_pos = np.array([1, 0])
# d2_strength = -1
# d2_orientation = np.array([-1, 0])
#
# # Calculate the electric field components at each point
# R1 = np.sqrt((X - d1_pos[0])**2 + (Y - d1_pos[1])**2)
# R2 = np.sqrt((X - d2_pos[0])**2 + (Y - d2_pos[1])**2)
#
# E1_x = (d1_strength*(X - d1_pos[0]) - 3*(X - d1_pos[0])*(X - d1_pos[0])*(X - d1_pos[0] - 3*d1_orientation[0]*(Y - d1_pos[1])**2)/R1**5)/R1**3
# E1_y = (d1_strength*(Y - d1_pos[1]) - 3*(Y - d1_pos[1])*(Y - d1_pos[1])*(Y - d1_pos[1] - 3*d1_orientation[1]*(X - d1_pos[0])**2)/R1**5)/R1**3
#
# E2_x = (d2_strength*(X - d2_pos[0]) - 3*(X - d2_pos[0])*(X - d2_pos[0])*(X - d2_pos[0] - 3*d2_orientation[0]*(Y - d2_pos[1])**2)/R2**5)/R2**3
# E2_y = (d2_strength*(Y - d2_pos[1]) - 3*(Y - d2_pos[1])*(Y - d2_pos[1])*(Y - d2_pos[1] - 3*d2_orientation[1]*(X - d2_pos[0])**2)/R2**5)/R2**3
#
# # Add the electric fields from the two dipoles
# E_x = E1_x + E2_x
# E_y = E1_y + E2_y
#
# # Plot the electric field lines
# plt.streamplot(X, Y, E_x, E_y, density=1.5)
#
# # Set the axis labels and limits
# plt.xlabel('x')
# plt.ylabel('y')
# plt.xlim(-5, 5)
# plt.ylim(-5, 5)
#
# # Display the plot
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define the grid
x = np.linspace(-20, 20, 41)
y = np.linspace(-20, 20, 41)
X, Y = np.meshgrid(x, y)

# Define the charges
q1 = 1
q2 = -1
d = 10  # separation between dipoles

# Define the dipole moments
p1 = np.array([0, 0.5, 0])
p2 = np.array([0, -0.5, 0])

# Define the electric field function
def E_field(x, y):
    softening_parameter = 0.01   # Prevent divide by zero errors

    r1 = np.array([x, y, 0]) - d/2 * np.array([0, 1, 0])
    r2 = np.array([x, y, 0]) + d/2 * np.array([0, 1, 0])
    r1_norm = np.linalg.norm(r1) + softening_parameter**2
    r2_norm = np.linalg.norm(r2) + softening_parameter**2
    E1 = q1 * (3*r1.dot(p1)*r1/r1_norm**5 - p1/r1_norm**3)
    E2 = q2 * (3*r2.dot(p2)*r2/r2_norm**5 - p2/r2_norm**3)
    return E1 + E2

# Compute the electric field
def compute_E(X, Y):
    Ex, Ey, Ez = np.zeros_like(X), np.zeros_like(Y), np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            E = E_field(x[i], y[j])
            Ex[j,i], Ey[j,i], Ez[j,i] = E
    return Ex, Ey, Ez

Ex, Ey, Ez = compute_E(X, Y)

# Plot the electric field
fig, ax = plt.subplots()
ax.streamplot(x, y, Ex, Ey, density=2, color='black', linewidth=1.5, arrowstyle='->', arrowsize=1.5)

# Add labels and title
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Electric field of two separated dipoles')

# Show the plot
plt.show()
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
#
# # Define the grid
# x = np.linspace(-20, 20, 41)
# y = np.linspace(-20, 20, 41)
# X, Y = np.meshgrid(x, y)
#
# # Define the charges
# q1 = 1
# q2 = -1
#
# # Define the dipole moments
# p1 = np.array([0, 0.5, 0])
# p2 = np.array([0, -0.5, 0])
#
# # Define the initial separation between dipoles
# d0 = 1
# d = d0
#
# # Define the speed at which the dipoles are moving apart
# v = 0.1
#
# # Define the electric field function
# def E_field(x, y):
#     r1 = np.array([x, y, 0]) - d/2 * np.array([0, 1, 0])
#     r2 = np.array([x, y, 0]) + d/2 * np.array([0, 1, 0])
#     r1_norm = np.linalg.norm(r1)
#     r2_norm = np.linalg.norm(r2)
#     E1 = q1 * (3*r1.dot(p1)*r1/r1_norm**5 - p1/r1_norm**3)
#     E2 = q2 * (3*r2.dot(p2)*r2/r2_norm**5 - p2/r2_norm**3)
#     return E1 + E2
#
# # Define the update function for the animation
# def update(frame):
#     global d
#     d = d0 + v * frame
#     Ex, Ey, Ez = np.zeros_like(X), np.zeros_like(Y), np.zeros_like(X)
#     for i in range(len(x)):
#         for j in range(len(y)):
#             E = E_field(x[i], y[j])
#             Ex[j,i], Ey[j,i], Ez[j,i] = E
#     print(E)
#     ax.clear()
#     ax.streamplot(x, y, Ex, Ey, density=2, color='black', linewidth=1.5, arrowstyle='->', arrowsize=1.5)
#     ax.set_xlabel('x (m)')
#     ax.set_ylabel('y (m)')
#     ax.set_title('Electric field of two moving dipoles (d={:.2f}m)'.format(d))
#     return ax,
#
# # Set up the figure and axis
# fig, ax = plt.subplots()
#
# # Create the animation
# ani = animation.FuncAnimation(fig, update, frames=10, interval=1000, blit=True)
#
# # Save the animation to a file
# ani.save('dipole_field_animation_2.mp4', writer='ffmpeg')
#
# print("done")
#
# # Show the plot (not needed for the saved animation)
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
#
# # Define the parameters
# Q1 = 1  # Dipole moment of the first charge
# Q2 = -1  # Dipole moment of the second charge
# d = 1  # Separation between the charges
# v = 0.1  # Velocity of the charges
#
# # Define the x range
# x_min = -20
# x_max = 20
# y_min = -20
# y_max = 20
#
# # Define the number of points in the x and y directions
# n_points = 50
#
# # Create a grid of x and y points
# x = np.linspace(x_min, x_max, n_points)
# y = np.linspace(y_min, y_max, n_points)
# X, Y = np.meshgrid(x, y)
#
# # Define the function for the magnetic field
# def Bx(x, y):
#     r1 = np.sqrt((x - d / 2) ** 2 + y ** 2)
#     r2 = np.sqrt((x + d / 2) ** 2 + y ** 2)
#     Bx1 = Q1 * (y * r1 ** 2 - 3 * x * y * r1) / (r1 ** 5)
#     Bx2 = Q2 * (y * r2 ** 2 - 3 * x * y * r2) / (r2 ** 5)
#     return Bx1 + Bx2
#
# def By(x, y):
#     r1 = np.sqrt((x - d / 2) ** 2 + y ** 2)
#     r2 = np.sqrt((x + d / 2) ** 2 + y ** 2)
#     By1 = Q1 * (x * r1 ** 2 - 3 * x ** 2 * r1) / (r1 ** 5)
#     By2 = Q2 * (x * r2 ** 2 - 3 * x ** 2 * r2) / (r2 ** 5)
#     return By1 + By2
#
# # Create a figure and axis
# fig, ax = plt.subplots()
#
# # Define the quiver plot
# line, = ax.plot([], [], lw=1.5)
#
# # Define the initialization function for the animation
# def init():
#     line.set_data([], [])
#     return line,
#
# # Define the update function for the animation
# def update(num):
#     x_shift = v * num
#     Bx_shifted = Bx(X - x_shift, Y)
#     By_shifted = By(X - x_shift, Y)
#     norm = np.sqrt(Bx_shifted ** 2 + By_shifted ** 2)
#     Bx_norm = Bx_shifted / norm
#     By_norm = By_shifted / norm
#     line.set_data(X.ravel(), Y.ravel())
#     line.set_color(norm.ravel())
#     line.set_linewidth(1.5)
#     line.set_antialiased(False)
#     line.set_alpha(0.8)
#     return line,
#
# # Create the animation
# anim = animation.FuncAnimation(fig, update, init_func=init, frames=100, interval=50, blit=True)
#
# # Save the animation to a file
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# anim.save('dipole_animation.mp4', writer=writer)