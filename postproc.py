import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load the saved data from the correct path
psi_time_evolution_real = np.load('psi_time_evolution_real.npy')
psi_time_evolution_imag = np.load('psi_time_evolution_imag.npy')

# Reconstruct the complex wave function from its real and imaginary parts
psi_time_evolution = psi_time_evolution_real + 1j * psi_time_evolution_imag

# Ensure the global variables are defined or redefined here if necessary
x_start, x_end, Nx = 0.0, 10.0, 100
x = np.linspace(x_start, x_end, Nx)
t_start, t_end, Nt = 0.0, 1.0, 500
t = np.linspace(t_start, t_end, Nt)

# Function to plot the probability density at specific time points
def plot_probability_density(x, psi_time_evolution, time_points, t):
    for time_point in time_points:
        plt.figure(figsize=(10, 6))
        probability_density = np.abs(psi_time_evolution[:, time_point])**2
        plt.plot(x, probability_density, label=f't = {t[time_point]:.2f}')
        plt.xlabel('Position')
        plt.ylabel('Probability Density')
        plt.title('Probability Density at Different Times')
        plt.legend()
        plt.show()

# Select time points to visualize
time_points = [0, Nt//4, Nt//2, 3*Nt//4, Nt-1]
plot_probability_density(x, psi_time_evolution, time_points, t)

# Create an animation showing the evolution of the wave packet over time
fig, ax = plt.subplots(figsize=(10, 6))

def animate(i):
    ax.clear()
    probability_density = np.abs(psi_time_evolution[:, i])**2
    ax.plot(x, probability_density)
    ax.set_xlabel('Position')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Wave Packet Evolution at t = {t[i]:.2f}')

# Create the animation
ani = FuncAnimation(fig, animate, frames=Nt, interval=50)

# Note for environments that do not support inline animation viewing:
# The animation is saved to a file, which can be downloaded and viewed externally.
ani.save('wave_packet_evolution.gif', writer='pillow')


# Provide the path to the saved animation for download
animation_file_path = 'wave_packet_evolution.gif'
print(animation_file_path)
