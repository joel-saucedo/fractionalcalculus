# Load the saved data
psi_time_evolution_real = np.load('psi_time_evolution_real.npy')
psi_time_evolution_imag = np.load('psi_time_evolution_imag.npy')

# Reconstruct the complex wave function from its real and imaginary parts
psi_time_evolution = psi_time_evolution_real + 1j * psi_time_evolution_imag

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

# To display the animation inline, you would typically use the IPython display utilities
# However, in this text interface, we cannot directly display the animation.
# You can use the following code to save the animation as a video or GIF file.
ani.save('wave_packet_evolution.mp4', writer='ffmpeg')

# Provide the path to the saved animation
animation_file_path = 'wave_packet_evolution.mp4'
animation_file_path
