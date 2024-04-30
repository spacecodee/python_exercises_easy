import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_numbers_with_phrase():
    # Sample data
    numbers = np.array([1, 2, 9, 5, 10, 11, 3, 4])
    phrase = "Hello AI"

    # Scatter plot with random y-axis values
    plt.scatter(range(len(numbers)), [np.random.random() for _ in numbers])

    # Text placement with spacing
    char_spacing = 1
    for i, char in enumerate(phrase):
        x = i * char_spacing
        y = 0.5  # Constant y-axis value for demonstration
        plt.text(x, y, char, ha="center", va="center", fontsize=16)

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Numbers with Phrase")
    plt.show()


def show_in_cartesian_plane():
    bandwidth_size = 80
    formats = [
        {"center": 800, "bandwidth": 100},
        {"center": 1200, "bandwidth": 130},
        {"center": 2400, "bandwidth": 300},
    ]

    # Plotting range (adjust as needed)
    plt.ylim(200, 3500)

    for formant in formats:
        center_freq = formant["center"]
        bandwidth = formant["bandwidth"]

        # Generate random dots around center frequency with specified bandwidth
        dots = np.random.normal(center_freq, bandwidth, bandwidth_size)  # Adjust number of dots as needed
        plt.scatter(np.random.rand(bandwidth_size), dots, marker='o', alpha=0.6)  # Random x positions for distinction

    # Labels and title (customize as needed)
    plt.xlabel("Time (arbitrary)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Vocal Formants")
    plt.legend(["F1", "F2", "F3"])  # Add legends for formats
    plt.show()


# use numpy to create arrays multidimensional
def numpy_matplotlib_2d():
    # Create a 2D array (matrix) with numpy
    matrix = np.array([[1, 2, 3, 10], [4, 5, 6, 11], [7, 8, 9, 12]])

    # Create x and y coordinates for the points in the matrix
    x, y = np.meshgrid(range(matrix.shape[1]), range(matrix.shape[0]))

    # Flatten the matrix and the coordinates
    x, y, z = x.flatten(), y.flatten(), matrix.flatten()

    # Create a scatter plot
    plt.scatter(x, y, c=z, s=100)

    # Add a colorbar
    plt.colorbar(label='Value')

    # Set labels
    plt.xlabel('X')
    plt.ylabel('Y')

    # Show the plot
    plt.show()


def numpy_matplotlib_3d():
    # Create a 3D array (tensor) with numpy
    tensor = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                       [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                       [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])

    # Create x, y, and z coordinates for the points in the tensor
    x, y, z = np.meshgrid(range(tensor.shape[2]), range(tensor.shape[1]), range(tensor.shape[0]))

    # Flatten the tensor and the coordinates
    x, y, z, value = x.flatten(), y.flatten(), z.flatten(), tensor.flatten()

    # Create a scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=value, s=100)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()


def gauss_bell():
    data = np.random.randn(1000)

    # Calculate mean and standard deviation
    mu, std = np.mean(data), np.std(data)

    # Plot a histogram with the data
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g')

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = 1 / (np.sqrt(2 * np.pi)) * np.exp(-0.5 * (1 / xmax * (x)) ** 2)

    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)

    plt.show()


def plot_parabola():
    a = 1
    b = -2
    c = 1

    # Define range of x values
    x_range = np.linspace(-10, 10, 400)

    # Calculate y values
    y_values = a * x_range ** 2 + b * x_range + c

    # Plot the parabola
    plt.plot(x_range, y_values)

    # Set labels
    plt.xlabel('X')
    plt.ylabel('Y')

    # Show the plot
    plt.show()


def plot_3d_parabola():
    a = 1
    b = 0
    c = 0

    # Define range of x and y values
    x_range = [-10, 10]

    # Create a meshgrid of x and y values
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(x_range[0], x_range[1], 100)
    x, y = np.meshgrid(x, y)

    # Calculate corresponding z values for the parabola
    z = a * x ** 2 + b * y + c

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()


def heatmap_plane():
    # Generate a random 10x10 matrix
    data = np.random.rand(15, 15)

    # Create a heatmap
    sns.heatmap(data, annot=True)

    # Show the plot
    plt.show()


if __name__ == '__main__':
    plot_numbers_with_phrase()
    show_in_cartesian_plane()
    numpy_matplotlib_2d()
    numpy_matplotlib_3d()
    gauss_bell()
    plot_parabola()
    plot_3d_parabola()
    heatmap_plane()
