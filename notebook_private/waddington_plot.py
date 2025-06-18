import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

def waddington_plot(branches=[1, 2, 4],
                    horizontal_distribution=None,
                    horizontal_distortion=None,
                    vertical_distribution=None,
                    vertical_distortion=None,
                    bottom_width=2000,
                    top_width=None,
                    ridge_count=50,
                    ridge_height=5,
                    sns_palette="Spectral",
                    line_color="black", line_type='-', line_size=0.1, line_alpha=0.2,
                    theme_void=True, hide_legend=True, do_return=False):
    
    # Default top width
    if top_width is None:
        top_width = 0.3 * bottom_width
    
    # Set up horizontal distribution
    if horizontal_distribution is None:
        horizontal_distribution = [np.ones(b) for b in branches]

    print(f"Horizontal distribution: {horizontal_distribution}")
    
    # Set up horizontal distortion
    if horizontal_distortion is None:
        horizontal_distortion = [np.repeat(b, 2) for b in horizontal_distribution]
    else:
        cosine_left = horizontal_distortion[::2]
        cosine_right = horizontal_distortion[1::2]
        horizontal_distortion = []
        for i in range(len(horizontal_distribution)):
            level_distortion = []
            for j in range(len(horizontal_distribution[i])):
                left_dist = cosine_left.pop(0) / (cosine_left.pop(0) + cosine_right.pop(0))
                right_dist = 1 - left_dist
                level_distortion.extend([left_dist * horizontal_distribution[i][j], right_dist * horizontal_distribution[i][j]])
            horizontal_distortion.append(level_distortion)

    print(f"Horizontal distortion post: {horizontal_distortion}")
    
    # Set up vertical distribution
    if vertical_distribution is None:
        vertical_distribution = np.ones(len(branches) - 1)
    
    print(f"Vertical distribution: {vertical_distribution}")
    vertical_distribution = np.round(ridge_count * vertical_distribution / np.sum(vertical_distribution)).astype(int)
    print(f"Vertical distribution post: {vertical_distribution}")

    # Set up vertical distortion
    if vertical_distortion is None:
        vertical_distortion = [np.repeat(b / 2, 2) for b in vertical_distribution]
    else:
        cosine_up = vertical_distortion[::2]
        cosine_dn = vertical_distortion[1::2]
        vertical_distortion = []
        for i in range(len(vertical_distribution)):
            up_dist = cosine_up.pop(0) / (cosine_up.pop(0) + cosine_dn.pop(0))
            dn_dist = 1 - up_dist
            vertical_distortion.append([up_dist * vertical_distribution[i], dn_dist * vertical_distribution[i]])
    
    print(f"Vertical distortion post: {vertical_distortion}")
    
    # Create the skeleton curves
    skeleton_curves = []
    for i in range(len(branches)):
        bias = horizontal_distortion[i]
        sections = np.quantile(np.arange(1, bottom_width + 1), np.cumsum(bias) / np.sum(bias), interpolation='nearest')
        curves = []
        for j in range(0, len(sections), 2):
            left_curve = np.cos(np.linspace(0, np.pi, sections[j]))
            right_curve = np.cos(np.linspace(np.pi, 2 * np.pi, sections[j + 1]))
            curves.extend(left_curve.tolist() + right_curve.tolist())
        skeleton_curves.append(np.array(curves[:bottom_width]))

    # Prepare data for plotting
    gg_data = []
    waves_sum = np.sum([sum(vd) for vd in vertical_distortion])
    waves_cur = waves_sum
    ratio = 1 - top_width / bottom_width
    
    for i in range(len(vertical_distortion)):
        curve1 = skeleton_curves[i]
        curve2 = skeleton_curves[i + 1]
        curve_mid = (curve1 + curve2) / 2
        
        # Upper part of the ridge
        for j in range(int(vertical_distortion[i][0])):
            alpha = j / vertical_distortion[i][0]
            curve = (1 - alpha) * curve1 + alpha * curve_mid
            x_shift = ratio * (bottom_width / 2) * waves_cur / waves_sum
            x = np.arange(1, bottom_width + 1) * (1 - ratio * waves_cur / waves_sum) + x_shift
            gg_data.append((x, waves_cur, curve))
            waves_cur -= 1

        # Lower part of the ridge
        for j in range(int(vertical_distortion[i][1])):
            alpha = j / vertical_distortion[i][1]
            curve = (1 - alpha) * curve_mid + alpha * curve2
            x_shift = ratio * (bottom_width / 2) * waves_cur / waves_sum
            x = np.arange(1, bottom_width + 1) * (1 - ratio * waves_cur / waves_sum) + x_shift
            gg_data.append((x, waves_cur, curve))
            waves_cur -= 1

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    palette = sns.color_palette(sns_palette, n_colors=len(gg_data))
    for i, (x, y, curve) in enumerate(gg_data):
        ax.plot(x, y + ridge_height * curve, color=line_color, linestyle=line_type, linewidth=line_size, alpha=line_alpha)
        ax.fill_between(x, y-10, y + ridge_height * curve, color=palette[i], alpha=line_alpha)
    
    if theme_void:
        ax.axis('off')
    
    if hide_legend:
        ax.legend().set_visible(False)
    
    plt.show()

    if do_return:
        return fig, ax


def landscape_potential(x, y, wells):
    z = np.zeros_like(x)
    
    for (x0, y0, depth, width) in wells:
        z += -depth * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * width**2))
    
    return z


def calculate_distribution(wells, temperature=0.1, num_points=1000, grid_size=500, grid_range=(-3, 3), seed=42):
    # Set the random seed
    np.random.seed(seed)
    # Define the grid
    x = np.linspace(grid_range[0], grid_range[1], grid_size)
    y = np.linspace(grid_range[0], grid_range[1], grid_size)
    x, y = np.meshgrid(x, y)

    # Calculate the Z values (potential energy of the landscape)
    z = landscape_potential(x, y, wells)

    # Calculate the Boltzmann distribution
    prob_density = np.exp(-z / temperature)

    # Normalize the probability density
    prob_density /= np.sum(prob_density)

    # Flatten the grid and probability density for sampling
    x_flat = x.flatten()
    prob_density_flat = prob_density.flatten()

    # Sample points based on the probability density
    indices = np.random.choice(np.arange(len(x_flat)), size=num_points, p=prob_density_flat)

    return x, y, z, indices



def plot_waddington_landscape(x, y, z, indices, color='black', cmap='Spectral', contour_alpha = 0.8, point_alpha = 0.5, save_path=None, s=5, dpi=300, figsize=(8, 6), image_format='png'):
    # Flatten the grid arrays
    x_flat = x.flatten()
    y_flat = y.flatten()

    # Extract the sampled points using the indices
    x_samples = x_flat[indices]
    y_samples = y_flat[indices]

    # Plot the contours of the landscape
    fig, ax = plt.subplots(figsize=figsize)
    contour = ax.contourf(x, y, z, levels=50, cmap=cmap, alpha=contour_alpha)
    ax.contour(x, y, z, levels=10, colors='black', linewidths=0.5)

    # Plot the sampled points with colors by group
    ax.scatter(x_samples, y_samples, color=color, s=s, alpha=point_alpha)

    # cbar = fig.colorbar(contour)
    # cbar.set_label('Potential Energy')

    # Customize the plot
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks

    ax.set_xlabel('')   # Remove x-axis label
    ax.set_ylabel('')   # Remove y-axis label
    ax.set_aspect('equal')

    # Save the plot if a save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, format=image_format)

    plt.show()



def plot_waddington_landscape_3d(x, y, z, indices=None, z_adjust=0.0, surface_alpha = 0.7, point_size=5, point_alpha=0.8, elev=30, azim=45, cmap='Spectral', save_path=None, figsize=(10, 8), dpi=300, image_format='png'):
    
    # Create a 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(x, y, z, cmap=cmap, edgecolor='none', alpha=surface_alpha, zorder=1)
    
    # Optionally, plot the sampled points on the surface
    if indices is not None:
        # Flatten the grid arrays
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()
        # Extract the sampled points using the indices
        x_samples = x_flat[indices]
        y_samples = y_flat[indices]
        z_samples = z_flat[indices]

        # Interpolate the z values for the sampled points
        #z_samples = griddata((x.flatten(), y.flatten()), z.flatten(), (x_samples, y_samples), method='linear')
        z_samples = [z+z_adjust for z in z_samples]

        #ax.scatter(x_samples, y_samples, z_samples, color='black', s=point_size, alpha=point_alpha)
        ax.plot(x_samples, y_samples, z_samples, color='black', marker='o', linestyle='None', markersize=point_size, alpha=point_alpha, zorder=3)

    # Add color bar
    #fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Customize the plot ax.set_xticks([])  # Remove x-axis ticks
    ax.set_xticks([])  # Remove y-axis ticks
    ax.set_yticks([])  # Remove z-axis ticks
    ax.set_zticks([])  # Remove z-axis ticks
    ax.grid(True)

    ax.set_xlabel('')   # Remove x-axis label
    ax.set_ylabel('')   # Remove y-axis label
    ax.set_zlabel('')   # Remove z-axis label
    
    ax.view_init(elev=elev, azim=azim)
    # Save the plot if a save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, format=image_format)

    plt.show()


def xyz_to_coords(x, y, z):
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    coords_grid = np.vstack((x_flat, y_flat, z_flat))
    return coords_grid


def coords_to_xyz(coords_grid):
    """
    Convert flattened coordinates back into x, y, z arrays for a square grid.

    Parameters:
    - coords_grid: A 3xN numpy array where N is the number of points (flattened grid).

    Returns:
    - x: 2D numpy array for the x-coordinates.
    - y: 2D numpy array for the y-coordinates.
    - z: 2D numpy array for the z-coordinates.
    """
    # Number of points (N)
    num_points = coords_grid.shape[1]

    # Calculate grid_size as sqrt of the number of points
    grid_size = int(np.sqrt(num_points))

    # Ensure that num_points is a perfect square
    assert grid_size ** 2 == num_points, "Number of points is not a perfect square!"

    # Extract the x, y, z components from coords_grid
    x_flat = coords_grid[0, :]
    y_flat = coords_grid[1, :]
    z_flat = coords_grid[2, :]

    # Reshape the flattened arrays back into grid_size x grid_size 2D arrays
    x = x_flat.reshape((grid_size, grid_size))
    y = y_flat.reshape((grid_size, grid_size))
    z = z_flat.reshape((grid_size, grid_size))

    return x, y, z


def rotate_coordinates(coords, theta_x=0, theta_y=0, theta_z=0):
    """
    Rotates coordinates around the x, y, and z axes.

    Parameters:
    - coords: numpy array of shape (3, N), where N is the number of points.
    - theta_x: Rotation angle around the x-axis in radians.
    - theta_y: Rotation angle around the y-axis in radians.
    - theta_z: Rotation angle around the z-axis in radians.

    Returns:
    - rotated_coords: numpy array of shape (3, N), rotated coordinates.
    """
    # Rotation matrix around x-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])

    # Rotation matrix around y-axis
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])

    # Rotation matrix around z-axis
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix
    R = Rz @ Ry @ Rx

    # Apply rotation
    rotated_coords = R @ coords

    return rotated_coords



def shift_coordinates(coords, dx=0, dy=0, dz=0):
    """
    Shifts coordinates along the x, y, and z axes.

    Parameters:
    - coords: numpy array of shape (3, N), where N is the number of points.
    - dx: Shift along the x-axis.
    - dy: Shift along the y-axis.
    - dz: Shift along the z-axis.

    Returns:
    - shifted_coords: numpy array of shape (3, N), shifted coordinates.
    """
    # Create shift vector
    shift_vector = np.array([[dx], [dy], [dz]])
    
    # Apply shift
    shifted_coords = coords + shift_vector

    return shifted_coords



def plot_two_waddington_landscapes_3d(x1, y1, z1, indices1, 
                                      x2, y2, z2, indices2, 
                                      color1='blue', color2='red',
                                      cmap1='Blues_r', cmap2='YlOrRd_r',
                                      show_projection=False, projection_z=0,
                                      surface_alpha=0.7,
                                      s=5, elev=30, azim=45, 
                                      show_grid=False,
                                      figsize=(10, 8), dpi=300, save_path=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the first surface
    surf1 = ax.plot_surface(x1, y1, z1, cmap=cmap1, edgecolor='none', alpha=surface_alpha)
    
    # Plot the second surface
    surf2 = ax.plot_surface(x2, y2, z2, cmap=cmap2, edgecolor='none', alpha=surface_alpha)
    
    # Plot the sampled points on the first surface
    if indices1 is not None:
        x_samples1 = x1.flatten()[indices1]
        y_samples1 = y1.flatten()[indices1]
        z_samples1 = z1.flatten()[indices1]
        ax.scatter(x_samples1, y_samples1, z_samples1, color=color1, s=s, alpha=0.8, label='Domain 1 Samples')
        
        # Optionally plot projections onto the x-y plane
        if show_projection:
            ax.scatter(x_samples1, y_samples1, projection_z, color=color1, s=s, alpha=0.8, marker='o')
            for xs, ys, zs in zip(x_samples1, y_samples1, z_samples1):
                ax.plot([xs, xs], [ys, ys], [projection_z, zs], color=color1, linewidth=0.5, alpha=0.1)
    
    # Plot the sampled points on the second surface
    if indices2 is not None:
        x_samples2 = x2.flatten()[indices2]
        y_samples2 = y2.flatten()[indices2]
        z_samples2 = z2.flatten()[indices2]
        ax.scatter(x_samples2, y_samples2, z_samples2, color=color2, s=s, alpha=0.8, label='Domain 2 Samples')
        
        # Optionally plot projections onto the x-y plane
        if show_projection:
            ax.scatter(x_samples2, y_samples2, projection_z, color=color2, s=s, alpha=0.8, marker='o')
            for xs, ys, zs in zip(x_samples2, y_samples2, z_samples2):
                ax.plot([xs, xs], [ys, ys], [projection_z, zs], color=color2, linewidth=0.5, alpha=0.1)
    
    # Customize the plot


    # Optionally show the grid
    if show_grid:
        ax.grid(True)
        ax.xaxis._axinfo["grid"].update({"color": "lightgray", "linewidth": 0.5})
        ax.yaxis._axinfo["grid"].update({"color": "lightgray", "linewidth": 0.5})
        ax.zaxis._axinfo["grid"].update({"color": "lightgray", "linewidth": 0.5})
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    
    ax.view_init(elev=elev, azim=azim)
    
    # Add a legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Domain 1 Samples',
                              markerfacecolor=color1, markersize=5),
                       Line2D([0], [0], marker='o', color='w', label='Domain 2 Samples',
                              markerfacecolor=color2, markersize=5)]
    ax.legend(handles=legend_elements)
    
    # Save the plot if a save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    
    plt.show()



def plot_two_waddington_landscapes_3d_plotly(x1, y1, z1, indices1,
                                             x2, y2, z2, indices2,
                                             s=5, elev=30, azim=45, save_path=None):
    import plotly.graph_objs as go
    from scipy.interpolate import griddata
    
    # Create the first surface trace
    surface1 = go.Surface(
        x=x1,
        y=y1,
        z=z1,
        colorscale='Viridis',
        opacity=0.7,
        name='Landscape 1',
        showscale=False
    )
    
    # Create the second surface trace
    surface2 = go.Surface(
        x=x2,
        y=y2,
        z=z2,
        colorscale='Plasma',
        opacity=0.7,
        name='Landscape 2',
        showscale=False
    )

    x_samples1 = x1.flatten()[indices1]
    y_samples1 = y1.flatten()[indices1]
    z_samples1 = z1.flatten()[indices1]
    
    # Create scatter traces for the sample points
    scatter1 = go.Scatter3d(
        x=x_samples1,
        y=y_samples1,
        z=z_samples1,
        mode='markers',
        marker=dict(size=s, color='black'),
        name='Dataset 1 Samples'
    )

    x_samples2 = x2.flatten()[indices2]
    y_samples2 = y2.flatten()[indices2]
    z_samples2 = z2.flatten()[indices2]
    
    scatter2 = go.Scatter3d(
        x=x_samples2,
        y=y_samples2,
        z=z_samples2,
        mode='markers',
        marker=dict(size=s, color='red'),
        name='Dataset 2 Samples'
    )
    
    # Combine all traces
    data = [surface1, surface2, scatter1, scatter2]
    
    # Set up the layout
    layout = go.Layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(
                eye=dict(x=1.25, y=1.25, z=1.25),
                up=dict(x=0, y=0, z=1)
            ),
            aspectratio=dict(x=1, y=1, z=0.7),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=True,
        legend=dict(x=0, y=1)
    )
    
    # Adjust camera view
    layout.scene.camera = dict(
        eye=dict(x=np.cos(np.radians(azim)) * np.cos(np.radians(elev)),
                 y=np.sin(np.radians(azim)) * np.cos(np.radians(elev)),
                 z=np.sin(np.radians(elev)))
    )
    
    fig = go.Figure(data=data, layout=layout)
    
    # Save the plot if save_path is provided
    if save_path:
        fig.write_image(save_path)
    
    fig.show()
