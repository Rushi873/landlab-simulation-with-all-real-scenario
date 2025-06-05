import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from landlab import RasterModelGrid
from landlab.components import (
    FlowAccumulator,
    FastscapeEroder,
    LinearDiffuser,
    SteepnessFinder,
)
from landlab.io.netcdf import write_raster_netcdf
from landlab.components.landslides import LandslideProbability
from landlab.components.spatial_precip import SpatialPrecipitationDistribution
from bmi_topography import Topography
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib.colors import ListedColormap
import os
from scipy.ndimage import uniform_filter

# --- DEM and grid setup (from run.py) ---
api_key = "Your-API-key"
north, south, east, west = 34.6, 33.8, 77.5, 76.3
buffer = 0.2

center_lat = (north + south) / 2
center_lon = (east + west) / 2
side = max(north - south, east - west)
half_side = side / 2

north_sq = center_lat + half_side
south_sq = center_lat - half_side
east_sq = center_lon + half_side
west_sq = center_lon - half_side

params = Topography.DEFAULT.copy()
params["north"] = north_sq
params["south"] = south_sq
params["east"] = east_sq
params["west"] = west_sq
params["dem_type"] = "SRTMGL3"
params["output_format"] = "GTiff"
params["api_key"] = api_key

region = Topography(**params)
region.fetch()
region.load()

da = region.da.isel(band=0)
z_data = da.values[::6, ::6]
nrows, ncols = z_data.shape
dx = abs(da['x'][1] - da['x'][0]) * 6
dy = abs(da['y'][1] - da['y'][0]) * 6
dxy = dx * 111000

assert np.isclose(dx, dy), "Cells must be square"
mg = RasterModelGrid((nrows, ncols), dxy)
z1 = mg.add_field("topographic__elevation", z_data.flatten().astype(float), at="node", copy=True)
mg.set_watershed_boundary_condition("topographic__elevation")
print("Grid shape (rows, cols):", mg.shape)
print("Number of nodes:", mg.number_of_nodes)
print("Number of active (core) nodes:", mg.number_of_core_nodes)
print("Cell spacing (dxy):", dxy, "meters")

s = mg.add_zeros("soil__depth", at="node", dtype=float)
mg.at_node["soil__depth"][mg.core_nodes] += 0.5
mg.add_zeros("bedrock__elevation", at="node")
mg.at_node["bedrock__elevation"][:] = mg.at_node["topographic__elevation"]
mg.at_node["topographic__elevation"][:] += mg.at_node["soil__depth"]
print("Grid successfully initialized")
mg.set_closed_boundaries_at_grid_edges(
    bottom_is_closed=False,
    left_is_closed=True,
    right_is_closed=True,
    top_is_closed=True,
)
print("Boundary conditions set")

# Parameters:
K_sp = 5e-6       # Stream power eroder coefficient (m^(1-2m) / yr)
D = 0.7          # Hillslope diffusivity (m^2/yr)
m_sp = 0.5         # Area exponent
n_sp = 1.0         # Slope exponent
uplift_rate = 0.004  # Uplift rate (m/yr)

def generate_rainfall(elevation, base_rain=1.0, orographic_factor=0.001, noise_std=0.9):
    """
    Generate rainfall array with orographic effect and noise.
    - elevation: array of node elevations (meters)
    - base_rain: base rainfall in mm/hr
    - orographic_factor: increase in rainfall per meter elevation
    - noise_std: standard deviation of Gaussian noise
    Returns: rainfall array (mm/hr) for all nodes
    """
    rain = base_rain + orographic_factor * (elevation - np.mean(elevation))
    noise = np.random.normal(0, noise_std, size=elevation.shape)
    rainfall = rain + noise
    rainfall = np.clip(rainfall, 0, None)  # No negative rainfall
    return rainfall



def plot_3d_topography_dual_views(
    mg, 
    surface="topographic__elevation", 
    title="3D Topography", 
    filename_prefix=None, 
    smooth_window=3
):
    """
    Plots and saves two smooth 3D topography plots from opposite viewpoints.
    
    Parameters:
        mg: Landlab ModelGrid object
        surface: field name to plot (default: 'topographic__elevation')
        title: plot title (used as base for both views)
        filename_prefix: if provided, files are saved as
                         {prefix}_view1.png and {prefix}_view2.png
        smooth_window: smoothing window size (default: 3)
    """
    # Prepare data
    Z = mg.at_node[surface].reshape(mg.shape)
    X = mg.x_of_node.reshape(mg.shape)
    Y = mg.y_of_node.reshape(mg.shape)
    Z_smooth = uniform_filter(Z, size=smooth_window)
    norm = (Z_smooth - Z_smooth.min()) / (Z_smooth.max() - Z_smooth.min())
    colors = cm.terrain(norm)

    # Define the two viewpoints
    view_settings = [
        {"elev": 30, "azim": -60, "suffix": "view1", "view_name": "Front-Left"},
        {"elev": 30, "azim": 120, "suffix": "view2", "view_name": "Back-Right"},
    ]

    for view in view_settings:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(
            X, Y, Z_smooth, facecolors=colors, rstride=1, cstride=1,
            linewidth=0, antialiased=True
        )
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_zlabel('Elevation (m)')
        ax.set_title(f"{title} ({view['view_name']})")
        ax.view_init(elev=view["elev"], azim=view["azim"])
        if filename_prefix:
            plt.savefig(f"{filename_prefix}_{view['suffix']}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)





# --- Human interaction field ---
# human_interaction = mg.add_zeros("human__interaction", at="node", dtype=int)
# np.random.seed(42)
# urban_fraction = 0.05
# agri_fraction = 0.15
# num_nodes = mg.number_of_nodes
# urban_nodes = np.random.choice(num_nodes, int(urban_fraction * num_nodes), replace=False)
# remaining_nodes = list(set(range(num_nodes)) - set(urban_nodes))
# agri_nodes = np.random.choice(remaining_nodes, int(agri_fraction * num_nodes), replace=False)
# mg.at_node["human__interaction"][urban_nodes] = 1
# mg.at_node["human__interaction"][agri_nodes] = 2
# mg.at_node["soil__depth"][urban_nodes] = 0.1
# mg.at_node["soil__depth"][agri_nodes] = 0.3

urban_fraction_initial = 0.01  # 1% urban initially
agri_fraction_initial = 0.05   # 5% agriculture initially
urban_growth_rate = 0.025      # 2.5% growth per timestep
agri_growth_rate = 0.015       # 1.5% growth per timestep

num_nodes = mg.number_of_nodes

# --- LandslideProbability fields (synthetic plausible values) ---
mg.add_field('topographic__slope', np.random.uniform(0.1, 0.5, mg.number_of_nodes), at='node')
mg.add_field('topographic__specific_contributing_area', np.random.uniform(10, 100, mg.number_of_nodes), at='node')
mg.add_field('soil__transmissivity', np.random.uniform(0.1, 1.0, mg.number_of_nodes), at='node')
mg.add_field('soil__saturated_hydraulic_conductivity', np.random.uniform(0.1, 1.0, mg.number_of_nodes), at='node')
mg.add_field('soil__mode_total_cohesion', np.random.uniform(100, 500, mg.number_of_nodes), at='node')
mg.add_field('soil__minimum_total_cohesion', mg.at_node['soil__mode_total_cohesion'] - 50, at='node')
mg.add_field('soil__maximum_total_cohesion', mg.at_node['soil__mode_total_cohesion'] + 50, at='node')
mg.add_field('soil__internal_friction_angle', np.random.uniform(20, 40, mg.number_of_nodes), at='node')
mg.add_field('soil__density', np.full(mg.number_of_nodes, 2000.0), at='node')
mg.add_field('soil__thickness', np.random.uniform(1, 5, mg.number_of_nodes), at='node')

# --- Landlab components ---
fa = FlowAccumulator(mg, flow_director='D8')
spe = FastscapeEroder(mg, K_sp=K_sp, m_sp=m_sp, n_sp=n_sp)
ld = LinearDiffuser(mg, linear_diffusivity=D)
sf = SteepnessFinder(mg)
ls_prob = LandslideProbability(
    mg,
    number_of_iterations=50,
    groundwater__recharge_distribution='uniform',
    groundwater__recharge_min_value=20.0,
    groundwater__recharge_max_value=120.0,
    seed=42
)
rain = SpatialPrecipitationDistribution(
    mg,
    number_of_years=1,
)


# --- Simulation parameters ---
total_years = 500000
dt = 1000
num_steps = int(total_years // dt)
i = 0

output_dir = f"Final ksp={K_sp}, D={D}, uplift={uplift_rate}"
os.makedirs(output_dir, exist_ok=True)

# Initialize lists for time series data
years = []
mean_elevation = []
max_elevation = []
mean_steepness = []
mean_drainage = []
annual_rainfall = []
storm_intensities = []

# Initialize human interaction tracking
urban_areas = []
agriculture_areas = []

mean_ls_probs = []
max_ls_probs = []

# To monitor the elevation change
elev_change_rain = []   # Rainfall effect (proxy: mean rainfall per step)
elev_change_human = []  # Human activity effect (proxy: total human nodes per step)
elev_change_spe = []    # FastscapeEroder
elev_change_ld = []     # LinearDiffuser
elev_change_uplift = [] 


for t in range(num_steps):
    current_year = (t+1)*dt
    print(f'Running timestep {t+1}/{num_steps} (Year {current_year})')
    
    # Generate rainfall and hydrology
    # Generate rainfall with orographic effects
    rainfall = generate_rainfall(mg.at_node['topographic__elevation']) # No negative rainfall
    
    # Track statistics
    annual_rainfall.append(rainfall.mean() * 24 * 365)   # Convert mm/hr to annual
    #storm_intensities.append(rain.storm_intensity)
    
    # Apply rainfall to hydrology
    mg.at_node['water__unit_flux_in'] = rainfall * 0.5  # Adjust for arid conditions
    mg.at_node['rainfall__flux'] = rainfall
    
    # Add rainfall visualization
    plt.figure(figsize=(10, 6))
    mg.imshow('rainfall__flux', cmap='Blues', vmax=10, colorbar_label="Rainfall intensity (mm/hr)")  # mm/hr
    plt.title(f'Rainfall Pattern - Year {(t+1)*dt}')
    plt.savefig(f'{output_dir}/rainfall_pattern_{t+1}.png')
    plt.close()
    
    # Run geomorphic processes
    fa.run_one_step()
    spe.run_one_step(dt)
    ld.run_one_step(dt)
    sf.calculate_steepnesses()
    
    # Apply tectonic uplift
    mg.at_node["topographic__elevation"][mg.core_nodes] += uplift_rate * dt

    # Store time series data
    years.append(current_year)
    mean_elevation.append(mg.at_node['topographic__elevation'].mean())
    max_elevation.append(mg.at_node['topographic__elevation'].max())
    mean_steepness.append(mg.at_node['channel__steepness_index'].mean())
    mean_drainage.append(mg.at_node['drainage_area'].mean())


    # # After geomorphic and rainfall updates in each timestep:
    # ls_prob.calculate_landslide_probability()
    # landslide_prob = mg.at_node['landslide__probability_of_failure']
    
    # # Save spatial plot
    # plt.figure(figsize=(10, 8))
    # mg.imshow('landslide__probability_of_failure', cmap='viridis', vmin=0, vmax=1)
    # plt.title(f'Landslide Probability - Year {current_year}')
    # plt.colorbar(label='Probability')
    # plt.savefig(f'{output_dir}/landslide_prob_{t+1}.png')
    # plt.close()
    
    # # Track statistics
    # mean_ls_prob = np.mean(landslide_prob)
    # max_ls_prob = np.max(landslide_prob)
    # mean_ls_probs.append(mean_ls_prob)
    # max_ls_probs.append(max_ls_prob)



    # Exponential growth for human land use
    urban_fraction = urban_fraction_initial * np.exp(urban_growth_rate * t)
    agri_fraction = agri_fraction_initial * np.exp(agri_growth_rate * t)
    
    # Ensure fractions don't exceed 1
    urban_fraction = min(urban_fraction, 0.2)  # cap at 20% and not more than 20% for people living
    agri_fraction = min(agri_fraction, 0.5)    # cap at 50%

    # Assign human interaction
    if not mg.has_field('human__interaction', at='node'):
        mg.add_zeros('human__interaction', at='node', dtype=int)
    
    mg.at_node['human__interaction'][:] = 0  # reset
    num_urban = int(urban_fraction * num_nodes)
    num_agri = int(agri_fraction * num_nodes)
    
    # Urban nodes
    urban_nodes = np.random.choice(num_nodes, num_urban, replace=False)
    remaining_nodes = list(set(range(num_nodes)) - set(urban_nodes))
    agri_nodes = np.random.choice(remaining_nodes, num_agri, replace=False)
    
    mg.at_node["human__interaction"][urban_nodes] = 1
    mg.at_node["human__interaction"][agri_nodes] = 2
    mg.at_node["soil__depth"][urban_nodes] = 0.1
    mg.at_node["soil__depth"][agri_nodes] = 0.3
    
    #Human interaction plot
    plt.figure(figsize=(8, 6))
    mg.imshow('human__interaction', cmap=ListedColormap(['lightgrey', 'red', 'yellow']), colorbar_label='Human Interaction Type')
    plt.title(f'Human Interaction Map - Year {current_year}')
    plt.savefig(f'{output_dir}/human_interaction_snapshot_{t+1}.png')
    plt.close()
    
    # Track areas
    urban_areas.append(np.sum(mg.at_node['human__interaction'] == 1))
    agriculture_areas.append(np.sum(mg.at_node['human__interaction'] == 2))

    # Save data to CSV
    data = {
        'node': np.arange(mg.number_of_nodes),
        'elevation': mg.at_node['topographic__elevation'],
        'drainage_area': mg.at_node['drainage_area'],
        'steepness_index': mg.at_node['channel__steepness_index'],
        'human_interaction': mg.at_node['human__interaction'],
        'rainfall_flux': rainfall
    }
    df = pd.DataFrame(data)
    df.to_csv(f'{output_dir}/timestep_{t+1}_data.csv', index=False)

    # Modified drainage area plot
    plt.figure(figsize=(10, 8))
    mg.imshow('drainage_area', cmap='viridis', colorbar_label='Log Drainage Area (m²)', norm=LogNorm())
    plt.title(f'Drainage Network Development - Year {current_year}')
    plt.savefig(f'{output_dir}/drainage_area_timestep_{t+1}.png')
    plt.close()

    # Erosion/Deposition patterns
    if t > 0:
        elevation_change = mg.at_node['topographic__elevation'] - prev_elevation
        plt.figure(figsize=(10, 8))
        mg.imshow(elevation_change, cmap='coolwarm_r', 
                 colorbar_label='Elevation Change (m)', vmin=-10, vmax=10)
        plt.title(f'Erosion/Deposition Patterns - Year {current_year}')
        plt.savefig(f'{output_dir}/erosion_deposition_timestep_{t+1}.png')
        plt.close()
    prev_elevation = mg.at_node['topographic__elevation'].copy()

    #write_raster_netcdf(f"{output_dir}/topography_{t+1:04d}.nc", mg, names=["topographic__elevation"])

    # Save as PNG
    plt.figure(figsize=(8, 6))
    mg.imshow('topographic__elevation', cmap='terrain', colorbar_label='Elevation(m)')
    plt.title(f'Topography at Timestep {t+1}')
    #plt.colorbar(label='Elevation (m)')
    plt.savefig(f"{output_dir}/topography_{t+1:04d}.png")
    plt.close()

    # 3D Plot
    plot_3d_topography_dual_views(
        mg, 
        title="3D Topography at Initial stage",
        filename_prefix=f"{output_dir}/topography3d_first.png"
    )

    # --- Track rainfall and human activity proxies ---
    elev_change_rain.append(np.mean(rainfall))  # Mean rainfall (mm/hr or mm/year)
    elev_change_human.append(np.sum(mg.at_node['human__interaction'] > 0))  # Number of human-affected nodes

    # --- Track elevation change by component ---
    # FastscapeEroder
    elev_before_spe = mg.at_node['topographic__elevation'].copy()
    spe.run_one_step(dt)
    elev_after_spe = mg.at_node['topographic__elevation'].copy()
    elev_change_spe.append(np.mean(elev_after_spe - elev_before_spe))

    # LinearDiffuser
    elev_before_ld = mg.at_node['topographic__elevation'].copy()
    ld.run_one_step(dt)
    elev_after_ld = mg.at_node['topographic__elevation'].copy()
    elev_change_ld.append(np.mean(elev_after_ld - elev_before_ld))

    # Uplift
    elev_before_uplift = mg.at_node['topographic__elevation'].copy()
    mg.at_node["topographic__elevation"][mg.core_nodes] += uplift_rate * dt
    elev_after_uplift = mg.at_node['topographic__elevation'].copy()
    elev_change_uplift.append(np.mean(elev_after_uplift - elev_before_uplift))


    
    if i==0:
        component_plots = [
    ('drainage_area', 'Blues', 'Drainage Area (m²)', 'Flow Accumulation'),
    ('topographic__elevation', 'gist_earth', 'Elevation (m)', 'Stream Power Erosion'),
    ('soil__depth', 'YlGn', 'Soil Depth (m)', 'Linear Diffusion'),
    ('channel__steepness_index', 'viridis', 'Steepness Index', 'Steepness Finder')
]

        for field, cmap, label, title in component_plots:
            plt.figure(figsize=(10, 8))
            mg.imshow(field, cmap=cmap, colorbar_label=label)
            #plt.colorbar(label=label)
            plt.title(f'{title} - Initial State')
            plt.savefig(f'{output_dir}/initial_{field}.png')
            plt.close()
            
        i +=1

# Create summary time series plots
plt.figure(figsize=(12, 8))
plt.plot(years, max_elevation, label='Max Elevation')
plt.xlabel('Years')
plt.ylabel('Elevation (m)')
plt.title('Elevation Change Over Time')
plt.legend()
plt.savefig(f'{output_dir}/elevation_trends.png')
plt.close()


plt.figure(figsize=(12, 8))
plt.plot(years, mean_steepness, label='Mean Steepness')
plt.xlabel('Years')
plt.ylabel('Metric Value')
plt.title('Steepness Evolution')
plt.legend()
plt.savefig(f'{output_dir}/channel_evolution.png')
plt.close()

plt.figure(figsize=(12, 8))
plt.plot(years, mean_drainage, label='Mean Drainage Area')
plt.xlabel('Years')
plt.ylabel('Metric Value')
plt.title('Drainage area Evolution')
plt.legend()
plt.savefig(f'{output_dir}/channel_evolution.png')
plt.close()

# 3. Component-specific plots
component_plots = [
    ('drainage_area', 'Blues', 'Drainage Area (m²)', 'Flow Accumulation'),
    ('topographic__elevation', 'gist_earth', 'Elevation (m)', 'Stream Power Erosion'),
    ('soil__depth', 'YlGn', 'Soil Depth (m)', 'Linear Diffusion'),
    ('channel__steepness_index', 'viridis', 'Steepness Index', 'Steepness Finder')
]

for field, cmap, label, title in component_plots:
    plt.figure(figsize=(10, 8))
    mg.imshow(field, cmap=cmap, colorbar_label=label)
    #plt.colorbar(label=label)
    plt.title(f'{title} - Final State')
    plt.savefig(f'{output_dir}/final_{field}.png')
    plt.close()

# 2. Human Interaction Evolution
# After the loop, plot time series
plt.figure(figsize=(12, 6))
plt.bar(years, urban_areas, width=dt * 0.8, color='red', alpha=0.7, label='Urban Areas')
plt.xlabel('Year')
plt.ylabel('Number of Urban Nodes')
plt.title('Urban Land Use Evolution in Leh Ladakh')
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/urban_land_use.png')
plt.close()

plt.figure(figsize=(12, 6))
plt.bar(years, agriculture_areas, width=dt * 0.8, color='orange', alpha=0.7, label='Agriculture Areas')
plt.xlabel('Year')
plt.ylabel('Number of Agriculture Nodes')
plt.title('Agriculture Land Use Evolution in Leh Ladakh')
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/agriculture_land_use.png')
plt.close()


# 1. Rainfall vs Timestep
plt.figure(figsize=(12, 6))
plt.plot(years, annual_rainfall, 'b-', linewidth=2)
plt.xlabel('Year')
plt.ylabel('Annual Rainfall (mm)')
plt.title('Rainfall Variability Over Time')
plt.grid(True)
plt.savefig(f'{output_dir}/rainfall_timeseries.png')
plt.close()

# 3D Plot
plot_3d_topography_dual_views(
    mg, 
    title="3D Topography at Final Timestep",
    filename_prefix=f"{output_dir}/topography3d_last.png"
)


# Elevation change other component
plt.figure(figsize=(12, 7))
plt.plot(years, np.cumsum(elev_change_spe), label='Cumulative Fluvial Incision (FastscapeEroder)', linewidth=2)
plt.plot(years, np.cumsum(elev_change_ld), label='Cumulative Hillslope Diffusion (LinearDiffuser)', linewidth=2)
plt.plot(years, np.cumsum(elev_change_uplift), label='Cumulative Uplift', linewidth=2)
plt.xlabel('Year')
plt.ylabel('Cumulative Elevation Change (m)')
plt.title('Elevation Change Over Time by Geomorphic Components')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{output_dir}/component_elevation_change.png')
plt.close()


# plt.figure(figsize=(12, 6))
# plt.plot(years, mean_ls_probs, label='Mean Landslide Probability')
# plt.plot(years, max_ls_probs, label='Max Landslide Probability')
# plt.xlabel('Year')
# plt.ylabel('Probability')
# plt.title('Landslide Probability Evolution')
# plt.legend()
# plt.savefig(f'{output_dir}/landslide_prob_evolution.png')
# plt.close()



print('Simulation complete. Data and plots saved in', output_dir)

