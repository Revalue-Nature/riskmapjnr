#===============================================================
# Deforestation Risk Assessment
# VERRA JNR Riskmap Methodology
#===============================================================

#--------------------------------------
# Import relevant packages and library
#--------------------------------------
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import contextily as ctx
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import geopandas as gpd
import matplotlib

from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Patch
from matplotlib import colors, patches
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tabulate import tabulate
from osgeo import gdal
from riskmapjnr.dist_edge_threshold import dist_edge_threshold
from riskmapjnr.makemap import makemap

#-----------------------------
# Specify input variables
#-----------------------------
# 1. Shapefile and rasters
# shapefile file directory
jurisdiction_directory = "./riskmapjnr/data/Sofala_Jurisdiction_32737.shp"
# forest cover raster file directory
ForestCoverChange_directory = "./riskmapjnr/data/sofala_fcc123.tif"
# Output folder
output_directory = "./riskmapjnr/output"
# output directory
out_dir = os.path.expanduser(output_directory+'/jnr_riskmap_16_21')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# 2. Configurations
# distance range
# step = interval of distance from deforestation/forest edge)
distance_range = np.arange(0, 3000, step=1)
# sets of window size
# Based on Verra guidance the optimal window size usually between 600 x 600m (20 x 20 pixels, 1 pixel size = 30 m) and 30,000 x 30,000 m (1000 x 1000 pixels)
window_size_multiProcess = np.arange(23, 30, step=2)
# sets of slicing algorithm
slice_algorithm_multiProcess = ["Equal Interval", "Equal Area", "Natural Breaks"]
# grid size for validation process
grid_size = 80
# total number years of calibration and validation period. Example [3,3] = 3 years calibration period and 3 years validation period
years_period_interval = [3,3]

#------------------------------------
# Function
#------------------------------------
# 1. Function for generating maps forest cover change
def generate_fcc_maps(raster_file_path, bounds_file_path, output_file_path):
    # import fcc raster
    raster = rasterio.open(raster_file_path)
    bounds_layer = gpd.read_file(bounds_file_path)

    # import geo
    colors_fcc = ['#e16e70','#BD0026','#0A7762']
    cmap_fcc = colors.ListedColormap(colors_fcc)
    norm_fcc = colors.BoundaryNorm([1, 2.5, 3.5], ncolors=3)

    legend_elements_fcc = [Line2D([0], [0], color='black', lw=1, label='Jurisdiction Boundary'),
                           Patch(facecolor='#e16e70', edgecolor= None, label='Deforestation 2016-2018'),
                           Patch(facecolor='#BD0026', edgecolor= None, label = 'Deforestation 2019-2021'),
                           Patch(facecolor='#0A7762', edgecolor= None, label = 'Remnant Forest in 2021')]

    fig_fcc, ax_fcc = plt.subplots(figsize=(20,20))
    x, y, arrow_length = 0.08, 0.1, 0.05
    ax_fcc.annotate('N', color='white', xy=(x, y), xytext=(x, y-arrow_length),                                               
                         arrowprops=dict(facecolor='white', width=5, headwidth=15),                      
                         ha='center', va='center', fontsize=20,                      
                         xycoords=ax_fcc.transAxes)
    
    # legend
    legend_roi_fcc = fig_fcc.legend(handles=legend_elements_fcc,facecolor="white", prop={'size': 12}, loc='lower right', title = "Legends")
    plt.setp(legend_roi_fcc.get_title(),fontsize='14')
    # title
    ax_fcc.set_title('Forest Cover Change 2017-2022 ',family='sans-serif', fontweight = 'semibold', fontsize=20)
    # scalebar
    ax_fcc.add_artist(ScaleBar(1, box_alpha=0.6, location = 'lower left'))
    plt.yticks(rotation='vertical')
    
    # layers
    ax_fcc = bounds_layer.plot(figsize=(20, 20), color = None, edgecolor = 'Black', linewidth = 1, facecolor="none", ax = ax_fcc)
    ax_fcc = ctx.add_basemap(source=ctx.providers.CartoDB.Positron, crs=bounds_layer.crs.to_string(), ax = ax_fcc )
    ax_fcc = rasterio.plot.show(raster, ax=ax_fcc, cmap = cmap_fcc, norm = norm_fcc)
    fig_fcc.savefig(output_file_path, bbox_inches ="tight")
    plt.close(fig_fcc)

# 2. Generate riskmap layer
def generate_defrisk_map(riskmap_file_path, bounds_file_path, output_file_path):
    # import riskmap raster
    riskmap_raster = rasterio.open(riskmap_file_path)
    # import shapefile boundary
    shp_layer = gpd.read_file(bounds_file_path)

    # get unique data
    list_value_riskmap = np.unique(riskmap_raster.read(1)).tolist()
    list_value_riskmap.remove(255)
    
    # resample the length of color based on unique raster value
    cmap = plt.get_cmap('RdYlGn_r').resampled(len(list_value_riskmap))
    # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be green revalue them
    cmaplist[0] = np.array([10/256, 119/256, 98/256, 1])
    
    # create the new color map
    cmap_riskmap = colors.LinearSegmentedColormap.from_list(    
        'Custom cmap', cmaplist, cmap.N)
    # define the bins and normalize
    bounds = np.linspace(0, max(list_value_riskmap), len(list_value_riskmap))
    norm_riskmap = colors.BoundaryNorm(bounds, cmap.N)

    fig_riskmap, ax_risk_map = plt.subplots(figsize=(20,20))
    x, y, arrow_length = 0.08, 0.1, 0.05
    ax_risk_map.annotate('N', color='white', xy=(x, y), xytext=(x, y-arrow_length),                                               
                         arrowprops=dict(facecolor='white', width=5, headwidth=15),                      
                         ha='center', va='center', fontsize=20,                      
                         xycoords=ax_risk_map.transAxes)
    
    # title
    ax_risk_map.set_title('Deforestation Risk at the Remnant Forest 2021',family='sans-serif', fontweight = 'semibold', fontsize=20)
    # scalebar
    ax_risk_map.add_artist(ScaleBar(1, box_alpha=0.6, location = 'lower left'))
    plt.yticks(rotation='vertical')\
    
    # layers
    ax_risk_map = shp_layer.plot(figsize=(20, 20), color = None, edgecolor = 'Black', linewidth = 1, facecolor="none", ax = ax_risk_map)
    basemap = ctx.add_basemap(source=ctx.providers.CartoDB.Positron, crs=shp_layer.crs.to_string(), ax = ax_risk_map )
    risk_map = rasterio.plot.show(riskmap_raster, ax=ax_risk_map, cmap = cmap_riskmap, norm = norm_riskmap)
    
    # add colorbar
    divider = make_axes_locatable(ax_risk_map)
    cax_riskmap = divider.append_axes("bottom", size="1%", pad=0.1)
    cbar = fig_riskmap.colorbar(matplotlib.cm.ScalarMappable(norm=norm_riskmap,cmap=cmap_riskmap), 
                                cax=cax_riskmap, ax=ax_risk_map, orientation ='horizontal', 
                                ticks=[0, 1, max(list_value_riskmap)],                                                                
                                label="Deforestation Risk Level in 2021")
    cbar.ax.set_xticklabels([r'Insignificant' "\n" r'Risk', r'Low' "\n" r'Risk', r'High' "\n" r'Risk'])  # horizontal colorbar
    fig_riskmap.savefig(output_file_path, bbox_inches ="tight")

#------------------------------------
# Process - Generate best riskmap
#------------------------------------
# 1. Distance to forest edge estimation during the Historical Reference Period of the FREL
distance_to_forestedge_file = f"{out_dir}/prec_def_dist.png"
threshold_distance = dist_edge_threshold(
    fcc_file = ForestCoverChange_directory,
    defor_values = [1,2],
    dist_file = f"{out_dir}/dist_edge.tif",
    dist_bins = distance_range,
    tab_file_dist = f"{out_dir}/tab_dist.csv",
    fig_file_dist = distance_to_forestedge_file,
    blk_rows = 128, verbose = False
)
dist_threshold = threshold_distance['dist_thresh']
print(f"The distance threshold during the historical reference period is {dist_threshold} m")

# 2. Multiprocessing
print(f"total maps generated: {len(slice_algorithm_multiProcess) * len(window_size_multiProcess)}")
print(f"Generate the maps with window size configuration in (pixel counts): {window_size_multiProcess}")
print(f"Generate the maps with slicing algorihtm: {slice_algorithm_multiProcess}")

# set up cpu
ncpu = mp.cpu_count() - 1
print(f"Total number of CPUs: {ncpu}.")

# multiprocessing 
results_makemap = makemap(
    fcc_file = ForestCoverChange_directory,
    time_interval = years_period_interval,
    output_dir = out_dir,
    clean = False,
    dist_bins = distance_range,
    win_sizes = window_size_multiProcess,
    ncat = 30,
    parallel = True,
    ncpu = ncpu,
    methods = slice_algorithm_multiProcess,
    csize = grid_size,
    figsize = (6.4, 4.8),
    dpi = 100,
    blk_rows = 1200,
    no_quantity_error = True,
    verbose = True
)

# 3. Identify the best model with window size and slicing algorithm
# the best window size and slicing algorithm is the config that produces the lowest wRMSE
model_comparison_df = pd.read_csv(f"{out_dir}/modcomp/mod_comp.csv")
model_comparison_df = model_comparison_df.sort_values(by=['wRMSE', 'ws'], ascending=True)
ws_hat = model_comparison_df.iloc[0]['ws']
m_hat = model_comparison_df.iloc[0]['m']

# export the maps
generate_fcc_maps(ForestCoverChange_directory, jurisdiction_directory, f"{out_dir}/FCC_maps.png")
generate_defrisk_map(f"{out_dir}/endval/riskmap_ws{ws_hat}_{m_hat}_ev.tif", jurisdiction_directory, f"{out_dir}/risksmaps.png")