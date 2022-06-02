#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
# author          :Ghislain Vieilledent
# email           :ghislain.vieilledent@cirad.fr, ghislainv@gmail.com
# web             :https://ecology.ghislainv.fr
# python_version  :>=3
# license         :GPLv3
# ==============================================================================


# Third party imports
import numpy as np
from osgeo import gdal

# Local application imports
from misc import progress_bar, makeblock


# set_defor_cat_zero
def set_defor_cat_zero(input_file,
                       dist_file,
                       dist_thresh,
                       output_file="defor_cat_zero.tif",
                       blk_rows=128,
                       verbose=True):
    """Set a value of 10001 to pixels with zero deforestation risk. A
    risk of deforestation of zero is assumed when distance to forest
    edge is greater than the distance threshold.

    :param input_file: Input raster file of local deforestation
        rates. Deforestation rates are defined by integer values
        between 0 and 10000 (ten thousand). This file is typically
        obtained with function ``local_defor_rate()``.

    :param dist_file: Path to the distance to forest edge raster file.

    :param dist_thresh: The distance threshold. This distance
        threshold is used to identify pixels with zero deforestation
        risk.

    :param output_file: Output raster file. Default to
        "defor_cat_zero.tif" in the current working directory. Pixels
        with zero deforestation risk are assigned a value of 10001.

    :param blk_rows: If > 0, number of rows for computation by block.

    :param verbose: Logical. Whether to print messages or not. Default
        to ``True``.

    :return: None. A raster file identifying pixels with zero risk of
        deforestation (value 10001) will be created (see
        ``output_file``).

    """

    # ==============================================================
    # Input rasters: deforestation rates and distance to forest edge
    # ==============================================================

    # Get local deforestation rate (ldefrate) raster data
    ldefrate_ds = gdal.Open(input_file)
    ldefrate_band = ldefrate_ds.GetRasterBand(1)
    # Raster size
    xsize = ldefrate_band.XSize
    ysize = ldefrate_band.YSize

    # Get distance to forest edge (dist) raster data
    dist_ds = gdal.Open(dist_file)
    dist_band = dist_ds.GetRasterBand(1)

    # Make blocks
    blockinfo = makeblock(input_file, blk_rows=blk_rows)
    nblock = blockinfo[0]
    nblock_x = blockinfo[1]
    x = blockinfo[3]
    y = blockinfo[4]
    nx = blockinfo[5]
    ny = blockinfo[6]
    if verbose:
        print("Divide region in {} blocks".format(nblock))

    # ==================================
    # Zero category (beyond dist_thresh)
    # ==================================

    # Create cat_zero (catzero) raster
    driver = gdal.GetDriverByName("GTiff")
    catzero_ds = driver.Create(output_file, xsize, ysize, 1,
                               gdal.GDT_UInt16,
                               ["COMPRESS=LZW",
                                "PREDICTOR=2", "BIGTIFF=YES"])
    catzero_ds.SetProjection(ldefrate_ds.GetProjection())
    catzero_ds.SetGeoTransform(ldefrate_ds.GetGeoTransform())
    catzero_band = catzero_ds.GetRasterBand(1)
    catzero_band.SetNoDataValue(65535)

    # Loop on blocks of data
    for b in range(nblock):
        # Progress bar
        progress_bar(nblock, b + 1)
        # Position
        px = b % nblock_x
        py = b // nblock_x
        # Data
        catzero_data = ldefrate_band.ReadAsArray(x[px], y[py], nx[px], ny[py])
        dist_data = dist_band.ReadAsArray(x[px], y[py], nx[px], ny[py])
        # Replace nodata value in dist_data with 0
        dist_data[dist_data == 65535] = 0
        # Set zero category to value 10001
        catzero_data[dist_data >= dist_thresh] = 10001
        catzero_band.WriteArray(catzero_data, x[px], y[py])

    # Compute statistics
    if verbose:
        print("Compute statistics")
    catzero_band.FlushCache()  # Write cache data to disk
    catzero_band.ComputeStatistics(False)

    # Dereference drivers
    catzero_band = None
    del catzero_ds
    del ldefrate_ds, dist_ds

    return None


# # Test
# input_file = "outputs/ldefrate_ws7.tif"
# dist_file = "outputs/dist_edge.tif"
# dist_thresh = 390
# output_file = "outputs/defor_cat_zero.tif"
# blk_rows = 128
# verbose = True

# set_defor_cat_zero(input_file,
#                    dist_file,
#                    dist_thresh,
#                    output_file,
#                    blk_rows=128,
#                    verbose=True)

# End