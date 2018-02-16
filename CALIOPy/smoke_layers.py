#!/usr/bin/env python
"""
A script for pulling smoke layer information out of CALIOP L2 data
"""
import argparse
import cis
import os.path
import numpy as np
from iris.cube import Cube
from cis.data_io.gridded_data import make_from_cube
from iris.analysis import MAX, MIN
from cis.aggregation.collapse_kernels import MultiKernel
import operator
from multiprocessing import Pool


MaxMin = MultiKernel('max_min', [MAX, MIN])


def get_smoke_heights(caliop_file):
    from CALIOPy.utils import layer_type, layer_subtype, Feature, AerosolFeature

    d = cis.read_data(caliop_file, "Atmospheric_Volume_Description", "Caliop_L2_cube")

    # Unpack the data from the masked array and cast back to an unsigned 16bit integer.
    # The scale and offseting turns it into a float...
    data = d.data.data.astype('u2').T

    # Get layer types and subtypes
    feature_type = layer_type(data)
    feature_subtype = layer_subtype(data)

    # Get heights array
    heights_array = np.broadcast_to(d.coord('altitude').points[:, np.newaxis], feature_subtype.shape)

    smoke_heights = np.ma.array(heights_array, mask=(feature_subtype != AerosolFeature.SMOKE) |
                                                    (feature_type != Feature.AEROSOL))

    cube = Cube(smoke_heights, dim_coords_and_dims=[(d.coord('altitude'), 0), (d.coord('time'), 1)],
                aux_coords_and_dims=[(d.coord('latitude'), 1), (d.coord('longitude'), 1)], var_name='smoke_height',
                units='m', long_name="Height of CALIOP smoke layers along track")
    return make_from_cube(cube)


def process_file(f):
    print("Processing {}...".format(f))
    c = get_smoke_heights(f)
    outfile = os.path.splitext(f)[0] + args.outfile + '.nc'
    # Collapse and then remove the remaining length-one dimension
    max_min = c.collapsed('altitude', MaxMin)
    max_min.remove_coord('altitude')
    # Add the depth
    max_min.append(operator.sub(*max_min))
    # Rename
    for c, n in zip(max_min, ['max_smoke_height', 'min_smoke_height', 'smoke_depth']):
        c.rename(n)
    max_min.save_data(outfile)
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('infiles', help="Input files", nargs='*')
    parser.add_argument('-o', '--outfile', help="Output file suffix", default='smoke_top')
    parser.add_argument('-n', '--processes', help="Number of processes to run", default=1, type=int)

    # Gets command line args by default
    args = parser.parse_args()

    with Pool(args.processes) as p:
        p.map(process_file, args.infiles)
