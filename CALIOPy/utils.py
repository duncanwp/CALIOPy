"""
Utils for working with CALIOP data, particularly bit fields such as the Atmospheric_Volume_Description.

The flag conversion functions were essentially copied from https://github.com/vnoel/pycode/blob/master/calipso/level2.py
 It doesn't have a license but the author appears to allow free use in his readme.
 
 """
import cis
import matplotlib.pyplot as plt
import numpy as np
from cis.utils import apply_mask_to_numpy_array
from enum import IntEnum
from cis.data_io.Coord import Coord
from cis.data_io.ungridded_data import Metadata


class Feature(IntEnum):
    INVALID = 0
    CLEAN_AIR = 1
    CLOUD = 2
    AEROSOL = 3
    STRAT_FEATURE = 4
    SURFACE = 5
    SUB_SURFACE = 6
    NO_SIGNAL = 7


class AerosolFeature(IntEnum):
    UNKNOWN = 0
    CLEAN_MARINE = 1
    DUST = 2
    POLLUTED_CONTINENTAL = 3
    CLEAN_CONTINENTAL = 4
    POLLUTED_DUST = 5
    SMOKE = 6
    OTHER = 7


powers_of_2 = [2**n for n in range(8)]

FEATURE_TYPE_LABELS = ['Invalid', 'Clean air', 'Cloud', 'Aerosol', 'Stratosphere feature', 'Surface', 'Subsurface', 'No signal']

FEATURE_SUB_TYPE_LABELS = [[] for i in range(8)]
FEATURE_SUB_TYPE_LABELS[Feature.AEROSOL] = ['Not determined', 'Clean marine', 'Dust', 'Polluted continental',
                                            'Clean continental', 'Polluted dust', 'Smoke', 'Other']
FEATURE_SUB_TYPE_LABELS[Feature.CLOUD] = ["low overcast, transparent", "low overcast, opaque",
                                          "transition stratocumulus", "low, broken cumulus",
                                          "altocumulus(transparent)", "altostratus(opaque)",
                                          "cirrus(transparent)", "deep convective(opaque)"]

HORIZONTAL_AVERAGING_TYPES = ['N/A', '1/3 km', '1 km', '5 km', '20 km', '80 km']
HORIZONTAL_AVERAGING_TYPES = ['N/A', '5 km', '20 km', '80 km',
                              '5 km w/ subgrid feature detected at 1/3 km',
                              '20 km w/ subgrid feature detected at 1/3 km',
                              '80 km w/ subgrid feature detected at 1/3 km', 'N/A']
HORIZONTAL_AVERAGING_LENGTHS = np.array([-999, 5, 20, 80, 5, 20, 80, -999])


STRATOSPHERE_INDEX = 53
TOTAL_LAYERS = 399
VERTICAL_SPACING = Coord(np.concatenate([np.ones(STRATOSPHERE_INDEX) * 0.180,
                                         np.ones(TOTAL_LAYERS-STRATOSPHERE_INDEX) * 0.060]),
                         metadata=Metadata(units='km'))

unit_standardisation = {"per kilometer": 'km-1'}


def fix_units(data):
    if data.units in unit_standardisation:
        data.units = unit_standardisation[data.units]


def remove_air_pressure(data):
    """
    Remove the air_pressure coordinate from a CALIOP dataset without invoking the _post_process method so that the dataset
    retains it's 2D structure
    
    :param CommonData data: 
    :return: 
    """
    if data._coords.get_coords('air_pressure'):
        data._coords.pop(data._coords.index(data._coords.get_coord('air_pressure')))


def integrate_profile(data, spacing=VERTICAL_SPACING):
    """
    Integrage a CALIOP vertical profile using the standard (two level) vertical resolution
    
    :param UngriddedData data: 2D profile data *retaining the 2D array structure*
    :return UngriddedData: 1D integrated swath
    """
    # TODO: Ideally we would be able to just collapse the altitude dimension, but this isn't a gridded dataset...
    fix_units(data)
    extinction_surface = data[:, 0]
    integrated_data = extinction_surface
    integrated_data.data = np.ma.sum(data.data * spacing.data, axis=1)
    integrated_data.units = data.units * spacing.units

    # Take off the altitude coordinate so that I can combine it with the AOD variable
    # integrated_data._coords.pop(-1)

    return integrated_data


def _find_aerosol(cad_score, confidence):
    """
    Calculate an aerosol mask based on the given CAD score array.

    :param int confidence: The confidence above which to count a retrieval as an aerosol or not (should be +ve)
    :param ndarray cad_score:
    :return bool ndarray : aerosol mask, True values are where there *are* aerosols with given confidence
    """
    return -cad_score > confidence  # Note the negative sign (aerosols are stored as negative numbers)


def _find_clouds(cad_score, confidence):
    """
    Calculate a cloud mask based on the given CAD score array
    :param int confidence: The confidence above which to count a retrieval as an aerosol or not (should be +ve)
    :param ndarray cad_score:
    :return bool ndarray : cloud mask, True values are where there *are* clouds with given confidence
    """
    return cad_score > confidence  # Clouds are stored as +ve numbers


def find_good_aerosol_columns(cad_score, cad_confidence):
    """
    Calculate a boolean array where True values represent good CALIOP aerosol profile columns

    :param UngriddedData cad_score:
    :param int cad_confidence:
    :return boool ndarray: True represents a column with some (confident) aerosol, no clouds, and
    no 'special' cad scores in the column
    """
    clouds = _find_clouds(cad_score.data, 0)
    aerosols = _find_aerosol(cad_score.data, cad_confidence)
    bad_cads = np.abs(cad_score.data) > 100  # Special CAD flags

    # There must be some (confident) aerosol, no clouds, and no 'special' cad scores in the column
    return aerosols.any(axis=1) & ~clouds.any(axis=1) & ~bad_cads.any(axis=1)


def _find_converged_extinction_points(extinction_qc):
    return (extinction_qc < 2)  # Constrained retrieval or unconstrained with unchanged lidar ratio


def find_good_extinction_columns(extinction_qc):
    mask = _find_converged_extinction_points(extinction_qc.data)
    return mask.all(axis=1)


def mask_data(data, cad_score, extinction_qc, cad_confidence=20):
    """
    Default CAD confidence of 80 from doi:10.1002/2013JD019527

    The extinction QC values are::    
        Bit 	Value 	Interpretation
        1 	    0 	    unconstrained retrieval; initial lidar ratio unchanged during solution process
        1 	    1 	    constrained retrieval
        2 	    2 	    Initial lidar ratio reduced to prevent divergence of extinction solution
        3 	    4 	    Initial lidar ratio increased to reduce the number of negative extinction
                        coefficients in the derived solution
        4   	8 	    Calculated backscatter coefficient exceeds the maximum allowable value
        5   	16 	    Layer being analyzed has been identified by the feature finder as being totally
                        attenuating (i.e., opaque)
        6 	    32 	    Estimated optical depth error exceeds the maximum allowable value
        7 	    64 	    Solution converges, but with an unacceptably large number of negative values
        8 	    128 	Retrieval terminated at maximum iterations
        9 	    256 	No solution possible within allowable lidar ratio bounds
        16 	    32768 	Fill value or no solution attempted

    :param CommonDataList data: The data to be masked  
    :param cad_score: 
    :param extinction_qc: 
    :param cad_confidence: 
    :return: 
    """
    from cis.data_io.ungridded_data import UngriddedDataList

    column_mask = find_good_aerosol_columns(cad_score, cad_confidence) & find_good_extinction_columns(extinction_qc)

    # Now do the full profiles. Pull out the valid parts of the aerosol and extinction masks
    good_extinctions = _find_converged_extinction_points(extinction_qc.data[column_mask])
    aerosols = _find_aerosol(cad_score.data[column_mask], cad_confidence)

    # First create the aerosol masked data (which is a shared mask)
    compressed_data = UngriddedDataList()
    for d in data:
        if d.data.shape[0] != column_mask.shape[0]:
            # This only outputs a warning in numpy currently
            raise ValueError("The data shape doesn't match the mask shape")
        c = d[column_mask]
        # If the data has (an extended) second dimension
        if len(c.shape) > 1 and c.shape[1] > 1:
            # Apply the aerosol (2D) mask
            c.data = apply_mask_to_numpy_array(c.data, ~aerosols)
            if c.name().startswith('Extinction'):
                # Apply the good extinction (2D) mask
                c.data = apply_mask_to_numpy_array(c.data, ~good_extinctions)
        compressed_data.append(c)
        print("Valid {} points: {}".format(c.name(), c.count()))

    return compressed_data


def layer_type(flags):
    """
    Returns the layer type from the feature classification flag

    0 = invalid (bad or missing data)
    1 = "clear air"
    2 = cloud
    3 = aerosol
    4 = stratospheric feature
    5 = surface
    6 = subsurface
    7 = no signal (totally attenuated)

    """
    # type flag : bits 1 to 3
    return flags & 7


def layer_subtype(flags):
    """
    Returs the layer subtype, as identified from the feature
    classification flag

    for clouds (feature type == layer_type == 2)
    0 = low overcast, transparent
    1 = low overcast, opaque
    2 = transition stratocumulus
    3 = low, broken cumulus
    4 = altocumulus (transparent) 
    5 = altostratus (opaque)
    6 = cirrus (transparent)
    7 = deep convective (opaque)
    """
    # subtype flag : bits 10 to 12
    return (flags & 3584) >> 9


def layer_subtype_qa(flags):
    """
    Returns the layer subtype quality flag, as identified from the feature
    classification flag
    """
    # subtype qa flag : bit 13
    return (flags & 4096) >> 12


def horizontal_average(flags):
    """
    
    :param flags: 
    :return: 
    """
    # horizontal averaging flag : bits 14-16
    return (flags & 57344) >> 13


def layer_type_qa(flags):
    """
    Returns the quality flag for the layer type, as identified from the
    feature classification flag
    """
    return (flags & 24) >> 3


def phase(flags):
    """
    Returns the layer thermodynamical phase, as identified from the
    feature classification flag

    0 = unknown / not determined 1 = randomly oriented ice
    2 = water
    3 = horizontally oriented ice
    """
    # 96 = 0b1100000, bits 6 to 7
    return (flags & 96) >> 5


def phase_qa(flags):
    """
    Returns the quality flag for the layer thermodynamical phase,
    as identified from the feature classification flag

    0 = none
    1 = low
    2 = medium 3 = high
    """
    return (flags & 384) >> 7


def create_horizontal_average(avd):
    """
    Given the Atmospheric Volume Description return a valid UngriddedData object describing the horizontal averaging
    :param UngriddedData avd:
    :return UngriddedData:
    """
    # Ensure the AVD is of the right type for the bit-wise operation
    avd.data = avd.data.astype('u2')
    horizontal_flags = horizontal_average(avd.data)
    horizontal_values = np.ma.masked_less(HORIZONTAL_AVERAGING_LENGTHS[horizontal_flags], 0.0)
    horizontal_av = avd.copy(data=horizontal_values)
    horizontal_av.var_name = 'horizontal_averaging'
    horizontal_av.long_name = 'The horizontal length-scale the point value is an average from'
    horizontal_av.units = 'km'
    return horizontal_av


if __name__ == '__main__':

    # Read the data in without the pressure coordinate (which has missing values and leads
    #  to the data being flattened)
    d = cis.read_data("CAL_LID_L2_05kmAPro-Prov-V3-01.2009-12-31T23-36-08ZN.hdf",
                      "Atmospheric_Volume_Description", "Caliop_L2_NO_PRESSURE")

    # Unpack the data from the masked array and cast back to an unsigned 16bit integer.
    # The scale and offseting turns it into a float...
    data = d.data.data.astype('u2')

    feature_type = layer_type(data)
    feature_type_QA = layer_type_qa(data)
    ice_water_phase = phase(data)
    ice_water_phase_QA = phase_qa(data)
    feature_subtype = layer_subtype(data)
    feature_subtype_QA = layer_subtype_qa(data)
    horizontal_averaging = horizontal_average(data)

    plt.figure(figsize=(12, 6))

    # Set the feauture type to plot
    feature_to_plot = Feature.AEROSOL
    number_of_features = len(FEATURE_SUB_TYPE_LABELS[feature_to_plot])

    aerosol_types = np.ma.array(feature_subtype[::-1], mask=feature_type[::-1] != feature_to_plot)

    x = plt.pcolormesh(d.coord('latitude').points, d.coord('altitude').points, aerosol_types,
                       cmap='Accent', vmin=0, vmax=number_of_features)

    cb = plt.colorbar(x)
    cb.set_ticks([np.arange(number_of_features)+0.5])
    cb.set_ticklabels(FEATURE_SUB_TYPE_LABELS[feature_to_plot])
    plt.title('Aerosol subtype')
    plt.show()

    aerosol_averaging = np.ma.array(horizontal_averaging[::-1], mask=feature_type[::-1] != feature_to_plot)

    x = plt.pcolormesh(d.coord('latitude').points, d.coord('altitude').points, aerosol_averaging,
                       cmap='Accent', vmin=0, vmax=len(HORIZONTAL_AVERAGING_TYPES))

    cb = plt.colorbar(x)
    cb.set_ticks([np.arange(len(HORIZONTAL_AVERAGING_TYPES))+0.5])
    cb.set_ticklabels(HORIZONTAL_AVERAGING_TYPES)
    plt.title('Aerosol horizontal averaging')
    plt.show()
