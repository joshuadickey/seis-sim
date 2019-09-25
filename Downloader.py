import pandas as pd
import io
import requests
import os
from UTILS import yrmm_range, cat_to_dat, get_cat, agg_dat

# CREATE DIRECTORY STRUCTURE
cat_dir = 'data/catalogs'
dat_dir = 'data/datasets'

if not os.path.exists(dat_dir):
    os.makedirs(dat_dir)
if not os.path.exists(cat_dir):
    os.makedirs(cat_dir)


##################################################################################
#                                  DOWNLOAD DATASET                              #
##################################################################################

# DEFINE LOCATION
lat, lon = [37, -100]
sta_rad = 25
evt_rad = 25

# DEFINE TIMEFRAME
st = '2007-01'
en = '2017-01'

sta_list = sorted(pd.read_csv(io.StringIO(requests.get('http://ds.iris.edu/files/earthscope/usarray/ALL-StationList.txt').content.decode('utf-8')), sep='\t', index_col='STA').index)

for yr, mm in yrmm_range(st, en):
    cat = get_cat(yr, mm, lat, lon, sta_rad, evt_rad, cat_dir, sta_list=sta_list)
    cat_to_dat(cat, os.path.join(dat_dir, f'{yr}{mm:02d}'))


##################################################################################
#                            AGGREGATE TRAINING SET                              #
##################################################################################


# DEFINE CONUS COORDS
inc_coords = {}
inc_coords['maxlat'] = 50
inc_coords['minlat'] = 25
inc_coords['maxlon'] = -75
inc_coords['minlon'] = -125


# DEFINE HOLDOUTS FROM TRAINING (STATIONS)
exc_stalst = [ 'Y58A', 'M63A', 'W60A', 'O53A', 'Q52A', 'P56A', 'V58A', 'L42A',
               'O56A', 'M04C', 'F62A', 'K22A', '545B', 'E56A', 'TUQ', 'M60A',
               'Y60A', 'J58A', 'L48A', 'R57A', 'WHTX', 'N56A', 'N57A', 'R54A',
               'V59A', 'U49A', 'Q24A', 'SPMN', 'Y57A', 'IKP', 'K63A', 'J56A',
               '121A', 'X18A', 'S60A', 'D55A', 'M57A', 'Z58A', 'L60A', 'V56A',
               'T45B', 'SCI2', 'U32A', 'L62A', 'K60A', 'MDND', '451A', 'I59A',
               'SNCC', 'I42A' ]

# DEFINE HOLDOUTS FROM TRAINING (LOCATIONS)
exc_coords = {}
exc_coords['maxlat'] = 47
exc_coords['minlat'] = 45
exc_coords['maxlon'] = -106
exc_coords['minlon'] = -108

# BUILD TRAINING SET
# agg_dat('2007-01', '2014-01', dat_dir, dataset_name='USArray_07-13')
# agg_dat('2007-01', '2014-01', dat_dir, 'USArray_07-13_holdout', exc_stalst, exc_coords)
agg_dat('2007-01', '2014-01', dat_dir, 'USArray_07-13_conus_holdout', exc_stalst, exc_coords, inc_coords)


##################################################################################
#                               AGGREGATE VALIDATION SET                         #
##################################################################################



# BUILD VALIDATION SET
# agg_dat('2014-01', '2015-01', dat_dir, dataset_name='USArray_14')
agg_dat('2014-01', '2015-01', dat_dir, 'USArray_14_conus', inc_coords=inc_coords)

##################################################################################
#                                AGGREGATE TESTING SET                           #
##################################################################################


# BUILD TEST SETS
agg_dat('2015-01', '2017-01', dat_dir, dataset_name='USArray_15-16_conus', inc_coords=inc_coords)

