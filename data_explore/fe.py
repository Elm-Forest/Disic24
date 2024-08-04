import warnings

import pandas as pd
from openfe import OpenFE
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

warnings.filterwarnings(action='ignore')
main_key = ['isic_id', 'target']
numeric_columns = [
    'age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B',
    'tbp_lv_Bext', 'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L',
    'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean',
    'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm',
    'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM', 'tbp_lv_nevi_confidence', 'tbp_lv_norm_border',
    'tbp_lv_norm_color', 'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL',
    'tbp_lv_stdLExt', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z'
]
categorical_columns = [
    'sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location',
    'tbp_lv_location_simple', 'attribution'
]

csv_file = 'K:/dataset/Disic2024challenge/train-metadata.csv'

ofe = OpenFE()

df = pd.read_csv(csv_file, low_memory=False)
df = df[numeric_columns + categorical_columns + main_key]
for col in categorical_columns:
    df[col] = df[col].fillna('Unknown')

if __name__ == '__main__':
    train_x, test_x, train_y, test_y = train_test_split(df[numeric_columns + categorical_columns], df['target'],
                                                        test_size=0.2, random_state=1)

    features = ofe.fit(data=train_x, label=train_y, n_jobs=4, verbose=0)  # generate new features
    # train_x, test_x = transform(train_x, test_x, features,
    #                             n_jobs=4, verbose=0)  # transform the train and test data according to generated features.
    # print(train_x)
fetch_california_housing