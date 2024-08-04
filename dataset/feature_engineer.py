import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


def feature_engineering(df, train_copy):
    # New features to try...
    df["lesion_size_ratio"] = df["tbp_lv_minorAxisMM"] / df["clin_size_long_diam_mm"]
    df["lesion_shape_index"] = df["tbp_lv_areaMM2"] / (df["tbp_lv_perimeterMM"] ** 2)
    df["hue_contrast"] = (df["tbp_lv_H"] - df["tbp_lv_Hext"]).abs()
    df["luminance_contrast"] = (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs()
    df["lesion_color_difference"] = np.sqrt(
        df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2)
    df["border_complexity"] = df["tbp_lv_norm_border"] + df["tbp_lv_symm_2axis"]
    df["color_uniformity"] = df["tbp_lv_color_std_mean"] / df["tbp_lv_radial_color_std_max"]
    df["3d_position_distance"] = np.sqrt(df["tbp_lv_x"] ** 2 + df["tbp_lv_y"] ** 2 + df["tbp_lv_z"] ** 2)
    df["perimeter_to_area_ratio"] = df["tbp_lv_perimeterMM"] / df["tbp_lv_areaMM2"]
    df["lesion_visibility_score"] = df["tbp_lv_deltaLBnorm"] + df["tbp_lv_norm_color"]
    #     df["combined_anatomical_site"] = df["anatom_site_general"] + "_" + df["tbp_lv_location"]

    df["symmetry_border_consistency"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_norm_border"]
    df["color_consistency"] = df["tbp_lv_stdL"] / df["tbp_lv_Lext"]

    df["size_age_interaction"] = df["clin_size_long_diam_mm"] * df["age_approx"]
    df["hue_color_std_interaction"] = df["tbp_lv_H"] * df["tbp_lv_color_std_mean"]
    df["lesion_severity_index"] = (df["tbp_lv_norm_border"] + df["tbp_lv_norm_color"] + df["tbp_lv_eccentricity"]) / 3
    df["shape_complexity_index"] = df["border_complexity"] + df["lesion_shape_index"]
    df["color_contrast_index"] = df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"] + df[
        "tbp_lv_deltaLBnorm"]
    df["log_lesion_area"] = np.log(df["tbp_lv_areaMM2"] + 1)
    df["normalized_lesion_size"] = df["clin_size_long_diam_mm"] / df["age_approx"]
    df["mean_hue_difference"] = (df["tbp_lv_H"] + df["tbp_lv_Hext"]) / 2
    df["std_dev_contrast"] = np.sqrt(
        (df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2) / 3)
    df["color_shape_composite_index"] = (df["tbp_lv_color_std_mean"] + df["tbp_lv_area_perim_ratio"] + df[
        "tbp_lv_symm_2axis"]) / 3
    df["3d_lesion_orientation"] = np.arctan2(train_copy["tbp_lv_y"], train_copy["tbp_lv_x"])
    df["overall_color_difference"] = (df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"]) / 3
    df["symmetry_perimeter_interaction"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_perimeterMM"]
    df["comprehensive_lesion_index"] = (df["tbp_lv_area_perim_ratio"] + df["tbp_lv_eccentricity"] + df[
        "tbp_lv_norm_color"] + df["tbp_lv_symm_2axis"]) / 4

    # 新しい特徴量

    # 1. 複合的な形状指標
    df['shape_complexity_ratio'] = df['tbp_lv_norm_border'] / df['lesion_shape_index']

    # 2. 色彩の変動性
    df['color_variability'] = df['tbp_lv_color_std_mean'] / df['tbp_lv_stdL']

    # 3. 境界の非対称性
    df['border_asymmetry'] = df['tbp_lv_norm_border'] * (1 - df['tbp_lv_symm_2axis'])

    # 4. 3D位置と大きさの関係
    df['3d_size_ratio'] = df['3d_position_distance'] / df['clin_size_long_diam_mm']

    # 5. 年齢と病変の特徴の相互作用
    df['age_lesion_interaction'] = df['age_approx'] * df['lesion_severity_index']

    # 6. 色彩コントラストの複合指標
    df['color_contrast_complexity'] = df['color_contrast_index'] * df['tbp_lv_radial_color_std_max']

    # 7. 形状と色彩の複合指標
    df['shape_color_composite'] = df['shape_complexity_index'] * df['color_uniformity']

    # 8. 病変の相対的な大きさ
    df['relative_lesion_size'] = df['clin_size_long_diam_mm'] / df['tbp_lv_minorAxisMM']

    # 9. 境界の複雑さと色彩の変動性の相互作用
    df['border_color_interaction'] = df['border_complexity'] * df['color_variability']

    # 10. 3D位置の極座標表現
    df['3d_radial_distance'] = np.sqrt(df['tbp_lv_x'] ** 2 + df['tbp_lv_y'] ** 2 + df['tbp_lv_z'] ** 2)
    df['3d_polar_angle'] = np.arccos(df['tbp_lv_z'] / df['3d_radial_distance'])
    df['3d_azimuthal_angle'] = np.arctan2(df['tbp_lv_y'], df['tbp_lv_x'])

    # 11. 病変の形状の複雑さと大きさの比
    df['shape_size_ratio'] = df['shape_complexity_index'] / df['clin_size_long_diam_mm']

    # 12. 色彩の非一様性と境界の複雑さの複合指標
    df['color_border_complexity'] = df['color_uniformity'] * df['border_complexity']

    # 13. 病変の可視性と大きさの相互作用
    df['visibility_size_interaction'] = df['lesion_visibility_score'] * np.log(df['clin_size_long_diam_mm'])

    # 14. 年齢調整済みの病変の特徴
    df['age_adjusted_lesion_index'] = df['comprehensive_lesion_index'] / np.log(df['age_approx'])

    # 15. 色彩コントラストの非線形変換
    df['nonlinear_color_contrast'] = np.tanh(df['color_contrast_index'])

    # 16. 病変の形状と位置の複合指標
    df['shape_location_index'] = df['lesion_shape_index'] * df['3d_position_distance']

    # 17. 境界の複雑さと非対称性の比率
    df['border_complexity_asymmetry_ratio'] = df['border_complexity'] / (df['tbp_lv_symm_2axis'] + 1e-5)

    # 18. 色彩の変動性と病変の大きさの相互作用
    df['color_variability_size_interaction'] = df['color_variability'] * np.log(df['tbp_lv_areaMM2'])

    # 19. 3D位置と病変の特徴の複合指標
    df['3d_lesion_composite'] = df['3d_position_distance'] * df['comprehensive_lesion_index']

    # 20. 病変の形状と色彩の非線形複合指標
    df['nonlinear_shape_color_composite'] = np.tanh(df['shape_color_composite'])

    new_num_cols = [
        "lesion_size_ratio", "lesion_shape_index", "hue_contrast",
        "luminance_contrast", "lesion_color_difference", "border_complexity",
        "color_uniformity", "3d_position_distance", "perimeter_to_area_ratio",
        "lesion_visibility_score", "symmetry_border_consistency", "color_consistency",

        "size_age_interaction", "hue_color_std_interaction", "lesion_severity_index",
        "shape_complexity_index", "color_contrast_index", "log_lesion_area",
        "normalized_lesion_size", "mean_hue_difference", "std_dev_contrast",
        "color_shape_composite_index", "3d_lesion_orientation", "overall_color_difference",
        "symmetry_perimeter_interaction", "comprehensive_lesion_index", "shape_complexity_ratio",
        "color_variability", "border_asymmetry", "3d_size_ratio", "age_lesion_interaction",
        "color_contrast_complexity", "shape_color_composite", "relative_lesion_size",
        "border_color_interaction", "3d_radial_distance", "3d_polar_angle", "3d_azimuthal_angle",
        "shape_size_ratio", "color_border_complexity", "visibility_size_interaction", "age_adjusted_lesion_index",
        "nonlinear_color_contrast", "shape_location_index", "border_complexity_asymmetry_ratio",
        "color_variability_size_interaction",
        "3d_lesion_composite", "nonlinear_shape_color_composite",
    ]
    #     new_cat_cols = ["combined_anatomical_site"]
    return df, new_num_cols


def get_meta_feature(df_train, df_test):
    train, test = df_train, df_test
    # 特征工程
    train, _ = feature_engineering(train.copy(), train.copy())
    test, _ = feature_engineering(test.copy(), train.copy())

    # 处理分类特征和二元特征
    cat_features = ["anatom_site_general", "tbp_lv_location_simple", "tbp_lv_location"]
    binary_column = "sex"
    numerical_columns = ['age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext',
                         'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext',
                         'tbp_lv_L', 'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio',
                         'tbp_lv_color_std_mean',
                         'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity',
                         'tbp_lv_minorAxisMM', 'tbp_lv_nevi_confidence', 'tbp_lv_norm_border', 'tbp_lv_norm_color',
                         'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL', 'tbp_lv_stdLExt',
                         'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z',
                         'lesion_size_ratio', 'lesion_shape_index', 'hue_contrast', 'lesion_color_difference',
                         'color_uniformity',
                         'perimeter_to_area_ratio', 'lesion_visibility_score', 'symmetry_border_consistency',
                         'color_consistency', 'size_age_interaction', 'lesion_severity_index', 'color_contrast_index',
                         'log_lesion_area', 'normalized_lesion_size', 'mean_hue_difference', '3d_lesion_orientation',
                         'overall_color_difference',
                         'symmetry_perimeter_interaction', 'comprehensive_lesion_index', 'shape_complexity_ratio',
                         'color_variability', 'border_asymmetry', '3d_size_ratio', 'age_lesion_interaction',
                         'color_contrast_complexity', 'shape_color_composite', 'relative_lesion_size',
                         'border_color_interaction', '3d_polar_angle', 'shape_size_ratio',
                         'visibility_size_interaction',
                         'age_adjusted_lesion_index', 'nonlinear_color_contrast', 'shape_location_index',
                         'border_complexity_asymmetry_ratio',
                         'color_variability_size_interaction', '3d_lesion_composite', 'nonlinear_shape_color_composite'
                         ]

    # 处理无限值和缺失值
    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.replace([np.inf, -np.inf], np.nan, inplace=True)

    train[numerical_columns] = train[numerical_columns].fillna(train[numerical_columns].median())
    test[numerical_columns] = test[numerical_columns].fillna(test[numerical_columns].median())

    # 生成和处理 n_images 特征
    train['n_images'] = train.patient_id.map(train.groupby(['patient_id']).isic_id.count())
    test['n_images'] = test.patient_id.map(test.groupby(['patient_id']).isic_id.count())
    train.loc[train['patient_id'] == -1, 'n_images'] = 1
    train['n_images'] = np.log1p(train['n_images'].values)
    test['n_images'] = np.log1p(test['n_images'].values)
    numerical_columns += ["n_images"]

    # 标准化数值特征
    scaler = MinMaxScaler()
    train[numerical_columns] = scaler.fit_transform(train[numerical_columns])
    test[numerical_columns] = scaler.transform(test[numerical_columns])

    # 处理分类和二元特征的缺失值
    simple_imputer = SimpleImputer(strategy='most_frequent')
    train[cat_features + [binary_column]] = simple_imputer.fit_transform(train[cat_features + [binary_column]])
    test[cat_features + [binary_column]] = simple_imputer.transform(test[cat_features + [binary_column]])
    train[binary_column] = train[binary_column].map({'male': 0, 'female': 1})
    test[binary_column] = test[binary_column].map({'male': 0, 'female': 1})

    # 对分类特征进行独热编码
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    train_encoded_df = pd.DataFrame(onehot_encoder.fit_transform(train[cat_features]))
    test_encoded_df = pd.DataFrame(onehot_encoder.transform(test[cat_features]))

    train_encoded_df.columns = onehot_encoder.get_feature_names_out(cat_features)
    test_encoded_df.columns = onehot_encoder.get_feature_names_out(cat_features)

    # 更新训练集和测试集
    train = train.drop(columns=cat_features).reset_index(drop=True)
    train = pd.concat([train, train_encoded_df], axis=1)
    test = test.drop(columns=cat_features).reset_index(drop=True)
    test = pd.concat([test, test_encoded_df], axis=1)

    cat_features = list(onehot_encoder.get_feature_names_out(cat_features))
    cat_features += ["sex"]

    meta_features = numerical_columns + cat_features
    return meta_features, train, test


if __name__ == '__main__':
    HOME = 'K:/dataset/Disic2024challenge/'
    train = pd.read_csv(HOME + "train-metadata.csv")
    test = pd.read_csv(HOME + "test-metadata.csv")
    get_meta_feature(train, test)
