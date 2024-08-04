import h5py

# 打开HDF5文件
file_path = 'K:/dataset/Disic2024challenge/train-image.hdf5'
with h5py.File(file_path, 'r') as hdf:

    # 列出文件中的所有组
    def print_structure(name, obj):
        print(name)

    print("文件中的组和数据集列表:")
    hdf.visititems(print_structure)

    # 示例: 读取某个数据集
    dataset_name = 'example_dataset'  # 替换为实际的数据集名称
    if dataset_name in hdf:
        dataset = hdf[dataset_name]
        print(f"\n数据集 '{dataset_name}' 的内容预览:")
        print(dataset[:10])  # 预览前10条记录

