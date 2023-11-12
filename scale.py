import os
import nrrd
from scipy.ndimage import zoom
import SimpleITK as sitk
from multiprocessing import Pool

# 定义函数用于检查和缩放NRRD文件
def check_and_scale_nrrd(file_path, output_path):
    # 读取原始的 NRRD 文件
    original_nrrd = sitk.ReadImage(file_path)

    # 获取原始的 spacing
    original_spacing = original_nrrd.GetSpacing()

    # 获取原始数据的 size
    original_size = original_nrrd.GetSize()

    # 检查数据尺寸，确定是否需要放大
    if original_size[0] == 512 and original_size[1] == 512:
        # 计算新的 spacing（只对高度和宽度减半，深度保持不变）
        new_spacing = (original_spacing[0] / 2, original_spacing[1] / 2, original_spacing[2])

        # 计算新的 size
        new_size = (original_size[0] * 2, original_size[1] * 2, original_size[2])

        # 使用最近邻插值方法进行 upsampling
        upsampled_nrrd = sitk.Resample(original_nrrd, new_size, sitk.Transform(), 
                                       sitk.sitkNearestNeighbor, original_nrrd.GetOrigin(), 
                                       new_spacing, original_nrrd.GetDirection(), 0, 
                                       original_nrrd.GetPixelID())
    else:
        # 尺寸正确，无需变化
        upsampled_nrrd = original_nrrd

    # 保存新的 NRRD 文件
    sitk.WriteImage(upsampled_nrrd, output_path)

    # 读取新文件以验证尺寸
    upsampled_nrrd_check = sitk.ReadImage(output_path)

    # 打印新文件的形状
    print("New shape of the upsampled file is:", upsampled_nrrd_check.GetSize())
# 创建data_1文件夹如果不存在
raw_data_path = 'data/raw'
output_data_path = 'data_1/raw'
if not os.path.exists(output_data_path):
    os.makedirs(output_data_path)

# 遍历raw文件夹中的每个case_xx文件夹
# for case_dir in sorted(os.listdir(raw_data_path)):
    
def process_case(case_dir):
    case_path = os.path.join(raw_data_path, case_dir)
    if os.path.isdir(case_path):
        # 创建data_1中对应的文件夹
        output_case_path = os.path.join(output_data_path, case_dir)
        if not os.path.exists(output_case_path):
            os.makedirs(output_case_path)

        # 处理img.nrrd文件
        img_path = os.path.join(case_path, 'img.nrrd')
        output_img_path = os.path.join(output_case_path, 'img.nrrd')
        check_and_scale_nrrd(img_path, output_img_path)

        # 创建structures文件夹
        structures_path = os.path.join(case_path, 'structures')
        output_structures_path = os.path.join(output_case_path, 'structures')
        if not os.path.exists(output_structures_path):
            os.makedirs(output_structures_path)

        # 遍历structures文件夹中的nrrd文件
        for structure_file in os.listdir(structures_path):
            if structure_file.endswith('.nrrd'):
                structure_path = os.path.join(structures_path, structure_file)
                output_structure_path = os.path.join(output_structures_path, structure_file)
                check_and_scale_nrrd(structure_path, output_structure_path)

pids = os.listdir(raw_data_path)

with Pool(processes=10) as pool:
    pool.map(process_case, pids)

print("Upsampling complete.")

