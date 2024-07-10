import os

# 定义输入文件的路径
input_file = '/root/autodl-tmp/PIDNet_sota_origin/data/list/cocostuff/val.txt'

# 定义新文件路径保存的文件名
output_file = '/root/autodl-tmp/PIDNet_sota_origin/data/list/cocostuff/new_val.txt'

# 存储新的文件路径
new_lines = []

# 读取txt文件中的每一行
with open(input_file, 'r') as file:
    lines = file.readlines()

# 遍历每一行，生成对应的文件路径
for line in lines:
    # 去除换行符，并根据空格分割路径
    image_path, label_path = line.strip().split()

    # 修改label路径
    label_dir, label_filename = os.path.split(label_path)
    label_name, label_ext = os.path.splitext(label_filename)
    new_label_name = label_name + '_labelTrainIds' + label_ext
    new_label_path = os.path.join(label_dir, new_label_name)

    # 将新的文件路径保存到列表中
    new_line = f"{image_path} {new_label_path}\n"
    new_lines.append(new_line)

# 将新的文件路径写入新的文件中
with open(output_file, 'w') as file:
    file.writelines(new_lines)

print(f"New file paths saved to {output_file}.")
