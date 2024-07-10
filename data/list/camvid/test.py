import os

# 读取txt文件的内容
file_path = '/root/autodl-tmp/PIDNet_sota_origin/data/list/camvid/val.lst'  # 替换成你的txt文件路径
with open(file_path, 'r') as file:
    lines = file.readlines()

# 替换每一行的内容
new_lines = []
for line in lines:
    parts = line.split()
    if len(parts) == 2:
        image_path = parts[0]
        label_path = parts[1]
        # 获取文件名
        image_filename = os.path.basename(image_path)
        label_filename = os.path.basename(label_path)
        # 判断label文件名是否以'_L'结尾
        if label_filename.endswith('_L.png'):
            # 替换label文件名中的'_L.png'为'.png'
            label_filename = label_filename.replace('_L.png', '.png')
            # 替换该行中的label文件名
            label_path = os.path.join(os.path.dirname(label_path), label_filename)
            # 更新行内容
            new_line = f"{image_path} {label_path}\n"
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

# 将新内容写入文件
output_file_path = 'output.txt'  # 替换成你想要保存的文件路径
with open(output_file_path, 'w') as file:
    file.writelines(new_lines)

print("处理完成！")
