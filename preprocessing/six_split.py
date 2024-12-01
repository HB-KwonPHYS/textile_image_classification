import os
from PIL import Image
root_dir = '/raid/coss04/textile_origin_split/Textile_origin_split'  
save_dir = '/raid/coss04/textile_split/Textile_six'



for class_folder in os.listdir(root_dir):
    class_path = os.path.join(root_dir, class_folder)
    save_class_path = os.path.join(save_dir, class_folder)
    if not os.path.isdir(class_path):
        continue  # 클래스 폴더가 아닌 경우 넘어감
    
    os.makedirs(save_class_path, exist_ok=True)
    for filename in os.listdir(class_path):
        filepath = os.path.join(class_path, filename)
        if not filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # 이미지 파일이 아니면 스킵
        img = Image.open(filepath)
        width, height = img.size
        # image split
        if height > width:  # 세로가 더 긴 경우
            part_width = width // 2
            part_height = height // 3
        else:  # 가로가 길거나 동일한 경우
            part_width = width // 3
            part_height = height // 2
        for i in range(3 if height > width else 2):
            for j in range(2 if height > width else 3):
                left = j * part_width
                upper = i * part_height
                right = (j + 1) * part_width if (height <= width or j < 1) else width
                lower = (i + 1) * part_height if (height > width or i < 1) else height
                if i == (2 if height > width else 1) and j == (1 if height > width else 2):
                    new_upper = upper - 75
                    new_lower = lower - 75
                    part = img.crop((left, new_upper, right, new_lower))
                else:
                    part = img.crop((left, upper, right, lower))
                # 부분 이미지를 지정된 폴더에 저장, 파일명에는 원본 이미지의 이름과 부분 번호가 포함
                part_filename = f"{filename.split('.')[0]}_{i * (2 if height > width else 3) + j + 1}.png"
                part_filepath = os.path.join(save_class_path, part_filename)
                part.save(part_filepath)
