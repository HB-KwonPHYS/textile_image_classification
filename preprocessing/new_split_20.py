import os
from PIL import Image
root_dir = 'C:/Users/USER/Dropbox/빅웨이브AI_preprocess/Textile_origin_split'  # 여기에 폴더의 경로를 적어주세요
save_dir = 'C:/Users/USER/Dropbox/빅웨이브AI/NEW_Textile_20/'
# 각 서브 폴더를 순회
for tvt_folder in os.listdir(root_dir):
    class_dir = os.path.join(root_dir, tvt_folder)
    save_class_dir = os.path.join(save_dir, tvt_folder)
    if not os.path.isdir(class_dir):
        continue
        
    for class_folder in os.listdir(class_dir):
        class_path = os.path.join(class_dir, class_folder)
        save_class_path = os.path.join(save_class_dir, class_folder)
        if not os.path.isdir(class_path):
            continue  # 클래스 폴더가 아닌 경우 스킵
        os.makedirs(save_class_path, exist_ok=True)  # 저장 폴더 생성
        for filename in os.listdir(class_path):
            filepath = os.path.join(class_path, filename)
            if not filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue  # 이미지 파일이 아니면 스킵
            img = Image.open(filepath)
            width, height = img.size
            # image split
            if height > width:  # 세로가 더 긴 경우
                part_width = width // 4
                part_height = height // 5
                parts_x = 4
                parts_y = 5
            else:  # 가로가 길거나 같은 경우
                part_width = width // 5
                part_height = height // 4
                parts_x = 5
                parts_y = 4
            for i in range(5 if height > width else 4):
                for j in range(4 if height > width else 5):
                    left = j * part_width
                    upper = i * part_height
                    right = (j + 1) * part_width if j < part_width - 1 else width
                    lower = (i + 1) * part_height if i < part_height - 1 else height
                    
                    if i == parts_y - 1 and j == parts_x - 1:
                        new_upper = max(upper - 75, 0)
                        new_lower = max(lower - 75, 0)
                        part = img.crop((left, new_upper, right, new_lower))
                    else:
                        part = img.crop((left, upper, right, lower))

                    part_filename = f"{filename.split('.')[0]}_{i * parts_x + j + 1}.png"
                    part_filepath = os.path.join(save_class_path, part_filename)
                    part.save(part_filepath)
                    
                
