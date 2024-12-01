import os
from PIL import Image

root_dir = '/home/bsy/data/Textile_origin/Textile'  # 원본 이미지 폴더의 경로
save_root_dir = '/home/bsy/data/Textile_origin/Textile_center'  # 저장 폴더의 경로
crop_width, crop_height = 224, 224  # 크롭 사이즈

for class_folder in os.listdir(root_dir):
    class_path = os.path.join(root_dir, class_folder)
    save_class_path = os.path.join(save_root_dir, class_folder)
    if not os.path.isdir(class_path):
        continue  # 클래스 폴더가 아닌 경우 스킵
    os.makedirs(save_class_path, exist_ok=True)  # 저장 폴더 생성
    # 각 폴더 내의 이미지 파일을 순회
    for filename in os.listdir(class_path):
        filepath = os.path.join(class_path, filename)
        if not filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # 이미지 파일이 아닌 경우 스킵
        img = Image.open(filepath)
        width, height = img.size
        center_x, center_y = width // 2, height // 2
        offsets_x = [crop_width // 2, 0, -crop_width // 2, -crop_width]  # 4개의 오프셋
        offsets_y = [crop_height // 2, 0, -crop_height // 2]  # 3개의 오프셋
        for i in range(len(offsets_y)):
            for j in range(len(offsets_x)):
                left = center_x - crop_width // 2 + offsets_x[j]
                upper = center_y - crop_height // 2 + offsets_y[i]
                right = left + crop_width
                lower = upper + crop_height
                part = img.crop((left, upper, right, lower))
                part_filename = f"{filename.split('.')[0]}_{i * len(offsets_x) + j + 1}_center.png"
                part_filepath = os.path.join(save_class_path, part_filename)
                part.save(part_filepath)
