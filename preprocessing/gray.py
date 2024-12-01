import os
from PIL import Image
root_dir = '/home/bsy/data/Textile_origin/Textile_six' 
save_root_dir = '/home/bsy/data/Textile_origin/Textile_gray_six' 
rgb = False
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
        # 이미지를 불러와서 그레이 스케일로 변환
        img = Image.open(filepath)
        gray_img = img.convert('L')
        save_img = gray_img
        color_md = 'GRAY'
        # 그레이 스케일 이미지를 다시 RGB로 변환
        if rgb:
            save_img = gray_img.convert('RGB')
            color_md = 'RGB'
        # 변환된 이미지를 지정된 폴더에 저장
        save_filepath = os.path.join(save_class_path, f'{color_md}_{filename}')
        save_img.save(save_filepath)
