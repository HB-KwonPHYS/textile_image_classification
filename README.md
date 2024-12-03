# Textile Image Classification with Limited Data

# abstract

본 프로젝트는 데이터 셋이 제한된 상태에서 직물 현미경 이미지의 경량화 된 분류 모델 설계와 성능을 높이기 위한 학습 방법인 스플릿 - soft voting을 제시합니다. 직물 이미지는 회전, 미러 등의 데이터 증강 방식이 제한됩니다. 반복되는 패턴을 가지고 있기에 이를 이용한 데이터 증강 방식으로 split을 제시합니다. 이를 통해 원본 이미지를 리사이즈 할 때 손실되는 정보량을 줄이고, 모델을 더 강건하게 학습시킬 수 있습니다. 스플릿과 데이터 손실의 관계는 이미지의 information entropy를 통해 확인 가능합니다. 나눠진 이미지를 통해 soft voting을 하여, 성능을 향상시켰고 가벼운 모델로도 높은 성능을 달성 했습니다.

# Introduction

직물의 현미경 이미지 분류기를 제작하는 일입니다. 다만, 직물 이미지의 데이터 셋이 굉장히 제한되어 있어 바닐라 CNN으로 under 55%의 F1 score를 얻었습니다. 이 문제를 해결하기 위한 데이터 증강 방식과 학습 방식을 제시합니다. 이를 통해 더 가벼운 모델로도 높은 성능을 달성 하였습니다. 우리는 이미지를 압축하는 과정에서 발생하는 데이터 손실에 주목하였습니다. 프리 트레이닝 된 이미지 분류기를 사용하기 위해 244*244로 이미지를 줄여 학습을 하였습니다. 하지만 이런 경우 이미지가 가진 디테일과 다양한 정보들을 잃게 됩니다. 이미지의 해상도와 분류기의 성능에 관해서는 상관관계가 있으며 저품질 이미지의 경우 모델의 성능을 올리기 위해 초해상도 이미지(super resolution) 방법이 제시된 바 있습니다.

 정보 손실을 최소화 하기 위해 직물 이미지가 가진 특성을 사용합니다. 일반적으로 사전학습된 모델을 사용하기 위해 244*244 이미지로 리사이즈 하는 것은 정보의 손실을 야기합니다. 직물 이미지는 패턴의 반복이며 같은 unit cell 이상의 블럭으로 나눌 수 있습니다. 따라서 이러한 나눠진 블럭을 다시 244*244로 리사이즈 하여 사용하면 데이터 손실을 줄이며 강건한 모델을 만들 수 있습니다. 이러한 나누는 과정을 비교하여 컴퓨팅 자원과 성능을 고려해 최적의 방법을 탐구 하였습니다. 그리고 이러한 경우 같은 이미지에서 나눠진 블럭들은 같은 클래스를 가지며, 이미지의 추론 시 voting을 할 수 있습니다. 최신 모델과의 비교를 통해 이 방법이 가지는 이점을 비교하였습니다.
 
# 문제 제기 및 방법 제안
각 클래스별로 기타 103, 평직 161, 능직 337 개의 매우 적은 데이터셋이며, class별로 불균형한 데이터를 고려하여 weighted F1 score를 평가지표로 선정.  파이토치의 바닐라 CNN을 이용 하였을 때, 52% under의 accuracy. 각 이미지는 회전 등의 데이터 증강 기법이 제한되어 있는 상황임. 따라서 성능을 높이기 위한 여러 학습 방법 시도. 

0. new model : cnn_mod and cnn_Deep

   바닐라 CNN을 개량하여 가벼우면서도 조금 더 깊고 설명력이 강한 모델을 제작.
2. gray scale

    컬러채널 평균내어 연산량 줄임
4. split

   반복되는 패턴의 최소단위인 Unit cell 이상의 블럭으로 이미지를 잘라 학습시킴
6. voting

   자른 이미지는 모두 같은 class이므로 voting 수행. 여기서 확률값을 더하는 soft voting과 각 class의 진리값을 더하는 hard voting을 비교해 봄
8. benchmarking

   CNN기반이 아닌 트렌스포머 기반 이미지 모델 등 최신 모델도 함께 비교하여 이러한 방법이 가지는 효과 측정.

# 데이터 개요 및 split 예시
한국섬유개발 연구원(KOREA TEXTILE DEVELOPMENT INSTITUTE)에서 촬영한 직물 구조 현미경 촬영 2160*2880 해상도 이미지 601건 

평직 161, 능직(우) 164, 능직(좌) 173, 기타 분류 103건

사전학습 모델 활용과 원활한 학습을 위한 224x224 resize.
![image](https://github.com/user-attachments/assets/bb90e2e2-72de-49b0-93e1-29f23c08c01b)
[평직 , 능직(우) , 능직(좌) , 기타] 

능직(우)와 능직(좌)는 같은 class로 통합

![image](https://github.com/user-attachments/assets/c8f348f5-7430-4cf6-a5ae-b43d9b34fe7b)

원본 이미지

6/12/20/35/48장 split

## 실험 환경 및 과정
ubuntu 22.04 LTS ,python 3.8 , NVIDIA RTX™ A6000 (Ampere) * 4 

# new model 구축
## CNN_deep
![CNN_deep](https://github.com/user-attachments/assets/65519437-1007-4151-bae0-fb7a75dff75b)
## CNN_mod
![CNN_mod_graph](https://github.com/user-attachments/assets/03e709f6-0059-44f2-aa3a-82d000a2ba39)


# task flow and Installation and Usage
train : validation : test = 7:1:2 , early stopping and then upload at Wandb
![image](https://github.com/user-attachments/assets/272999b8-9361-44a2-9f32-6e80b1ef51e8)

Usage : preprocessing 폴더에 있는 코드를 통해 split or gary scale 등 수행 후  tarining_cnn.py 를 실행해 mobilenetv2 | xception | deit_tiny | cnn | cnn_deep를 학습시킴. voting and scoring 폴더의 코드로 test 수행.


# 실험 결과

## performance by image split
![image](https://github.com/user-attachments/assets/5086fd89-6185-47be-9900-a3ebae360fbf)

cnn의 경우 split 수가 늘어날 수록 성능이 높아지는 경향을 보임.

## performance by image split and voting
![image](https://github.com/user-attachments/assets/e984e792-47e7-4de8-93af-69cd2a381905)


cnn의 경우 split과 함께 voting을 수행하였을 때, 예외 없이 더 높은 성능을 보임.




![HB-KwonPHYS](https://github.com/HB-KwonPHYS/textile_image_classification/blob/main/plot/all%20.png)



