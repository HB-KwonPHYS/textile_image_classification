# Textile Image Classification with Limited Data

# abstract

본 프로젝트는 데이터 셋이 제한된 상태에서 직물 현미경 이미지의 경량화 된 분류 모델 설계와 성능을 높이기 위한 학습 방법인 스플릿 - soft voting을 제시합니다. 직물 이미지는 회전, 미러 등의 데이터 증강 방식이 제한됩니다. 반복되는 패턴을 가지고 있기에 이를 이용한 데이터 증강 방식으로 split을 제시합니다. 이를 통해 원본 이미지를 리사이즈 할 때 손실되는 정보량을 줄이고, 모델을 더 강건하게 학습시킬 수 있습니다. 스플릿과 데이터 손실의 관계는 이미지의 information entropy를 통해 확인 가능합니다. 나눠진 이미지를 통해 soft voting을 하여, 성능을 향상시켰고 가벼운 모델로도 높은 성능을 달성 했습니다.

# Introduction

직물의 현미경 이미지 분류기를 제작하는 일입니다. 다만, 직물 이미지의 데이터 셋이 굉장히 제한되어 있어 바닐라 CNN으로 under 55%의 F1 score를 얻었습니다. 이 문제를 해결하기 위한 데이터 증강 방식과 학습 방식을 제시합니다. 이를 통해 더 가벼운 모델로도 높은 성능을 달성 하였습니다. 우리는 이미지를 압축하는 과정에서 발생하는 데이터 손실에 주목하였습니다. 프리 트레이닝 된 이미지 분류기를 사용하기 위해 244*244로 이미지를 줄여 학습을 하였습니다. 하지만 이런 경우 이미지가 가진 디테일과 다양한 정보들을 잃게 됩니다. 이미지의 해상도와 분류기의 성능에 관해서는 상관관계가 있으며 저품질 이미지의 경우 모델의 성능을 올리기 위해 초해상도 이미지(super resolution) 방법이 제시된 바 있습니다. 이는 저해상도 이미지를 업스케일링 하여 딥러닝을 수행하는 작업입니다.

 이 프로젝트에서는 정보 손실을 최소화 하기 위해 직물 이미지가 가진 특성을 사용합니다. 사전학습된 모델을 사용하기 위해 244*244 이미지로 리사이즈 하는 것은 정보의 손실을 야기합니다. 직물 이미지는 패턴의 반복이며 같은 unit cell 이상의 블럭으로 나눌 수 있습니다. 따라서 이러한 나눠진 블럭을 다시 244*244로 리사이즈 하여 사용하면 데이터 손실을 줄이며 강건한 모델을 만들 수 있습니다. 이러한 나누는 과정을 비교하여 컴퓨팅 자원과 성능을 고려해 최적의 방법을 탐구 하였습니다. 그리고 이러한 경우 같은 이미지에서 나눠진 블럭들은 같은 클래스를 가지며, 이미지의 추론 시 voting을 할 수 있습니다. 최신 모델과의 비교를 통해 이 방법이 가지는 이점을 비교하였습니다.
 
# 문제 제기 및 방법 제안
각 클래스별로 기타 103, 평직 161, 능직 337 개의 매우 적은 데이터셋이며, class별로 불균형한 데이터를 고려하여 weighted F1 score를 평가지표로 선정.  파이토치의 바닐라 CNN을 이용 하였을 때, 52% under의 accuracy. 각 이미지는 회전 등의 데이터 증강 기법이 제한되어 있는 상황임. 따라서 성능을 높이기 위한 여러 학습 방법 시도. 

0. new model : cnn_mod and cnn_Deep

   바닐라 CNN을 개량하여 가벼우면서도 조금 더 깊고 설명력이 강한 모델을 제작. gray scale 용 cnn 모델 제작 등.
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

바닐라 CNN 보다 깊고 큰 fully connected layer을 가진 모델 생성. 

# task flow and Installation and Usage
train : validation : test = 7:1:2 , early stopping and then upload at Wandb
![image](https://github.com/user-attachments/assets/272999b8-9361-44a2-9f32-6e80b1ef51e8)

Usage : preprocessing 폴더에 있는 코드를 통해 split or gary scale 등 수행 후  tarining_cnn.py 를 실행해 mobilenetv2 | xception | deit_tiny | cnn | cnn_deep를 학습시킴. voting and scoring 폴더의 코드로 test 수행.


# 실험 결과

## CNN performance by image split
![image](https://github.com/user-attachments/assets/5086fd89-6185-47be-9900-a3ebae360fbf)
cnn의 경우 split 수가 늘어날 수록 성능이 높아지는 경향을 보임. 

## CNN performance by image split and voting
![image](https://github.com/user-attachments/assets/e984e792-47e7-4de8-93af-69cd2a381905)
cnn의 경우 split과 함께 voting을 수행하였을 때, 더 높은 성능을 보임. 특히 hard voting보다 soft voting의 성능이 더 높게 나옴.

## performance by image split
![image](https://github.com/user-attachments/assets/4018dc76-ca76-4ef2-9b62-ecf17df19199)
트랜스포머 기반 최신 모델을 같이 학습시킨 경우, CNN기반 모델과 달리 split의 영향이 크지 않음. 파라미터 수가 적은 모바일넷의 경우 스플릿의 효과가 두드러짐.

## performance by image split and voting
![image](https://github.com/user-attachments/assets/81cefb05-7537-4b09-8bd0-ad3f41fc29c7)
![HB-KwonPHYS](https://github.com/HB-KwonPHYS/textile_image_classification/blob/main/plot/all%20.png)
soft voting을 수행한 경우, 각 모델들이 최고 성능을 달성할 수 있었음. 

## gary scale 
![image](https://github.com/user-attachments/assets/a29f17a7-e26a-4cb1-8cd2-28739a998836)

cnn_deep을 gray scale에 맞게 입력층을 수정 하였으나 컴퓨팅 리소스 측면에서 컬러채널이 있는것과 차이가 없어 폐기함.

# computing power
![image](https://github.com/user-attachments/assets/2841784f-b142-4616-a45c-3a3f8f0a0e2a)
![image](https://github.com/user-attachments/assets/c8463647-5460-4aca-928e-c48a56b972a1)

각 모델이 각 층을 통과할때 메모리 사이즈와 conv layer의 파라미터 개수를 비교한 것입니다. 이러한 학습 방법과 voting을 통해 작은 모델로도, 높은 성능을 달성할 수 있습니다.

# 결론
"Garbage in, garbage out" 이란 말이 있듯, 딥러닝과 컴퓨터 비전에서 가장 중요한 것은 입력 데이터의 품질이다. 하지만 어떤 이유로 데이터가 매우 제한된 상황이라면 문제를 해결하기 위한 기법들이 필요하다. 이 연구에서는 기존 초해상도 방법론을 이미지 데이터 증강에 사용하였다. 초해상도 기법은 저품질- 고품질 이미지의 변환이 핵심이다. 우리가 사용한 방법은 저품질의 압축 이미지를 사용하는 것 대신 고품질의 원본 이미지를 이용한 데이터 증강이라는 점에서 차이가 있다.

직물 이미지가 가진 특성인 unit cell의 반복을 통해 데이터를 증강하여 강건하고 높은 성능의 모델을 얻을 수 있었으며, 이 과정에서 split의 증가는 성능을 높여주는 주요 요소임을 알 수 있었다. 이미지의 엔트로피 관점에서는 데이터 손실을 줄인 학습방법이라 할 수 있다. 추가적으로 voting을 통해 더 높은 성능을 얻었고 soft voting이 최고의 성능을 달성하는 것을 확인하였다. 사전 학습된 Mobilenetv2의 경우 49% point의 성능 향상을 보였으며 이 실험을 위해 만든 사전 학습되지 않은 가벼운 모델들도 높은 성능 향상을 보였다. 

vision Transformer계열인 DeiT와  xception의 경우에는 split and voting을 수행한 경우 최고의 성능을 보였지만 아무것도 적용하지 않아도 높은 성능을 보였다. 앞의 모델들과 비교하면 훨씬 무거우며 사전학습되었다는 차이점이 존재하며, 이러한 방법을 사용 하였을때 CNN_deep과 같은 가벼운 모델로도 같은 성능을 달성할 수 있었다. 



# 레퍼런스
Z. Wang, S. Chang, Y. Yang, D. Liu and T. S. Huang, "Studying Very Low Resolution Recognition Using Deep Networks," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 2016, pp. 4792-4800, doi: 10.1109/CVPR.2016.518.

Saeed Anwar, Salman Khan, and Nick Barnes. 2020. A Deep Journey into Super-resolution: A Survey. ACM Comput. Surv. 53, 3, Article 60 (May 2021), 34 pages. https://doi.org/10.1145/3390462

Y. Hou, Z. Ma, C. Liu and C. C. Loy, "Learning Lightweight Lane Detection CNNs by Self Attention Distillation," 2019 IEEE/CVF International Conference on Computer Vision (ICCV), Seoul, Korea (South), 2019, pp. 1013-1021, doi: 10.1109/ICCV.2019.00110.

Min Yong Park, & Hyun Kwon (2023-02-08). Evaluation of Image Classification Performance based on Resolution Difference. Proceedings of Symposium of the Korean Institute of communications and Information Sciences.

Sungho Shin, Joosoon Lee, Junseok Lee, Seungjun Choi, & Kyoobin Lee. (2020). Low-Resolution Image Classification Using Knowledge Distillation From High-Resolution Image Via Self-Attention Map. Journal of KIISE, 47(11), 1027-1031, 10.5626/JOK.2020.47.11.1027
