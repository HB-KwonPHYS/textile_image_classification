# Textile Image Classification with Limited Data

# abstract

본 프로젝트는 데이터 셋이 제한된 상태에서 직물 현미경 이미지의 경량화 된 분류 모델 설계와 성능을 높이기 위한 학습 방법인 스플릿 - soft voting을 제시합니다. 직물 이미지는 회전, 미러 등의 데이터 증강 방식이 제한됩니다. 반복되는 패턴을 가지고 있기에 이를 이용한 데이터 증강 방식으로 split을 제시합니다. 이를 통해 원본 이미지를 리사이즈 할 때 손실되는 정보량을 줄이고, 모델을 더 강건하게 학습시킬 수 있습니다. 스플릿과 데이터 손실의 관계는 이미지의 information entropy를 통해 확인 가능합니다. 나눠진 이미지를 통해 soft voting을 하여, 성능을 향상시켰고 가벼운 모델로도 높은 성능을 달성 했습니다.

# Introduction

직물의 현미경 이미지 분류기를 제작하는 일입니다. 다만, 직물 이미지의 데이터 셋이 굉장히 제한되어 있어 바닐라 CNN으로 under 55%의 F1 score를 얻었습니다. 이 문제를 해결하기 위한 데이터 증강 방식과 학습 방식을 제시합니다. 이를 통해 더 가벼운 모델로도 높은 성능을 달성 하였습니다. 우리는 이미지를 압축하는 과정에서 발생하는 데이터 손실에 주목하였습니다. 프리 트레이닝 된 이미지 분류기를 사용하기 위해 244*244로 이미지를 줄여 학습을 하였습니다. 하지만 이런 경우 이미지가 가진 디테일과 다양한 정보들을 잃게 됩니다. 이러한 문제의식에서 고해상도 이미지 학습에 대한 방법이 제시된 바 있습니다.

본 연구도 이와 비슷하게 손실된 정보를 최소화 하기 위해 직물 이미지가 가진 특성을 사용합니다. 직물 이미지는 패턴의 반복이며 같은 unit cell 이상의 블럭으로 나눌 수 있습니다. 이러한 나누는 과정을 비교하여 컴퓨팅 자원과 성능을 고려해 최적의 방법을 탐구 하였습니다. 그리고 이러한 경우 같은 이미지에서 나눠진 블럭들은 같은 클래스를 가지며, 이미지의 추론 시 voting을 할 수 있습니다. 최신 모델과의 비교를 통해 이 방법이 가지는 이점을 비교하였습니다.

# 데이터 개요
직접 수집한 직물 구조 현미경 촬영 이미지 601건 활용

평직 161, 능직(우) 164, 능직(좌) 173, 기타 분류 103건

고해상도 이미지의 원활한 학습을 위한 resize 실시, 224x224 해상도 활용
![image](https://github.com/user-attachments/assets/bb90e2e2-72de-49b0-93e1-29f23c08c01b)



# Information Entropy, Image Entropy, Image Compression, Information Loss

## 1. Information Entropy

Information entropy quantifies the uncertainty or randomness in a probability distribution, representing the average number of bits needed to encode information.

**Formula**:
\[
H(X) = -\sum_{i=1}^{n} P(x_i) \log_2 P(x_i)
\]
Where:
- \( H(X) \): Entropy of the random variable \( X \)
- \( P(x_i) \): Probability of event \( x_i \)
- \( n \): Number of possible values of \( X \)

---

## 2. Entropy of an Image

The entropy of an image is calculated based on the probability distribution of pixel values, representing the information content or complexity of the image.

**Formula**:
\[
H = -\sum_{i=0}^{L-1} p_i \log_2 p_i
\]
Where:
- \( L \): Range of possible pixel values (e.g., \( L=256 \) for 8-bit grayscale images)
- \( p_i \): Probability of the \( i \)-th pixel value

---

## 3. Image Compression

Image compression reduces the storage size of an image by eliminating redundancy or perceptually insignificant data. It is classified into:
1. **Lossless Compression**: Retains all data while removing redundancy (e.g., PNG, TIFF).
2. **Lossy Compression**: Removes some data to achieve higher compression ratios (e.g., JPEG).

**Compression Ratio**:
\[
CR = \frac{\text{Original Size}}{\text{Compressed Size}}
\]

---

## 4. Information Loss

Lossy compression may result in the loss of some information, which can be quantified using metrics like **Peak Signal-to-Noise Ratio (PSNR)**.

**PSNR Formula**:
\[
PSNR = 10 \cdot \log_{10} \left( \frac{\text{MAX}^2}{\text{MSE}} \right)
\]
Where:
- \( \text{MAX} \): Maximum possible pixel value (e.g., 255 for 8-bit images)
- \( \text{MSE} \): Mean Squared Error between the original and compressed images:
\[
\text{MSE} = \frac{1}{MN} \sum_{i=0}^{M-1} \sum_{j=0}^{N-1} \left[ I(i, j) - K(i, j) \right]^2
\]
    - \( I(i, j) \): Original image pixel value
    - \( K(i, j) \): Compressed image pixel value
# 결과

![HB-KwonPHYS](https://github.com/HB-KwonPHYS/textile_image_classification/blob/main/plot/all%20.png)



