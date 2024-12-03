# Textile Image Classification with Limited Data

# Abstract

This project proposes a lightweight classification model for textile microscope images under limited datasets and introduces a training method called **split-soft voting** to enhance performance. Textile images have limitations in data augmentation methods such as rotation and mirroring due to their inherent patterns. Leveraging their repetitive patterns, we propose a split data augmentation method. This approach reduces information loss when resizing the original images and allows the model to learn more robustly. The relationship between splitting and data loss is verified through the information entropy of the images. By performing soft voting on the split images, we improved performance and achieved high accuracy even with lightweight models.

# Introduction

We are developing a classifier for microscope images of textiles. However, the dataset of textile images is extremely limited, resulting in a weighted F1 score under **52%** with a vanilla CNN. To address this problem, we propose data augmentation and training methods. Through these methods, we achieved high performance even with lighter models.

We focused on the data loss that occurs during the image resizing process. To use a pre-trained image classifier, we reduced the images to 224×224 pixels for training. However, this resizing leads to a loss of details and various inherent information in the images. There is a correlation between image resolution and classifier performance, and super-resolution methods have been proposed to improve model performance for low-quality images by upscaling them.

In this project, we utilize the characteristics of textile images to minimize information loss. Textile images have repeating patterns and can be divided into blocks larger than the unit cell. By splitting the images into these blocks and resizing them to 224×224 pixels, we can reduce data loss and create a more robust model. We explored the optimal method by comparing different splitting processes, considering computing resources and performance. Moreover, since the blocks split from the same image belong to the same class, we can perform voting during inference. We compared the advantages of this method with state-of-the-art models.

# Problem Statement and Proposed Method

Given a highly limited and unbalanced dataset with 103 images of 'Others', 161 of 'Plain Weave', and 337 of 'Twill', we selected the weighted F1 score as the evaluation metric. Using PyTorch's vanilla CNN resulted in a weighted F1 score under **52%**. Data augmentation techniques such as rotation are limited for these images due to their structural patterns. Therefore, we attempted various training methods to improve performance:

1. **New Models: `cnn_mod` and `cnn_deep`**

   - Modified the vanilla CNN to create lightweight yet deeper models with greater explanatory power. Developed a CNN model specifically for grayscale images, among others.

2. **Grayscale Conversion**

   - Reduced computational load by averaging the color channels to convert images to grayscale.

3. **Split**

   - Trained the model by splitting images into blocks larger than the unit cell, the minimum repeating pattern in textiles.

4. **Voting**

   - Since all split images belong to the same class, we performed voting during inference. We compared **soft voting**, which sums probability values, and **hard voting**, which sums the predicted class labels.

5. **Benchmarking**

   - Compared our methods with the latest models, such as transformer-based image models, to measure the effectiveness of our approach.

# Data Overview and Split Examples

- **Dataset:** 601 microscope images of textile structures with a resolution of 2160×2880 pixels, provided by the **Korea Textile Development Institute**.
- **Categories:** 161 'Plain Weave', 164 'Twill (Right)', 173 'Twill (Left)', 103 'Others'.

To utilize pre-trained models and ensure efficient training, images were resized to 224×224 pixels.

![image](https://github.com/user-attachments/assets/bb90e2e2-72de-49b0-93e1-29f23c08c01b)

*[From left to right: Plain Weave, Twill (Right), Twill (Left), Others]*

- Combined 'Twill (Right)' and 'Twill (Left)' into the same class 'Twill'.

![image](https://github.com/user-attachments/assets/c8f348f5-7430-4cf6-a5ae-b43d9b34fe7b)

- **Original Image**
- **Split into 6, 12, 20, 35, and 48 pieces**

## Experimental Environment and Process

- **Operating System:** Ubuntu 22.04 LTS
- **Programming Language:** Python 3.8
- **Hardware:** NVIDIA RTX™ A6000 (Ampere) ×4 GPUs

# Building the New Model

## `cnn_deep`

![CNN_deep](https://github.com/user-attachments/assets/65519437-1007-4151-bae0-fb7a75dff75b)

- Created a model with a deeper architecture and larger fully connected layers than the vanilla CNN.

# Task Flow, Installation, and Usage

- **Data Split:** Train : Validation : Test = 7:1:2
- **Training Process:** Implemented early stopping; models were uploaded to Wandb.

![image](https://github.com/user-attachments/assets/272999b8-9361-44a2-9f32-6e80b1ef51e8)

**Usage:** Use the code in the `preprocessing` folder to perform splitting or grayscale conversion. Then, run `training_cnn.py` to train models like `mobilenetv2`, `xception`, `deit_tiny`, `cnn`, and `cnn_deep`. Use the code in the `voting_and_scoring` folder for testing.

# Experimental Results

## CNN Performance by Image Split

![image](https://github.com/user-attachments/assets/5086fd89-6185-47be-9900-a3ebae360fbf)

- The CNN model showed improved performance as the number of splits increased.

## CNN Performance by Image Split and Voting

![image](https://github.com/user-attachments/assets/e984e792-47e7-4de8-93af-69cd2a381905)

- When voting was performed along with splitting in the CNN model, higher performance was observed. Soft voting showed higher performance than hard voting.

## Performance by Image Split

![image](https://github.com/user-attachments/assets/4018dc76-ca76-4ef2-9b62-ecf17df19199)

- When training with the latest transformer-based models, unlike CNN-based models, splitting had little effect. In the case of MobileNetV2, which has fewer parameters, the effect of splitting was more pronounced.

## Performance by Image Split and Voting

![image](https://github.com/user-attachments/assets/81cefb05-7537-4b09-8bd0-ad3f41fc29c7)
![Performance Comparison](https://github.com/HB-KwonPHYS/textile_image_classification/blob/main/plot/all%20.png)

- When soft voting was performed, each model achieved its highest performance.

## Grayscale

![image](https://github.com/user-attachments/assets/a29f17a7-e26a-4cb1-8cd2-28739a998836)

- Modified the input layer of `cnn_deep` to accommodate grayscale images but discarded this approach as there was no significant difference in computational resources compared to using color channels.

# Computing Power

![image](https://github.com/user-attachments/assets/2841784f-b142-4616-a45c-3a3f8f0a0e2a)
![image](https://github.com/user-attachments/assets/c8463647-5460-4aca-928e-c48a56b972a1)

- Compared the memory usage and number of parameters in convolutional layers as each model processes the layers.
- Through this training method and voting, we can achieve high performance even with lightweight models.

# Conclusion

As the saying goes, "Garbage in, garbage out," the quality of input data is crucial in deep learning and computer vision. However, when data is severely limited, techniques are needed to address the problem. In this study, we employed a data augmentation method inspired by super-resolution techniques. While super-resolution focuses on converting low-quality images to high-quality ones, our method differs by using high-quality original images to augment data instead of relying on compressed low-quality images.

By augmenting data through the repetition of unit cells—a characteristic of textile images—we obtained robust and high-performance models. We found that increasing the number of splits was a major factor in improving performance. From the perspective of image entropy, this method reduces data loss during training. Additionally, we achieved higher performance through voting and confirmed that soft voting achieved the best results. In the case of the pre-trained MobileNetV2, we observed a **49-percentage point** improvement in performance. The lightweight models we created without pre-training also showed significant performance improvements.

Vision transformer models like DeiT and Xception showed the best performance when split and voting were applied but also demonstrated high performance even without these methods. Compared to the previous models, they are much heavier and benefit from pre-training. By using our proposed method, we were able to achieve similar performance with lightweight models like `cnn_deep`.

# Future Work

While our proposed methods have shown significant improvements in textile image classification with limited data, there are several avenues for future research:

1. **Exploring Optimal Splitting Boundaries**: Although we attempted splits up to 48 pieces, intuitively, the performance is likely to degrade when the splits become smaller, especially below the unit cell size. However, we were unable to experiment with this. In future work, we aim to find the boundary for optimal performance by experimenting with various split sizes, including those smaller than the unit cell.

2. **Detailed Analysis of Image Entropy**: By conducting a deeper analysis of the relationship between image entropy and model performance, we can gain more profound insights into data loss during preprocessing and help design better preprocessing pipelines. However, image entropy does not provide insights into different resolutions. We can compare and analyze factors such as image entropy density and color-considered image entropy for each split.

3. **Application to Other Textile Types**: By extending our methodology to classify additional types of textile patterns or to detect defects, we can broaden the applicability of the model in the textile industry. Such textile classifiers can be widely used in the recycling industry or the SPA (Specialty retailer of Private label Apparel) industry. Additionally, by improving the model, it can be used as a defect detection model in textile factories.

4. **Optimization for Real-Time Deployment**: Investigating methods such as model compression or quantization to optimize the model for real-time processing would make it more practical for industrial applications. The model can be optimized for industrial embedded inference machines like NVIDIA® Jetson™.


# Application Areas
Both crystal structures and textiles are characterized by the repetition of a fundamental unit called the unit cell. In crystallography, a unit cell is the smallest repeating structure that, when duplicated in three-dimensional space, forms the entire crystal lattice. Similarly, textiles are composed of repeating patterns where the unit cell represents the minimal design element that, when repeated, creates the full fabric pattern.

Material Science and Crystallography: The methodologies developed in this study can be applied to microscopic images of crystalline materials. By focusing on the repetition of unit cells, we can improve the classification and analysis of crystal structures, aiding in material identification and quality control.

Automated Defect Detection: Understanding the regularity of unit cell patterns allows for the detection of anomalies or defects in both textiles and crystalline materials. This can be crucial for quality assurance in manufacturing processes where structural integrity is vital. In particular, the crystal structure or repeating structure can be used for defect detection in semiconductor chips, semiconductor wafers, and super alloys.


# References and Documentation
Z. Wang, S. Chang, Y. Yang, D. Liu and T. S. Huang, "Studying Very Low Resolution Recognition Using Deep Networks," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 2016, pp. 4792-4800, doi: 10.1109/CVPR.2016.518.

Saeed Anwar, Salman Khan, and Nick Barnes. 2020. A Deep Journey into Super-resolution: A Survey. ACM Comput. Surv. 53, 3, Article 60 (May 2021), 34 pages. https://doi.org/10.1145/3390462

Y. Hou, Z. Ma, C. Liu and C. C. Loy, "Learning Lightweight Lane Detection CNNs by Self Attention Distillation," 2019 IEEE/CVF International Conference on Computer Vision (ICCV), Seoul, Korea (South), 2019, pp. 1013-1021, doi: 10.1109/ICCV.2019.00110.

Min Yong Park, & Hyun Kwon (2023-02-08). Evaluation of Image Classification Performance based on Resolution Difference. Proceedings of Symposium of the Korean Institute of communications and Information Sciences.

Sungho Shin, Joosoon Lee, Junseok Lee, Seungjun Choi, & Kyoobin Lee. (2020). Low-Resolution Image Classification Using Knowledge Distillation From High-Resolution Image Via Self-Attention Map. Journal of KIISE, 47(11), 1027-1031, 10.5626/JOK.2020.47.11.1027
