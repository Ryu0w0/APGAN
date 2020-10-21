# Attention Progressive growing of GAN
A PyTorch implementation of APGAN based on CelebA dataset. It is runnable with CelebA dataset. It would be helpful for those seeking the implementation of Progressive GAN (PGAN) or Self-Attention GAN (SAGAN).
# Description
## APGAN
This model was proposed by Ali, Mohamed and Mahdy (2019). APGAN incorporates techniques from PGAN (Karras et al., 2018) and SAGAN (Zhang et al., 2019) as below;

- From PGAN
    - Progressive growing structure
    - Equalized learning rate
- From SAGAN
    - Self-attention module
    - Pixel-wise feature vector normalization
    - Standard deviation-based feature map

**References**

Ali, I. S., Mohamed, M. F. and Mahdy, Y. B. (2019) ‘Data Augmentation for Skin Lesion using Self-Attention based Progressive Generative Adversarial Network’. Available at: http://arxiv.org/abs/1910.11960.

Karras, T. et al. (2018) ‘Progressive growing of GANs for improved quality, stability, and variation’, 6th International Conference on Learning Representations, ICLR 2018 - Conference Track Proceedings, pp. 1–26.

Zhang, H. et al. (2019) ‘Self-attention generative adversarial networks’, 36th International Conference on Machine Learning, ICML 2019, 2019-June, pp. 12744–12753.

## How to run

1. Clone the repository to the local.
2. Download `Align&Cropped Images` from [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
3. Unzip the obtained zip file and locate it such that ./files/input/dataset/img_align_celeba/*.jpg
4. Call "python ./apgan_train_main.py"

## Outputs

- Synthetic images
    - Location: ./apgan/files/output/gans/images
    - Real and synthetic images are periodically produced here while training for monitoring the changes of the appearance of images
- Text-based Log files
    - Location: ./files/output/gans/logs
- TensorBoard-based log files
    - ./files/output/gans/board
    - Losses in G and D are periodically produced for monitoring
- Weights of generator
    - ./files/output/gans/model
    - Weights of generator is produced as *.ptn file
    
## Configuration
There are 2 locations to configure to specify the behavior of APGAN.
### JSON file 
Location: ./gans/model/apgan/config/default_ch3.json

It controls the activation of techniques below;
- equalized learning rate
- pixel-wise normalization
- self-attention module.

Also, it specifies the learning rate of optimizers.

### Arguments of main.py
At the head of scripts in apgan_train_main.py, there are arguments of the program.

Most of the parameters can be used with default, whereas `batch_size_list` should be modify according to the available computational resource.
 
