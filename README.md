# Attention Progressive growing of GAN
A PyTorch implementation of Attention Progressive growing of GAN (APGAN). It is runnable with CelebA dataset. This repository would be helpful for those seeking the implementation of Progressive GAN (PGAN) or Self-Attention GAN (SAGAN).
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

## Structure

The structure of APGAN is progressively growing as the training goes. The figure shown below is the maximum size (fully grown) of the structure that the generator produces 256x256 synthetic images and the discriminator accepts 256x256 real or synthetic images. A self-attention module is attached in the block of resolution of 128x128 denoted as the orange layer. Standard deviation-based feature map is concatinated in the last block in the discriminator denoted as the green layer.

<img src="https://github.com/Ryu0w0/meta_repository/blob/master/APGAN/images/structure.PNG" width=60%>

## How to run

1. Clone the repository to the local.
2. Download `Align&Cropped Images` from [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
3. Unzip the obtained zip file and locate it such that ./files/input/dataset/img_align_celeba/*.jpg
4. Call "python ./apgan_train_main.py"

## Configuration
There are 2 locations for the configuration to specify the behavior of APGAN.
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
 
## Outputs
- Synthetic images
    - Location: ./apgan/files/output/gans/images
    - Synthetic images are periodically produced from a generator at the moment while training for monitoring the changes of the appearance of images.
    - Examples of the synthetic images produced by APGAN when speficying 600000 iter per stage (It was actually not enough to produce high quality images): <img src="https://github.com/Ryu0w0/meta_repository/blob/master/APGAN/images/fake.jpg" width=60%>
- Text-based Log files
    - Location: ./files/output/gans/logs
- TensorBoard-based log files
    - ./files/output/gans/board
    - Losses in G and D at the moment are periodically produced for monitoring
- Weights of generator
    - ./files/output/gans/model
    - Weights of generator is produced as *.ptn file
    
