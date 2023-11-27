# Facial Gender Classification Using CNNs: A Transfer Learning Approach

#### Background
**Transfer learning** has proved an immensely powerful technique in the context of building deep models using a limited dataset and/or limited computational resources. An alternative to training a model from scratch, transfer learning leverages knowledge from solving one task to improve performance on a different but related task. Within the context of Computer Vision, **Convolutional Neural Networks (CNNs)** can benefit greatly from the technique, as hierarchical features extracted by the convolutional base can be relevant for use in other tasks.

Here, I aim to build a facial recognition classifier to identify images as either "Male" or "Female". I have chosen to do so by utilizing the convolutional base of the **"VGGFace"** model as a **pre-trained base** within my model, on top of which I will build and train my own custom classifier of dense layers. VGGFace is a CNN model that is specifically trained for face recognition tasks, having been designed to recognize and classify faces in images.

It is predicted that these two tasks (gender classification and general facial recognition) might share many underlying patterns or features which may be exploited to train an effective classifier. Thus, by using the pre-trained base of the VGGFace model, we should be able to successfully leverage the power of transfer learning and build a deep learning model capable of carrying out the task of facial gender classification to a high level of predictive performance.

#### The dataset

The dataset consists of 20,000 celebrity face images, where the "Male" class is labelled "1", and "Female" labelled  "0". Each image is zoomed into and centred upon the subject's face. Due to limited computational resources, I chose to extract just a subset of **6,500 images**, where I implemented a Train/Validation/Test split of 5000/750/750.

CelebFace dataset collated by https://www.kaggle.com/jessicali9530. <br>
Dataset preprocessed at and downloaded from https://www.kaggle.com/datasets/ashishjangra27/gender-detection-20k-images-celeba.

#### Model comparison

I will also be testing the performance of two more CNNs with different convolutional bases, in addition to that built upon the VGGFace pre-trained base.

Firstly, I will build my own basic, **untrained** convolutional base. It is predicted that this model will perform significantly worse than that with the VGGFace pre-trained base, as it will only be able to use the data at hand here to learn feature extraction, as opposed to the large dataset of images used in training VGGFace.

Secondly, I will investigate the effect of using the pre-trained base of the more generalised **"VGG16"** model, which is trained on the **ImageNet** database (1,000 images each of the 1,000 most common objects). Note that the architecture ("VGG16" architecture) is the same as the VGGFace model; the models simply differ in their learned weights. It is predicted that, while still having learned to extract many useful features, the VGG16 base will not perform quite as well as the VGGFace base, which is trained on a task considerably more similar to that of this project than the VGG16 model.

Thus, the three models investigated in this project are summarised below:

<br>

- **"Custom":** Architecture = Custom, Weights = Untrained

<br>

- **"VGG16":** Architecture = VGG16, Weights = ImageNet

<br>

- **"VGGFace":** Architecture = VGG16, Weights = VGGFace

---

The **full project report** is provided here in both **.html** and **.ipynb** format.

N.B. you will have to open the .ipynb file in a Jupyter notebook to make use of the section links within the ToC...
