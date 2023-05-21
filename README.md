# Abnormal-Activity-Detection-Deep-Learning
## Unveiling Abnormality: Real-time Anomaly Detection with LRCN Deep Learning

- The Abnormal-Activity-Detection-using Deep-Learning LRCN project focuses on developing a deep learning model for detecting abnormal behavior in videos. The project utilizes the Long-term Recurrent Convolutional Network (LRCN), which combines the power of convolutional neural networks (CNNs) and recurrent neural networks (RNNs).

- The LRCN model offers the advantage of reduced computational time compared to previous models, making it suitable for real-time detection. By leveraging 11 layers instead of the time-consuming 16 layers of the VGG-16 model, the LRCN model achieves efficient processing without sacrificing accuracy.

- To optimize memory usage, the project team resized video frames from 224px to 64px while maintaining sufficient detail for anomaly detection. The dataset used for training the model consists of videos containing both abnormal behavior, such as fighting, and normal behavior, including walking and running. This diverse dataset enhances the model's ability to accurately identify and classify anomalous activities.

- With an achieved accuracy of 82% on their custom dataset, the Abnormal-Activity-Detection-using Deep-Learning LRCN project showcases the potential of deep learning techniques to detect and analyze abnormal behavior in videos.

![](doc_imgs/main_page_ss.png)

# Tools:
- [Tensorflow](https://www.tensorflow.org/io)
- [Keras](https://keras.io/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [moviepy](https://zulko.github.io/moviepy/)
- [Matplotlib](https://matplotlib.org/)

# Long-term recurrent convolutional network (LRCN):

- [LRCN](https://sh-tsang.medium.com/brief-review-lrcn-long-term-recurrent-convolutional-networks-for-visual-recognition-and-9542bc7e8a79)

# Getting Started:
After cloning the repository to local machine.
To run the project locally, run the following CLI commands.
```
pip install -r requirements.txt ## Install all the dependencies
```
