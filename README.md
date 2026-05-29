# 🧠 Handwritten Digit Recognition CNN

An end-to-end Deep Learning repository utilizing a custom 2-layer Convolutional Neural Network (CNN) built via PyTorch to classify handwritten digits from the MNIST dataset. 

This model achieves a peak evaluation accuracy of **99.05%** on 10,000 unseen test images and is deployed locally via an asynchronous drawing canvas powered by Gradio.

## 📺 Project Demo

<video src="./assets/demo.mp4" controls width="100%" autoplay loop muted>
  Your browser does not support the video tag.
</video>

---

## 🔬 Deep Learning Architecture & Mathematical Pipeline

Rather than relying on a classic Multilayer Perceptron (MLP) which flattens image arrays and discards vital spatial correlations, this architecture exploits 2D spatial feature extraction via localized receptive fields.

### 📐 Structural Dimensions & Core Math

The matrix transformation pipeline tracks the structural dimensions of the tensor as it passes through the network:

1. **Input Matrix**: Grayscale image tensor tracking at $1 \times 28 \times 28$ pixels.
2. **Convolutional Layer 1 (`conv1`)**: 
   * Applies 16 unique $3 \times 3$ filters with a stride of 1 and padding of 1.
   * Spatial formula: 
     $$Out = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1 = \left\lfloor \frac{28 - 3 + 2(1)}{1} \right\rfloor + 1 = 28$$
   * Output tensor dimension: $16 \times 28 \times 28$.
3. **Max Pooling 1 (`pool`)**: 
   * Downsamples via $2 \times 2$ kernels with a stride of 2, taking the peak activation value to achieve minor translation invariance.
   * Output tensor dimension: $16 \times 14 \times 14$.
4. **Convolutional Layer 2 (`conv2`)**: 
   * Extracts deeper, combined geometric configurations using 32 filters of $3 \times 3$.
   * Output tensor dimension: $32 \times 14 \times 14$.
5. **Max Pooling 2 (`pool`)**: 
   * Secondary spatial downsampling step.
   * Output tensor dimension: $32 \times 7 \times 7$.
6. **Flattening Step**:
   * The matrix spatial channels are unrolled linearly prior to classification dense layers:
     $$\text{Linear Inputs} = 32 \times 7 \times 7 = 1568 \text{ elements}$$
7. **Fully Connected Classifier (`fc1` & `fc2`)**:
   * Maps 1568 features down to a latent 128-neuron dense array, utilizing a final linear transformation to output 10 raw non-normalized logs (logits), matching digits 0–9.

### 📈 Non-Linear Activation & Mathematical Optimization

* **Rectified Linear Unit (ReLU)**: Injected after every convolution and dense layer to enforce structural non-linearity, enabling the model to learn complex high-dimensional boundaries without suffering from vanishing gradients:
  $$f(x) = \max(0, x)$$
* **Optimization & Loss Tracking**: Trained using **Cross-Entropy Loss** combined with the **Adam (Adaptive Moment Estimation)** optimizer. Adam utilizes first and second-order moments of gradients to dynamically adjust learning rates per individual parameter, speeding up stochastic gradient descent convergence safely.

---

## 📊 Training Convergence Logs

The model was optimized using mini-batch stochastic tracking over 5 epochs (Batch Size = 64, Learning Rate = 0.001):

* **Epoch 1**: Initial Loss: 0.2147 $\rightarrow$ Average Convergence Loss: **0.1810**
* **Epoch 2**: Average Convergence Loss: **0.0506**
* **Epoch 3**: Average Convergence Loss: **0.0362**
* **Epoch 4**: Average Convergence Loss: **0.0273**
* **Epoch 5**: Final Convergence Loss: **0.0209**

### 🎯 Final Evaluation Metric
```text
Evaluating model on test dataset...
Accuracy of the model on the 10,000 test images: 99.05%
