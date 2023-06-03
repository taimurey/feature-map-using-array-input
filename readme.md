# Convolutional Neural Network (CNN) Feature Extraction

This repository contains a Python code snippet that demonstrates the process of feature extraction using a convolutional neural network (CNN). The code applies a 3x3 filter to a 9x9 input image, generates a feature map, applies rectified linear unit (ReLU) activation, performs max pooling, and displays the results step by step.

## Installation

To run the code, you need to have the following dependencies installed:

- Python 3.x
- NumPy
- python-docx

You can install the required dependencies using pip:

```shell
pip install numpy python-docx
```

## Usage

To use the code, follow these steps:

1. Clone this repository or download the code file `cnn_feature_extraction.py`.
2. Open a terminal or command prompt and navigate to the directory where the code file is located.
3. Run the code using the following command:

```shell
python cnn_feature_extraction.py
```

4. The code will generate a Word document named `output.docx` that contains the input image, filter, feature map, and pooled feature map. You can open the document to view the results.

## Methodology

The code performs the following steps:

1. Defines an input image and a 3x3 filter.
2. Applies the filter to the input image to generate a feature map using convolution.
3. Applies the rectified linear unit (ReLU) activation to the feature map.
4. Performs max pooling on the activated feature map using a 2x2 pooling filter and a stride of 2.
5. Displays each step, including the input image, filter, feature map, and pooled feature map.

## Contributing

Contributions to this repository are welcome. If you have any suggestions or improvements, please feel free to submit a pull request.

## License

This code is released under the [MIT License](LICENSE).

## Acknowledgements

This code is inspired by the concepts of convolutional neural networks (CNNs) and feature extraction. Special thanks to the authors and contributors of the Python libraries used in this code.
