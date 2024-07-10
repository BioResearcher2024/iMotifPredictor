
# iMotifPredictor

## Overview

iMotifPredictor is a convolutional neural network (CNN) designed to predict i-motif (iM) structures in DNA by integrating multiple data sources. iMs are non-canonical structures that form in single-stranded DNA and have been linked to various cellular functions and diseases. This project includes trained models for predicting iM structures using different data sources.

## Contents

- **iMotifPredictor___ACM_BCB.pdf**: Research paper detailing the development and evaluation of iMotifPredictor.
- **Model files**:
  - **atac_model_gen.h5**
  - **atac_model_rand.h5**
  - **atacandmicro_model_gen.h5**
  - **atacandmicro_model_rand.h5**
  - **iMPropensity.h5**
  - **micro_model_gen.h5**
  - **micro_model_perm.h5**
  - **micro_model_rand.h5**
  - **sequence_model_gen.h5**
  - **sequence_model_perm.h5**
  - **sequence_model_rand.h5**

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Pandas

## Usage

The provided models can be used to predict iM structures based on different types of input data. Below are the instructions on how to use these models. Note that the provided code example is a suggested way to use the models, but you can also load sequences (in the appropriate format) and use the `predict` function to make predictions and save results as you prefer.

### Example Usage

Here is an example of how to load a model and make predictions using the provided models:

```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define the function to load the model and make predictions
def predict_and_save(chunk_file, model_path, model_type):
    # Load the chunk data
    data = pd.read_csv(chunk_file)
    
    # Preprocess the sequences (placeholder, adjust as necessary)
    def encode_sequence(sequence):
        mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
        return np.array([mapping.get(base, [0, 0, 0, 0]) for base in sequence], dtype=np.int8)

    sequences = np.array([encode_sequence(seq) for seq in data['Sequence']])
    signal = np.array(data['signal']).reshape(-1, 1)
    atac_signal = np.array(data['atac_signal']).reshape(-1, 1)

    # Load the model
    model = load_model(model_path)

    # Make predictions based on the model type
    if model_type == 'sequence_model':
        predictions = model.predict(sequences, batch_size=128)
    elif model_type == 'micro_model':
        predictions = model.predict([sequences, signal], batch_size=128)
    elif model_type == 'atac_model':
        predictions = model.predict([sequences, atac_signal], batch_size=128)
    elif model_type == 'atacandmicro_model':
        predictions = model.predict([sequences, atac_signal, signal], batch_size=128)
    else:
        raise ValueError("Invalid model type")

    # Add the predictions to the dataframe
    prediction_column_name = model_path.split('/')[-1].replace('.h5', '')
    data[prediction_column_name] = predictions

    # Save the dataframe to the same file
    data.to_csv(chunk_file, index=False)

# Example usage
predict_and_save('input_chunk.csv', 'sequence_model_gen.h5', 'sequence_model')
```

### Input Format

- **DNA Sequences**: CSV file containing a column named `Sequence` with DNA sequences of the required length (either 60 or 124, depending on the model).
- **Microarray Signal**: Column named `signal` containing iM propensity scores predicted by the microarray model.
- **ATAC Signal**: Column named `atac_signal` containing open-chromatin signals.

### How to Use the Models

1. **Prepare Input Data**: Ensure your input data is in a CSV format with the necessary columns (`Sequence`, `signal`, and/or `atac_signal`).

2. **Load Model**: Use TensorFlow's `load_model` function to load the required model.

3. **Make Predictions**: Pass the input data to the model and obtain predictions.

4. **Save Predictions**: Add the predictions to your input data and save the results.

### Models Description

- **sequence_model_gen.h5**: Model trained using genomic sequences of length 124 encoded in one-hot format.
- **sequence_model_perm.h5**: Model trained using permuted genomic sequences of length 124.
- **sequence_model_rand.h5**: Model trained using randomly selected genomic sequences of length 124.
- **micro_model_gen.h5**: Model trained using microarray data to predict iM propensity, requires sequences of length 60 encoded in one-hot format.
- **micro_model_perm.h5**: Model trained using permuted microarray data.
- **micro_model_rand.h5**: Model trained using randomly selected microarray data.
- **atac_model_gen.h5**: Model trained using genomic sequences and ATAC-seq data, requires sequences of length 124 encoded in one-hot format and ATAC signals.
- **atac_model_rand.h5**: Model trained using randomly selected ATAC-seq data.
- **atacandmicro_model_gen.h5**: Model trained using genomic sequences, microarray data, and ATAC-seq data, requires sequences of length 124 encoded in one-hot format, microarray signals, and ATAC signals.
- **atacandmicro_model_rand.h5**: Model trained using randomly selected genomic, microarray, and ATAC-seq data.
- **iMPropensity.h5**: Model trained to predict iM propensity based on high-throughput microarray data, requires sequences of length 60 encoded in one-hot format.

### Using iMPropensity Model

The `iMPropensity.h5` model is used to predict the propensity of iM formation based on microarray data. This model receives a DNA sequence of length 60 encoded in one-hot format. Here's an example of how to use the `iMPropensity.h5` model:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define a function to encode sequences
def encode_sequence(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    return np.array([mapping.get(base, [0, 0, 0, 0]) for base in sequence], dtype=np.int8)

# Load the model
model = load_model('iMPropensity.h5')

# Example sequence
sequence = 'ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT'
encoded_sequence = np.array([encode_sequence(sequence)])

# Make prediction
prediction = model.predict(encoded_sequence)
print(f'iM Propensity: {prediction[0][0]}')
```

## Conclusion

iMotifPredictor provides a powerful tool for predicting i-motif structures in DNA using a combination of sequence information, microarray data, and open-chromatin information. By leveraging convolutional neural networks, iMotifPredictor achieves high accuracy and provides valuable insights into iM formation mechanisms.


