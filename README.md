# iMotifPredictor


## Overview

iMotifPredictor is a convolutional neural network (CNN) designed to predict i-motif (iM) structures in DNA by integrating multiple data sources. iMs are non-canonical structures that form in single-stranded DNA and have been linked to various cellular functions and diseases. This project includes the trained models, scripts, and necessary instructions to utilize iMotifPredictor for predicting iM structures.

## Contents

- **iMotifPredictor___ACM_BCB.pdf**: Research paper detailing the development and evaluation of iMotifPredictor.
- **iMotifPredictor_seq.py**: Script for predicting iM structures using only DNA sequence information.
- **iMotifPredictor_atac.py**: Script for predicting iM structures using DNA sequence and open-chromatin information (ATAC-seq).
- **iMotifPredictor_microarray.py**: Script for predicting iM propensity using microarray data.
- **iMotifPredictor_microarray_and_atac.py**: Script for predicting iM structures using a combination of DNA sequence, microarray data, and open-chromatin information.
- **iMPropensity.py**: Script for computing iM propensity scores.
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
- Biopython
- Matplotlib
- seaborn
- pybedtools
- h5py

## Usage

### Predicting iM Propensity

The `iMPropensity.py` script is used to predict the propensity of iM formation based on microarray data. This model receives a DNA sequence of length 60 encoded in one-hot format.

To use the script:
```python
python iMPropensity.py --input input_sequence.fasta --model iMPropensity.h5 --output output_predictions.txt
```

### Predicting iM Structures

#### Using Sequence Information

The `iMotifPredictor_seq.py` script predicts iM structures using only DNA sequence information. The model requires DNA sequences of length 124 encoded in one-hot format.

To use the script:
```python
python iMotifPredictor_seq.py --input input_sequence.fasta --model sequence_model_gen.h5 --output output_predictions.txt
```

#### Using Sequence and Open-Chromatin Information

The `iMotifPredictor_atac.py` script predicts iM structures using DNA sequence and open-chromatin information (ATAC-seq). The sequence input should be of length 124 encoded in one-hot format, and the open-chromatin signal should be provided as a scalar.

To use the script:
```python
python iMotifPredictor_atac.py --input input_sequence.fasta --atac atac_signal.txt --model atac_model_gen.h5 --output output_predictions.txt
```

#### Using Sequence, Microarray Data, and Open-Chromatin Information

The `iMotifPredictor_microarray_and_atac.py` script predicts iM structures using a combination of DNA sequence, microarray data, and open-chromatin information. The sequence input should be of length 124 encoded in one-hot format. The microarray-based iM propensity and open-chromatin signal should be provided as scalars.

To use the script:
```python
python iMotifPredictor_microarray_and_atac.py --input input_sequence.fasta --microarray microarray_signal.txt --atac atac_signal.txt --model atacandmicro_model_gen.h5 --output output_predictions.txt
```

### Input Format

- **DNA Sequences**: FASTA file containing DNA sequences of the required length (either 60 or 124, depending on the model).
- **Microarray Signal**: Text file containing iM propensity scores predicted by the microarray model.
- **ATAC Signal**: Text file containing open-chromatin signals.

### Model Training

The models were trained using the following datasets and parameters:
- **Sequence Model**: Trained on DNA sequences of length 124.
- **ATAC Model**: Trained on DNA sequences and ATAC-seq data.
- **Microarray Model**: Trained on microarray data to predict iM propensity.
- **Combined Model**: Trained on DNA sequences, microarray data, and ATAC-seq data.

### Examples

```python
# Predicting iM propensity
python iMPropensity.py --input example_sequence.fasta --model iMPropensity.h5 --output propensity_predictions.txt

# Predicting iM structures using sequence information
python iMotifPredictor_seq.py --input example_sequence.fasta --model sequence_model_gen.h5 --output sequence_predictions.txt

# Predicting iM structures using sequence and ATAC information
python iMotifPredictor_atac.py --input example_sequence.fasta --atac atac_signal.txt --model atac_model_gen.h5 --output atac_predictions.txt

# Predicting iM structures using sequence, microarray, and ATAC information
python iMotifPredictor_microarray_and_atac.py --input example_sequence.fasta --microarray microarray_signal.txt --atac atac_signal.txt --model atacandmicro_model_gen.h5 --output combined_predictions.txt
```

## Conclusion

iMotifPredictor provides a powerful tool for predicting i-motif structures in DNA using a combination of sequence information, microarray data, and open-chromatin information. By leveraging convolutional neural networks, iMotifPredictor achieves high accuracy and provides valuable insights into iM formation mechanisms.



