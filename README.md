# whitebox_aes

Python white box aes implementation (Chow)

## Introduction

Reference: [A Tutorial on White-box AES](https://eprint.iacr.org/2013/104.pdf). If you want to understand how it works,
you should read the paper carefully.

I implement it step by step through 4 constructions:

1. pure lookup tables
2. lookup tables with input and output encodings applied (bijections)
3. lookup tables with mixing bijections applied
4. lookup tables with both input and output encodings and mixing bijections applied (TODO)

## Requirements

- Python 3.12 (for reference)
- `pip install numpy galois more-itertools pycryptodomex scipy`

## Usage

`aes/`: aes implementation

`aes/aes_1d.py`: view state as 1-dimensional array

`aes/aes_2d.py`: view state as 2-dimensional array

`aes_numpy/`: aes implementation using numpy

`aes_numpy/aes_1d.py`: view state as 1-dimensional ndarray

`aes_numpy/aes_2d.py`: view state as 2-dimensional ndarray

`wb_aes/wb_aes.py`: white box aes implementation

`wb_aes_numpy/wb_aes.py`: white box aes implementation using numpy (less loops but harder to comprehend, you must be
familiar with numpy's features and tricks). I make it for speed's sake, but it turns out to have no significant
improvement compared to the `wb_aes/wb_aes.py`.