# GNN for Social Network Security

Enhanced Graph Neural Network implementation for malicious account detection in social networks.

## Files

- `improved_gcn.py` - Enhanced GCN with dropout, early stopping, and feature engineering
- `experiment_notebook.ipynb` - Model comparisons and realistic testing
- `Untitled1.ipynb` - old files
- `Untitled4.ipynb` - old files
- `gcn.py` - Basic GCN implementation (old files)
- `requirements.txt` - Dependencies

## Key Improvements

- **Architecture**: Added dropout layers and proper regularization
- **Training**: Implemented early stopping and weight decay
- **Features**: Enhanced with node degrees and normalization
- **Evaluation**: Realistic testing with noisy data

## Usage

```bash
python improved_gcn.py
jupyter notebook experiment_notebook.ipynb
```

## Dependencies
PyTorch, PyTorch Geometric, NetworkX, Jupyter
