# Fake News Detection System

## Overview
This project implements a state-of-the-art fake news detection system using BERT (Bidirectional Encoder Representations from Transformers). The system is designed to classify news articles as either genuine or fake with high accuracy, leveraging advanced natural language processing techniques.

## Key Features
- Utilizes pre-trained BERT model for text classification
- Implements advanced text preprocessing pipeline
- Provides model evaluation metrics (Accuracy, F1 Score)
- Includes trained model persistence and prediction API
- Comprehensive logging and result tracking

## Technical Details
- **Model Architecture**: BERT-base-uncased
- **Framework**: PyTorch with Hugging Face Transformers
- **Preprocessing**: Tokenization, Padding, Attention Masking
- **Evaluation Metrics**: Accuracy, F1 Score, Confusion Matrix
- **Training**: Fine-tuning with AdamW optimizer

## Usage
1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the detection system:
```python
python fake_news_detector.py
```

3. Make predictions:
```python
from fake_news_detector import predict

result = predict("Your news article text here")
print(result)  # Returns 'Real' or 'Fake'
```

## Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- KaggleHub
- NumPy
- Pandas
- Scikit-learn

## License
MIT License - See LICENSE file for details

## Contributing
Contributions are welcome! Please read our contribution guidelines before submitting pull requests.
