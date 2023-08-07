# 📝 Fake News Detection using LSTM

## 📝 Description 
This project aims to detect fake news using LSTM (Long Short-Term Memory) neural networks. The model is trained on a dataset containing both real and fake news articles. It preprocesses the text data, applies word embeddings, and then trains an LSTM model to classify news articles as either real or fake.

## ⌛ Dataset 
The dataset used in this project is a combination of two datasets: one containing real news articles and the other containing fake news articles. The real news dataset is sourced from True.csv and the fake news dataset is sourced from Fake.csv.

## 🎯 Approach 
1. Combining the real and fake news datasets into a single dataset for training.
2. Preprocessing the text data by converting to lowercase, removing stopwords, and keeping only alphabetic characters.
3. Creating word embeddings using pre-trained GloVe word vectors.
4. Tokenizing the text data and padding sequences to ensure a fixed length for LSTM input.
5. Building an LSTM-based deep learning model for fake news detection.
6. Training the model on the preprocessed data.
7. Evaluating the model performance using accuracy and ROC-AUC score.
8. Creating a function to predict the label of new news articles using the trained model.

## 📥 Installations 
- Python
- Libraries: pandas, numpy, matplotlib, seaborn, nltk, keras

## ⚙️ Setup 
1. Clone the repository: `git clone <repository-url>`
2. Navigate to the project directory: `cd NLPFakeNewsDetect`
3. Install the required libraries: `pip install -r requirements.txt`

## 🛠️ Requirements 
- Dataset: fake_news.csv (contains combined real and fake news data)
- Python 
- Internet connection to download pre-trained GloVe word vectors (if not already available)

## 🚀 Technology Used 
- Python
- Jupyter Notebook
- Natural Language Processing (NLP)
- Long Short-Term Memory (LSTM) Neural Network
- Word Embeddings (GloVe)

## ▶️ Run 
To run the project, follow these steps:
1. Ensure you have the required dataset (fake_news.csv) in the project directory.
2. Open the notebook in Jupyter Notebook.
3. Run each cell sequentially to execute the code and train the LSTM model.
4. After training, you can use the 'predict_text' function to predict the label of new news articles.

---

📝 I frequently create content focused on complete projects in the realm of data science using Python and R.

💬 Feel free to inquire about data science, computer vision, Python, and R.

---

<p align="Right">(◕‿◕) Thank you for exploring my GitHub project repository. 👋</p>
