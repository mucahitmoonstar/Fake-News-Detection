**ABOUT DATASET**

The dataset contains two types of articles fake and real News. This dataset was collected from realworld sources; the truthful articles were obtained by crawling articles from Reuters.com (News website). As for the fake news articles, they were collected from different sources. The fake news articles were collected from unreliable websites that were flagged by Politifact (a fact-checking organization in the USA) and Wikipedia. The dataset contains different types of articles on different topics, however, the majority of articles focus on political and World news topics.

The dataset consists of two CSV files. The first file named “True.csv” contains more than 12,600 articles from reuter.com. The second file named “Fake.csv” contains more than 12,600 articles from different fake news outlet resources. Each article contains the following information: article title, text, type and the date the article was published on. To match the fake news data collected for kaggle.com, we focused mostly on collecting articles from 2016 to 2017. The data collected were cleaned and processed, however, the punctuations and mistakes that existed in the fake news were kept in the text.

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

![DATASET news](https://github.com/user-attachments/assets/554baefa-3f02-4cee-9c14-a06aba59da57)



**GOAL**

Classify news articles as real or fake using machine learning" refers to a project focused on detecting true and false news articles by leveraging labeled datasets. This involves gathering extensive collections of news articles, each marked as either credible or misleading. The project entails preprocessing the text for analysis, utilizing techniques like TF-IDF for feature extraction, and employing algorithms such as Multinomial Naive Bayes for classification. By training the model on this dataset, the system can accurately assess and classify new articles, providing users with a reliable tool to identify misinformation and promote media literacy. Continuous improvement through user feedback enhances its effectiveness in the evolving landscape of news consumption.


**ALGORİTHM**

**Multinomial Naive Bayes;**

Multinomial Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem, which is particularly suited for discrete data, such as text classification.
It assumes that the features (in this case, words in the text) are conditionally independent given the class label.
Reasons for Choosing Multinomial Naive Bayes
Effectiveness for Text Classification:

High-Dimensional Data: Text data typically has a large number of features (words). MultinomialNB is designed to work well in high-dimensional spaces, which makes it effective for tasks like fake news detection.
It can handle datasets with many features without overfitting, as it focuses on the frequency of word occurrences.
Simplicity and Speed:

Fast Training and Prediction: The algorithm is computationally efficient, making it suitable for scenarios where you need quick training and prediction, especially important for applications like news classification.
The model can be trained quickly even on large datasets, which is beneficial for real-time applications.
Probabilistic Nature:

Output Probabilities: MultinomialNB not only classifies inputs but also provides probabilities of the predictions. This can help in understanding the model's confidence in its predictions, which can be valuable in certain applications where decision-making depends on certainty levels.
This aspect allows for more nuanced interpretations of the results, such as determining which articles are borderline cases.
Works Well with TF-IDF:

The algorithm is particularly effective when combined with TF-IDF (Term Frequency-Inverse Document Frequency) feature extraction, as seen in the code. TF-IDF provides a way to weigh words based on their importance, which enhances the performance of the Naive Bayes classifier by allowing it to focus on significant words in the context of the dataset.
Assumptions of Feature Independence:

While the assumption of conditional independence of features (words) may not hold perfectly in real-world data, the algorithm still often performs surprisingly well. This is because the Naive Bayes model is robust to violations of its assumptions, especially in text classification tasks.


![naive bias](https://github.com/user-attachments/assets/0fe83479-fe45-4fe8-b7a8-c430e7aaa2c9)



**RESULT**

The accuracy of 0.9347 indicates that the model successfully classified approximately 93.47% of the news articles correctly as either real or fake. This high accuracy demonstrates the effectiveness of the implemented machine learning approach in distinguishing between credible and misleading information. Such a strong performance not only reflects the quality of the training data and preprocessing steps but also highlights the robustness of the Multinomial Naive Bayes algorithm in handling text classification tasks. Achieving this level of accuracy is crucial in real-world applications, as it enhances user trust in the model’s ability to identify misinformation and supports efforts to promote informed news consumption. This result suggests that the model is a valuable tool in the ongoing battle against fake news.
