#Natural Language Processes 

This code performs text classification on the "SMSSpamCollection" dataset, focusing on the identification of spam and non-spam (ham) messages within SMS content. The workflow encompasses several key steps:

1. *Data Loading:* The code begins by importing the dataset from the "SMSSpamCollection.txt" file and organizing it into a structured format using the Pandas library. The dataset consists of labeled SMS messages, categorized as either spam or ham.

2. *Text Preprocessing:* The natural language processing (NLP) pipeline is initiated using the Natural Language Toolkit (NLTK). The textual content of the messages undergoes essential preprocessing stages. Tokenization segments the messages into distinct words, lemmatization reduces words to their base forms, and stopwords—common words with little contextual meaning—are excluded from the analysis. The result is a cleaned and standardized version of the text, referred to as the "processed_text."

3. *Feature Extraction:* The CountVectorizer, a text processing tool, converts the processed_text into numerical features that machine learning algorithms can interpret. The goal is to create a numerical representation of the text that captures its essence while maintaining its contextual meaning.

4. *Addressing Imbalanced Data:* Acknowledging the class imbalance issue—where one class (spam) has significantly fewer samples than the other (ham)—the code employs the RandomOverSampler technique. This approach creates synthetic instances of the minority class (spam) to balance out the distribution of the dataset and ensure a fair representation during training.

5. *Model Training and Evaluation:* Two classification models, Multinomial Naive Bayes and Logistic Regression, are chosen to classify messages as spam or ham. Each model is trained on the balanced dataset and then evaluated against a test set. This evaluation entails computing metrics such as accuracy, precision, recall, and the F1-score. These metrics provide insights into the models' performance in terms of correct classifications, false positives, and false negatives.

6. *Visualization:* The code employs visual aids, particularly confusion matrices represented as heatmaps, to offer an intuitive depiction of the classification results. These matrices illustrate how well the models are distinguishing between the two classes and provide a visual understanding of where they excel or struggle.

*Results:*
- The Multinomial Naive Bayes model:
  - Attains an accuracy of around 97%, implying that approximately 97% of the predictions are accurate.
  - Demonstrates balanced precision and recall for both spam and ham categories, indicating competent classification performance for both classes.
  - Achieves a weighted F1-score of approximately 97%, illustrating a harmonious trade-off between precision and recall across both categories.

- The Logistic Regression model:
  - Yields an accuracy of about 99%, signifying a high proportion of accurate predictions.
  - Displays robust precision and recall values for both spam and ham classes, akin to the Multinomial Naive Bayes model.
  - Records a weighted F1-score of about 99%, highlighting the model's proficiency in maintaining a balanced relationship between precision and recall.

In essence, the code employs NLP techniques to preprocess text, addresses class imbalance, trains classification models, and evaluates their performance. The outcomes showcase promising results in distinguishing between spam and ham messages within SMS data, underscoring the models' capability to discern between the two classes with commendable accuracy and equilibrium between precision and recall.
