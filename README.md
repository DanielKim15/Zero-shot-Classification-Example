# Zero-shot-Classification-Example

##What is Zero-shot Classification?

Zero-shot learning is a type of machine learning where instead of giving training examples, a high-level description of new categories is given so that the machine can relate it to existing categories that the machine has learned about2. The purpose of generative Zero-shot learning is to learn from seen classes, transfer the learned knowledge, and create samples of unseen classes from the description of these unseen categories1. With zero-shot learning, we can create a mapping connecting low-level features and a semantic description of auxiliary data to classify unknown classes of actions3.

##How does it work

Let's say you are looking for a certain category or specific type, such as traveling or cooking. You can create a list that contain the categories you want to look for, then the model will search through the texts to determine which category is the most likely of the ones you've chosen.

```
sequence_to_classify = "one day I will see the world"
candidate_labels = ['travel', 'cooking', 'dancing']
classifier(sequence_to_classify, candidate_labels)
#{'labels': ['travel', 'dancing', 'cooking'],
# 'scores': [0.9938651323318481, 0.0032737774308770895, 0.002861034357920289],
# 'sequence': 'one day I will see the world'}

```

Download data from this kaggle link: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
