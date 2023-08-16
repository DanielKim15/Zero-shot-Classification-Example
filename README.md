# Zero-shot-Classification-Example

## What is Zero-shot Classification?

Zero-shot learning is a type of machine learning where instead of giving training examples, a high-level description of new categories is given so that the machine can relate it to existing categories that the machine has learned about2. The purpose of generative Zero-shot learning is to learn from seen classes, transfer the learned knowledge, and create samples of unseen classes from the description of these unseen categories1. With zero-shot learning, we can create a mapping connecting low-level features and a semantic description of auxiliary data to classify unknown classes of actions3.

## How does it work

Let's say you are looking for a certain category or specific type, such as traveling or cooking. You can create a list that contain the categories you want to look for, then the model will search through the texts to determine which category is the most likely of the ones you've chosen.

```python

sequence_to_classify = "one day I will see the world"
candidate_labels = ['travel', 'cooking', 'dancing']
classifier(sequence_to_classify, candidate_labels)
#{'labels': ['travel', 'dancing', 'cooking'],
# 'scores': [0.9938651323318481, 0.0032737774308770895, 0.002861034357920289],
# 'sequence': 'one day I will see the world'}

```
Here we see that travel is the cateogry that most likely fit with the texts.

## How to get started

### Source of the NLP: Facebook

Let's download the bart-large-mnli from Hugging Face:

```python
from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

```

Then lets add our data in. The example that's being used is the Amazon food reviews from kaggle and 10,000 random rows will be selected for this demonstration:

```python
data = pd.read_csv("C:/Users/dkim.CENSEO/Downloads/Reviews.csv").sample(n=10000)
data

```

Once the data is ready, let's set up the model in Pytorch and lets start the classification:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline

model_name = "facebook/bart-large-mnli"
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)


classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer, device=0) # 0 for CUDA device

batch_size = 100
classifications = [] 
candidate_labels = ['dessert', 'meat', 'vegetable','fruit', 'bread','pasta','potato']


for i in range(0, len(data), batch_size): #The batching is to make the coding more efficient
    batch = data['Text'][i:i + batch_size].tolist() #bath is going to pick 100 rows and put them in a list
    classifications.extend(classifier(batch, candidate_labels)) #Then its going to run the zero shot classifier in the batch, then the process repeats until all has been finished

data['classification'] = classifications 

```

Now print out the results and you should have your results!

# Pros and Cons

## Pros

* Flexibility and Scalability: Zero-shot learning allows the model to classify data into new classes without requiring specific training examples for those classes. This flexibility makes extending the model to handle new classes easily without retraining.
  
* Reduced Data Annotation: Since zero-shot learning doesn’t require labelled examples for all classes, it can significantly reduce the need for extensive data annotation, which can be expensive and time-consuming in traditional supervised learning.
  
* Generalization to Unseen Classes: The model can generalize to classify data not seen during training as long as relevant semantic embeddings or descriptions are provided. This makes zero-shot learning useful for applications where new classes emerge frequently.
  
* Cross-Domain Transfer: Zero-shot learning can effectively transfer knowledge from one domain to another, leveraging pre-trained models’ semantic understanding to classify data in a related field with minimal adaptation.
  
* Adaptability to Multi-Label Classification: Zero-shot learning can handle multi-label classification naturally. It can assign multiple labels to an input instance if the class descriptions allow that.

Disadvantages
Performance Dependence on Class Descriptions: The quality and relevance of the provided class descriptions significantly impact zero-shot learning’s performance. The model might struggle to classify instances if the descriptions are inadequate or ambiguous correctly.
Limited Performance on Dissimilar Classes: Zero-shot learning might not perform as well in classes that are very different from those seen during pre-training. The model’s performance degrades for classes that lack semantic similarity with the training data.
Overfitting Risk: If the class descriptions are too specific or similar to the training data, the model might inadvertently overfit the given descriptions, leading to poorer generalization on unseen data.
Sensitivity to Input Variations: Zero-shot learning can be sensitive to slight variations in the input, as the model heavily relies on semantic embeddings and context. Small changes in the input text might lead to different predictions.
Evaluation Challenges: Measuring the performance of zero-shot learning can be challenging, as standard evaluation metrics used in supervised learning might not directly apply. Evaluating truly unseen classes can be complex and might require the careful design of evaluation protocols.

Download data from this kaggle link: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
