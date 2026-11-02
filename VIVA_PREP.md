# Viva Preparation

Use these as short speaking notes. Explain what you actually implemented; do not claim that the model proves a review is fake.

## A 30-second project introduction

“My project is a fake and deceptive product-review detection prototype. It combines review text with simple behavior signals from a graph connecting reviewers to products. I trained a TF–IDF and logistic-regression model, added reviewer and product history features, evaluated both known-reviewer and unseen-reviewer cases, and built a small Streamlit demo. The supplied labels are reviewer classifications, so I describe the output as a triage signal rather than proof of fraud.”

## Likely questions

### 1. What problem are you solving?

Deceptive reviews can distort the information shoppers use. I built a model that flags reviews whose text and available reviewer/product history resemble examples with the supplied fake label.

### 2. What is the objective of the project?

To compare text-only review classification with a hybrid approach that also uses reviewer–product graph history, then demonstrate the result in an interactive app.

### 3. What data did you use?

The supplied labeled file has 80,281 reviews, 36,764 reviewers, and labels 0 and 1. I matched the review text and reviewer ID to the full review corpus to attach product IDs. I did not include the dataset in the public repository.

### 4. What do labels 0 and 1 mean?

In this project, 0 means reviewer-labeled honest and 1 means reviewer-labeled fake, following the supplied classification field. They are weak labels, not independent expert judgments of each review.

### 5. What is TF–IDF?

TF–IDF converts text into numeric features. It gives weight to words and word pairs that help distinguish reviews while reducing the influence of terms that appear everywhere.

### 6. Why did you use logistic regression?

It is a practical classifier for sparse text features, trains quickly, and produces class probabilities. I used class weighting to account for the unequal label counts.

### 7. What is the graph in your project?

It is a bipartite graph: one node set contains reviewers, the other contains products, and an edge means a reviewer reviewed a product.

### 8. What graph features did you use?

The number of prior reviews by the reviewer, the number of distinct products they had reviewed, and the number of prior reviewers connected to that product.

### 9. How did you reduce data leakage?

For the hybrid evaluation, graph features are built only from training rows. For each training example, its own reviewer–product edge is excluded. The reviewer-disjoint evaluation holds out whole reviewers and uses text only for those unseen reviewers.

### 10. What were your results?

“On the random review split, the hybrid model reached 0.7889 accuracy and 0.8417 fake-class F1. On the reviewer-disjoint cold-start split, the text-only model reached 0.7235 accuracy and 0.7958 fake-class F1. These measure agreement with the supplied labels, not verified real-world fraud.”

### 11. Why report two evaluation settings?

Because a system with known reviewer history and a system seeing a new reviewer are different cases. Combining them into one score could hide how much the graph history helps.

### 12. What does precision mean here?

Fake-class precision asks: among reviews the model flagged as fake, what fraction matched the supplied fake label?

### 13. What does recall mean here?

Fake-class recall asks: among reviews with the supplied fake label, what fraction did the model flag?

### 14. Does the model prove a review is fake?

No. It estimates similarity to patterns in the training labels. The labels may be noisy, and a high activity count can describe a legitimate reviewer. A person should check other evidence.

### 15. Why did you not report BERT scores?

The project includes an optional Sentence-BERT backend, but the results in this run are from the TF–IDF baseline. I would only report BERT performance after actually training and evaluating that backend on the same splits.

### 16. What are the main limitations?

The labels are not independently verified; the graph signals are simple counts; the evaluation is not chronological; and performance may change for another product category, marketplace, or time period.

### 17. What would you improve next?

I would obtain independently adjudicated labels, add a chronological test split, compare the optional BERT embeddings, and investigate coordinated groups while checking that legitimate reviewers are not unfairly flagged.

### 18. How does the demo work?

The user enters a review and optional prior-history values. The saved model combines the text vector with standardized graph features and returns an estimated probability. If history is unknown, the demo accepts zero values.

## If you do not know an answer

Say what you implemented, identify what is still a limitation, and describe how you would test it. Do not invent a source, feature, experiment, or result.
