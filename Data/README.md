Datasets

1. Regression Dataset : The UCI Wine Quality dataset lists 11 chemical measurements of 4898 white wine samples as well as an 
  overall quality per sample, as determined by wine connoisseurs. See winequality- white.csv. We split the data into training, 
  validation and test sets in the preprocessing code. You will use linear regression to predict wine quality from the chemical 
  measurement features.
  
2. Classification Dataset : MNIST is one of the most well-known datasets in computer vision, consisting of images of 
  handwritten digits from 0 to 9. We will be working with a subset of the official version of MNIST, denoted as mnist subset. 
  In particular,cwe randomly sampled 700 images from each category and split them into training, validation, and test sets. 
  This subsetccorresponds to a JSON file named mnist subset.json. JSON is a lightweight data-interchange format, similar to a 
  dictionary. After load- ing the file, you can access its training, validation, and test splits using the keys ‘train’, 
  ‘valid’, and ‘test’, respectively. For example, if we load mnist subset.json to the variable x, x[′train′] refers to the 
  training set of mnist subset. This set is a list with two elements: x[′train′][0] containing the features of size N (samples)
  ×D (dimension of features), and x[′train′][1] containing the corresponding labels of size N.
