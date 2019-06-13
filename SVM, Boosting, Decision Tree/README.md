Assignment 3

Problem 1 : Pegasos: a stochastic gradient based solver for linear SVM

1.1 Finish the implementation of the objective function in the primal formulation of SVM.
1.2 Implement the Pegasos Algorithm.
1.3 After training the model, run your classifier on the test set and report the accuracy.
1.4 Run the Pegasos algorithm for 500 iterations with 6 settings (mini-batch size K = 100 with different λ ∈ {0.01, 0.1, 1} 
    and λ = 0.1 with different K ∈ {1, 10, 1000}), and output a pegasos.json that records the test accuracy and the value of 
    objective function at each iteration during the training process.

Problem 2 : Boosting

- Implement decision stumps as weak classifiers and AdaBoost as the boosting algorithm. 
- Boosting algorithms construct a strong (binary) classifier based on iteratively adding one weak (binary) classifier 
  into it.
  
Problem 3 : Decision Tree

3.1 Conditional Entropy function implementation
3.2 Tree construction : The building of a decision tree involves growing and pruning. For simplicity, in this programming 
    set you are only asked to grow a tree.
