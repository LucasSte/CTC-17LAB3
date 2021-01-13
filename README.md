# ID3-Python

This is an implementation of the ID3 algorithm to build a decision tree.
This repository depends on the following packages:
```
numpy
pandas
sklearn
```

```tree_test.py``` loads a csv, deletes variables with low information gain and runs the algorithm.
In the end, it prints the decision tree as a json dictionary and runs a test case. 
Instructions for use can be found inside the file.