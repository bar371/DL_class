r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1. False. The in-sample error is estimated by the training loss :
2. False. A solid minimal value should be used for both training and test, to give an extreme example, taking only 
 the last 5 samples out of 50000 to be your test set does not represent your accuracy on the task. Moreover, 
 taking only 5 samples for training and the rest for test would mean little to no learned distribution on the data,
 and thus achieve poor accuracy. A good split would be 80:20 on large data-sets.
3. True. In order to keep the test-set 'pure' and our hyper-parameter tuned with the validation set correct we should
keep the test set separate from the cross-validation.
4. False. The validation set is used for hyper-parameter tunning and not for estimation of the model's generalisation, 
that task is saved for the test-set.
 
"""

part1_q2 = r"""
** The approach is not justified. By re-training the training-set with different values from the hyper param 
lambda while trying to find the best possible result on the test-set leads to over-fitting your test-set, The 
Test set should always be sepereated from any hyper tunning and/or training. The correct approch would be
to split the training set into training and validation (by cross-validation) and finding the best value 
of lambda on the validation set.:**
"""

# ==============
# Part 2 answers

part2_q1 = r""" 
Yes. A higher K would lead to improved generalization for unseen data.   
Up to the point of including all the training samples. While this would be a very generalized predictor, 
it would also have a very high bias as instead of fitting the test sample to it's neighbors it matches it to the whole 
dataset.   
"""

part2_q2 = r"""
1. In terms of the train-set accuracy --- by doing cross validation we actually introduce or model to more variances of data, 
allowing it to generalize better as we chose the best mean accuracy between the folds. 

2. In terms of the test-set accuracy --- using cross-validation allows you to find the best hyper-params (k in our case) 
without overfitting the testset. This way we are able to see which k optimizes our validation set, 
and apply it on the test set. Thus we grantee that the test-set was not contaimed and our results are accurate.
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
"""

part3_q2 = r"""
**Your answer:**
"""

part3_q3 = r"""
**Your answer:**
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**
"""

part4_q2 = r"""
**Your answer:**
"""

part4_q3 = r"""
**Your answer:**
"""

# ==============
