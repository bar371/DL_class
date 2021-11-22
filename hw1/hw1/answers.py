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

part2_q1 = r"""It depends to what point. with lower K's , A higher K would lead to improved generalization for unseen 
data as it takes into account more samples to vote from. However, this is true ip to the point of including the 
number of samples from the most frequent class of the training samples. This would be lead to very poor generalization  
as it will always predict the majority class from the training data, which would likely lead to bad accuracy. """

part2_q2 = r"""1. In terms of the train-set accuracy --- by doing cross validation we actually introduce or model to 
more variances of data, allowing it to generalize better as we chose the best mean accuracy between the folds. 

2. In terms of the test-set accuracy --- using cross-validation allows you to find the best hyper-params (k in our case) 
without overfitting the testset. This way we are able to see which k optimizes our validation set, 
and apply it on the test set. Thus we grantee that the test-set was not contaminated and our results are accurate.
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
** The Delta Hyper-Param allows for some margin of error between the score of the predicted class and the actual class.
However, thanks to the regularization term that penalizes large weights even if we will choose a higher margin, 
the SVM model will still attempt to find a matrix W that minimizes the loss, making the Delta param arbitrary. :**
"""

part3_q2 = r"""
** 1. The linear model learns a weight Matrix W that represent the distribution of each number class, by applying
 W to X (input image) so the pixels that 'agree' with W will get a high value. This is represented by the x_score,
 the highest scoring class index will be chosen as the predicted class. Following this, you can see that
  the mistakes the model has made were on numbers close enough in shape to the predicted class rather the then actual
  label. 
  2. A KNN model will look-up the closest resembling images and that the k-majority vote from them. Doing so 
  does not learn a representation of the distribution for each label from the data like linear regression,
  in that regard it is different. However in terms of similarity we would imaging that images the linear regression
   was mistaken on will also be challenging for the KNN classifier --- 
   as these images will have similar shapes but different classes. **
"""

part3_q3 = r"""
** 1. The learning rate we chose based on the graph result is too high, evidence of this is the 'dip' in accuracy or spike 
in loss during ~epoch 16 while still reaching high over-all accuracy/low bias. A Good Graph will achieve same or better
accuracy with no such 'dips'. A too-low learning rate will be sub-par in accuracy results as it took too long to 
learn the matrix W with given epochs.

2. The model is slightly(+) over-fitted to the training set, the difference between it (~94% accuracy) and the test-set 
 accuracy (~85% accuracy) is non-trivial. The reason for the overfitting is a high learning rate with limited regularization,
 A possible solution would be a lower learning rate with more training epochs and increased regularization. :**
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
Idealy the residual plot should have a fairly random pattern around the horizontal axis (close to 0).

The final plot after the CV is fit better than the plot for the top 5-features because it appears that there is a greater amount of residuals closer to to the horizontal axis.
"""

part4_q2 = r"""
1. Yes. Adding non-linear features still keeps a linear regression model linear as the learn weight matrix W is still doing a linear transformtion to the input data. We do note that the dimensional space will be higher given non-linear for the input features.
2. Yes. We can fit a non-linear function to the original features, taking them into a higher dimention and then apply a linear regression model (in a higher space then the original input space).
3. Imagine a linear classification model. As we saw in Part 3, the parameters $\mat{W}$ of such a model define a hyperplane representing the decision boundary. How would adding non-linear features affect the decision boundary of such a classifier? Would it still be a hyperplane? Why or why not?
The decision boundry of such a hyperplane would grow as we increased the working dimitional space, it would still be  a hyperplane,
as a linear classifier always builds a hyperplane attepmting to seperate the diffrent classes from on another. The hyper itself would lie in a higher space.
"""

part4_q3 = r"""
1. We use np.logspace beacuse it leads to a higher range of different variables to attempt.
2. Total amount of times the model was fitted to data: degree_range * lambda_range * k_fold

Our Model:
degree_range: 3
lambda_range: 20
k_fold: 3
= 3 * 20 * 3 = 180 
"""
# ==============
