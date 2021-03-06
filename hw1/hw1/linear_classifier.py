import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = torch.normal(mean=0, std=weight_std, size=(n_features,n_classes))

        # ====== YOUR CODE: ======
        
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        class_scores = x @ self.weights
        y_pred = torch.argmax(class_scores, dim=1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        assert(len(y) == len(y_pred))
        acc = len(y[y == y_pred]) / len(y)
        # ========================

        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):


        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
            train_b_counter, val_b_counter = 0, 0

            total_correct = 0
            average_loss = 0


            for train_batch in dl_train:
                # training vars
                x_b_train = train_batch[0]
                y_b_train = train_batch[1]
                y_b_pred, x_scores = self.predict(x_b_train)

                # get loss
                average_loss += loss_fn.loss(x_b_train,y_b_train, x_scores, y_b_pred)
                total_correct += self.evaluate_accuracy(y_b_train, y_b_pred)
                train_b_counter += 1

                # update weights , inc. regulaziation
                self.weights = self.weights  - learn_rate * (loss_fn.grad() + weight_decay * self.weights)

            train_res.accuracy.append(total_correct / train_b_counter)
            train_res.loss.append(average_loss / train_b_counter)

            total_correct, average_loss = 0,0
            for valid_batch in dl_valid:

                # validation vars
                x_b_val = valid_batch[0]
                y_b_val = valid_batch[1]
                y_b_pred_val, x_scores_val = self.predict(x_b_val)
                total_correct += self.evaluate_accuracy(y_b_val, y_b_pred_val)
                val_b_counter += 1
                # get loss
                average_loss += loss_fn.loss(x_b_val, y_b_val, x_scores_val, y_b_pred_val)


            valid_res.accuracy.append(total_correct / val_b_counter)
            valid_res.loss.append(average_loss / val_b_counter)
            # ========================
            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        var = self.weights[1:,:] if has_bias else self.weights # skip the first feature if has_bais
        w_images = var.T.view((self.n_classes, *img_shape))

        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0.0, learn_rate=0.0, weight_decay=0.0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    hp['weight_std'] = 0.1
    hp['learn_rate'] = 0.1
    hp['weight_decay'] = 0.001
    # ========================

    return hp
