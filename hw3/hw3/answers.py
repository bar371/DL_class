r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size']=128
    hypers['seq_len']=99
    hypers['h_dim']=128
    hypers['n_layers']=3
    hypers['dropout']=0.27
    hypers['learn_rate']=0.001
    hypers['lr_sched_factor']=0.1
    hypers['lr_sched_patience']=1

    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq='ACT 1'
    # start_seq = 'O Romeo, Romeo! wherefore art thou Romeo?'
    temperature = 0.00001
    # ========================
    return start_seq, temperature


part1_q1 = r"""
If the sequence is too large then the RNN can fall into the vanishing gradient problem. To avoid this we split the data into smaller sequences. Another reason is that using one sample can cause the model to overfit therefore we break it up into smaller sequences. """

part1_q2 = r"""
**Your answer:**
"""

part1_q3 = r"""
We don't shuffle the order while training because the beginning text affects the later text and we want the order to be maintained. 
"""

part1_q4 = r"""
1. The tempature hyperparameter affects the probablities of the chars from the softmax. During training we want the model to choose the most likely chars and by lowering the tempature we are giving a higher probability to those chars.  
2. When the tempature is very high there tends to be more mistakes but there is also more diversity. 
3. When we lower the tempature we our increasing our models's confidence but also causing it to be more conservative in it's samples because it sticks to the more probable possiblities. When the tempature approaches 0 it is most likely to get stuck in an infinite loop, this happens because your model becomes very confident and doesn't diversify the text. 
"""
# ==============

