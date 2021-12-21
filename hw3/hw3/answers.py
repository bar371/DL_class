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
    hypers['batch_size']=32
    hypers['seq_len']=78
    hypers['h_dim']=256
    hypers['n_layers']=3
    hypers['dropout']=0.3
    hypers['learn_rate']=0.05
    hypers['lr_sched_factor']=0.01
    hypers['lr_sched_patience']=1
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq='ACT 1'
    # start_seq = 'ACT 1: Juliet: Wherefore Art Thou Romeo'
    temperature = 0.00001
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**
"""

part1_q2 = r"""
**Your answer:**
"""

part1_q3 = r"""
**Your answer:**
"""

part1_q4 = r"""
**Your answer:**
"""
# ==============

