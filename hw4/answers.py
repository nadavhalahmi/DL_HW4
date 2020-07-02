r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(batch_size=32,
              gamma=0.99,
              beta=0.5,
              learn_rate=0.0015,
              eps=1e-8,
              )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(batch_size=32,
              gamma=0.99,
              beta=0.9,
              delta=1.,
              learn_rate=1e-3,
              eps=1e-8,
              )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======
    hp['batch_size'] = 8
    hp['learn_rate'] = 0.002
    hp['hidden_dim'] = 128
    hp['gamma'] = 0.945
    hp['beta'] = 0.9
    hp['delta'] = 0.004
    # ========================
    return hp


part1_q1 = r"""
**Your answer:**

Subtracting baseline helps reducing the variance if the baseline is defined wisely. For instance, in our case we 
defined the baseline as the mean of all q-values which is the mean of all estimated state-values. Choosing it that way, 
helps us normalize the results of the loss based on the current state such that the results of it are closer to 0.
This makes the model more stable and with less noise.

An example for a situation where it can help is when there is a safe way to land the spaceship. In this case we would 
like to prior this way and by using baseline which helps to avoid a sequence of "bad" actions and makes the decision 
more independent.


"""


part1_q2 = r"""
**Your answer:**

We always get a valid approximation thanks to the connection between the terms $v_\pi(s)$ and $q_\pi(s,a)$. 
Actually, $v_\pi(s)$ can be expressed as $\sum_{a \in A}\pi(a|s)q_\pi(s,a)$. It means that $v_\pi(s)$ is actually 
takes into consideration all the possible q-values given state $s$ with respect to the agent behavior. 
So defining the baseline that way gives us good results since q-values close to this baseline will lead to lower 
loss values just like we want.
The value of the loss in this case is actually the lowest when the value estimator is equal to the expected 
value of the loss of all trajectories from the given state.

"""


part1_q3 = r"""
**Your answer:**

1. As we can see, the bpg and cpg perforemed the best way,
and where very stable thanks to the baseline. epg and vpg perforemed
in a similar way, but were very sensitive, and at some point,
exploded. when epg converges to 0, we can conclude that we slighly
over-fitted, and the action is poorly randomized (the entropy
is low).
2. AAC got just a bit better results than cpg, during training and "playing"
with AAC, it felt like it's more stable than cpg and studying at
a high rate.

"""
