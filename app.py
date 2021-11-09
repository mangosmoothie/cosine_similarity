"""
Author: Nathan Lloyd

Demo for using cosine similarity to compare sentence vectors

to run you  must install packages then use spacy to download
an nlp model

$ python -m pip install numpy matplotlib spacy
$ spacy download en_core_web_sm

Example usage:

>> run([1,2], [3, 5])

>> vec_a = nlp('I like pizza.').vector
>> vec_b = nlp('I like tomatoes.').vector
>> cosine_similarity(vec_a, vec_b)
Cosine Similarity between A and B:0.8697508573532104
"""
import numpy as np
import matplotlib.pyplot as plt
import spacy


def magnitude(v):
    return np.linalg.norm(v)


def make_plot(v1, v2, min_plot=0):
    a = np.array(v1)
    b = np.array(v2)

    ax = plt.axes()
    ax.arrow(0.0, 0.0, a[0], a[1], head_width=0.4, head_length=0.5)
    plt.annotate(f"A({a[0]},{a[1]})", xy=(a[0], a[1]), xytext=(a[0] + 0.5, a[1]))
    ax.arrow(0.0, 0.0, b[0], b[1], head_width=0.4, head_length=0.5)
    plt.annotate(f"B({b[0]},{b[1]})", xy=(b[0], b[1]), xytext=(b[0] + 0.5, b[1]))
    plt.xlim(min_plot, 10)
    plt.ylim(min_plot, 10)
    plt.show()
    plt.close()


def cosine_similarity(a, b):
    """
    calculate and print cosine similarity for two vectors: a & b
    """
    cos_sim = np.dot(a, b) / (magnitude(a) * magnitude(b))
    print(f"Cosine Similarity between A and B:{cos_sim}")


def run(a, b, min_plot=0):
    """
  Make two 2-D vectors - plot them and compare them with cosine similarity

    :param a: 2-D vector
    :param b: 2-D vector
    :param min_plot: min window on plot
    """
    make_plot(a, b, min_plot)
    cosine_similarity(a, b)


nlp = spacy.load('en_core_web_sm')