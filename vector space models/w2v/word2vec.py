#!/usr/bin/env python

import argparse
import numpy as np
import random
from utils.gradcheck import gradcheck_naive, grad_tests_softmax, grad_tests_negsamp
from utils.utils import normalizeRows, softmax



def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

def naiveSoftmaxLossAndGradient(centerWordVec,outsideWordIdx,outsideVectors,dataset):

    #loss
    y_hat = softmax(np.dot(outsideVectors, centerWordVec))
    loss = -np.log(y_hat[outsideWordIdx]) #outsideindex (o of uo) 
    # Compute gradients
    delta = y_hat.copy()
    delta[outsideWordIdx] -= 1 
    
    gradCenterVec = np.dot(outsideVectors.T, delta)
    gradOutsideVecs = np.outer(delta, centerWordVec)
    return loss, gradCenterVec, gradOutsideVecs

def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(centerWordVec,outsideWordIdx,outsideVectors,dataset,K=10):
    

    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    loss = 0.0
    gradCenterVec = np.zeros_like(centerWordVec)
    gradOutsideVecs = np.zeros_like(outsideVectors)

    # Compute loss and gradients for the positive sample
    u_o = outsideVectors[outsideWordIdx]
    z = np.dot(u_o, centerWordVec)
    loss -= np.log(sigmoid(z))
    gradCenterVec += (sigmoid(z) - 1) * u_o
    gradOutsideVecs[outsideWordIdx] += (sigmoid(z) - 1) * centerWordVec

    # Compute loss and gradients for the negative samples
    for k in range(K):
        neg_idx = indices[k + 1]
        u_k = outsideVectors[neg_idx]
        z = np.dot(u_k, centerWordVec)
        loss -= np.log(sigmoid(-z))
        gradCenterVec += sigmoid(z) * u_k
        gradOutsideVecs[neg_idx] += sigmoid(z) * centerWordVec
    
    return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,centerWordVectors, outsideVectors, dataset, word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    
    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

   
    for outside_word in outsideWords:
        outside_word_idx = word2Ind[outside_word]
        current_loss, current_gradCenterVec, current_gradOutsideVecs = word2vecLossAndGradient(
                                centerWordVec=centerWordVectors[word2Ind[currentCenterWord]],
                                outsideWordIdx=outside_word_idx,
                                outsideVectors=outsideVectors,
                                dataset=dataset)
        loss += current_loss
        gradCenterVecs[word2Ind[currentCenterWord]] += current_gradCenterVec.reshape(-1)
        gradOutsideVectors += current_gradOutsideVecs

    
    return loss, gradCenterVecs, gradOutsideVectors


#############################################
#         Testing functions below           # 
#############################################

def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset,
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N/2),:]
    outsideVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad[:int(N/2), :] += gin / batchsize
        grad[int(N/2):, :] += gout / batchsize

    return loss, grad

def test_sigmoid():
    """ Test sigmoid function """
    print("=== Sanity check for sigmoid ===")
    assert sigmoid(0) == 0.5
    assert np.allclose(sigmoid(np.array([0])), np.array([0.5]))
    assert np.allclose(sigmoid(np.array([1,2,3])), np.array([0.73105858, 0.88079708, 0.95257413]))
    print("Tests for sigmoid passed!")

def getDummyObjects():
    """ Helper method for naiveSoftmaxLossAndGradient and negSamplingLossAndGradient tests """

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]

    dataset = type('dummy', (), {})()
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    return dataset, dummy_vectors, dummy_tokens

def test_naiveSoftmaxLossAndGradient():
    """ Test naiveSoftmaxLossAndGradient """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for naiveSoftmaxLossAndGradient ====")
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = naiveSoftmaxLossAndGradient(vec, 1, dummy_vectors, dataset)
        return loss, gradCenterVec
    gradcheck_naive(temp, np.random.randn(3), "naiveSoftmaxLossAndGradient gradCenterVec")

    centerVec = np.random.randn(3)
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = naiveSoftmaxLossAndGradient(centerVec, 1, vec, dataset)
        return loss, gradOutsideVecs
    gradcheck_naive(temp, dummy_vectors, "naiveSoftmaxLossAndGradient gradOutsideVecs")

def test_negSamplingLossAndGradient():
    """ Test negSamplingLossAndGradient """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for negSamplingLossAndGradient ====")
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = negSamplingLossAndGradient(vec, 1, dummy_vectors, dataset)
        return loss, gradCenterVec
    gradcheck_naive(temp, np.random.randn(3), "negSamplingLossAndGradient gradCenterVec")

    centerVec = np.random.randn(3)
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = negSamplingLossAndGradient(centerVec, 1, vec, dataset)
        return loss, gradOutsideVecs
    gradcheck_naive(temp, dummy_vectors, "negSamplingLossAndGradient gradOutsideVecs")

def test_skipgram():
    """ Test skip-gram with naiveSoftmaxLossAndGradient """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient),
        dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")
    grad_tests_softmax(skipgram, dummy_tokens, dummy_vectors, dataset)

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient),
        dummy_vectors, "negSamplingLossAndGradient Gradient")
    grad_tests_negsamp(skipgram, dummy_tokens, dummy_vectors, dataset, negSamplingLossAndGradient)

def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    test_sigmoid()
    test_naiveSoftmaxLossAndGradient()
    test_negSamplingLossAndGradient()
    test_skipgram()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test your implementations.')
    parser.add_argument('function', nargs='?', type=str, default='all',
                        help='Name of the function you would like to test.')

    args = parser.parse_args()
    if args.function == 'sigmoid':
        test_sigmoid()
    elif args.function == 'naiveSoftmaxLossAndGradient':
        test_naiveSoftmaxLossAndGradient()
    elif args.function == 'negSamplingLossAndGradient':
        test_negSamplingLossAndGradient()
    elif args.function == 'skipgram':
        test_skipgram()
    elif args.function == 'all':
        test_word2vec()
