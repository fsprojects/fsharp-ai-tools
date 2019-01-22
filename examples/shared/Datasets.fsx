module Datasets

module MNIST = 
    //CVDF mirror of http://yann.lecun.com/exdb/mnist/
    let baseUrl = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    let trainImages = "train-images-idx3-ubyte.gz"
    let trainLabels = "train-labels-idx1-ubyte.gz"
    let testImages  = "t10k-images-idx3-ubyte.gz"
    let testLabels  = "t10k-labels-idx1-ubyte.gz"
    let allFiles = [|trainImages; trainLabels; testImages; testLabels|]
