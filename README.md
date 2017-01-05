# Go Feed Forward
With the emphasis on simple, this is a feed forward libary implemented in Go.

## Theory of operation
A feed forward network is designed to classify inputs using the combination
of weighted sums of those inputs.  The weights for the sums are calculated 
from random initial values using backpropagation.  

backpropagation basically assigns credit to each weight for the amount of 
error that weighted contributed to the final error.  Using a combination
of the derivative of the error function and the magnitude of the weight, the weights
are updated to reduce the error between the inputs and expected outputs
from a set of known samples.

See [Backpropagation on Wikipedia](https://en.wikipedia.org/wiki/Backpropagation)
for a fairly detailed discussion of the principles of backpropagation.

## Creating a network
Creating a network is relatively simple.  The <code>MakeNetwork</code> function
creates a network and all its internal layers.

```golang
network := MakeNetwork(2, 4, 1)
network.Randomize()
```

This code will create a network with two input neurons, 1 output neuron, and 4
neurons in the hidden layer.  This is actually 15 weights, as both the hidden 
layer and the output layer are biased.  Once the network is created, it must be
randomized to set the neruons to a random value between -0.5 to 0.5.

## Training a network
The next step is training the network.  This requires a set of training examples
and some traing parameters (the most significant of which is alpha - the learning
rate).  For example, the "xor" function can be expressed by the following code:

```golang
td := TrainingData{
		TrainingDatum{Inputs: []float64{1.0, 0.0}, Expected: []float64{0.9}},
		TrainingDatum{Inputs: []float64{0.0, 1.0}, Expected: []float64{0.9}},
		TrainingDatum{Inputs: []float64{0.0, 0.0}, Expected: []float64{0.1}},
		TrainingDatum{Inputs: []float64{1.0, 1.0}, Expected: []float64{0.1}},
	}
```

Xor is a good demonstration of a training data set because the classes cannot be
linearly separated.  Perceptrons (or basically single layer networks) cannot 
learn non-linearly separable functions.  

Note that the expected values are in the range of 0.1 to 0.9.  That's because the 
output of the network is "squashed" by the transfer function (a differentiable 
function that keeps the output in a specifc range like 0.0 to 1.0).  This network uses
the standard [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) function so 
the output will never reach either 0.0 or 1.0.  Instead the expected values are 0.1 and
0.9.

The trainer is initialized using as a struct.  Training will run forever unless there
is a stoping critera defined.  As a convenience function, <code>AddSimpleStoppingCriteria</code>
will terminate training once either the number of iterations is reached or the mean
squared error falls below a certain theshold.  It is also possible to register an 
end of iteration callback and use some other critera for stopping.

```golang 
trainer := Trainer{}
trainer.AddSimpleStoppingCriteria(50000, 0.01)
trainer.Train(&network, td)
```

Finally, the call to <code>Train</code> will train the network. 