/*
BSD 2-Clause License

Copyright (c) 2016, Darc Inc
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Package gofeedforward provides simple feed forward neural network
// evaluation and training.  It uses fully connected networks and the sigmoid
// transfer function.
//
// Creating a network uses the MakeNetwork(size ...int) function to create
// a fully connected, feed forward network with zero values for all the
// weights.  The layers are biased, so a 2, 4, 1 network will create a
// network with 17 trainable weights.  ((2 + 1) x 4 + (4 + 1) x 1).
//
// Assuming you have training data already loaded into an array of TrainingDatum
// structs with inputs and expected values, the process of training
// the network is fairly simple.
//
// td := GetMyTrainingDataFromSomewhere()
// net := MakeNetwork(2, 4, 1)
// net.Randomize()
//
// trainer := Trainer{}
// trainer.AddSimpleStoppingCriteria(50000, 0.001)
//
// err := trainer.Train(&net, td)
package gofeedforward

import "fmt"

// Network represents a neural network.  It is composed of its layers and the
// output from the last inputs presented.  The last output value is important
// training but may also be useful in other contexts.
type Network struct {
	Layers  []Layer
	Outputs []float64
}

// BasicClassifier is the signature for a classifer that turns an array of
// network outputs into an array of class names.  It returns an array of
// class names to cover the possiblity that a valid translation of network
// outputs to classes may produce mutliple classes.
type BasicClassifier func([]float64) ([]string, error)

// MakeNetwork returns a neural network with the given size layers.  For example, a 2, 4, 1
// network will have two inputs, 4 hidden layer neurons, and 1 output neurons.
// There is no upper limit on the network size but the larger the network, the more
// difficult it is to train.  It may also be the case that a larger network is
// overfitted to the problem and may fail to generalize.
func MakeNetwork(sizes ...int) Network {
	result := Network{}
	for i := 1; i < len(sizes); i++ {
		inputs := sizes[i-1]
		outputs := sizes[i]
		result.Layers = append(result.Layers, MakeLayer(inputs, outputs))
	}
	return result
}

// Randomize updates the weights in the network to random values between 0.5 and
// -0.5.  This uses Go's built in random number generated without any
// initialization.  It is recommended that the random number generator be
// initialized prior to randomizing the network.
func (n *Network) Randomize() {
	for _, layer := range n.Layers {
		layer.Randomize()
	}
}

// Process takes the given input and produces a set of outputs for the network.
// It returns the output and any error, retaining a copy of the output in
// the network.
func (n *Network) Process(inputs []float64) ([]float64, error) {
	n.Outputs = nil
	var err error
	temp := inputs
	for idx := range n.Layers {
		temp, err = n.Layers[idx].Process(temp)
		if err != nil {
			return nil, err
		}
	}
	n.Outputs = make([]float64, len(temp))
	copy(n.Outputs, temp)
	return n.Outputs, nil
}

// InputSize returns the network input size.  When presenting data
// to the network, the array of values must be exactly this size.
func (n Network) InputSize() int {
	return n.Layers[0].Weights.InputSize() - 1
}

// OutputSize returns the network output size.  The network will
// always return a vector of this size.
func (n Network) OutputSize() int {
	return n.Layers[len(n.Layers)-1].Weights.OutputSize()
}

// Classify executes the network returning the output as a class
// member instead of a raw numeric value.
func (n *Network) Classify(inputs []float64, classifer BasicClassifier) ([]string, error) {
	rawValues, err := n.Process(inputs)

	if err != nil {
		return nil, err
	}

	return classifer(rawValues)
}

// MakeBestOfClassifier creates a classifier that translates a class to the highest scoring
// value in the array of values passed in.  Only returns one value.
func MakeBestOfClassifier(classes []string) BasicClassifier {
	return func(rawValues []float64) ([]string, error) {
		if len(rawValues) != len(classes) {
			return nil, fmt.Errorf("Unable to classify becuase there are %d outputs and %d classes", len(rawValues), len(classes))
		}

		maxIdx := 0
		maxValue := rawValues[0]
		for idx, val := range rawValues {
			if val > maxValue {
				maxIdx = idx
				maxValue = val
			}
		}

		return []string{classes[maxIdx]}, nil
	}
}

// MakeThresholdClassifier creates a classifier that returns a class for each input above
// the given threshold.  It can return 0, 1 or many classes, depending on how many of the
// inputs are above the threshold.
func MakeThresholdClassifier(classes []string, threshold float64) BasicClassifier {
	return func(rawValues []float64) ([]string, error) {
		if len(rawValues) != len(classes) {
			return nil, fmt.Errorf("Unable to classify becuase there are %d outputs and %d classes", len(rawValues), len(classes))
		}

		result := []string{}
		for idx, val := range rawValues {
			if val > threshold {
				result = append(result, classes[idx])
			}
		}
		return result, nil
	}
}
