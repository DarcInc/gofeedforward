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

package gofeedforward

import (
	"fmt"
	"math"
	"math/rand"
)

// IterationCallback is a function prototype for a callback that, when registered, is called
// at the end of every training iteration.  Multiple callbacks can be registered.
type IterationCallback func(*Trainer, SquaredError, int, error)

// TrainingCallback is a function prototype for a callback that, when registered, is called
// at the start or end of the training.  Multiple callbacks can be registered.
type TrainingCallback func(*Trainer)

// Trainer is a network trainer that trains a network.  The Alpha is the learning rate
// and has a default of 0.1.  BatchUpdate indicates if updates should occur in a batch or
// with each presentation.  ShuffleRounds indicates the number of rounds to shuffle the
// training data before presenting it to the network.
type Trainer struct {
	endOfIterationHandlers []IterationCallback
	startTrainingHandlers  []TrainingCallback
	endTrainingHandlers    []TrainingCallback
	requestTerminate       bool
	Alpha                  float64
	BatchUpdate            bool
	ShuffleRounds          int
}

// TrainingDatum is a training example and is composed of a set of inputs and the
// expected network outputs.
type TrainingDatum struct {
	Inputs   []float64
	Expected []float64
}

// TrainingData is a collection of training datum.
type TrainingData []TrainingDatum

// Shuffle is a very crude shuffling algorithm which randomizes the order
// of the training data.
func (td TrainingData) Shuffle(rounds int) {
	for roundIdx := 0; roundIdx < rounds; roundIdx++ {
		for rowIdx := 0; rowIdx < len(td); rowIdx++ {
			left := rand.Int() % len(td)
			right := rand.Int() % len(td)

			td[left], td[right] = td[right], td[left]
		}
	}
}

// Scale takes a set of input columns and scales them from 0 to 1.0 where 0 is
// the minimum value found in the collection of columns and 1.0 is the maximum
// value.  For example, Scale(0, 2) will scale the contents of the training data
// inputs in column 0 and column 2, so that the max of column 0 or 2 is 1.0 and
// the min of either column 0 or 2 is 0.0.
func (td TrainingData) Scale(cols ...int) error {
	for _, col := range cols {
		if col >= len(td[0].Inputs) || col < 0 {
			return fmt.Errorf("Unable to scale column %d", col)
		}
	}

	low := td[0].Inputs[cols[0]]
	high := td[0].Inputs[cols[0]]

	for idx := range td {
		for _, col := range cols {
			if td[idx].Inputs[col] < low {
				low = td[idx].Inputs[col]
			}

			if td[idx].Inputs[col] > high {
				high = td[idx].Inputs[col]
			}
		}
	}

	for idx := range td {
		for _, col := range cols {
			td[idx].Inputs[col] = (td[idx].Inputs[col] - low) / (high - low)
		}
	}

	return nil
}

// Split divides the training data such that at least the given percentage winds up on the
// left hand side and the remainder on the right hand side.  For example, if there are 15
// elements in the training data and the fraction in 0.5, 8 will wind up in the left and
// 7 will wind up in the right.
func (td TrainingData) Split(fraction float64) (TrainingData, TrainingData, error) {
	if fraction > 1.0 || fraction < 0.0 {
		return nil, nil, fmt.Errorf("Spliting data requires a fraction between 0.0 and 1.0, not: %0.4f", fraction)
	}
	leftCount := int(math.Ceil(float64(len(td)) * fraction))
	return td[:leftCount], td[leftCount:], nil
}

func calculateDeltas(nextDeltas []float64, layer Layer) []float64 {
	thisDeltas := make([]float64, len(layer.Inputs))
	for nextLayerInputIdx := range layer.Inputs {
		sum := 0.0
		for nextDeltaIdx := range nextDeltas {
			for weightIdx := range layer.Weights {
				sum += nextDeltas[nextDeltaIdx] * layer.Weights[weightIdx][nextLayerInputIdx]
			}
		}
		thisDeltas[nextLayerInputIdx] = sum * layer.Inputs[nextLayerInputIdx] * (1 - layer.Inputs[nextLayerInputIdx])
	}
	return thisDeltas
}

func calculateUpdate(layer Layer, deltas []float64, alpha float64) Core {
	result := MakeCore(layer.Weights.InputSize(), layer.Weights.OutputSize())
	biasedInputs := append(layer.Inputs, 1.0)
	for row := range layer.Weights {
		for col := range layer.Weights[row] {
			result[row][col] = biasedInputs[col] * deltas[row] * -alpha
		}
	}
	return result
}

// OneIteration conducts a training iteration.  It takes  a network and some training data and
// returns the mean squared error array for all the network outputs.
func (t Trainer) OneIteration(net *Network, data TrainingData) (SquaredError, error) {
	deltas := [][]float64{}
	updates := []Core{}
	for _, layer := range net.Layers {
		deltas = append(deltas, make([]float64, layer.Weights.OutputSize()))
		updates = append(updates, MakeCore(layer.Weights.InputSize(), layer.Weights.OutputSize()))
	}

	if t.ShuffleRounds > 0 {
		data.Shuffle(t.ShuffleRounds)
	}

	total := SquaredError(make([]float64, net.OutputSize()))

	for _, datum := range data {
		outputs, err := net.Process(datum.Inputs)
		if err != nil {
			return nil, err
		}

		if len(datum.Expected) != len(outputs) {
			return nil, fmt.Errorf("Failed to processes data with length %d against expected output of length %d",
				len(datum.Expected), len(outputs))
		}

		sse, _ := CalcError(datum.Expected, outputs)
		total.Accumulate(sse)

		for i := 0; i < len(datum.Expected); i++ {
			deltas[len(net.Layers)-1][i] = (outputs[i] - datum.Expected[i]) * outputs[i] * (1 - outputs[i])
		}

		for i := len(net.Layers) - 2; i >= 0; i-- {
			deltas[i] = calculateDeltas(deltas[i+1], net.Layers[i+1])
		}

		for i := 0; i < len(net.Layers); i++ {
			update := calculateUpdate(net.Layers[i], deltas[i], t.Alpha)

			if !t.BatchUpdate {
				net.Layers[i].UpdateWeights(update)
			} else {
				update, err = updates[i].Add(update)
				updates[i] = update
			}
			if err != nil {
				return nil, err
			}
		}
	}

	if t.BatchUpdate {
		for idx := range net.Layers {
			net.Layers[idx].UpdateWeights(updates[idx])
		}
	}
	total.Average(len(data))
	return total, nil
}

// AddIterationEndHandler adds an end of iteration callback function.
func (t *Trainer) AddIterationEndHandler(handler IterationCallback) {
	t.endOfIterationHandlers = append(t.endOfIterationHandlers, handler)
}

// AddTrainingBeginHandler adds a callback to be executed when training starts
func (t *Trainer) AddTrainingBeginHandler(handler TrainingCallback) {
	t.startTrainingHandlers = append(t.startTrainingHandlers, handler)
}

// AddTrainingEndHandler adds a callback to be executed when training ends
func (t *Trainer) AddTrainingEndHandler(handler TrainingCallback) {
	t.endTrainingHandlers = append(t.endTrainingHandlers, handler)
}

// RequestTermination is called to break the training loop
func (t *Trainer) RequestTermination() {
	t.requestTerminate = true
}

// AddSimpleStoppingCriteria registers an end of iteration callback that will check to see
// if either the maximum iterations have been exceeded or the mean squared error is less
// than the minimum error.  If either of these conditions are met, then the callback
// requests termination.
func (t *Trainer) AddSimpleStoppingCriteria(maxIter int, minErr float64) {
	t.AddIterationEndHandler(func(t *Trainer, mse SquaredError, iter int, err error) {
		if iter > maxIter {
			t.RequestTermination()
		}

		if mse.Combine() < minErr {
			t.RequestTermination()
		}
	})
}

// Train conducts the training loop, taking a network and some training data.  It executes the
// start of training callbacks, then executes the training loop forever, unless termination is
// requested.  At the end of each iteration in calls the end of iteration callbacks.  When
// training is finished, it calls the end of training callbacks.
func (t *Trainer) Train(net *Network, td TrainingData) (err error) {
	if t.Alpha == 0.0 {
		t.Alpha = 0.1
	}

	if len(t.startTrainingHandlers) > 0 {
		for _, st := range t.startTrainingHandlers {
			st(t)
		}
	}

	iteration := 0
	for {
		iteration++
		mse, err := t.OneIteration(net, td)

		if len(t.endOfIterationHandlers) > 0 {
			for _, eoi := range t.endOfIterationHandlers {
				eoi(t, mse, iteration, err)
			}
		}

		if err != nil {
			return err
		}

		if t.requestTerminate {
			break
		}
	}

	if len(t.endTrainingHandlers) > 0 {
		for _, et := range t.endTrainingHandlers {
			et(t)
		}
	}

	return
}

// Evaluate a network returning the error values that can then be averaged
// or analyzed.  It executes the network for each example in the training
// data, returning the error for each example.
func Evaluate(net Network, td TrainingData) (AllErrors, error) {
	result := AllErrors{}
	for _, datum := range td {
		output, err := net.Process(datum.Inputs)
		if err != nil {
			return nil, err
		}

		se, err := CalcError(datum.Expected, output)
		if err != nil {
			return nil, err
		}
		result = append(result, se)
	}
	return result, nil
}

// ClassificationError calculates the error rate for a network that is used
// as a classifer.  The network and testing data are passed as the first two
// arguments.  The third argument is a classifer to translate the outputs to
// a specific class.  What is compared is the result of classifying the expected
// outputs vs classifying the actual outputs.
func ClassificationError(net Network, td TrainingData, classifier BasicClassifier) (float64, error) {
	failed := 0.0

	for _, datum := range td {
		expected, err := classifier(datum.Expected)
		if err != nil {
			return 0.0, err
		}

		outputs, err := net.Process(datum.Inputs)
		if err != nil {
			return 0.0, err
		}

		actual, err := classifier(outputs)
		if err != nil {
			return 0.0, err
		}

		if len(expected) != len(actual) {
			failed += 1.0
			continue
		}

		for idx := range expected {
			if actual[idx] != expected[idx] {
				failed += 1.0
				continue
			}
		}
	}

	return failed / float64(len(td)), nil
}
