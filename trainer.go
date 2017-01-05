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

import "fmt"

// IterationCallback is a function prototype for a callback that, when registered, is called
// at the end of every training iteration.  Multiple callbacks can be registered.
type IterationCallback func(*Trainer, SumOfSquaredErrors, int, error)

// TrainingCallback is a function prototype for a callback that, when registered, is called
// at the start or end of the training.  Multiple callbacks can be registered.
type TrainingCallback func(*Trainer)

// Trainer is a network trainer that trains a network.  The Alpha is the learning rate
// and has a default of 0.1.  BatchUpdate indicates if updates should occur in a batch or
// with each presentation.
type Trainer struct {
	endOfIterationHandlers []IterationCallback
	startTrainingHandlers  []TrainingCallback
	endTrainingHandlers    []TrainingCallback
	requestTerminate       bool
	Alpha                  float64
	BatchUpdate            bool
}

// TrainingDatum is a training example and is composed of a set of inputs and the
// expected network outputs.
type TrainingDatum struct {
	Inputs   []float64
	Expected []float64
}

// TrainingData is a collection of training datum.
type TrainingData []TrainingDatum

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
func (t Trainer) OneIteration(net *Network, data TrainingData) (SumOfSquaredErrors, error) {
	deltas := [][]float64{}
	updates := []Core{}
	for _, layer := range net.Layers {
		deltas = append(deltas, make([]float64, layer.Weights.OutputSize()))
		updates = append(updates, MakeCore(layer.Weights.InputSize(), layer.Weights.OutputSize()))
	}

	total := SumOfSquaredErrors(make([]float64, net.OutputSize()))

	for _, datum := range data {
		outputs, err := net.Process(datum.Inputs)
		if err != nil {
			return nil, err
		}

		if len(datum.Expected) != len(outputs) {
			return nil, fmt.Errorf("Failed to processes data with length %d against expected output of length %d",
				len(datum.Expected), len(outputs))
		}

		sse, _ := SumOfSquaredError(datum.Expected, outputs)
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
	t.AddIterationEndHandler(func(t *Trainer, mse SumOfSquaredErrors, iter int, err error) {
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
