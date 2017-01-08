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

import "testing"

func xorData() TrainingData {
	return TrainingData{
		TrainingDatum{Inputs: []float64{1.0, 0.0}, Expected: []float64{1.0}},
		TrainingDatum{Inputs: []float64{0.0, 1.0}, Expected: []float64{1.0}},
		TrainingDatum{Inputs: []float64{0.0, 0.0}, Expected: []float64{0.0}},
		TrainingDatum{Inputs: []float64{1.0, 1.0}, Expected: []float64{0.0}},
	}
}

func TestCalculateUpdate(t *testing.T) {
	layer := Layer{Weights: MakeCore(3, 1), Inputs: []float64{0.5, 0.5}}
	deltas := []float64{0.25, 0.25}

	updates := calculateUpdate(layer, deltas, 1.0)
	for _, row := range updates {
		for _, val := range row {
			if !outOfBoundsCheck(0.0, val, 0.001) {
				t.Errorf("The bounds check should not be zero for calculateUpdate")
			}
		}
	}
}

func TestTrainer_OneIteration(t *testing.T) {
	td := xorData()

	net := MakeNetwork(2, 3, 1)
	net.Randomize()

	trainer := Trainer{}
	sse, err := trainer.OneIteration(&net, td)
	if err != nil {
		t.Errorf("Failed to train network: %v", err)
	}

	if !outOfBoundsCheck(sse.Combine(), 0.0, 0.001) {
		t.Errorf("Should have returned some not zero value")
	}
}

func TestTrainer_Train(t *testing.T) {
	td := xorData()

	net := MakeNetwork(2, 4, 1)
	net.Randomize()

	trainer := Trainer{}
	trainer.AddSimpleStoppingCriteria(50000, 0.001)

	err := trainer.Train(&net, td)
	if err != nil {
		t.Errorf("Error during training: %v", err)
	}
}

func TestTrainer_TrainBatch(t *testing.T) {
	td := xorData()
	net := MakeNetwork(2, 4, 1)
	net.Randomize()

	trainer := Trainer{BatchUpdate: true, ShuffleRounds: 3}
	trainer.AddSimpleStoppingCriteria(50000, 0.001)

	err := trainer.Train(&net, td)
	if err != nil {
		t.Errorf("Error during training: %v", err)
	}
}

func TestTrainingData_Shuffle(t *testing.T) {
	td := TrainingData{
		TrainingDatum{Expected: []float64{0.0}, Inputs: []float64{0.0}},
		TrainingDatum{Expected: []float64{1.0}, Inputs: []float64{1.0}},
		TrainingDatum{Expected: []float64{2.0}, Inputs: []float64{2.0}},
		TrainingDatum{Expected: []float64{3.0}, Inputs: []float64{3.0}},
		TrainingDatum{Expected: []float64{4.0}, Inputs: []float64{4.0}},
	}

	td.Shuffle(10)
	if td[0].Inputs[0] < td[1].Inputs[0] && td[1].Inputs[0] < td[2].Inputs[0] && td[2].Inputs[0] < td[3].Inputs[0] {
		t.Error("The order was not disturbed")
	}
}

func TestEvaluate(t *testing.T) {
	td := TrainingData{
		TrainingDatum{Inputs: []float64{1.0, 0.0}, Expected: []float64{0.5, 0.5}},
		TrainingDatum{Inputs: []float64{2.0, 0.0}, Expected: []float64{0.4, 0.4}},
	}

	network := MakeNetwork(2, 2)
	allErrors, err := Evaluate(network, td)

	if err != nil {
		t.Errorf("Error evaluating network: %v", err)
	}

	if len(allErrors) != 2 {
		t.Errorf("There should be two errors after evaluation")
	}

	if outOfBoundsCheck(0.0, allErrors[0][0], 0.001) || outOfBoundsCheck(0.0, allErrors[0][1], 0.001) {
		t.Errorf("Error values are out of bounds")
	}

	if outOfBoundsCheck(0.01, allErrors[1][0], 0.001) || outOfBoundsCheck(0.01, allErrors[1][1], 0.001) {
		t.Errorf("Errors were not what was expected.")
	}
}

func TestTrainingData_Split(t *testing.T) {
	td := TrainingData{
		TrainingDatum{Inputs: []float64{1.0}, Expected: []float64{1.0}},
		TrainingDatum{Inputs: []float64{2.0}, Expected: []float64{2.0}},
		TrainingDatum{Inputs: []float64{3.0}, Expected: []float64{3.0}},
		TrainingDatum{Inputs: []float64{4.0}, Expected: []float64{4.0}},
		TrainingDatum{Inputs: []float64{5.0}, Expected: []float64{5.0}},
	}

	left, right, _ := td.Split(0.5)
	if len(left) != 3 {
		t.Errorf("There should have been 3 in the left side, but got: %d", len(left))
	}

	if len(right) != 2 {
		t.Errorf("There should have been 2 in the right side, but got: %d", len(right))
	}
}
