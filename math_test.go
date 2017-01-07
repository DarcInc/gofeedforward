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
	"math"
	"testing"
)

func outOfBoundsCheck(expected, actual, bound float64) bool {
	return math.Abs(expected-actual) > bound
}

func TestSigmoid(t *testing.T) {
	if outOfBoundsCheck(0.5, Sigmoid(0.0), 0.001) {
		t.Errorf("Invalid sigmoid result.  Expected 0.5 but got %0.5f", Sigmoid(0.0))
	}

	if outOfBoundsCheck(0.99, Sigmoid(10.0), 0.01) {
		t.Errorf("Sigmoid high value.  Expected 0.99 but got %0.5f", Sigmoid(10.0))
	}

	if outOfBoundsCheck(0.00, Sigmoid(-10.0), 0.01) {
		t.Errorf("Sigmoid low value. Expected 0.00 but got %0.5f", Sigmoid(-10.0))
	}
}

func TestDotProduct(t *testing.T) {
	left := []float64{1.0, 1.0, 1.0}
	right := []float64{1.0, 2.0, 3.0}

	actual, _ := DotProduct(left, right)
	if outOfBoundsCheck(6.0, actual, 0.01) {
		t.Errorf("DotProduct value incorrect, expected 6.0 but got %0.2f", actual)
	}
}

func TestDotProductUnequal(t *testing.T) {
	left := []float64{1.0, 1.0}
	right := []float64{1.0, 2.0, 3.0}

	_, err := DotProduct(left, right)

	if err == nil {
		t.Error("DotProduct did not report error, expected err got nil")
	}
}

func TestCalcError(t *testing.T) {
	expected := []float64{1.0, 2.0}
	actual := []float64{3.0, 3.0}

	value, err := CalcError(expected, actual)
	if err != nil {
		t.Errorf("Failed to calculate SSE: %v", err)
	}

	if outOfBoundsCheck(value.Combine(), 5.0, 0.001) {
		t.Errorf("Expected error to be 5.0 but got %0.4f", value.Combine())
	}
}

func TestAllErrors_Total(t *testing.T) {
	ae := AllErrors{
		SquaredError{1.0, 2.0},
		SquaredError{3.0, 4.0},
	}

	se := ae.Total()
	if outOfBoundsCheck(4.0, se[0], 0.001) || outOfBoundsCheck(6.0, se[1], 0.001) {
		t.Errorf("Invalid total error")
	}
}

func TestAllErrors_Average(t *testing.T) {
	ae := AllErrors{
		SquaredError{1.0, 2.0},
		SquaredError{3.0, 4.0},
	}

	se := ae.Average()
	if outOfBoundsCheck(2.0, se[0], 0.001) || outOfBoundsCheck(3.0, se[1], 0.001) {
		t.Errorf("Invalid average")
	}
}
