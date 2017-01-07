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
)

// SquaredError represents the squared sum of the errors between actual and
// estimated observations.
type SquaredError []float64

// AllErrors is a complete set of squared errors.
type AllErrors []SquaredError

// Sigmoid is a standard sigmoid squashing function.  It will produce an output
// value between 0.0 and 1.0.  Very large positive inputs will produce a value very near 1.0
// and very large negative inputs will produce a value near 0.0.  The output
// of the Sigmoid at 0.0 is 0.5.
func Sigmoid(input float64) float64 {
	return 1.0 / (1.0 + math.Exp(-input))
}

// DotProduct calculates a dot-product of two arrays.  Both array must be of
// the same size.
func DotProduct(left, right []float64) (float64, error) {
	if len(left) != len(right) {
		return math.NaN(), fmt.Errorf("Dot product arguments are of different length: %d vs %d",
			len(left), len(right))
	}

	sum := 0.0
	for i := 0; i < len(left); i++ {
		sum += left[i] * right[i]
	}
	return sum, nil
}

// CalcError calculates the sum of squared errors for the expected and actual
// values.
func CalcError(expected, actual []float64) (SquaredError, error) {
	if len(expected) != len(actual) {
		return nil, fmt.Errorf("Expected length = %d actual length = %d", len(expected), len(actual))
	}

	sum := SquaredError(make([]float64, len(expected)))
	for i := 0; i < len(expected); i++ {
		diff := expected[i] - actual[i]
		sum[i] += diff * diff
	}

	return sum, nil
}

// Accumulate adds the sum of squares error to the given sum of squares error.
func (sse SquaredError) Accumulate(new SquaredError) {
	for i := 0; i < len(sse); i++ {
		sse[i] += new[i]
	}
}

// Average divides the sum of squares by a number of degrees of freedom.
func (sse SquaredError) Average(nexample int) {
	for i := 0; i < len(sse); i++ {
		sse[i] = sse[i] / float64(nexample)
	}
}

// Combine returns the sum of the sum of squares error.
func (sse SquaredError) Combine() float64 {
	sum := 0.0
	for i := 0; i < len(sse); i++ {
		sum += sse[i]
	}
	return sum
}

// WeightedCombination adds together the components of the sum of squared errors using the
// given weights.
func (sse SquaredError) WeightedCombination(weights []float64) (float64, error) {
	if len(weights) != len(sse) {
		return 0.0, fmt.Errorf("Number of weights %d do not equal number of values %d", len(weights), len(sse))
	}

	totalWeights := 0.0
	for i := 0; i < len(weights); i++ {
		totalWeights += weights[i]
	}

	sum := 0.0
	for i := 0; i < len(sse); i++ {
		sum += (weights[i] / totalWeights) * sse[i]
	}
	return sum, nil
}

// Total computes the total error in a set of all errors.
func (s AllErrors) Total() SquaredError {
	sum := SquaredError(make([]float64, len(s[0])))
	for _, val := range s {
		sum.Accumulate(val)
	}
	return sum
}

// Average computes the average for each error in a set of errors.
func (s AllErrors) Average() SquaredError {
	avg := s.Total()
	for idx := range avg {
		avg[idx] = avg[idx] / float64(len(s))
	}
	return avg
}
