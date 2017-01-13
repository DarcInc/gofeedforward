package gofeedforward

import "fmt"

// Example demonstrates how to use the package to train a classifer on the
// venerable Iris data set.
func Example() {

	// Scale the raw data so that the input values are within the 0.0 to 1.0
	// range.  Elements 0, 1, and 2 are all related so they should all be
	// normalized together while 3 is an independant element and scaled
	// separately.
	IrisData.Scale(0, 1, 2)
	IrisData.Scale(3)

	// Create a new network and set the weights to random values between
	// -0.5 and 0.5.
	n := MakeNetwork(4, 6, 3)
	n.Randomize()

	// Shuffle the input data a few times to randomize the order and
	// split it into a test and evaluation set and a traing set.
	IrisData.Shuffle(10)
	training, tv, err := IrisData.Split(0.6667)
	if err != nil {
		fmt.Printf("Failed to split iris training set: %v", err)
	}
	if err != nil {
		fmt.Printf("Failed to split test and valiation: %v", err)
	}

	// Create a new trainer and train the network.  Stop at the end of
	// 10,000 iterations or when the mean squared error drops below 0.1.
	trainer := Trainer{Alpha: 0.1, BatchUpdate: true, ShuffleRounds: 1}
	trainer.AddSimpleStoppingCriteria(10000, 0.1)
	trainer.Train(&n, training)

	// Check the classification error using the test and evaluation
	// data.
	classError, err := ClassificationError(n, tv, MakeBestOfClassifier([]string{"virginica", "versicolor", "setosa"}))
	if err != nil {
		fmt.Printf("Failed to get classification error: %v", err)
	}

	if classError > 0.10 {
		fmt.Printf("Failed to properly train network")
	}
}
