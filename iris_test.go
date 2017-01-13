package gofeedforward

import "fmt"

// Example demonstrates how to use the package to train a classifer on the
// venerable Iris data set.
func Example() {
	IrisData.Scale(0, 1, 2)
	IrisData.Scale(3)

	n := MakeNetwork(4, 6, 3)
	n.Randomize()

	IrisData.Shuffle(10)
	training, tv, err := IrisData.Split(0.6667)
	if err != nil {
		fmt.Printf("Failed to split iris training set: %v", err)
	}
	if err != nil {
		fmt.Printf("Failed to split test and valiation: %v", err)
	}

	trainer := Trainer{Alpha: 0.1, BatchUpdate: true, ShuffleRounds: 1}
	trainer.AddSimpleStoppingCriteria(10000, 0.1)
	trainer.Train(&n, training)

	classError, err := ClassificationError(n, tv, MakeBestOfClassifier([]string{"virginica", "versicolor", "setosa"}))
	if err != nil {
		fmt.Printf("Failed to get classification error: %v", err)
	}

	if classError > 0.10 {
		fmt.Printf("Failed to properly train network")
	}
}
