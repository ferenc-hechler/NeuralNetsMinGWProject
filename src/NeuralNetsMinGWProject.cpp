//============================================================================
// Name        : FerisFirstMinGWProject.cpp
// Author      : feri
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "neuralnetwork.h"
#include "platform/utils.h"
#include "neuralnets/Vect.h"
#include "neuralnets/NN.h"
#include "neuralnets/NNUtils.h"

using namespace std;


int main() {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!

	Vect hidden = createVect(2, 10.0f, 10.0f);

	nn::initfcnn(3, &hidden, 2);

	Vect x = createVect(3, 2.0f, 3.0f ,4.0f);
	Vect y = createVect(2, 1.0f, 0.0f);

	float err = nn::train(&x, &y);
	logFloat(err);

	nn::predict(&x, &y);
	logFloat(y.get(0));
	log(", ");
	logFloat(y.get(1));

	nn::getBrain()->print();

	log("\nfinished");
	return 0;
}
