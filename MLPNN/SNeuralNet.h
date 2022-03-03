#pragma once

#ifndef SNEURALNET_H
#define SNEURALNET_H

#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <fstream>
#include <sstream>
#include <set>
#include <time.h>
#include "UnitNeurone.h"

using namespace std;

class Smodel
{

private:

	vector<vector<Neurone>> fullyconnectedLdata;
	vector<string> fullyactivations;
	vector<vector<vector<double>>> fMW;
	vector<vector<double>> fMB;
	vector<vector<vector<double>>> fVW;
	vector<vector<double>> fVB;

	double relu_ops(const double val);
	double Lrelu_ops(const double val);
	double SquaredError(const double targetval, const double predictedval);
	double sigmoid(const double& val);
	void softmax_ops(const int i);
	double fullyconnectedforwardpropagation(const int q, vector<vector<double>>& tdata, const string& loss_function, set<string>& fns, string fn);
	void fullyconnectedbackpropagation(vector<vector<vector<double>>>& weights, vector<vector<double>>& biases);
	void resize_W_B(vector<vector<vector<double>>>& weights, vector<vector<double>>& biases);
	void update_All_weights_biases(vector<vector<vector<double>>>& weights, vector<vector<double>>& biases, const string& optimizer, const double iteration, const double learning_rate, const double beta1, const double beta2, const double eplision, const double batchsize);
	double calculate_accuracy(vector<vector<double>>& testdata, const string& lossfun, set<string> fns);
	double cal_acc_with_mse(vector<vector<double>>& testdata, const string& lossfun, set<string> fns);
	double cal_Tmse(vector<vector<double>>& testdata, const string& lossfun, set<string> fns);
	double meanofcolumn(const int j, vector<vector<double>>& tdata);
	double stdv(const int j, const double mean, vector<vector<double>>& tdata);
	void z_score(const int j, vector<vector<double>>& tdata, const double mean, const double stdev);
	vector<pair<double, double>> standardization(vector<vector<double>>& tdata);


public:

	Smodel() {};

	string double_int_toString(const double a);
	vector<string> split(const string& str, const string& delim);
	void drop_rows(vector<vector<double>>& tdata, const int& i);
	void shuffle_rows(vector<vector<double>>& tdata, const time_t seed);
	void column_drop(vector<int> drops, vector<vector<double>>& tdata);
	void reposition_target_col(vector<vector<double>>& tdata, const int& current);
	set<string> featurize_data(vector<vector<string>>& tdata, vector<int>& alphafeature);
	vector<vector<double>> transform_catagories(vector<vector<string>> Stestd, set<string> cordinates);
	void fullyconnectedinputL(const int currentNneurone, const int nextlayerNnuerone);
	void fullyconnectedhiddenL(const int currentNneurone, const int nextlayerNnuerone, const string& acctivationfun);
	void outputLayer(const int currentNneurone, const string& acctivationfun);
	pair<vector<double>, vector<double>> train_neural_network(vector<vector<double>>& tdata, const string& Lossfunction, const string& optimizer, const double learningRate, const double beta1, const double beta2, const double eplision, const double epoches, const double batch_size, const double split_train_per, bool stdtrue);
	void save_model(const string& fn);
	set<string> load_model(const string& fn);
	vector<double> make_prediction(vector<vector<double>>& testdata, const string& lossfunction, set<string>& labels);



	~Smodel() {};

};

#endif