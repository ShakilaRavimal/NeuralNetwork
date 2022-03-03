#include "SNeuralNet.h"


string Smodel::double_int_toString(const double a)
{
	ostringstream temp;
	temp << a;
	return temp.str();
}

vector<string> Smodel::split(const string& str, const string& delim)
{
	vector<string> tokens;
	size_t prev = 0, pos = 0;
	do
	{
		pos = str.find(delim, prev);
		if (pos == string::npos)
		{
			pos = str.length();
		}
		string token = str.substr(prev, pos - prev);
		if (!token.empty())
		{
			tokens.push_back(token);
		}
		prev = pos + delim.length();
	} while (pos < str.length() && prev < str.length());

	return tokens;
}


void Smodel::drop_rows(vector<vector<double>>& tdata, const int& i)
{
	tdata[i].clear();
}


void Smodel::shuffle_rows(vector<vector<double>>& tdata, const time_t seed)
{
	srand((unsigned)seed);
	vector<double> saved;
	for (int i = 1; i < tdata.size(); i++)
	{
		int r = rand() % tdata.size();
		if (r != i && r != 0)
		{
			for (int j = 0; j < tdata[i].size(); j++)
			{
				saved.push_back(tdata[i][j]);
			}
			drop_rows(tdata, i);
			for (int j = 0; j < saved.size(); j++)
			{
				tdata[i].push_back(tdata[r][j]);
			}
			drop_rows(tdata, r);
			for (int j = 0; j < saved.size(); j++)
			{
				tdata[r].push_back(saved[j]);
			}
			saved.clear();

		}

	}
}


void Smodel::column_drop(vector<int> drops, vector<vector<double>>& tdata)
{
	sort(drops.begin(), drops.end());
	for (int k = 0; k < drops.size(); k++)
	{
		if (k > 0)
		{
			drops[k] = drops[k] - k;
		}
		for (int i = 0; i < tdata.size(); i++)
		{
			tdata[i].erase(tdata[i].begin() + drops[k]);
		}

	}
}


void Smodel::reposition_target_col(vector<vector<double>>& tdata, const int& current)
{
	for (int i = 0; i < tdata.size(); i++)
	{
		double td = tdata[i][current];
		tdata[i].erase(tdata[i].begin() + current);
		tdata[i].push_back(td);
	}

}

set<string> Smodel::featurize_data(vector<vector<string>>& tdata, vector<int>& alphafeature)
{
	set<string> data;
	for (int j = 0; j < alphafeature.size(); j++)
	{
		data.clear();
		for (int i = 0; i < tdata.size(); i++)
		{
			data.insert(tdata[i][alphafeature[j]]);
		}

		for (int c = 0; c < data.size(); c++)
		{
			for (int i = 0; i < tdata.size(); i++)
			{
				if (tdata[i][alphafeature[j]] == *next(data.begin(), c))
				{
					tdata[i][alphafeature[j]] = double_int_toString(c);

				}
			}
		}

	}
	return data;
}

vector<vector<double>> Smodel::transform_catagories(vector<vector<string>> Stestd, set<string> cordinates)
{
	for (int c = 0; c < cordinates.size(); c++)
	{
		for (int i = 0; i < Stestd.size(); i++)
		{
			if (Stestd[i][Stestd[i].size() - 1] == *next(cordinates.begin(), c))
			{
				Stestd[i][Stestd[i].size() - 1] = double_int_toString(c);

			}
		}
	}
	vector<vector<double>> ptdata(Stestd.size());
	for (int i = 0; i < Stestd.size(); i++)
	{
		for (int j = 0; j < Stestd[i].size(); j++)
		{
			ptdata[i].push_back(stod(Stestd[i][j]));
		}
	}

	return ptdata;
}

void Smodel::fullyconnectedinputL(const int currentNneurone, const int nextlayerNnuerone)
{
	vector<Neurone> currentlayer;
	mt19937 gen(time(0));
	uniform_real_distribution<double> dis(0.1, 0.5);


	for (int i = 0; i < (currentNneurone - 1); i++)
	{
		vector<double> weights;
		for (int j = 0; j < nextlayerNnuerone; j++)
		{
			weights.push_back(dis(gen));
		}
		Neurone new_neuro(0, 0, 0, 0, INFINITY, 0, weights);
		currentlayer.push_back(new_neuro);
	}
	fullyconnectedLdata.push_back(currentlayer);
	fullyactivations.push_back("no");
}

void Smodel::fullyconnectedhiddenL(const int currentNneurone, const int nextlayerNnuerone, const string& acctivationfun)
{
	vector<Neurone> currentlayer;
	mt19937 gen(time(0));
	uniform_real_distribution<double> dis(0.1, 0.5);
	double val = dis(gen);


	for (int i = 0; i < currentNneurone; i++)
	{
		vector<double> weights;
		for (int j = 0; j < nextlayerNnuerone; j++)
		{
			weights.push_back(dis(gen));
		}

		Neurone new_neuro(0, 0, 0, val, INFINITY, 0, weights);
		currentlayer.push_back(new_neuro);
	}
	fullyconnectedLdata.push_back(currentlayer);
	fullyactivations.push_back(acctivationfun);

}

void Smodel::outputLayer(const int currentNneurone, const string& acctivationfun)
{
	vector<Neurone> currentlayer;
	mt19937 gen(time(0));
	uniform_real_distribution<double> dis(0.1, 0.5);
	double val = dis(gen);
	vector<double> weights;

	for (int i = 0; i < currentNneurone; i++)
	{
		Neurone new_neuro(0, 0, 0, val, 0, 0, weights);
		currentlayer.push_back(new_neuro);

	}
	fullyconnectedLdata.push_back(currentlayer);
	fullyactivations.push_back(acctivationfun);
}

double Smodel::relu_ops(const double val)
{
	return max(0.0, val);
}

double Smodel::Lrelu_ops(const double val)
{
	return max(0.01 * val, val);
}

double Smodel::SquaredError(const double targetval, const double predictedval)
{
	return pow((targetval - predictedval), 2);
}

double Smodel::sigmoid(const double& val)
{
	return (1 / (1 + exp(-val)));
}

void Smodel::softmax_ops(const int i)
{
	double maxval = -INFINITY;
	for (int h = 0; h < fullyconnectedLdata[i].size(); h++)
	{
		if (maxval < fullyconnectedLdata[i][h].getinval())
		{
			maxval = fullyconnectedLdata[i][h].getinval();
		}

	}

	double Tval = 0;
	for (int v = 0; v < fullyconnectedLdata[i].size(); v++)
	{
		Tval += exp(fullyconnectedLdata[i][v].getinval() - maxval);
	}

	const double offset = maxval + log(Tval);
	for (int v = 0; v < fullyconnectedLdata[i].size(); v++)
	{
		fullyconnectedLdata[i][v].setoutval((exp(fullyconnectedLdata[i][v].getinval() - offset)));

	}
}

double Smodel::fullyconnectedforwardpropagation(const int q, vector<vector<double>>& tdata, const string& loss_function, set<string>& fns, string fn)
{
	double crossentropy = 0;
	double mseE = 0;

	for (int i = 0; i < fullyconnectedLdata.size(); i++)
	{
		for (int j = 0; j < fullyconnectedLdata[i].size(); j++)
		{
			fullyconnectedLdata[i][j].setinval(0);
			fullyconnectedLdata[i][j].setoutval(0);
		}
	}

	for (int i = 0; i < tdata[q].size() - 1; i++)
	{
		fullyconnectedLdata[0][i].setoutval(tdata[q][i]);
		fullyconnectedLdata[0][i].setinval(tdata[q][i]);

	}

	for (int i = 0; i < fullyconnectedLdata.size() - 1; i++)
	{
		for (int j = 0; j < fullyconnectedLdata[i].size(); j++)
		{
			for (int k = 0; k < fullyconnectedLdata[i][j].getweights().size(); k++)
			{
				fullyconnectedLdata[i + 1][k].setinval(fullyconnectedLdata[i + 1][k].getinval() + (fullyconnectedLdata[i][j].getoutval() * fullyconnectedLdata[i][j].getweights()[k]));
			}

		}
		for (int c = 0; c < fullyconnectedLdata[i + 1].size(); c++)
		{
			fullyconnectedLdata[i + 1][c].setinval((fullyconnectedLdata[i + 1][c].getinval() + fullyconnectedLdata[i + 1][c].getbias()));

		}
		for (int v = 0; v < fullyconnectedLdata[i + 1].size(); v++)
		{
			if (fullyactivations[i + 1] == "relu")
			{
				fullyconnectedLdata[i + 1][v].setoutval(relu_ops(fullyconnectedLdata[i + 1][v].getinval()));

			}
			if (fullyactivations[i + 1] == "Lrelu")
			{
				fullyconnectedLdata[i + 1][v].setoutval(Lrelu_ops(fullyconnectedLdata[i + 1][v].getinval()));
			}
			if (fullyactivations[i + 1] == "sigmoid")
			{
				fullyconnectedLdata[i + 1][v].setoutval(sigmoid(fullyconnectedLdata[i + 1][v].getinval()));
			}

		}

	}

	if (fullyactivations[fullyactivations.size() - 1] == "softmax")
	{
		softmax_ops(fullyactivations.size() - 1);
	}


	if (fn != "")
	{
		vector<double> ovec(fullyconnectedLdata[fullyconnectedLdata.size() - 1].size());

		if (loss_function == "Ccross")
		{
			for (int i = 0; i < fns.size(); i++)
			{
				if (*next(fns.begin(), i) == fn)
				{
					ovec[i] = 1;
					fullyconnectedLdata[fullyconnectedLdata.size() - 1][i].settargetval(1);

				}
				else
				{
					ovec[i] = 0;
					fullyconnectedLdata[fullyconnectedLdata.size() - 1][i].settargetval(0);
				}
			}

			for (int v = 0; v < fullyconnectedLdata[fullyconnectedLdata.size() - 1].size(); v++)
			{
				if (fullyconnectedLdata[fullyconnectedLdata.size() - 1][v].getoutval() == 0)
				{
					crossentropy += (ovec[v] * log(fullyconnectedLdata[fullyconnectedLdata.size() - 1][v].getoutval() + 1e-15));
				}
				else
				{
					crossentropy += (ovec[v] * log(fullyconnectedLdata[fullyconnectedLdata.size() - 1][v].getoutval()));
				}
				fullyconnectedLdata[fullyconnectedLdata.size() - 1][v].setderror(fullyconnectedLdata[fullyconnectedLdata.size() - 1][v].getoutval() - ovec[v]);

			}
			return -crossentropy;

		}
		if (loss_function == "Bcross")
		{
			for (int v = 0; v < fullyconnectedLdata[fullyconnectedLdata.size() - 1].size(); v++)
			{
				crossentropy += ((stof(fn) * log(fullyconnectedLdata[fullyconnectedLdata.size() - 1][v].getoutval())) + ((1 - stof(fn)) * log(1 - fullyconnectedLdata[fullyconnectedLdata.size() - 1][v].getoutval())));
				fullyconnectedLdata[fullyconnectedLdata.size() - 1][v].setderror((fullyconnectedLdata[fullyconnectedLdata.size() - 1][v].getoutval() - fullyconnectedLdata[fullyconnectedLdata.size() - 1][v].gettargetvalval()) / (fullyconnectedLdata[fullyconnectedLdata.size() - 1][v].getoutval() * (1 - fullyconnectedLdata[fullyconnectedLdata.size() - 1][v].getoutval())));
			}
			return -crossentropy;

		}
	}

	if (loss_function == "mse")
	{
		if (fullyconnectedLdata[fullyconnectedLdata.size() - 1].size() == 1)
		{
			for (int v = 0; v < fullyconnectedLdata[fullyconnectedLdata.size() - 1].size(); v++)
			{
				double values = SquaredError((tdata[q][tdata[q].size() - 1]), (fullyconnectedLdata[fullyconnectedLdata.size() - 1][v].getoutval()));
				double dvalue = (fullyconnectedLdata[fullyconnectedLdata.size() - 1][v].getoutval() - tdata[q][tdata[q].size() - 1]);

				fullyconnectedLdata[fullyconnectedLdata.size() - 1][v].setderror(dvalue);
				mseE = values;

			}

			mseE = mseE / 2;

			return mseE;
		}

	}

	return 0;

}

void Smodel::fullyconnectedbackpropagation(vector<vector<vector<double>>>& weights, vector<vector<double>>& biases)
{
	for (int i = 0; i < fullyconnectedLdata.size(); i++)
	{
		for (int j = 0; j < fullyconnectedLdata[i].size(); j++)
		{
			double val = 1;
			if (fullyactivations[i] == "softmax")
			{
				if (fullyconnectedLdata[i][j].getderror() != INFINITY)
				{
					val *= fullyconnectedLdata[i][j].getderror();

				}
			}
			if (fullyactivations[i] == "sigmoid")
			{
				if (fullyconnectedLdata[i][j].getderror() != INFINITY)
				{
					val *= fullyconnectedLdata[i][j].getderror();

				}
				else
				{
					val *= (fullyconnectedLdata[i][j].getoutval() * (1 - fullyconnectedLdata[i][j].getoutval()));
				}
			}
			if (fullyactivations[i] == "relu" || fullyactivations[i] == "Lrelu")
			{
				if (fullyconnectedLdata[i][j].getderror() != INFINITY)
				{
					val *= fullyconnectedLdata[i][j].getderror();

				}
				else
				{
					if (fullyactivations[i] == "relu")
					{
						if (fullyconnectedLdata[i][j].getinval() >= 0)
						{
							val *= 1;
						}
						else
						{
							val *= 0;
						}
					}
					if (fullyactivations[i] == "Lrelu")
					{
						if (fullyconnectedLdata[i][j].getinval() >= 0)
						{
							val *= 1;
						}
						else
						{
							val *= 0.01;
						}
					}

				}
			}

			fullyconnectedLdata[i][j].setiderivative(val);

		}
	}

	for (int i = fullyconnectedLdata.size() - 2; i >= 0; i--)
	{
		for (int j = 0; j < fullyconnectedLdata[i].size(); j++)
		{
			double backval = 0;

			for (int k = 0; k < fullyconnectedLdata[i][j].getweights().size(); k++)
			{
				backval += (fullyconnectedLdata[i + 1][k].getiderivative() * fullyconnectedLdata[i][j].getweights()[k]);
			}

			fullyconnectedLdata[i][j].setiderivative(fullyconnectedLdata[i][j].getiderivative() * backval);

		}
	}

	for (int i = fullyconnectedLdata.size() - 2; i >= 0; i--)
	{
		for (int j = 0; j < fullyconnectedLdata[i].size(); j++)
		{
			double val = 0;
			for (int k = 0; k < fullyconnectedLdata[i][j].getweights().size(); k++)
			{
				val = (fullyconnectedLdata[i + 1][k].getiderivative() * fullyconnectedLdata[i][j].getoutval());
				weights[i][j][k] += val;

			}
		}
	}

	for (int i = 0; i < fullyconnectedLdata.size(); i++)
	{
		for (int j = 0; j < fullyconnectedLdata[i].size(); j++)
		{
			biases[i][j] += fullyconnectedLdata[i][j].getiderivative();

		}
	}

}

void Smodel::resize_W_B(vector<vector<vector<double>>>& weights, vector<vector<double>>& biases)
{
	weights.clear();
	biases.clear();

	weights.resize(fullyconnectedLdata.size());
	biases.resize(fullyconnectedLdata.size());

	for (int c = 0; c < fullyconnectedLdata.size(); c++)
	{
		weights[c].resize(fullyconnectedLdata[c].size());
		biases[c].resize(fullyconnectedLdata[c].size());
		for (int u = 0; u < fullyconnectedLdata[c].size(); u++)
		{
			weights[c][u].resize(fullyconnectedLdata[c][u].getweights().size());
		}
	}
}

void Smodel::update_All_weights_biases(vector<vector<vector<double>>>& weights, vector<vector<double>>& biases, const string& optimizer, const double iteration, const double learning_rate, const double beta1, const double beta2, const double eplision, const double batchsize)
{
	if (optimizer == "sgd")
	{
		for (int i = 0; i < fullyconnectedLdata.size(); i++)
		{
			for (int j = 0; j < fullyconnectedLdata[i].size(); j++)
			{
				for (int k = 0; k < fullyconnectedLdata[i][j].getweights().size(); k++)
				{
					vector<double> w = fullyconnectedLdata[i][j].getweights();
					w[k] = w[k] - (learning_rate * (weights[i][j][k] / batchsize));
					fullyconnectedLdata[i][j].setweights(w);
				}
			}
		}

		for (int i = 0; i < fullyconnectedLdata.size(); i++)
		{
			for (int j = 0; j < fullyconnectedLdata[i].size(); j++)
			{
				fullyconnectedLdata[i][j].setbias(fullyconnectedLdata[i][j].getbias() - (learning_rate * (biases[i][j] / batchsize)));
			}
		}
	}
	if (optimizer == "adam")
	{
		for (int i = 0; i < fullyconnectedLdata.size(); i++)
		{
			for (int j = 0; j < fullyconnectedLdata[i].size(); j++)
			{
				for (int k = 0; k < fullyconnectedLdata[i][j].getweights().size(); k++)
				{
					float mt = (beta1 * fMW[i][j][k]) + ((1 - beta1) * (weights[i][j][k] / batchsize));
					float vt = (beta2 * fVW[i][j][k]) + ((1 - beta2) * pow((weights[i][j][k] / batchsize), 2));

					float mthat = mt / (1 - pow(beta1, iteration));
					float vthat = vt / (1 - pow(beta2, iteration));

					vector<double> w = fullyconnectedLdata[i][j].getweights();
					w[k] = w[k] - ((learning_rate * mthat) / (sqrt(vthat) + eplision));
					fullyconnectedLdata[i][j].setweights(w);

					fMW[i][j][k] = mt;
					fVW[i][j][k] = vt;
				}
			}
		}

		for (int i = 0; i < fullyconnectedLdata.size(); i++)
		{
			for (int j = 0; j < fullyconnectedLdata[i].size(); j++)
			{
				float mt = (beta1 * fMB[i][j]) + ((1 - beta1) * (biases[i][j] / batchsize));
				float vt = (beta2 * fVB[i][j]) + ((1 - beta2) * pow((biases[i][j] / batchsize), 2));

				float mthat = mt / (1 - pow(beta1, iteration));
				float vthat = vt / (1 - pow(beta2, iteration));

				fullyconnectedLdata[i][j].setbias(fullyconnectedLdata[i][j].getbias() - ((learning_rate * mthat) / (sqrt(vthat) + eplision)));
				fMB[i][j] = mt;
				fVB[i][j] = vt;
			}
		}
	}
	if (optimizer == "m")
	{
		for (int i = 0; i < fullyconnectedLdata.size(); i++)
		{
			for (int j = 0; j < fullyconnectedLdata[i].size(); j++)
			{
				for (int k = 0; k < fullyconnectedLdata[i][j].getweights().size(); k++)
				{
					float mt = (beta1 * fMW[i][j][k]) + ((1 - beta1) * (weights[i][j][k] / batchsize));

					vector<double> w = fullyconnectedLdata[i][j].getweights();
					w[k] = w[k] - (learning_rate * mt);
					fullyconnectedLdata[i][j].setweights(w);

					fMW[i][j][k] = mt;

				}
			}
		}

		for (int i = 0; i < fullyconnectedLdata.size(); i++)
		{
			for (int j = 0; j < fullyconnectedLdata[i].size(); j++)
			{
				float mt = (beta1 * fMB[i][j]) + ((1 - beta1) * (biases[i][j] / batchsize));

				fullyconnectedLdata[i][j].setbias(fullyconnectedLdata[i][j].getbias() - (learning_rate * mt));
				fMB[i][j] = mt;

			}
		}
	}

	resize_W_B(weights, biases);
}

double Smodel::calculate_accuracy(vector<vector<double>>& testdata, const string& lossfun, set<string> fns)
{
	double match = 0;
	for (int k = 0; k < testdata.size(); k++)
	{
		double maxprob = -INFINITY;
		int index = 0;
		fullyconnectedforwardpropagation(k, testdata, lossfun, fns, double_int_toString(testdata[k][testdata[k].size() - 1]));
		for (int i = 0; i < fullyconnectedLdata[fullyconnectedLdata.size() - 1].size(); i++)
		{
			if (maxprob < fullyconnectedLdata[fullyconnectedLdata.size() - 1][i].getoutval())
			{
				maxprob = fullyconnectedLdata[fullyconnectedLdata.size() - 1][i].getoutval();

				index = i;
			}

		}

		if (fullyconnectedLdata[fullyconnectedLdata.size() - 1][index].gettargetvalval() == 1)
		{
			match++;
		}

	}

	return 100 * (match / testdata.size());

}

double Smodel::cal_acc_with_mse(vector<vector<double>>& testdata, const string& lossfun, set<string> fns)
{
	double match = 0;
	for (int i = 0; i < testdata.size(); i++)
	{
		fullyconnectedforwardpropagation(i, testdata, lossfun, fns, double_int_toString(testdata[i][testdata[i].size() - 1]));

		if (round(fullyconnectedLdata[fullyconnectedLdata.size() - 1][0].getoutval()) == testdata[i][testdata[i].size() - 1])
		{
			match++;
		}

	}

	return 100 * (match / testdata.size());

}

double Smodel::cal_Tmse(vector<vector<double>>& testdata, const string& lossfun, set<string> fns)
{
	double Terror = 0;
	for (int k = 0; k < testdata.size(); k++)
	{
		Terror += fullyconnectedforwardpropagation(k, testdata, lossfun, fns, double_int_toString(testdata[k][testdata[k].size() - 1]));
	}

	return (Terror / testdata.size());
}

double Smodel::meanofcolumn(const int j, vector<vector<double>>& tdata)
{
	double total = 0;
	for (int i = 0; i < tdata.size(); i++)
	{
		total += (tdata[i][j]);
	}

	return total / tdata.size();
}

double Smodel::stdv(const int j, const double mean, vector<vector<double>>& tdata)
{
	double sq = 0;
	for (int i = 0; i < tdata.size(); i++)
	{
		sq += pow(tdata[i][j] - mean, 2);
	}

	return sqrt(sq / tdata.size());
}

void Smodel::z_score(const int j, vector<vector<double>>& tdata, const double mean, const double stdev)
{
	for (int i = 0; i < tdata.size(); i++)
	{
		tdata[i][j] = (tdata[i][j] - mean) / stdev;
	}
}

vector<pair<double, double>> Smodel::standardization(vector<vector<double>>& tdata)
{
	vector<pair<double, double>> vals;
	for (int j = 0; j < tdata[0].size() - 1; j++)
	{
		double mean = meanofcolumn(j, tdata);
		double stdeviation = stdv(j, mean, tdata);
		vals.push_back(make_pair(mean, stdeviation));
		z_score(j, tdata, mean, stdeviation);

	}

	return vals;
}

pair<vector<double>, vector<double>> Smodel::train_neural_network(vector<vector<double>>& tdata, const string& Lossfunction, const string& optimizer, const double learningRate, const double beta1, const double beta2, const double eplision, const double epoches, const double batch_size, const double split_train_per, bool stdtrue = false)
{
	vector<pair<double, double>> standardized;
	if (stdtrue)
	{
		standardized = standardization(tdata);
	}

	vector<vector<double>> testdatasplited;

	const int testdataN = (tdata.size()) - ((split_train_per * (tdata.size())) / 100);
	testdatasplited.resize(testdataN);
	int y = 0;

	for (int i = tdata.size() - testdataN; i < tdata.size(); i++)
	{
		for (int j = 0; j < tdata[i].size(); j++)
		{
			testdatasplited[y].push_back(tdata[i][j]);
		}
		y++;

	}

	tdata.resize(tdata.size() - testdataN);
	tdata.shrink_to_fit();

	set<string> fns;
	for (int i = 0; i < tdata.size(); i++)
	{
		fns.insert(double_int_toString(tdata[i][tdata[i].size() - 1]));
	}

	vector<vector<vector<double>>> weights;
	vector<vector<double>> biases;

	resize_W_B(weights, biases);

	resize_W_B(fMW, fMB);
	resize_W_B(fVW, fVB);

	double losserror = 0;
	double items = 0;
	vector<double> losserrors;
	vector<double> accs;
	vector<double> testlosserrors;
	vector<double> specialaccses;
	for (int k = 0; k < epoches; k++)
	{
		double iteration = 0;
		double y = 1;
		items = 0;
		for (int i = 0; i < tdata.size(); i++)
		{
			losserror += fullyconnectedforwardpropagation(i, tdata, Lossfunction, fns, double_int_toString(tdata[i][tdata[i].size() - 1]));
			fullyconnectedbackpropagation(weights, biases);

			if ((i + 1) % (int)batch_size == 0 || i == (tdata.size() - 1))
			{
				iteration++;
				items += y;

				double acc = 0;
				double e = 0;
				double specialacc = 0;
				if (Lossfunction == "Ccross" || Lossfunction == "mse")
				{
					if (fullyconnectedLdata[fullyconnectedLdata.size() - 1].size() > 1 && Lossfunction == "mse")
					{
						acc = calculate_accuracy(testdatasplited, Lossfunction, fns);
						e = cal_Tmse(testdatasplited, Lossfunction, fns);
						cout << "epoch--> " << k + 1 << "/" << epoches << " batch--> " << items << "/" << tdata.size() << " T_acc--> " << acc << "%" << " T_loss--> " << e << " loss--> " << (losserror / (2 * y)) << '\n';
					}
					if (fullyconnectedLdata[fullyconnectedLdata.size() - 1].size() > 1 && Lossfunction == "Ccross")
					{
						acc = calculate_accuracy(testdatasplited, Lossfunction, fns);
						cout << "epoch--> " << k + 1 << "/" << epoches << " batch--> " << items << "/" << tdata.size() << " T_acc--> " << acc << " % " << " loss--> " << (losserror / y) << '\n';
					}

				}
				if (Lossfunction == "mse")
				{
					if (fullyconnectedLdata[fullyconnectedLdata.size() - 1].size() == 1)
					{
						specialacc = cal_acc_with_mse(testdatasplited, Lossfunction, fns);
						e = cal_Tmse(testdatasplited, Lossfunction, fns);
						cout << "epoch--> " << k + 1 << "/" << epoches << " batch--> " << items << "/" << tdata.size() << " T_acc--> " << specialacc << "%" << " T_loss--> " << e << " loss--> " << (losserror / (2 * y)) << '\n';

					}

				}

				update_All_weights_biases(weights, biases, optimizer, iteration, learningRate, beta1, beta2, eplision, y);

				losserrors.push_back(losserror / (2 * y));
				accs.push_back(acc);
				testlosserrors.push_back(e);
				specialaccses.push_back(specialacc);

				losserror = 0;
				y = 0;

			}

			y++;

		}
		shuffle_rows(tdata, time(0));

	}
	return make_pair(losserrors, accs);


}

void Smodel::save_model(const string& fn)
{
	ofstream file(fn);
	file << fullyconnectedLdata.size() << '\n';

	for (int i = 0; i < fullyconnectedLdata.size(); i++)
	{
		file << fullyconnectedLdata[i].size() << '#';
		file << fullyconnectedLdata[i][0].getweights().size() << '#';
		file << fullyactivations[i] << '\n';

	}
	file << '$' << '\n';
	for (int i = 0; i < fullyconnectedLdata.size() - 1; i++)
	{
		for (int j = 0; j < fullyconnectedLdata[i].size(); j++)
		{
			for (int k = 0; k < fullyconnectedLdata[i][j].getweights().size(); k++)
			{
				file << fullyconnectedLdata[i][j].getweights()[k] << '?';

			}
		}

	}
	file << '\n';
	file << '!' << '\n';
	for (int i = 0; i < fullyconnectedLdata.size(); i++)
	{
		for (int j = 0; j < fullyconnectedLdata[i].size(); j++)
		{
			file << fullyconnectedLdata[i][j].getbias() << '*';
		}
	}
	file.close();

	cout << "model successfully saved!" << endl;

}

set<string> Smodel::load_model(const string& fn)
{
	ifstream file1(fn);
	string line = "";
	bool found = false, found2 = false;
	int i = 0, N = 0;

	while (getline(file1, line))
	{
		if (line == "$")
		{
			found = true;
		}
		if (line == "!")
		{
			found2 = true;

		}
		if (i == 0)
		{
			fullyconnectedLdata.resize(atoi(line.c_str()));
		}
		else
		{
			if (found2)
			{
				if (line != "!")
				{
					int q = 0;
					vector<string> B = split(line, "*");
					for (int b = 0; b < fullyconnectedLdata.size(); b++)
					{
						for (int j = 0; j < fullyconnectedLdata[b].size(); j++)
						{
							fullyconnectedLdata[b][j].setbias(atof(B[q++].c_str()));
						}
					}

				}

			}
			if (!found && !found2)
			{
				vector<string> str = split(line, "#");
				fullyconnectedLdata[N].resize(atoi(str[0].c_str()));
				for (int c = 0; c < fullyconnectedLdata[N].size(); c++)
				{
					vector<double> w(atoi(str[1].c_str()));
					fullyconnectedLdata[N][c].setweights(w);
				}
				fullyactivations.push_back(str[2]);
				N++;
			}
			else
			{
				if (line != "$" && !found2)
				{
					int v = 0;
					vector<string> W = split(line, "?");
					for (int i = 0; i < fullyconnectedLdata.size() - 1; i++)
					{
						for (int j = 0; j < fullyconnectedLdata[i].size(); j++)
						{
							vector<double> w;
							for (int k = 0; k < fullyconnectedLdata[i][j].getweights().size(); k++)
							{
								w.push_back(atof(W[v++].c_str()));

							}
							fullyconnectedLdata[i][j].setweights(w);

						}

					}
				}

			}

		}

		i++;
	}
	file1.close();

	cout << "model successfully loaded!" << endl;
	set<string> nullobj;

	return  nullobj;

}

vector<double> Smodel::make_prediction(vector<vector<double>>& testdata, const string& lossfunction, set<string>& labels)
{
	cout << "making predictions....." << endl;
	if (lossfunction == "Ccross")
	{
		vector<double> Ccrosspre;
		for (int i = 0; i < testdata.size(); i++)
		{
			double max = -INFINITY;
			int index = 0;

			fullyconnectedforwardpropagation(i, testdata, lossfunction, labels, "");
			for (int j = 0; j < fullyconnectedLdata[fullyconnectedLdata.size() - 1].size(); j++)
			{
				if (max < fullyconnectedLdata[fullyconnectedLdata.size() - 1][j].getoutval())
				{
					max = fullyconnectedLdata[fullyconnectedLdata.size() - 1][j].getoutval();
					index = j;

				}

			}

			Ccrosspre.push_back(index);
		}

		return Ccrosspre;
	}
	if (lossfunction == "Bcross")
	{
		vector<double> Bcrosspre;
		for (int i = 0; i < testdata.size(); i++)
		{
			fullyconnectedforwardpropagation(i, testdata, lossfunction, labels, "");
			if (fullyconnectedLdata[fullyconnectedLdata.size() - 1][0].getoutval() >= 0.5)
			{
				Bcrosspre.push_back(1);
			}
			else
			{
				Bcrosspre.push_back(0);
			}
		}

		return Bcrosspre;

	}
	if (lossfunction == "mse")
	{
		vector<double> mseS;
		for (int i = 0; i < testdata.size(); i++)
		{
			fullyconnectedforwardpropagation(i, testdata, lossfunction, labels, "");
			mseS.push_back(fullyconnectedLdata[fullyconnectedLdata.size() - 1][0].getoutval());

		}

		return mseS;
	}

	return {};

}




