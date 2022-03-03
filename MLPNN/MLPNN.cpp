#include "SNeuralNet.h"

using namespace std;

pair<vector<vector<double>>, set<string>> readprepareTraindataset(const char* fname, Smodel& model)
{
	vector<string> data;
	vector<vector<string>> tdata;
	ifstream file(fname);
	string line = "";

	while (getline(file, line))
	{
		if (line != "")
		{
			data.push_back(line);
		}

	}

	file.close();

	tdata.resize(data.size() - 1);

	for (int i = 1; i < data.size(); i++)
	{
		vector<string> str = model.split(data[i], ",");

		for (int j = 0; j < str.size(); j++)
		{
			tdata[i - 1].push_back((str[j]));

		}
	}
	vector<int> alphaf = { 4 };

	set<string> zip = model.featurize_data(tdata, alphaf);

	vector<vector<double>> ptdata(tdata.size());

	for (int i = 0; i < tdata.size(); i++)
	{
		for (int j = 0; j < tdata[i].size(); j++)
		{
			ptdata[i].push_back(stod(tdata[i][j]));
		}
	}


	model.shuffle_rows(ptdata, time(0));

	return make_pair(ptdata, zip);

}

void model_training(Smodel* my_model, vector<vector<double>>& tdata)
{
	//must specify number of nodes in each layer according to following structure
	int input_layer_Nnodes = tdata[0].size();
	int second_layer_Nnodes = 5;
	int output_layer_Nnodes = 3;

	//can be created as many only fullyconnectedhiddenL as optimal
	my_model->fullyconnectedinputL(input_layer_Nnodes, second_layer_Nnodes);
	my_model->fullyconnectedhiddenL(second_layer_Nnodes, output_layer_Nnodes, "relu");
	my_model->outputLayer(output_layer_Nnodes, "softmax");
	//Ccross=Catagorical_cross_entropy and Bcross=binary...
	//can change lossfunction then change relevent other parameters.Algoritm will recognized relevent parameters according to the lossfunction specified
	//traindata and testdata must be in double datatype
	//split_train_percentage(prams) will be used to calculate accuracy and in this case it is 90% for traindata and 10% for validation.
	pair<vector<double>, vector<double>> lossnacc = my_model->train_neural_network(tdata, "Ccross", "adam", 0.05, 0.9, 0.999, 1e-8, 5, 3, 90, false);

	//saving the model
	my_model->save_model("MLNN.data");
}

void model_prediction(Smodel* my_model, set<string>& cordinates)
{
	//have to change following code for regression problem.below code is customizable
		//loading the model
	my_model->load_model("MLNN.data");
	vector<vector<string>> Stestd = { {"5.1","3.5","1.4","0.2","Iris-setosa"} };
	//must pass tdata.second reference for following relevent methods
	vector<vector<double>> Dtestd = my_model->transform_catagories(Stestd, cordinates);

	//making predictions
	vector<double> predictions = my_model->make_prediction(Dtestd, "Ccross", cordinates);

	for (int i = 0; i < predictions.size(); i++)
	{
		int p = predictions[i];
		string convertpredictiontotarget = *next(cordinates.begin(), p);
		std::cout << "Target: " << Stestd[i][Stestd[i].size() - 1] << " ---> Prediction: " << convertpredictiontotarget << std::endl;
	}

}

int main(void)
{
	////rememeber to include Sneuralnet.h
	////creating an instance from neural network class(Smodel)
	Smodel* my_model = new Smodel();
	////must included target_column in traindataset
	pair<vector<vector<double>>, set<string>> tdata = readprepareTraindataset("iris.data", *my_model);

	////following method should comment after saving the preferred model.Then, model will load without errors when loading the saved model
	////model training and saving..
	model_training(my_model, tdata.first);


	////following method should comment when saving a new model.
	////loading the model...
	//model_prediction(my_model, tdata.second);

}
