#pragma once

#ifndef UNITNEURONE_H
#define UNITNEURONE_H

using namespace std;

class Neurone
{

private:

	double inval;
	double outval;
	double targetval;
	double bias;
	double derror;
	double individualderivative;
	vector<double> weights;

public:

	Neurone() {};

	Neurone(const double inval, const int outval, const double targetval, const double bias, const double derror, const double iderivative, vector<double>& weights)
	{
		this->inval = inval;
		this->outval = outval;
		this->targetval = targetval;
		this->bias = bias;
		this->derror = derror;
		this->individualderivative = iderivative;
		this->weights = weights;
	}

	double getinval()
	{
		return this->inval;
	}
	void setinval(const double inval)
	{
		this->inval = inval;
	}
	double getoutval()
	{
		return this->outval;
	}
	void setoutval(const double outval)
	{
		this->outval = outval;
	}
	double gettargetvalval()
	{
		return this->targetval;
	}
	void settargetval(const double targetval)
	{
		this->targetval = targetval;
	}
	double getbias()
	{
		return this->bias;
	}
	void setbias(const double bias)
	{
		this->bias = bias;
	}
	double getderror()
	{
		return this->derror;
	}
	void setderror(const double derror)
	{
		this->derror = derror;
	}
	double getiderivative()
	{
		return this->individualderivative;
	}
	void setiderivative(const double iderivative)
	{
		this->individualderivative = iderivative;
	}
	vector<double> getweights()
	{
		return this->weights;
	}
	void setweights(vector<double>& weights)
	{
		this->weights = weights;
	}


	~Neurone() {};

};

#endif