/*******************************************************************************
* Tools for computing complex matrix			Version 1.0
* Copyright 2018 Bin Yu, UCAS&NLPR, Beijing.	[yubin2017@ia.ac.cn]
*******************************************************************************/
#include <opencv2/opencv.hpp>
#include "ComplexMat.h"
#include <vector>

using namespace cv;
using namespace std;

float sqrnorm(const vector<complex_mat> &input, int fea, int channel)
{
	float out = 0, sum1, sum2;
	int c;
	Mat real, img, temp;
	if (fea == 0 & channel == 1)c = 1;
	else c = 4;
	for (int i = 0; i < c; ++i)
	{
		real = input[i].real.clone();
		img = input[i].img.clone();
		multiply(real, real, temp);
		sum1 = sum(temp)[0];
		multiply(img, img, temp);
		sum2 = sum(temp)[0];
		out += sum1 + sum2;
	}
	out = out / (input[0].img.cols*input[0].img.rows);
	return out;
}
vector<complex_mat> sqrmag(const vector<complex_mat> &input, int fea, int channel)
{
	int c;
	if (fea == 0 & channel == 1)c = 1;
	else c = 4;
	vector<complex_mat> out(c);
	Mat temp(input[0].img.rows, input[0].img.cols, CV_32FC1);
	temp.setTo(0);
	for (int i = 0; i < c; ++i)
	{
		out[i].real = input[i].real.mul(input[i].real) + input[i].img.mul(input[i].img);
		out[i].img = temp;
	}
	return out;
}
vector<complex_mat> sqrmag(const vector<complex_mat> &input, const vector<complex_mat> &input2, int fea, int channel)
{
	int c;
	if (fea == 0 & channel == 1)c = 1;
	else c = 4;
	vector<complex_mat> out(c);
	for (int i = 0; i < c; ++i)
	{
		out[i].real = input[i].real.mul(input2[i].real) + input[i].img.mul(input2[i].img);
		out[i].img = input[i].img.mul(input2[i].real) - input[i].real.mul(input2[i].img);
	}
	return out;
}
struct complex_mat mul_com(const complex_mat &input1, const complex_mat &input2)
{
	complex_mat out;
	out.real = input1.real.mul(input2.real) - input1.img.mul(input2.img);
	out.img = input1.img.mul(input2.real) + input1.real.mul(input2.img);
	return out;
}
struct complex_mat mul_com(const complex_mat &input1, const float &input2)
{
	complex_mat out;
	out.real = input1.real * input2;
	out.img = input1.img * input2;
	return out;
}
struct complex_mat conj_com(const complex_mat &input1)
{
	complex_mat out;
	out.real = input1.real.clone();
	out.img = input1.img * (-1.0);
	return out;
}
struct complex_mat plus_com(const complex_mat &input1, const float &input2)
{
	complex_mat out;
	out.real = input1.real + input2;
	out.img = input1.img.clone();
	return out;
}
struct complex_mat plus_com(const complex_mat &input1, const complex_mat &input2)
{
	complex_mat out;
	out.real = input1.real + input2.real;
	out.img = input1.img + input2.img;
	return out;
}
struct complex_mat div_com(const complex_mat &input1, const complex_mat &input2)
{
	complex_mat out, den, num;
	den = mul_com(input2, conj_com(input2));
	num = mul_com(input1, conj_com(input2));
	divide(num.real, den.real, out.real);
	divide(num.img, den.real, out.img);
	return out;
}