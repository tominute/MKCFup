/*******************************************************************************
* Tools for computing complex matrix			Version 1.0
* Copyright 2018 Bin Yu, UCAS&NLPR, Beijing.	[yubin2017@ia.ac.cn]
*******************************************************************************/
#ifndef  COMPLEXMAT_H
#define COMPLEXMAT_H
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

struct complex_mat
{
	Mat real;
	Mat img;
};

float sqrnorm(const vector<complex_mat> &input, int fea, int channel);
vector<complex_mat> sqrmag(const vector<complex_mat> &input, int fea, int channel);
vector<complex_mat> sqrmag(const vector<complex_mat> &input, const vector<complex_mat> &input2, int fea, int channel);
struct complex_mat mul_com(const complex_mat &input1, const complex_mat &input2);
struct complex_mat mul_com(const complex_mat &input1, const float &input2);
struct complex_mat conj_com(const complex_mat &input1);
struct complex_mat plus_com(const complex_mat &input1, const float &input2);
struct complex_mat plus_com(const complex_mat &input1, const complex_mat &input2);
struct complex_mat div_com(const complex_mat &input1, const complex_mat &input2);
#endif