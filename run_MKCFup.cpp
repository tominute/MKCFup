/*******************************************************************************
* MKCFup										Version 1.0
* Copyright 2018 Bin Yu, UCAS&NLPR, Beijing.	[yubin2017@ia.ac.cn]
* Paper [High-speed Tracking with Multi-kernel Correlation Filters]
*******************************************************************************/
#include <fftw3.h>
#include <opencv2\opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include "fhog.hpp"
#include "cnfeat.hpp"
#include "omp.h"
#include "ComplexMat.h"
#include "Params.h"

using namespace cv;
using namespace std;

typedef struct _COMPLEX
{
	float real;
	float img;
}COMPLEX;
struct model {
	complex_mat alphaf;
	float d[2];
};
float **m;
COMPLEX **fm;
float **ffm;
COMPLEX **fmy;
float **ffmy;
float** allocfloat(float **mem, int h, int w)
{
	mem = (float **)malloc(sizeof(float *) * h);
	mem[0] = (float *)malloc(sizeof(float) * h * w);
	for (int i = 1; i<h; i++)
	{
		mem[i] = mem[i - 1] + w;
	}
	return(mem);
}
COMPLEX** alloccomplex(COMPLEX **mem, int h, int w)
{
	mem = (COMPLEX **)malloc(sizeof(COMPLEX *) * h);
	mem[0] = (COMPLEX *)malloc(sizeof(COMPLEX) * h * w);
	for (int i = 1; i<h; i++)
	{
		mem[i] = mem[i - 1] + w;
	}
	return(mem);
}
void FFT2(float **input, COMPLEX **output, int height, int width)
{
	fftwf_plan planR;
	fftwf_complex *inR, *outR;
	inR = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * height * width);
	outR = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * height * width);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			inR[i * width + j][0] = input[i][j];
			inR[i * width + j][1] = 0;
		}
	planR = fftwf_plan_dft_2d(height, width, inR, outR, FFTW_FORWARD, FFTW_ESTIMATE);

	fftwf_execute(planR);

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			output[i][j]._COMPLEX::real = outR[i * width + j][0];
			output[i][j]._COMPLEX::img = outR[i * width + j][1];
		}
	fftwf_destroy_plan(planR);
	fftwf_free(inR);
	fftwf_free(outR);
}
void IFFT2(COMPLEX **input, float **output, int height, int width)
{
	fftwf_plan planR;
	fftwf_complex *inR, *outR;
	inR = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * height * width);
	outR = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * height * width);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			inR[i * width + j][0] = input[i][j]._COMPLEX::real;
			inR[i * width + j][1] = input[i][j]._COMPLEX::img;
		}
	planR = fftwf_plan_dft_2d(height, width, inR, outR, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftwf_execute(planR);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			output[i][j] = outR[i * width + j][0] / (height*width);
		}
	fftwf_destroy_plan(planR);
	fftwf_free(inR);
	fftwf_free(outR);
}
struct complex_mat newfft2(const Mat &input)
{
	complex_mat out;
	/*m = allocfloat(m, input.rows, input.cols);
	fm = alloccomplex(fm, input.rows, input.cols);*/
	Mat real(input.rows, input.cols, CV_32FC1);
	Mat img(input.rows, input.cols, CV_32FC1);
	Mat temp = input.clone();
	for (int i = 0; i < input.rows; i++)
	{
		float *data = temp.ptr<float>(i);
		for (int j = 0; j < input.cols; j++)
		{
			m[i][j] = data[j];
		}
	}

	FFT2(m, fm, input.rows, input.cols);

	for (int i = 0; i < input.rows; i++)
	{
		float *data = real.ptr<float>(i);
		float *data1 = img.ptr<float>(i);
		for (int j = 0; j < input.cols; j++)
		{
			data[j] = fm[i][j]._COMPLEX::real;
			data1[j] = fm[i][j]._COMPLEX::img;
		}
	}
	out.real = real.clone();
	out.img = img.clone();


	return out;
}
struct complex_mat newfft2(const Mat &input, const Mat &cos_window)
{
	complex_mat out;
	Mat real(input.rows, input.cols, CV_32FC1);
	Mat img(input.rows, input.cols, CV_32FC1);
	Mat temp = input.mul(cos_window);
	for (int i = 0; i < input.rows; i++)
	{
		float *data = temp.ptr<float>(i);
		for (int j = 0; j < input.cols; j++)
		{
			m[i][j] = data[j];
		}
	}

	FFT2(m, fm, input.rows, input.cols);

	for (int i = 0; i < input.rows; i++)
	{
		float *data = real.ptr<float>(i);
		float *data1 = img.ptr<float>(i);
		for (int j = 0; j < input.cols; j++)
		{
			data[j] = fm[i][j]._COMPLEX::real;
			data1[j] = fm[i][j]._COMPLEX::img;
		}
	}
	out.real = real.clone();
	out.img = img.clone();

	return out;
}
Mat newifft2(const complex_mat &input)
{
	Mat real = input.real.clone();
	Mat img = input.img.clone();
	/*ffm = allocfloat(ffm, real.rows, real.cols);
	fm = alloccomplex(fm, real.rows, real.cols);*/
	Mat tempp(real.rows, real.cols, CV_32FC1);
	for (int i = 0; i < real.rows; i++)
	{
		float *data = real.ptr<float>(i);
		float *data1 = img.ptr<float>(i);
		for (int j = 0; j < real.cols; j++)
		{
			fm[i][j]._COMPLEX::real = data[j];
			fm[i][j]._COMPLEX::img = data1[j];
		}
	}
	IFFT2(fm, ffm, real.rows, real.cols);
	for (int i = 0; i < real.rows; i++)
	{
		float *data = tempp.ptr<float>(i);
		for (int j = 0; j < real.cols; j++)
		{
			data[j] = ffm[i][j];
		}
	}

	return tempp;
}
Mat newifft2_for_y(const complex_mat &input)
{
	Mat real = input.real.clone();
	Mat img = input.img.clone();
	Mat tempp(real.rows, real.cols, CV_32FC1);
	for (int i = 0; i < real.rows; i++)
	{
		float *data = real.ptr<float>(i);
		float *data1 = img.ptr<float>(i);
		for (int j = 0; j < real.cols; j++)
		{
			fmy[i][j]._COMPLEX::real = data[j];
			fmy[i][j]._COMPLEX::img = data1[j];
		}
	}
	IFFT2(fmy, ffmy, real.rows, real.cols);
	for (int i = 0; i < real.rows; i++)
	{
		float *data = tempp.ptr<float>(i);
		for (int j = 0; j < real.cols; j++)
		{
			data[j] = ffmy[i][j];
		}
	}

	return tempp;
}
Mat cosine_window_function(int dim1, int dim2)
{
	Mat m1(1, dim1, CV_32FC1), m2(dim2, 1, CV_32FC1);
	double N_inv = 1. / (static_cast<double>(dim1) - 1.);
	for (int i = 0; i < dim1; ++i)
		m1.at<float>(i) = 0.5*(1. - cos(2. * CV_PI * static_cast<double>(i) * N_inv));
	N_inv = 1. / (static_cast<double>(dim2) - 1.);
	for (int i = 0; i < dim2; ++i)
		m2.at<float>(i) = 0.5*(1. - cos(2. * CV_PI * static_cast<double>(i) * N_inv));
	Mat ret = m2*m1;
	return ret;
}
Mat scale_window_function(int dim1)
{
	Mat m1(1, dim1, CV_32FC1);
	double N_inv = 1. / (static_cast<double>(dim1) - 1.);
	for (int i = 0; i < dim1; ++i)
		m1.at<float>(i) = 0.5*(1. - cos(2. * CV_PI * static_cast<double>(i) * N_inv));
	return m1;
}
Mat circshift(const Mat &patch, int x_rot, int y_rot)
{
	Mat rot_patch(patch.size(), CV_32FC1);
	Mat tmp_x_rot(patch.size(), CV_32FC1);

	//circular rotate x-axis
	if (x_rot < 0) {
		//move part that does not rotate over the edge
		Range orig_range(-x_rot, patch.cols);
		Range rot_range(0, patch.cols - (-x_rot));
		patch(Range::all(), orig_range).copyTo(tmp_x_rot(Range::all(), rot_range));

		//rotated part
		orig_range = Range(0, -x_rot);
		rot_range = Range(patch.cols - (-x_rot), patch.cols);
		patch(Range::all(), orig_range).copyTo(tmp_x_rot(Range::all(), rot_range));
	}
	else if (x_rot > 0) {
		//move part that does not rotate over the edge
		Range orig_range(0, patch.cols - x_rot);
		Range rot_range(x_rot, patch.cols);
		patch(Range::all(), orig_range).copyTo(tmp_x_rot(Range::all(), rot_range));

		//rotated part
		orig_range = Range(patch.cols - x_rot, patch.cols);
		rot_range = Range(0, x_rot);
		patch(Range::all(), orig_range).copyTo(tmp_x_rot(Range::all(), rot_range));
	}
	else {    //zero rotation
			  //move part that does not rotate over the edge
		Range orig_range(0, patch.cols);
		Range rot_range(0, patch.cols);
		patch(Range::all(), orig_range).copyTo(tmp_x_rot(Range::all(), rot_range));
	}

	//circular rotate y-axis
	if (y_rot < 0) {
		//move part that does not rotate over the edge
		Range orig_range(-y_rot, patch.rows);
		Range rot_range(0, patch.rows - (-y_rot));
		tmp_x_rot(orig_range, Range::all()).copyTo(rot_patch(rot_range, Range::all()));

		//rotated part
		orig_range = Range(0, -y_rot);
		rot_range = Range(patch.rows - (-y_rot), patch.rows);
		tmp_x_rot(orig_range, Range::all()).copyTo(rot_patch(rot_range, Range::all()));
	}
	else if (y_rot > 0) {
		//move part that does not rotate over the edge
		Range orig_range(0, patch.rows - y_rot);
		Range rot_range(y_rot, patch.rows);
		tmp_x_rot(orig_range, Range::all()).copyTo(rot_patch(rot_range, Range::all()));

		//rotated part
		orig_range = Range(patch.rows - y_rot, patch.rows);
		rot_range = Range(0, y_rot);
		tmp_x_rot(orig_range, Range::all()).copyTo(rot_patch(rot_range, Range::all()));
	}
	else { //zero rotation
		   //move part that does not rotate over the edge
		Range orig_range(0, patch.rows);
		Range rot_range(0, patch.rows);
		tmp_x_rot(orig_range, Range::all()).copyTo(rot_patch(rot_range, Range::all()));
	}

	return rot_patch;
}
Mat gaussian_shaped_labels(double sigma, int dim1, int dim2)
{

	Mat labels(dim2, dim1, CV_32FC1);
	int range_y[2] = { -dim2 / 2, dim2 - dim2 / 2 };
	int range_x[2] = { -dim1 / 2, dim1 - dim1 / 2 };
	double sigma_s = sigma*sigma;
	for (int y = range_y[0], j = 0; y < range_y[1]; ++y, ++j)
	{
		float * row_ptr = labels.ptr<float>(j);
		double y_s = y*y;
		for (int x = range_x[0], i = 0; x < range_x[1]; ++x, ++i)
		{
			row_ptr[i] = std::exp(-0.5 * (y_s + x*x) / sigma_s);
		}
	}
	//rotate so that 1 is at top-left corner (see KCF paper for explanation)
	Mat rot_labels = circshift(labels, range_x[0], range_y[0]);
	assert(rot_labels.at<float>(0, 0) >= 1.f - 1e-10f);
	return rot_labels;
}
Mat get_subwindow(const Mat &input, int cx, int cy, int width, int height, float currentScaleFactor)
{
	cv::Size sz, model_sz;
	cv::Point centerCoor;
	model_sz.width = floor(width);
	model_sz.height = floor(height);
	sz.width = floor(width*currentScaleFactor);
	sz.height = floor(height*currentScaleFactor);
	cv::Mat subWindow;
	centerCoor.x = cx;
	centerCoor.y = cy;
	cv::Point lefttop(min(input.cols - 2, max(-sz.width + 1, centerCoor.x - cvFloor(float(sz.width) / 2.0) + 1)),
		min(input.rows - 2, max(-sz.height + 1, centerCoor.y - cvFloor(float(sz.height) / 2.0) + 1)));
	cv::Point rightbottom(lefttop.x + sz.width, lefttop.y + sz.height);
	cv::Rect border(-min(lefttop.x, 0), -min(lefttop.y, 0),
		max(rightbottom.x - (input.cols - 1), 0), max(rightbottom.y - (input.rows - 1), 0));
	cv::Point lefttopLimit(max(lefttop.x, 0), max(lefttop.y, 0));
	cv::Point rightbottomLimit(min(rightbottom.x, input.cols - 1), min(rightbottom.y, input.rows - 1));
	cv::Rect roiRect(lefttopLimit, rightbottomLimit);
	input(roiRect).copyTo(subWindow);
	if (border != cv::Rect(0, 0, 0, 0))
		cv::copyMakeBorder(subWindow, subWindow, border.y, border.height, border.x, border.width, cv::BORDER_REPLICATE);
	Mat out;
	resize(subWindow, out, model_sz);
	return out;
}
vector<Mat> average_faeture_region(const vector<Mat> &input, int region_size, int size1, int size2, int channels)
{
	float maxval = 1;
	vector<Mat> out(channels);
	float region_area = region_size*region_size;
#pragma omp parallel for
	for (int n = 0; n < channels; ++n)
	{
		//cout << input.rows << " "<<input.cols<<endl;
		Mat region_image(size1, size2, CV_32FC1);
		Mat iImage;
		integral(input[n], iImage, CV_32FC1);
		/*cout << iImage.type() << endl;
		cout << iImage.rows << " " << iImage.cols << endl;
		cout << region_image.rows << " " << region_image.cols << endl;*/

		int *i1 = new int[size1];
		int *i2 = new int[size2];
		for (int i = 0; i < size1; ++i)
		{
			i1[i] = region_size*(i + 1);
			//cout << i1[i] << endl;
		}
		for (int i = 0; i < size2; ++i)
			i2[i] = region_size*(i + 1);
		for (int i = 0; i < size1; ++i)
		{
			float *data1 = iImage.ptr<float>(i1[i]);
			float *data2 = iImage.ptr<float>(i1[i] - region_size);
			float *out = region_image.ptr<float>(i);
			for (int j = 0; j < size2; ++j)
			{
				out[j] = (data1[i2[j]] - data1[i2[j] - region_size] - data2[i2[j]] + data2[i2[j] - region_size]) / (region_area * maxval);
			}
		}
		out[n] = region_image.clone();
		//cout << region_image << endl;
		delete[]i1;
		delete[]i2;
	}
	return out;
}
Mat get_scale_subwindow(const Mat &input, int cx, int cy, int width, int height, float *scaleFactors, int *scale_model_sz, float currentScaleFactor, int nScales, int featureRatio, int num_hog_fea)
{
	Mat out_pca;
	cv::Size outsize;
	outsize.width = scale_model_sz[1];
	outsize.height = scale_model_sz[0];
	int imchannel = input.channels();
	int dim_scale = floor(scale_model_sz[1] / featureRatio)*floor(scale_model_sz[0]/ featureRatio) * num_hog_fea;
	int dimm = floor(scale_model_sz[1] / featureRatio)*floor(scale_model_sz[0] / featureRatio);
	cv::Point centerCoor;
	centerCoor.x = cx;
	centerCoor.y = cy;
	out_pca = Mat::zeros(nScales, dim_scale, CV_32FC1);
	omp_set_num_threads(num_threads);
	#pragma omp parallel for
	for (int i = 0; i < nScales; ++i)
	{
		vector<Mat> hog_feat(num_hog_fea);
		cv::Size sz;
		sz.width = floor(width*scaleFactors[i]* currentScaleFactor);
		sz.height = floor(height*scaleFactors[i]* currentScaleFactor);
		cv::Mat subWindow;
		cv::Point lefttop(min(input.cols - 2, max(-sz.width + 1, centerCoor.x - cvFloor(float(sz.width) / 2.0) + 1)),
			min(input.rows - 2, max(-sz.height + 1, centerCoor.y - cvFloor(float(sz.height) / 2.0) + 1)));
		cv::Point rightbottom(lefttop.x + sz.width, lefttop.y + sz.height);
		cv::Rect border(-min(lefttop.x, 0), -min(lefttop.y, 0),
			max(rightbottom.x - (input.cols - 1), 0), max(rightbottom.y - (input.rows - 1), 0));
		cv::Point lefttopLimit(max(lefttop.x, 0), max(lefttop.y, 0));
		cv::Point rightbottomLimit(min(rightbottom.x, input.cols - 1), min(rightbottom.y, input.rows - 1));
		cv::Rect roiRect(lefttopLimit, rightbottomLimit);
		input(roiRect).copyTo(subWindow);
		if (border != cv::Rect(0, 0, 0, 0))
			cv::copyMakeBorder(subWindow, subWindow, border.y, border.height, border.x, border.width, cv::BORDER_REPLICATE);
		Mat out, temp;
		resize(subWindow, out, outsize);
		if (imchannel == 3)
		{
			cvtColor(out, temp, COLOR_BGR2GRAY);
			hog_feat = FHoG::extract(out);
		}
		else
			hog_feat = FHoG::extract(out);
		float *data = out_pca.ptr<float>(i);
		int p = 0;
		for (int j = 0; j < num_hog_fea; ++j)
		{
			hog_feat[j] = hog_feat[j].t();
			float *data2 = hog_feat[j].ptr<float>(0);
			for (int q = 0; q < dimm; ++q)
				data[p+q] = data2[q];
			p += dimm;
		}
	}
	return out_pca.t();
}
Mat new_gaussian_correlation(const vector<complex_mat> &xf, const vector<complex_mat> &yf, double sigma, bool auto_correlation, int fea, int channel, int num_pca_fea)
{
	float xf_sqr_norm = sqrnorm(xf, fea, channel);
	float yf_sqr_norm = auto_correlation ? xf_sqr_norm : sqrnorm(yf, fea, channel);
	float numel_xf_inv;
	vector<complex_mat> xyf = auto_correlation ? sqrmag(xf, fea, channel) : sqrmag(xf, yf, fea, channel);


	//ifft2 and sum over 3rd dimension, we dont care about individual channels
	Mat xy_sum(xf[0].img.rows, xf[0].img.cols, CV_32FC1);
	complex_mat temp;
	temp.real = Mat::zeros(xf[0].img.rows, xf[0].img.cols, CV_32FC1);
	temp.img = Mat::zeros(xf[0].img.rows, xf[0].img.cols, CV_32FC1);
	xy_sum.setTo(0);
	if (fea == 0 & channel == 1)
	{
		for (int i = 0; i < 1; ++i)
		{
			temp.img += xyf[i].img;
			temp.real += xyf[i].real;
		}
		numel_xf_inv = 1.f / (xf[0].img.rows * xf[0].img.cols );
	}
	else
	{
		for (int i = 0; i < num_pca_fea; ++i)
		{
			temp.img += xyf[i].img;
			temp.real += xyf[i].real;
		}
		numel_xf_inv = 1.f / (xf[0].img.rows * xf[0].img.cols * num_pca_fea);
	}
	xy_sum = newifft2(temp);

	Mat tmp;
	//complex_mat temp;
	cv::exp(-1.f / (sigma * sigma) * cv::max((xf_sqr_norm + yf_sqr_norm - 2 * xy_sum) * numel_xf_inv, 0), tmp);
	return tmp;
}
struct model newtrainmodel(complex_mat &alphaf_num1, complex_mat &alphaf_num2, complex_mat &alphaf_den1, complex_mat &alphaf_den2, float &d_num1, float &d_num2, float &d_den1, float &d_den2,
	const Mat &k_hog, const Mat &k_cn, const complex_mat &yf, int frame, int start_frame, float learning_rate_cn, float learning_rate_hog, const Mat &y, int imchannel, float lambda)
{
	complex_mat kf_hog = newfft2(k_hog);
	complex_mat kf_cn = newfft2(k_cn);
	struct model p;
	p.d[0] = 0.5;
	p.d[1] = 0.5;
	float prevD[2] = { 0.5, 0.5 }, deltaD[2];
	int stop = 0;
	int count = 0;
	float threshold = 0.03;
	float d_new_num1, d_new_num2, d_new_den1, d_new_den2, d_num11, d_num22, d_den11, d_den22;
	Mat alpha, temp1, temp2, temp, out_temp, prevalpha, deltaalpha;
	complex_mat new_num1, new_num2, new_den1, new_den2, alphaf_num11, alphaf_num22, alphaf_den11, alphaf_den22, alphaf_num, alphaf_den;
	while (stop == 0)
	{

		//train alpha
		new_num1 = mul_com(yf, mul_com(kf_cn, p.d[0]));
		new_num2 = mul_com(yf, mul_com(kf_hog, p.d[1]));
		new_den1 = mul_com(mul_com(kf_cn, p.d[0]) , plus_com(mul_com(conj_com(kf_cn),p.d[0]),lambda));
		new_den2 = mul_com(mul_com(kf_hog, p.d[1]), plus_com(mul_com(conj_com(kf_hog), p.d[1]), lambda));
		if (frame == start_frame)
		{
			alphaf_num11 = new_num1;
			alphaf_num22 = new_num2;
			alphaf_num = plus_com(alphaf_num11 , alphaf_num22);
			alphaf_den11 = new_den1;
			alphaf_den22 = new_den2;
			alphaf_den = plus_com(alphaf_den11, alphaf_den22);
		}
		else
		{
			alphaf_num11 = plus_com(mul_com(alphaf_num1, (1 - learning_rate_cn)) , mul_com(new_num1, learning_rate_cn));
			alphaf_num22 = plus_com(mul_com(alphaf_num2, (1 - learning_rate_hog)), mul_com(new_num2, learning_rate_hog));
			alphaf_den11 = plus_com(mul_com(alphaf_den1, (1 - learning_rate_cn)), mul_com(new_den1, learning_rate_cn));
			alphaf_den22 = plus_com(mul_com(alphaf_den2, (1 - learning_rate_hog)), mul_com(new_den2, learning_rate_hog));
			alphaf_num = plus_com(alphaf_num11,  alphaf_num22);
			alphaf_den = plus_com(alphaf_den11, alphaf_den22);
		}
		p.alphaf = div_com(alphaf_num , alphaf_den);

		//train D
		temp1 = newifft2(mul_com(conj_com(kf_cn),p.alphaf));
		temp2 = newifft2(mul_com(conj_com(kf_hog),p.alphaf));
		temp = 2 * y - alpha*lambda;
		multiply(temp, temp1, out_temp);

		d_new_num1 = sum(out_temp)[0];
		multiply(temp, temp2, out_temp);
		d_new_num2 = sum(out_temp)[0];
		multiply(temp1, temp1, out_temp);
		d_new_den1 = sum(2 * out_temp)[0];
		multiply(temp2, temp2, out_temp);
		d_new_den2 = sum(2 * out_temp)[0];
		alpha = newifft2(p.alphaf);
		if (frame == start_frame)
		{
			d_num11 = d_new_num1;
			d_num22 = d_new_num2;
			d_den11 = d_new_den1;
			d_den22 = d_new_den2;
		}
		else
		{
			d_num11 = d_num1*(1 - learning_rate_cn) + learning_rate_cn*d_new_num1;
			d_num22 = d_num2*(1 - learning_rate_hog) + learning_rate_hog*d_new_num2;
			d_den11 = d_den1*(1 - learning_rate_cn) + learning_rate_cn*d_new_den1;
			d_den22 = d_den2*(1 - learning_rate_hog) + learning_rate_hog*d_new_den2;
		}

		p.d[0] = d_num11 / d_den11;
		p.d[1] = d_num22 / d_den22;

		float summ = p.d[0] + p.d[1];
		p.d[0] = p.d[0] / summ;
		p.d[1] = p.d[1] / summ;

		//iteration
		count++;
		if (count > 1)
		{
			deltaalpha = abs(alpha - prevalpha);
			deltaD[0] = abs(p.d[0] - prevD[0]);
			deltaD[1] = abs(p.d[1] - prevD[1]);
			if (sum(deltaalpha)[0] <= threshold*sum(abs(prevalpha))[0] && (deltaD[0] + deltaD[1]) <= threshold*(prevD[0] + prevD[1]))
				stop = 1;
		}
		prevalpha = alpha;
		prevD[0] = p.d[0];
		prevD[1] = p.d[1];
		if (count > 100)break;
	}

	alphaf_num1 = alphaf_num11;
	alphaf_num2 = alphaf_num22;
	alphaf_den1 = alphaf_den11;
	alphaf_den2 = alphaf_den22;
	d_num1 = d_num11;
	d_num2 = d_num22;
	d_den1 = d_den11;
	d_den2 = d_den22;
	return p;
}
struct complex_mat resizeDFT2(const complex_mat &input, const int *sz)
{
	int imsz[2], minsz[2], mids[2], mide[2];
	imsz[0] = input.img.rows;
	imsz[1] = input.img.cols;
	minsz[0] = imsz[0];
	minsz[1] = imsz[1];
	float scaling = sz[0] * sz[1] / (imsz[0] * imsz[1]);
	//cout << minsz[0] << " " << minsz[1] << endl;
	//cout << sz[0] << " " << sz[1] << endl;

	complex_mat resizeddft, imf;
	imf.real = input.real.clone();
	imf.img = input.img.clone();

	resizeddft.real = Mat::zeros(sz[0], sz[1], CV_32FC1);
	resizeddft.img = Mat::zeros(sz[0], sz[1], CV_32FC1);//cout << resizeddft.real.rows << " " << resizeddft.real.cols << endl;
	//cout << imf.real << endl;
	mids[0] = ceil(minsz[0] / 2.0);
	mids[1] = ceil(minsz[1] / 2.0);
	mide[0] = floor((minsz[0] - 1) / 2.0 - 1);
	mide[1] = floor((minsz[1] - 1) / 2.0 - 1);

	for (int i = 0; i < mids[0]; ++i)
	{
		float *data1 = imf.real.ptr<float>(i);
		float *data2 = imf.img.ptr<float>(i);
		float *out1 = resizeddft.real.ptr<float>(i);
		float *out2 = resizeddft.img.ptr<float>(i);
		for (int j = 0; j < mids[1]; ++j)
		{
			out1[j] = scaling * data1[j];
			out2[j] = scaling * data2[j];
		}
	}


	for (int i = 0; i < mids[0]; ++i)
	{
		float *data1 = imf.real.ptr<float>(i);
		float *data2 = imf.img.ptr<float>(i);
		float *out1 = resizeddft.real.ptr<float>(i);
		float *out2 = resizeddft.img.ptr<float>(i);
		for (int j = sz[1] - mide[1] -1; j < sz[1]; ++j)
		{
			out1[j] = scaling * data1[j - sz[1] + minsz[1]];
			out2[j] = scaling * data2[j - sz[1] + minsz[1]];
		}
	}

	for (int i = sz[0] - mide[0] - 1; i < sz[0]; ++i)
	{
		float *data1 = imf.real.ptr<float>(i - sz[0]+ minsz[0] );
		float *data2 = imf.img.ptr<float>(i - sz[0] + minsz[0]);
		float *out1 = resizeddft.real.ptr<float>(i);
		float *out2 = resizeddft.img.ptr<float>(i);
		for (int j = 0; j < mids[1]; ++j)
		{
			out1[j] = scaling * data1[j];
			out2[j] = scaling * data2[j];
		}
	}
	for (int i = sz[0] - mide[0] - 1; i < sz[0]; ++i)
	{
		float *data1 = imf.real.ptr<float>(i - sz[0] + minsz[0]);
		float *data2 = imf.img.ptr<float>(i - sz[0] + minsz[0]);
		float *out1 = resizeddft.real.ptr<float>(i);
		float *out2 = resizeddft.img.ptr<float>(i);
		for (int j = sz[1] - mide[1] - 1; j < sz[1]; ++j)
		{
			out1[j] = scaling * data1[j - sz[1] + minsz[1]];
			out2[j] = scaling * data2[j - sz[1] + minsz[1]];
		}
	}
	return resizeddft;
}
struct complex_mat resizeDFT(const complex_mat &input, const int n)
{
	int imsz, minsz, mids, mide;
	imsz = input.img.cols;
	minsz = imsz;
	float scaling = float(n) / float(imsz);

	complex_mat resizeddft, imf;
	imf.real = input.real.clone();
	imf.img = input.img.clone();

	resizeddft.real = Mat::zeros(1, n, CV_32FC1);
	resizeddft.img = Mat::zeros(1, n, CV_32FC1);
	mids = ceil(minsz / 2.0);
	mide = floor((minsz - 1) / 2.0 - 1);
	float *data1 = imf.real.ptr<float>(0);
	float *data2 = imf.img.ptr<float>(0);
	float *out1 = resizeddft.real.ptr<float>(0);
	float *out2 = resizeddft.img.ptr<float>(0);
	for (int i = 0; i < mids; ++i)
	{
		out1[i] = scaling * data1[i];
		out2[i] = scaling * data2[i];
	}
	for (int i = n - mide - 1; i < n; ++i)
	{
		out1[i] = scaling * data1[i - n + minsz];
		out2[i] = scaling * data2[i - n + minsz];
	}
	return resizeddft;
}

int main()
{
	//choose one params set for the certain benchmark
	params.choose_benchmarks("OTB2013");
	int featureRatio					= params.featureRatio;
	double padding						= params.get_padding();
	double output_sigma_factor			= params.output_sigma_factor;
	double cnsigma_color				= params.cnsigma_color;
	double hogsigma_color				= params.hogsigma_color;
	double learning_rate_cn_color		= params.get_learning_rate_cn_color();
	double learning_rate_hog_color		= params.get_learning_rate_hog_color();
	double cnsigma_gray					= params.get_cnsigma_gray();
	double hogsigma_gray				= params.get_hogsigma_gray();
	double learning_rate_cn_gray		= params.get_learning_rate_cn_gray();
	double learning_rate_hog_gray		= params.get_learning_rate_hog_gray();
	double lambda						= params.lambda;
	double scale_step					= params.scale_step;
	double scale_sigma_factor			= params.scale_sigma_factor;
	double scale_model_max_area			= params.scale_model_max_area;
	double translation_model_max_area	= params.translation_model_max_area;
	double translation_model_min_area	= params.translation_model_min_area;
	double scale_interp_factor			= params.scale_interp_factor;
	bool use_dsst						= params.use_dsst;
	int gap								= params.gap;
	bool visualization					= params.visualization;
	int num_cn_fea						= params.num_cn_fea;
	int num_hog_fea						= params.num_hog_fea;
	int num_pca_fea						= params.num_pca_fea;
	int nScales							= params.nScales;
	int nScalesInterp					= params.nScalesInterp;
	string sequences[] = { "Jogging-2" };
	int total_num = sizeof(sequences) / sizeof(string);
	//set the openmp set
	omp_set_num_threads(num_threads);
	float fps = 0;
	for (int seq_num = 0; seq_num < total_num; ++seq_num)
	{
		//load_video_info
		char buf[100];
		sprintf(buf, "res/results_%s.txt", sequences[seq_num].c_str());
		FILE *res = fopen(buf, "w+");
		const char *name;
		name = sequences[seq_num].c_str();
		if (res == NULL) cout << "Cannot read the res file." << endl;
		sprintf(buf, "sequences/%s/%s_gt.txt", name, name);
		FILE *fp_gt = fopen(buf, "r");
		int init_gt[4];
		for (int i = 0; i < 4; ++i)init_gt[i] = 0;
		if (fp_gt == NULL) cout << "Cannot read the gt file." << endl;
		else
		{
			fscanf(fp_gt, "%d,%d,%d,%d", &init_gt[0], &init_gt[1], &init_gt[2], &init_gt[3]);
			fclose(fp_gt);
		}
		if (init_gt[3] == 0)
		{
			fp_gt = fopen(buf, "r");
			fscanf(fp_gt, "%d %d %d %d", &init_gt[0], &init_gt[1], &init_gt[2], &init_gt[3]);
			fclose(fp_gt);
		}
		sprintf(buf, "sequences/%s/%s_frames.txt", name, name);
		FILE *fp_frame = fopen(buf, "r");
		int start_frame, end_frame;
		if (fp_frame == NULL) cout << "Cannot read the file." << endl;
		else
		{
			fscanf(fp_frame, "%d,%d", &start_frame, &end_frame);
			fclose(fp_frame);
		}
		//size initialize
		int target_sz[2];
		target_sz[0] = init_gt[3];
		target_sz[1] = init_gt[2];

		int pos[2];
		pos[0] = init_gt[1];
		pos[1] = init_gt[0];
		int init_pos[2];
		init_pos[0] = pos[0] + target_sz[0] / 2;
		init_pos[1] = pos[1] + target_sz[1] / 2;
		pos[0] = init_pos[0];
		pos[1] = init_pos[1];
		int wsize[2];
		wsize[0] = target_sz[0];
		wsize[1] = target_sz[1];

		int init_target_sz[2];
		init_target_sz[0] = wsize[0];
		init_target_sz[1] = wsize[1];
		float currentScaleFactor;
		if (init_target_sz[0] * init_target_sz[1] > translation_model_max_area)
			currentScaleFactor = sqrt(init_target_sz[0] * init_target_sz[1] / float(translation_model_max_area));
		else
			currentScaleFactor = 1.0;
		int base_target_sz[2];
		base_target_sz[0] = target_sz[0] / currentScaleFactor;
		base_target_sz[1] = target_sz[1] / currentScaleFactor;
		int sz[2];
		sz[0] = base_target_sz[0] * (1 + padding);
		sz[1] = base_target_sz[1] * (1 + padding);

		//adjust size
		while (1)
		{
			if (((sz[1]) % 8) == 0)break;
			else sz[1] ++;
		}
		while (1)
		{
			if (((sz[0]) % 8) == 0)break;
			else sz[0] ++;
		}

		float output_sigma = sqrt(base_target_sz[0] / featureRatio * base_target_sz[1] / featureRatio) * output_sigma_factor;
		int use_sz[2];
		use_sz[0] = sz[0] / featureRatio;
		use_sz[1] = sz[1] / featureRatio;
		int interp_sz[2];
		interp_sz[0] = use_sz[0] * featureRatio;
		interp_sz[1] = use_sz[1] * featureRatio;
		//allocate fft space
		ffm = allocfloat(ffm, use_sz[0], use_sz[1]);
		fm = alloccomplex(fm, use_sz[0], use_sz[1]);
		m = allocfloat(m, use_sz[0], use_sz[1]);
		ffmy = allocfloat(ffmy, interp_sz[0], interp_sz[1]);
		fmy = alloccomplex(fmy, interp_sz[0], interp_sz[1]);

		Mat y = gaussian_shaped_labels(output_sigma, use_sz[1], use_sz[0]);
		Mat kk, kk2, temp;
		vector<Mat> channels(2);
		complex_mat yf = newfft2(y);
		Mat cos_window = cosine_window_function(use_sz[1], use_sz[0]);
		sprintf(buf, "./sequences/%s/img/%04d.jpg", name, start_frame);
		Mat img = imread(buf, -1);
		int imchannel = img.channels();
		if (img.empty()) cout << "Can't load image！" << endl;
		//distinguish image channel
		float cnsigma, hogsigma, learning_rate_hog, learning_rate_cn;
		int modnum, cn_channel, cn_out_channel;
		if (imchannel == 3)
		{
			cnsigma = cnsigma_color;
			hogsigma = hogsigma_color;
			learning_rate_hog = learning_rate_hog_color;
			learning_rate_cn = learning_rate_cn_color;
			modnum = gap;
			cn_channel = num_cn_fea;
			cn_out_channel = num_pca_fea;
		}
		else
		{
			cnsigma = cnsigma_gray;
			hogsigma = hogsigma_gray;
			learning_rate_hog = learning_rate_hog_gray;
			learning_rate_cn = learning_rate_cn_gray;
			modnum = 1;
			cn_channel = 1;
			cn_out_channel = 1;
		}
		vector<Mat> cn_feat(cn_channel), x_cn2(cn_out_channel), z_cn2(cn_out_channel);
		vector<complex_mat> zf_cn(cn_out_channel), xf_cn(cn_out_channel);
		Mat proj_cn(cn_out_channel, cn_channel, CV_32FC1), proj_hog(num_pca_fea, num_hog_fea, CV_32FC1);

		//Scale initialize
		float scale_model_factor = 1, scale_sigma = float(nScalesInterp) * scale_sigma_factor;
		float min_scale_factor, max_scale_factor;
		int scale_model_sz[2];
		complex_mat mega_ysf, sf_den, sf_num_test;
		Mat mega_scale_window;
		float *interpScaleFactors= new float[nScalesInterp], *interp_scale_exp= new float[nScalesInterp], *interp_scale_exp_shift= new float[nScalesInterp], *scaleSizeFactors_test= new float[nScales],
			*scaleSizeFactors= new float[nScales], *scale_exp= new float[nScales], *scale_exp_shift= new float[nScales];
		if (use_dsst)
		{
			for (int i = -floor((nScales - 1) / 2.); i <= ceil((nScales - 1) / 2.); ++i)
				scale_exp[int(i + floor((nScales - 1) / 2.))] = i * nScalesInterp / float(nScales);
			for (int i = 0; i < nScales - floor((nScales - 1) / 2.); ++i)
				scale_exp_shift[i] = scale_exp[int(floor((nScales - 1) / 2.) + i)];
			for (int i = nScales - floor((nScales - 1) / 2.); i < nScales; ++i)
				scale_exp_shift[i] = scale_exp[int(i - (nScales - floor((nScales - 1) / 2.)))];
			for (int i = -floor((nScalesInterp - 1) / 2.); i <= ceil((nScalesInterp - 1) / 2.); ++i)
				interp_scale_exp[int(i + floor((nScalesInterp - 1) / 2.))] = i;
			for (int i = 0; i < nScalesInterp - floor((nScalesInterp - 1) / 2.); i++)
				interp_scale_exp_shift[i] = interp_scale_exp[int(floor((nScalesInterp - 1) / 2.) + i)];
			for (int i = nScalesInterp - floor((nScalesInterp - 1) / 2.); i < nScalesInterp; ++i)
				interp_scale_exp_shift[i] = interp_scale_exp[int(i - (nScalesInterp - floor((nScalesInterp - 1) / 2.)))];
			for (int i = 0; i < nScales; ++i)
				scaleSizeFactors[i] = pow(scale_step, scale_exp[i]);
			for (int i = 0; i < nScales; ++i)
				scaleSizeFactors_test[i] = pow(scale_step, scale_exp_shift[i]);
			for (int i = 0; i < nScalesInterp; ++i)
				interpScaleFactors[i] = pow(scale_step, interp_scale_exp_shift[i]);
			Mat ys(1, nScales, CV_32FC1);
			float *data = ys.ptr<float>(0);
			for (int i = 0; i < nScales; ++i)
				data[i] = exp(-0.5 * (scale_exp_shift[i] * scale_exp_shift[i]) / (scale_sigma*scale_sigma));
			dft(ys, kk, DFT_COMPLEX_OUTPUT);
			split(kk, channels);
			complex_mat ysf;
			ysf.real = channels[0].clone();
			ysf.img = channels[1].clone();
			Mat scale_window = scale_window_function(nScales);
			if (scale_model_factor*scale_model_factor *init_target_sz[0] * init_target_sz[1] > scale_model_max_area)
				scale_model_factor = sqrt(scale_model_max_area / float(init_target_sz[0] * init_target_sz[1]));

			scale_model_sz[0] = floor(init_target_sz[0] * scale_model_factor);
			scale_model_sz[1] = floor(init_target_sz[1] * scale_model_factor);

			mega_scale_window = repeat(scale_window, floor(scale_model_sz[1] / featureRatio)*floor(scale_model_sz[0] / featureRatio) * num_hog_fea, 1);
			mega_ysf.real = repeat(ysf.real, floor(scale_model_sz[1] / featureRatio)*floor(scale_model_sz[0] / featureRatio) * num_hog_fea, 1);
			mega_ysf.img = repeat(ysf.img, floor(scale_model_sz[1] / featureRatio)*floor(scale_model_sz[0] / featureRatio) * num_hog_fea, 1);

			min_scale_factor = pow(scale_step, ceil(log(max(5. / float(sz[0]), 5. / float(sz[1]))) / log(scale_step)));
			max_scale_factor = pow(scale_step, floor(log(min(float(img.rows) / float(base_target_sz[0]), float(img.cols) / float(base_target_sz[1]))) / log(scale_step)));
		}

		//model initialize
		float d_num1 = 0, d_num2 = 0, d_den1 = 0, d_den2 = 0;
		complex_mat alphaf_num1, alphaf_num2, alphaf_den1, alphaf_den2;
		struct model p;
		vector<Mat> x_hog, x_cn, hog_feat, z_hog, z_cn, x_hog2(num_pca_fea), z_hog2(num_pca_fea);
		vector<complex_mat> zf_hog(num_pca_fea), xf_hog(num_pca_fea);
		int use_num = use_sz[0] * use_sz[1];
		Mat x_hog_pca(num_hog_fea, use_num, CV_32FC1);
		Mat z_hog_pca(num_hog_fea, use_num, CV_32FC1);
		Mat x_cn_pca(cn_channel, use_num, CV_32FC1);
		Mat z_cn_pca(cn_channel, use_num, CV_32FC1);
		Mat x_channel(use_sz[0], use_sz[1], CV_32FC1);
		Mat proj_hog_t, proj_cn_t, U1, U2, s_num;
		for (int i = 0; i < num_pca_fea; ++i)
		{
			x_hog2[i] = x_channel.clone();
			if (imchannel == 3)
				x_cn2[i] = x_channel.clone();
		}
		if (imchannel != 3)
			x_cn2[0] = x_channel.clone();
		double total_time = 0;
		clock_t startTime, endTime;

		//start tracking
		for (int frame = start_frame; frame <= end_frame; ++frame)
		{
			//load image
			sprintf(buf, "./sequences/%s/img/%04d.jpg", name, frame);
			Mat im = imread(buf, -1);
			if (im.empty())
			{
				cout << "Can't load image！" << endl;
				break;
			}

			startTime = clock();
			//Detection
			if (frame > start_frame)
			{
				//get feature
				Mat patch = get_subwindow(im, pos[1], pos[0], sz[1], sz[0], currentScaleFactor);
				hog_feat = FHoG::extract(patch);
				if (imchannel == 3)
					cn_feat = CNFeat::extract(patch);
				else
				{
					patch.convertTo(patch, CV_32FC1, 1.0 / 255);
					cn_feat[0] = patch - 0.5;
				}
				cn_feat = average_faeture_region(cn_feat, num_pca_fea, use_sz[0], use_sz[1], cn_channel);
				z_hog = hog_feat;
				z_cn = cn_feat;

				//PCA
				#pragma omp parallel for
				for (int i = 0; i < num_hog_fea; ++i)
				{
					float *data = z_hog_pca.ptr<float>(i);
					float *data2 = z_hog[i].ptr<float>(0);
					for (int q = 0; q < use_num; ++q)
						data[q] = data2[q];
				}
				Mat z_hog_pca_t = z_hog_pca.t();
				Mat z_hog2_temp = z_hog_pca_t * proj_hog_t;
				z_hog2_temp = z_hog2_temp.t();
				Mat z_cn2_temp;
				if (imchannel == 3)
				{
					#pragma omp parallel for
					for (int i = 0; i < num_cn_fea; ++i)
					{
						float *data = z_cn_pca.ptr<float>(i);
						float *data2 = z_cn[i].ptr<float>(0);
						for (int q = 0; q < use_num; ++q)
							data[q] = data2[q];
					}
					Mat z_cn_pca_t = z_cn_pca.t();
					z_cn2_temp = z_cn_pca_t * proj_cn_t;
					z_cn2_temp = z_cn2_temp.t();
				}
				for (int i = 0; i < num_pca_fea; ++i)
				{
					z_hog2[i] = x_channel.clone();
					float *data = z_hog2_temp.ptr<float>(i);
					float *data2 = z_hog2[i].ptr<float>(0);
					for (int q = 0; q < use_num; ++q)
						data2[q] = data[q];
					z_hog2[i] = z_hog2[i].reshape(0, use_sz[0]);
					if (imchannel == 3)
					{
						z_cn2[i] = x_channel.clone();
						float *data3 = z_cn2_temp.ptr<float>(i);
						float *data4 = z_cn2[i].ptr<float>(0);
						for (int q = 0; q < use_num; ++q)
							data4[q] = data3[q];
						z_cn2[i] = z_cn2[i].reshape(0, use_sz[0]);
					}
				}
				//compute fft and kernel
				for (int i = 0; i < num_pca_fea; ++i)
				{
					zf_hog[i] = newfft2(z_hog2[i], cos_window);
					if (imchannel == 3)
						zf_cn[i] = newfft2(z_cn2[i], cos_window);
				}
				if (imchannel != 3)zf_cn[0] = newfft2(z_cn[0], cos_window);

				Mat kzf_hog = new_gaussian_correlation(xf_hog, zf_hog, hogsigma, 0, 1, imchannel, num_pca_fea);
				Mat kzf_cn = new_gaussian_correlation(xf_cn, zf_cn, cnsigma, 0, 0, imchannel, num_pca_fea);
				//get response
				complex_mat responsef = mul_com(conj_com(newfft2(p.d[1] * kzf_hog + p.d[0] * kzf_cn)), p.alphaf);
				responsef = resizeDFT2(responsef, interp_sz);
				Mat response = newifft2_for_y(responsef);
				//target location
				Point2i max_loc;
				minMaxLoc(response, 0, 0, 0, &max_loc);
				if ((max_loc.x + 1) > (response.cols / 2))
					max_loc.x = max_loc.x - response.cols;
				if ((max_loc.y + 1) > (response.rows / 2))
					max_loc.y = max_loc.y - response.rows;
				pos[1] += round(max_loc.x*currentScaleFactor);
				pos[0] += round(max_loc.y*currentScaleFactor);

				//scale detection
				if (use_dsst)
				{
					Mat xs_pca = get_scale_subwindow(im, pos[1], pos[0], base_target_sz[1], base_target_sz[0], scaleSizeFactors, scale_model_sz, currentScaleFactor, nScales, featureRatio, num_hog_fea);
					Mat temp_den = xs_pca;
					temp_den = temp_den.mul(mega_scale_window);

					dft(temp_den, kk, -DFT_ROWS);
					split(kk, channels);
					complex_mat xsf_test;
					xsf_test.real = channels[0];
					xsf_test.img = channels[1];

					complex_mat temp_sf_test = mul_com(sf_num_test, xsf_test);
					complex_mat temp_scale_responsef_test;
					reduce(temp_sf_test.real, temp_scale_responsef_test.real, 0, REDUCE_SUM, CV_32FC1);
					reduce(temp_sf_test.img, temp_scale_responsef_test.img, 0, REDUCE_SUM, CV_32FC1);

					complex_mat scale_responsef = div_com(temp_scale_responsef_test, plus_com(sf_den, lambda));
					complex_mat redft = resizeDFT(scale_responsef, nScalesInterp);

					channels[0] = redft.real.clone();
					channels[1] = redft.img.clone();
					merge(channels, kk);
					idft(kk, kk2);
					split(kk2, channels);
					Mat interp_scale_response = channels[0].clone();
					minMaxLoc(interp_scale_response, 0, 0, 0, &max_loc);
					currentScaleFactor = currentScaleFactor * interpScaleFactors[max_loc.x];
					if (currentScaleFactor < min_scale_factor)
						currentScaleFactor = min_scale_factor;
					else if (currentScaleFactor > max_scale_factor)
						currentScaleFactor = max_scale_factor;
				}
			}

			//Training
			//get feature
			Mat patch = get_subwindow(im, pos[1], pos[0], sz[1], sz[0], currentScaleFactor);
			hog_feat = FHoG::extract(patch);
			if (imchannel == 3)
				cn_feat = CNFeat::extract(patch);
			else
			{
				patch.convertTo(patch, CV_32FC1, 1.0 / 255);
				cn_feat[0] = patch - 0.5;
			}
			cn_feat = average_faeture_region(cn_feat, num_pca_fea, use_sz[0], use_sz[1], cn_channel);
			//Update appearance
			if (frame == start_frame)
			{
				x_hog = hog_feat;
				x_cn = cn_feat;
			}
			else
			{
				for (int i = 0; i < num_hog_fea; ++i)
					x_hog[i] = (1 - learning_rate_hog) * x_hog[i] + learning_rate_hog * hog_feat[i];
				for (int i = 0; i < cn_channel; ++i)
					x_cn[i] = (1 - learning_rate_cn) * x_cn[i] + learning_rate_cn * cn_feat[i];
			}
			//PCA
			#pragma omp parallel for
			for (int i = 0; i < num_hog_fea; ++i)
			{
				float *data = x_hog_pca.ptr<float>(i);
				float *data2 = x_hog[i].ptr<float>(0);
				for (int q = 0; q < use_num; ++q)
					data[q] = data2[q];
			}
			Mat x_hog_pca_t = x_hog_pca.t();
			Mat W, V;
			SVD::compute(x_hog_pca*x_hog_pca_t, W, U1, V);
			U1 = U1.t();
			Mat x_cn_pca_t;
			if (imchannel == 3)
			{
				#pragma omp parallel for
				for (int i = 0; i < cn_channel; ++i)
				{
					float *data = x_cn_pca.ptr<float>(i);
					float *data2 = x_cn[i].ptr<float>(0);
					for (int q = 0; q < use_num; ++q)
						data[q] = data2[q];
				}
				x_cn_pca_t = x_cn_pca.t();
				SVD::compute(x_cn_pca*x_cn_pca_t, W, U2, V);
				U2 = U2.t();
			}
			for (int i = 0; i < num_pca_fea; ++i)
			{
				float *data = proj_hog.ptr<float>(i);
				float *data2 = U1.ptr<float>(i);
				for (int q = 0; q < num_hog_fea; ++q)
					data[q] = data2[q];
				if (imchannel == 3)
				{
					float *data3 = proj_cn.ptr<float>(i);
					float *data4 = U2.ptr<float>(i);
					for (int q = 0; q < cn_channel; ++q)
						data3[q] = data4[q];
				}
			}
			proj_hog_t = proj_hog.t();
			Mat x_hog2_temp = x_hog_pca_t * proj_hog_t;
			x_hog2_temp = x_hog2_temp.t();
			Mat x_cn2_temp;
			if (imchannel == 3)
			{
				proj_cn_t = proj_cn.t();
				x_cn2_temp = x_cn_pca_t * proj_cn_t;
				x_cn2_temp = x_cn2_temp.t();
			}
			#pragma omp parallel for
			for (int i = 0; i < num_pca_fea; ++i)
			{
				float *data = x_hog2_temp.ptr<float>(i);
				float *data2 = x_hog2[i].ptr<float>(0);
				for (int q = 0; q < use_num; ++q)
					data2[q] = data[q];
				x_hog2[i] = x_hog2[i].reshape(0, use_sz[0]);
				if (imchannel == 3)
				{
					float *data3 = x_cn2_temp.ptr<float>(i);
					float *data4 = x_cn2[i].ptr<float>(0);
					for (int q = 0; q < use_num; ++q)
						data4[q] = data3[q];
					x_cn2[i] = x_cn2[i].reshape(0, use_sz[0]);
				}
			}
			//compute fft
			for (int i = 0; i < num_pca_fea; ++i)
			{
				xf_hog[i] = newfft2(x_hog2[i], cos_window);
				if (imchannel == 3)xf_cn[i] = newfft2(x_cn2[i], cos_window);
			}
			if (imchannel != 3)xf_cn[0] = newfft2(x_cn[0], cos_window);
			//compute kernel and train model
			if (frame == start_frame)
			{
				Mat kf_hog = new_gaussian_correlation(xf_hog, xf_hog, hogsigma, 1, 1, imchannel, num_pca_fea);
				Mat kf_cn = new_gaussian_correlation(xf_cn, xf_cn, cnsigma, 1, 0, imchannel, num_pca_fea);
				p = newtrainmodel(alphaf_num1, alphaf_num2, alphaf_den1, alphaf_den2, d_num1, d_num2, d_den1, d_den2, kf_hog, kf_cn, yf, frame, start_frame, learning_rate_cn, learning_rate_hog, y, imchannel, lambda);
			}
			else if (frame%gap == 0)
			{
				Mat kf_hog = new_gaussian_correlation(xf_hog, xf_hog, hogsigma, 1, 1, imchannel, num_pca_fea);
				Mat kf_cn = new_gaussian_correlation(xf_cn, xf_cn, cnsigma, 1, 0, imchannel, num_pca_fea);
				p = newtrainmodel(alphaf_num1, alphaf_num2, alphaf_den1, alphaf_den2, d_num1, d_num2, d_den1, d_den2, kf_hog, kf_cn, yf, frame, start_frame, learning_rate_cn, learning_rate_hog, y, imchannel, lambda);
				//cout << p.d[0] << " " << p.d[1] << endl;
			}
			//scale training
			if (use_dsst)
			{
				Mat xs_pca = get_scale_subwindow(im, pos[1], pos[0], base_target_sz[1], base_target_sz[0], scaleSizeFactors, scale_model_sz, currentScaleFactor, nScales, featureRatio, num_hog_fea);
				if (frame == start_frame)
					s_num = xs_pca.clone();
				else
					s_num = (1 - scale_interp_factor) * s_num + scale_interp_factor * xs_pca;
				Mat bigY = s_num.clone();
				Mat bigY_den = xs_pca.clone();

				Mat temp = bigY.mul(mega_scale_window);
				Mat temp_den = bigY_den.mul(mega_scale_window);

				dft(temp, kk, -DFT_ROWS);
				split(kk, channels);
				complex_mat sf_proj_test;
				sf_proj_test.real = channels[0].clone();
				sf_proj_test.img = channels[1].clone();
				dft(temp_den, kk, -DFT_ROWS);

				split(kk, channels);
				complex_mat xsf_test;
				xsf_test.real = channels[0].clone();
				xsf_test.img = channels[1].clone();

				sf_num_test = mul_com(conj_com(sf_proj_test), mega_ysf);
				complex_mat temp_sf_test = mul_com(xsf_test, conj_com(xsf_test));
				complex_mat new_sf_den_test;
				reduce(temp_sf_test.real, new_sf_den_test.real, 0, REDUCE_SUM, CV_32FC1);
				reduce(temp_sf_test.img, new_sf_den_test.img, 0, REDUCE_SUM, CV_32FC1);
				if (frame == start_frame)
				{
					sf_den.real = new_sf_den_test.real.clone();
					sf_den.img = new_sf_den_test.img.clone();
				}
				else
					sf_den = plus_com(mul_com(sf_den, (1 - scale_interp_factor)), mul_com(new_sf_den_test, scale_interp_factor));
			}
			endTime = clock();
			total_time += (double)(endTime - startTime);

			//save results
			target_sz[0] = base_target_sz[0] * currentScaleFactor;
			target_sz[1] = base_target_sz[1] * currentScaleFactor;
			fprintf(res, "%d,%d,%d,%d\n", pos[0], pos[1], target_sz[0], target_sz[1]);
			//printf("seq: %s has completed: %.2lf%%\r", name, frame * 100 / float(end_frame - start_frame));
			//visualization
			if (visualization)
			{
				rectangle(im, Point(pos[1] - target_sz[1] / 2, pos[0] - target_sz[0] / 2), Point(pos[1] + target_sz[1] / 2, pos[0] + target_sz[0] / 2), Scalar(255, 0, 0), 1, 1, 0);
				sprintf(buf, "Seq: %s" , name);
				namedWindow(buf);
				imshow(buf, im);
				waitKey(1);
			}
		}
		//free space
		fclose(res);
		delete[] interpScaleFactors;
		delete[] interp_scale_exp;
		delete[] interp_scale_exp_shift;
		delete[] scaleSizeFactors_test;
		delete[] scaleSizeFactors;
		delete[] scale_exp;
		delete[] scale_exp_shift;
		free(ffm[0]);
		free(fm[0]);
		free(m[0]);
		free(ffm);
		free(fm);
		free(m);
		free(ffmy[0]);
		free(fmy[0]);
		free(ffmy);
		free(fmy);
		destroyWindow(buf);
		//show speed
		total_time = total_time / CLOCKS_PER_SEC;
		float temp_fps = (end_frame - start_frame + 1) / total_time;
		fps += temp_fps;
		cout << "No. "<<seq_num+1<<" seq, speed: " << temp_fps <<" fps         "<< endl;
	}
	//(175fps on OTB2013 at PC of Intel(R) Core(TM) i7-7700 CPU @ 3.60Ghz)
	cout << "Average FPS: " << fps / total_num << endl;
	waitKey(0);
	return 0;
}
