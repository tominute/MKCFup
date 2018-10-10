#pragma once
class Params
{
	public:
		bool visualization									= false;	//show the realtime results
		int featureRatio									= 4;		//ratio between the size of the original image and hog features
		double output_sigma_factor							= 1. / 16.;	//the sigma factor of the expected output y
		double cnsigma_color								= 0.65;		//the kernel sigma of CN features in color sequences
		double hogsigma_color								= 0.6;		//the kernel sigma of HOG features in color sequences
		double lambda										= 0.01;		//the regular coefficient 
		int num_cn_fea										= 10;		//channel number of CN features
		int num_hog_fea										= 31;		//channel number of HOG features
		int num_pca_fea										= 4;		//channel number after PCA
		int gap												= 6;		//train model every the gap frames
		
		//Scale
		bool use_dsst										= true;		//This module is the tool for DSST and we use cv::dft instead of fftw for simplicity and convenience
		int nScales											= 20;		//instead of 17
		int nScalesInterp									= 39;		//instead of 33
		double scale_step									= 1.02;		//the same as DSST
		double scale_sigma_factor							= 1. / 16.;	//the same as DSST
		double scale_model_max_area							= 512.0;	//the same as DSST
		double translation_model_max_area					= 50000;	//make sure not too large
		double translation_model_min_area					= 0;		//the same as DSST
		double scale_interp_factor							= 0.025;	//the same as DSST

		void choose_benchmarks(char* benchmark)
		{
			if (strcmp(benchmark, "OTB2013") == 0){}//default
			else if (strcmp(benchmark, "OTB2015") == 0)
			{
				cnsigma_gray = 0.45;
				hogsigma_gray = 0.37;
				learning_rate_cn_gray = 0.016;
				learning_rate_hog_gray = 0.020;
			}
			else if (strcmp(benchmark, "NfS") == 0)
			{
				padding = 1.0;	
				learning_rate_cn_color = 0.0045;
				learning_rate_hog_color = 0.0045;
			}
			else
			{
				cout << "Error input, you should choose one among OTB2013, OTB2015 and NfS." << endl;
				exit(0);
			}

		}
		double get_padding() {return padding;}
		double get_learning_rate_cn_color() { return learning_rate_cn_color; }
		double get_learning_rate_hog_color() { return learning_rate_hog_color; }
		double get_cnsigma_gray() { return cnsigma_gray; }
		double get_hogsigma_gray() { return hogsigma_gray; }
		double get_learning_rate_cn_gray() { return learning_rate_cn_gray; }
		double get_learning_rate_hog_gray() { return learning_rate_hog_gray; }

	private:
		double padding										= 1.5;		//search area outside the object
		double learning_rate_cn_color						= 0.017;	//the learning rate of CN model and appearence in color sequences
		double learning_rate_hog_color						= 0.016;	//the learning rate of HOG model and appearence in color sequences
		double cnsigma_gray									= 0.47;		//the kernel sigma of CN features in gray sequences
		double hogsigma_gray								= 0.37;		//the kernel sigma of HOG features in gray sequences
		double learning_rate_cn_gray						= 0.019;	//the learning rate of CN model and appearence in gray sequences
		double learning_rate_hog_gray						= 0.017;	//the learning rate of HOG model and appearence in gray sequences		
}params;

int num_threads = 8;//use omp, set the number of threads
