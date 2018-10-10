This is the implementation of our MKCFup paper.  We use DSST instead of fDSST in C++ version for higher speed.
We performed this implementation on a PC with Intel Core i7-7700 3.60GHz CPU and 8GB RAM.
The average FPS is 175 on OTB2013 with Release-x64 mode and our results are stored in ./res.
--
###Before running our code, check if you have finished the following steps.

1. Install visual studio 2015, fftw3.3.5 and opencv3.2 or better;
2. Make sure cn_data.cpp, ComplexMat.cpp and gradientMex.cpp have been added in your project;
3. Open the OpenMP support in your visual studio;
4. Use release mode.

Please run run_MKCF.cpp to use our tracker. If you encounter the speed problem, make sure the tracker gets 10x higher speed in release mode than in debug mode, otherwise, check if you have compiled all the cpp files right.

###Evaluations

1. You need to download OTB2013, OTB2015 or NfS in ./sequences to evaluate our tracker;
2. Choose the corresponding parameters set to achieve the best performance;
3. Files in ./utils can be used for evaluation with Matlab and results are stored in ./res.