// WaveDetector.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "WDetector.h"
#include <opencv2\opencv.hpp>
#include <opencv2\imgproc\types_c.h>

using namespace std;
using namespace cv;

#ifdef _DEBUG  
#pragma comment(lib,"opencv_world400d.lib")
#else  
#pragma comment(lib,"opencv_world400.lib")
#endif


int main()
{
	int nScale = 4;
	string strFile = "E:\\MyProject\\波形检测\\ErrorImg\\S2412\\OK2.bmp";
	//string strFile = "E:\\MyProject\\波形检测\\ErrorImg\\e5.jpg";
	Mat srcImg = imread(strFile);

	Mat resizedImg;
	resize(srcImg, resizedImg, cvSize(srcImg.cols / nScale, srcImg.rows / nScale));
	
	CWDetector cwd;
	Mat resImg;
	cwd.MainProc(resizedImg, resImg);

	//rectangle(resizedImg, r, Scalar(0, 0, 255));

	namedWindow("final");
	imshow("final", resImg);

	waitKey(0);

    return 0;
}

