#pragma once
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

//检测到的线的基本信息;
struct LineInfo
{
	int y;  //y坐标;
	int xs; //x坐标起始点;
	int xe; //x坐标终结点;
	int n;  //长度;
	LineInfo() {
		y = 0;
		n = 0;
		xs = 0;
		xe = 0;
	}
};

class CWDetector
{
public:

	CWDetector();
	virtual ~CWDetector();

	//入口主函数,resImg为返回的带了检测结果的图;
	void MainProc(Mat srcImg, Mat &resImg);

	//对输入的二值图bImg进行校正;
	float RectifyWaveRegion(Mat srcImg, Mat bImg);
	static double GenerateBImg(Mat srcImg, Mat & bImg);

	//查找波形图中的直线;
	void FindWaveLines_byHough(Mat srcImg, Mat outerImg);
	void FindWaveLines_byHist(Mat outerImg);
	//查找roiRect区域内的直线信息，并保存到vecLi变量中。
	void FindLines(Mat outerBImg, Rect roiRect, LineInfo & li);

	//查找波形图区域,并将该区域的二值图保存到bImg;
	Rect FindWaveRegion(Mat srcImg, Mat &bImg);

	//从输入的二值图bimg中，获得最外边的边界outimg;
	void GetOuterBoundary(Mat bimg, Mat outimg);
	//根据外边界图outimg,在srcImg图上画出外边界;
	void DrawOuterBourdary(Mat outerImg, Mat srcImg);


private:
	Rect m_rectWaveRegion;

};

