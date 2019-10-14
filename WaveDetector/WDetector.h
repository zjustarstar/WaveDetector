#pragma once
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

//��⵽���ߵĻ�����Ϣ;
struct LineInfo
{
	int y;  //y����;
	int xs; //x������ʼ��;
	int xe; //x�����ս��;
	int n;  //����;
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

	//���������,resImgΪ���صĴ��˼������ͼ;
	void MainProc(Mat srcImg, Mat &resImg);

	//������Ķ�ֵͼbImg����У��;
	float RectifyWaveRegion(Mat srcImg, Mat bImg);
	static double GenerateBImg(Mat srcImg, Mat & bImg);

	//���Ҳ���ͼ�е�ֱ��;
	void FindWaveLines_byHough(Mat srcImg, Mat outerImg);
	void FindWaveLines_byHist(Mat outerImg);
	//����roiRect�����ڵ�ֱ����Ϣ�������浽vecLi�����С�
	void FindLines(Mat outerBImg, Rect roiRect, LineInfo & li);

	//���Ҳ���ͼ����,����������Ķ�ֵͼ���浽bImg;
	Rect FindWaveRegion(Mat srcImg, Mat &bImg);

	//������Ķ�ֵͼbimg�У��������ߵı߽�outimg;
	void GetOuterBoundary(Mat bimg, Mat outimg);
	//������߽�ͼoutimg,��srcImgͼ�ϻ�����߽�;
	void DrawOuterBourdary(Mat outerImg, Mat srcImg);


private:
	Rect m_rectWaveRegion;

};

