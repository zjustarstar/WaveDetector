#include "WDetector.h"
#include <fstream>
#include <opencv2/imgproc/types_c.h>

#define SHOW_WAVE_REGION      0   //��ʾ��ֵ����Ĳ�������
#define SHOW_BINARY_RES       0   //��ʾ��ֵ���
#define SHOW_ROTATED_BOUNDARY 0   //��ʾ��ת��Ĳ�����߿�;
#define DRAW_MINAREARECT      0  //�Ƿ���ʾ�����������С��ӿ�;

CWDetector::CWDetector()
{
}


CWDetector::~CWDetector()
{

}

//�����ݶ�����2ֵͼ;
double CWDetector::GenerateBImg(Mat srcImg, Mat & bImg) {

	Mat gray;
	if (srcImg.channels() == 3)
		cvtColor(srcImg, gray, CV_BGR2GRAY);
	else
		gray = srcImg;

	Mat grad_x, grad_y, grad;
	Mat abs_grad_x, abs_grad_y;

	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	/// �� X�����ݶ�
	Sobel(gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	/// �� Y�����ݶ�
	Sobel(gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);

	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	//��ֵ��;
	double d = threshold(grad, bImg, 0, 255, THRESH_OTSU);

	return d;
}

//�ӵ�ǰ�Ķ�ֵͼ(�Ѿ�ȥ���������ķ�Ŀ������Ķ�ֵͼ)�У��������ߵı߽�;
//��߽���255��ʾ;
void CWDetector::GetOuterBoundary(Mat bimg, Mat outimg) {

	//��������;
	for (int y = 0; y < bimg.rows; y++)
	{
		//���;
		for (int x = 0; x < bimg.cols; x++)
		{
			int v = bimg.at<uchar>(y, x);
			if (v == 255)
			{
				outimg.at<uchar>(y, x) = 255;
				break;
			}
		}
		//�ұ�;
		for (int x = bimg.cols - 1; x > 0; x--)
		{
			int v = bimg.at<uchar>(y, x);
			if (v == 255)
			{
				outimg.at<uchar>(y, x) = 255;
				break;
			}
		}
	}

	//��������;
	for (int y = 0; y < bimg.cols; y++)
	{
		//�ϱ�;
		for (int x = 0; x < bimg.rows; x++)
		{
			int v = bimg.at<uchar>(x, y);
			if (v == 255)
			{
				outimg.at<uchar>(x, y) = 255;
				break;
			}
		}

		//�±�;
		for (int x = bimg.rows - 1; x > 0; x--)
		{
			int v = bimg.at<uchar>(x, y);
			if (v == 255)
			{
				outimg.at<uchar>(x, y) = 255;
				break;
			}
		}
	}
}

//��������߽߱�ͼouterImg,��ԭͼsrcImg�л����߽�����;
void CWDetector::DrawOuterBourdary(Mat outerImg, Mat srcImg) {
	//���߿�;
	for (int y = 0; y < outerImg.rows; y++)
		for (int x = 0; x < outerImg.cols; x++)
		{
			int label = outerImg.at<uchar>(y, x);
			if (label == 255)
				srcImg.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
		}
}

Rect CWDetector::FindWaveRegion(Mat srcImg, Mat &bImg) {
	Mat channels[3];

	split(srcImg, channels);

	Mat bimg;
	//double dthre = threshold(channels[1], bimg, 0, 255, THRESH_OTSU);
	GenerateBImg(srcImg, bimg);
	if (SHOW_BINARY_RES) {
		namedWindow("bimg");
		imshow("bimg", bimg);
	}

	Mat labels, stats, centroids;
	int nccomps = cv::connectedComponentsWithStats(bimg, labels, stats, centroids);
	int nMaxArea = stats.at<int>(1, cv::CC_STAT_AREA);//0�Ǳ�������;
	int nMaxIndex = 1;
	for (int i = 2; i < nccomps; i++)
	{
		int nCurArea = stats.at<int>(i, cv::CC_STAT_AREA);
		if (nCurArea > nMaxArea) {
			nMaxArea = nCurArea;
			nMaxIndex = i;
		}
	}

	//��ѡ����������Ե;
	Rect r;
	r.x = stats.at<int>(nMaxIndex, cv::CC_STAT_LEFT);
	r.y = stats.at<int>(nMaxIndex, cv::CC_STAT_TOP);
	r.width = stats.at<int>(nMaxIndex, cv::CC_STAT_WIDTH);
	r.height = stats.at<int>(nMaxIndex, cv::CC_STAT_HEIGHT);

	//ȥ��������ͨ����;
	for (int y = 0; y < bimg.rows; y++)
		for (int x = 0; x < bimg.cols; x++)
		{
			int label = labels.at<int>(y, x);
			if (label != nMaxIndex)
				bimg.at<unsigned char>(y, x) = 0;
		}

	if (SHOW_WAVE_REGION) {
		namedWindow("waveRegion");
		imshow("waveRegion", bimg);
	}

	bImg = bimg;

	return r;
}

//��wave�������У��;
float CWDetector::RectifyWaveRegion(Mat srcImg, Mat bImg) {
	//����С���rect;
	RotatedRect rr;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarcy;
	findContours(bImg, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	//����ж����߽�����,���䶼�����һ���߽磬���Ըñ߽�Ϊ�������м���;
	cout << "contours size=" << contours.size() << endl;
	for (int i = 1; i < contours.size(); i++)
	{
		vector<Point> vecPoint = contours[i];
		for (int m = 0; m < vecPoint.size(); m++)
		{
			contours[0].push_back(vecPoint[m]);
		}
	}

	rr = minAreaRect(Mat(contours[0]));
	if (DRAW_MINAREARECT)
	{
		Point2f pt[4];
		rr.points(pt);
		for (int j = 0; j < 4; j++)
			line(srcImg, pt[j], pt[(j + 1) % 4], Scalar(255, 0, 0), 2, 8);  //������С��Ӿ���ÿ����
	}

	float angle;
	cout << "angle=" << rr.angle << endl;
	angle = rr.angle;

	float fRoatedAngle = angle;
	//���÷���任������ת        ��һ�ַ�����͸�ӱ任
	 if (0 < abs(angle) && abs(angle) <= 45)
		 fRoatedAngle = angle;//������˳ʱ����ת
	 else if (45 < abs(angle) && abs(angle) < 90)
		 fRoatedAngle = 90 - abs(angle);//��������ʱ����ת
	 Point2f center = rr.center;  //������ת��������
	 Mat roateM = getRotationMatrix2D(center, fRoatedAngle, 1);  //�����ת����,˳ʱ��Ϊ������ʱ��Ϊ��

	 Mat rotatedBImg;
	 warpAffine(bImg, rotatedBImg, roateM, bImg.size()); //����任

	 if (SHOW_ROTATED_BOUNDARY) {
		 namedWindow("rotated");
		 imshow("rotated", rotatedBImg);
	 }

	 FindWaveLines_byHist(rotatedBImg);

	 return angle;
}

void CWDetector::FindLines(Mat outerBImg, Rect roiRect,LineInfo & li) {

	int nMargin = 4;

	Mat m = outerBImg(roiRect);

	int nW = m.cols;
	bool * pLineFlag = new bool[nW];  //���ڼ�¼��ǰ���ϵĵ��״̬;
	bool * pTempCopy = new bool[nW];
	memset(pLineFlag, 0, sizeof(bool)*nW);

	int nMax = 0;
	LineInfo finalLi;
	for (int r = 0; r < m.rows; r++)
	{
		memset(pLineFlag, 0, sizeof(bool)*nW);

		//������ÿһ��Ϊ���ĵģ������ؿ���nMargin�к���ܺ���Ϣ;
		for (int subr = r - nMargin; subr < r + nMargin; subr++)
		{
			if (subr < 0 || subr >= m.rows)
				continue;
			
			for (int c = 0; c < m.cols; c++)
			{
				int v = m.at<unsigned char>(subr, c);
				if (v != 0)
					pLineFlag[c] = 1;
			}
		}

		//���֮��Ŀ�϶;
		int nMaxGap = 3; //���С��ֱ������;
		memcpy(pTempCopy, pLineFlag, sizeof(bool)*nW);
		for (int c = 0; c < m.cols; c++)
		{
			if (pTempCopy[c])
			{
				for (int newc = c - nMaxGap; newc < c + nMaxGap; newc++)
				{
					if (newc < 0 || newc >= m.cols) 
						continue;
					pLineFlag[newc] = 1;
				}
			}
		}

		//�������һ����;
		LineInfo l;
		LineInfo temp;  //���������һ����;
		int sum = 0;
		for (int c = 0; c < m.cols; c++) {
			while (!pLineFlag[c]) c++;
			l.xs = c;
			while (pLineFlag[c]) c++;
			l.xe = c-1;
			l.n = l.xe - l.xs + 1;

			if (l.n > sum)
			{
				sum = l.n;
				temp = l;
			}
		}

		//�ҵ��������������һ����,����¼;
		temp.y = r;
		if (temp.n > finalLi.n) {
			finalLi = temp;
		}
	}

	finalLi.y = roiRect.y + finalLi.y;
	finalLi.xe += roiRect.x;
	finalLi.xs += roiRect.x;
	li = finalLi;

	delete[] pLineFlag;
	delete[] pTempCopy;
}


void CWDetector::FindWaveLines_byHist(Mat outerImg) {
	Rect roiRect = m_rectWaveRegion;
	roiRect.x -= 10;
	roiRect.width += 20;
	roiRect.y -= 10;
	roiRect.height += 20;
	Mat roiRegion = outerImg(roiRect);
	int nw = roiRegion.cols;
	int nh = roiRegion.rows;

	LineInfo upLi, dwLi;
	Rect r;
	r.x = 0;
	r.y = 0;
	r.width = roiRegion.cols;
	r.height = 25;
	//�ϰ벿��;
	FindLines(roiRegion, r, upLi);
	//�°벿��;
	r.y = roiRegion.rows - 25;
	FindLines(roiRegion, r, dwLi);

	upLi.y += roiRect.y;
	upLi.xe += roiRect.x;
	upLi.xs += roiRect.x;
	dwLi.y += roiRect.y;
	dwLi.xs += roiRect.x;
	dwLi.xe += roiRect.x;

	Mat tempB = outerImg.clone();
	cvtColor(tempB, tempB, COLOR_GRAY2BGR);

	rectangle(tempB, roiRect, Scalar(0, 255, 0));
	Point pt1 = Point(upLi.xs, upLi.y);
	Point pt2 = Point(upLi.xe,upLi.y);
	line(tempB, pt1, pt2, Scalar(255, 0, 0));
	 pt1 = Point(dwLi.xs, dwLi.y);
	 pt2 = Point(dwLi.xe, dwLi.y);
	line(tempB, pt1, pt2, Scalar(0, 0, 255));

	namedWindow("line");
	imshow("line", tempB);

	//�ϰ벿�ֵ�ͳ��;
	/*Mat histUp = Mat::zeros(cvSize(1,nh/2), CV_16S);   //�洢ÿ�еķ���ͳ��ֵ
	int nMax = 0;
	int nMaxIndex = 0;
	for (int nRow = 0; nRow < nh/2; nRow++)
	{
		Rect rr;
		rr.x = 0;
		rr.width = nw;
		rr.height = 2 * nMargin + 1; //�����¸���Margin�����ؽ���ͳ��;
		rr.y = nRow - nMargin;
		if (rr.y < 0)
		{
			rr.y = 0;
			rr.height = nMargin + nRow + 1;
		}
		else if (rr.y+rr.height >= nh) 
			rr.height = nh - 1 - nRow + nMargin;

		int n;
		n = countNonZero(roiRegion(rr));
		histUp.at<short>(0,nRow) = n;
		if (n > nMax) {
			nMax = n;
			nMaxIndex = nRow;
		}
	}

	double maxx, minn;
	cv::Point min_loc, max_loc;
	cv::minMaxLoc(histUp, &minn, &maxx, &min_loc, &max_loc);

	int y = max_loc.y + m_rectWaveRegion.y;
	Point pt1(m_rectWaveRegion.x, y);
	Point pt2(m_rectWaveRegion.x + m_rectWaveRegion.width, y);
	line(outerImg, pt1, pt2, Scalar(255, 255, 255));

	//�°벿�ֵ�ͳ��;
	Mat histDw = Mat::zeros(cvSize(1, nh / 2), CV_16S);   //�洢ÿ�еķ���ͳ��ֵ
	nMax = 0;
	nMaxIndex = 0;
	for (int nRow = nh/2; nRow < nh; nRow++)
	{
		Rect rr;
		rr.x = 0;
		rr.width = nw;
		rr.height = 2 * nMargin + 1; //�����¸���Margin�����ؽ���ͳ��;
		rr.y = nRow - nMargin;
		if (rr.y < 0) rr.y = 0;
		else if (rr.y + rr.height >= nh) {
			rr.height = nh -1 - nRow + nMargin;
		}

		int n;
		n = countNonZero(roiRegion(rr));
		histDw.at<short>(0, nRow-nh/2) = n;
		if (n > nMax) {
			nMax = n;
			nMaxIndex = nRow;
		}
	}

	cv::minMaxLoc(histDw, &minn, &maxx, &min_loc, &max_loc);

	y = max_loc.y + m_rectWaveRegion.y + nh/2;
	pt1 = Point(m_rectWaveRegion.x, y);
	pt2 = Point(m_rectWaveRegion.x + m_rectWaveRegion.width, y);
	line(outerImg, pt1, pt2, Scalar(255, 255, 255));
	

	namedWindow("line");
	imshow("line", outerImg);*/
}

//�������ֱ��
void CWDetector::FindWaveLines_byHough(Mat srcImg, Mat outerImg) {
	vector<Vec4i> lines;
	//��HoughLines��ͬ���ǣ�HoughLinesP�õ�lines���Ǻ���ֱ���ϵ������ģ�����������л���ʱ�Ͳ�����Ҫ�Լ������������ȷ��Ψһ��ֱ����
	HoughLinesP(outerImg, lines, 1, CV_PI / 180, 20, 30, 5);//ע������������Ϊ��ֵ

	cvtColor(outerImg, outerImg, CV_GRAY2BGR);
	cout << "lines size = " << lines.size() << endl;
	//���λ���ÿ���߶�
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		//ֻ������;
	    if (abs(l[1]-l[3])<5)
			line(outerImg, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 255)); //Scalar�������ڵ����߶���ɫ
	}

	namedWindow("line");
	imshow("line", outerImg);
}

void CWDetector::MainProc(Mat srcImg, Mat &resImg) {

	Mat bImg;
	
	//�ҵ�����ͼ����;
	m_rectWaveRegion = FindWaveRegion(srcImg, bImg);

	//��ȡ�������������Χ�߽�;
	Mat outerBoundary = Mat::zeros(bImg.size(), bImg.type());
	GetOuterBoundary(bImg, outerBoundary);

	//���б߿�У��;
	RectifyWaveRegion(srcImg, outerBoundary);

	resImg = srcImg;
	DrawOuterBourdary(outerBoundary, resImg);

	//�ڽ��ͼ��ʾwave region��
	rectangle(resImg, m_rectWaveRegion, Scalar(255, 0, 0));
}