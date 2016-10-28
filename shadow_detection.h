#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"

#define TEXTURE_WINDOW_WIDTH 11

using namespace std;
using namespace cv;

bool testLab(Vec3f img, Vec3f bg);
bool testGradient(float img_mag, float img_ori, float bg_mag, float bg_ori);
void computeDeltaP(Mat& img_dx, Mat& img_dy, Mat& img_mag, Mat& img_ori, Mat& bg_dx, Mat& bg_dy, Mat& bg_mag, Mat& bg_ori, Mat& out);
bool myTestTexture(Mat& window);

bool testTextureAvgDiff(Mat& in, Mat& avgDiff);

void computeWindowAvgDiff(Mat& deltaP, Mat& windowAvgDiff);
void countNonZeroInRows(Mat& in, Mat& out, int width);

Mat takeCandidateCoords(Mat& mask);
Mat takeForegroundCoords(Mat& mask);
RotatedRect refineEllipse(Mat& mask, Mat& out);

void detectShadows(Mat original, Mat src, Mat bg, Mat& mask, Mat& img_dx, Mat& img_dy, Mat& img_mag, Mat& img_ori, Mat& bg_dx, Mat& bg_dy, Mat& bg_mag, Mat& bg_ori, Rect roi);
