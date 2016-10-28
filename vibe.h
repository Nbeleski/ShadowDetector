// Segmentation.h

#include <iostream>
#include <random>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"

#define WIDTH 960
#define HEIGHT 540

using namespace std;
using namespace cv;

int getRandNeighbourJ(int j);
int getRandNeighbourI(int i);

void initBackground(Mat src);
void vibe(Mat src, Mat& dst);