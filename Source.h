#include "iostream"
#include <array>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"

#include "utils.h"
#include "shadow_detection.h"

#define WIDTH 960
#define HEIGHT 540

#define FILENAME "test2.avi"
#define WINDOW_TITLE "Output"
#define ESC_KEY 27 // O código ASCII do ESC.

#define INTERVAL 25
#define SAMPLES 11

using namespace std;
using namespace cv;