#include "iostream"
#include <array>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"

#include "vibe.h"
#include "utils.h"
#include "shadow_detection.h"

// Our  video  is  test2.avi, W:960 H:540
// Sanin video 1 is vid1.avi, W:320 H:240
// Sanin video 2 is vid2.avi, W:320 H:240

#define WIDTH 960
#define HEIGHT 540

#define RESIZE true

#define FILENAME "test2.avi"
#define WINDOW_TITLE "Output"
#define ESC_KEY 27 // O código ASCII do ESC.

#define INTERVAL 25
#define SAMPLES 11

#define COMPARE 1
#define VIBE 0

using namespace std;
using namespace cv;

float calcAccuracy(RotatedRect r, Point center, float axis1, float axis2, float angle, Mat &resp);
