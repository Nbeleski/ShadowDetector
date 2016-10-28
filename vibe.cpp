// vibe.cpp

#include "vibe.h"

/* Parameters ViBe ***************************************************/
#define N  20		// samples per pixel
int R = 20;		// radius of the sphere
int close = 2;	// number of close samples for being part of bg
int phi = 16;	// ammount of random subsampling

float floatR = 0.14f;

uchar foreground = 255;
uchar background = 0;

/* Data **************************************************************/
Mat samples[N];
Mat segMap = Mat::zeros(Size(WIDTH, HEIGHT), CV_8U);

/* Random ************************************************************/
random_device rd;
mt19937 eng(rd());
uniform_int_distribution<> distribution(0, phi - 1);
uniform_int_distribution<> rand_sample(0, N - 1);
uniform_int_distribution<> rand_neighbour(-R, R);
int jr, ir;

int getRandNeighbourJ(int j)
{
	int new_j = j + rand_neighbour(eng);
	return min(max(new_j, 0), HEIGHT - 1);
}

int getRandNeighbourI(int i)
{
	int new_i = i + rand_neighbour(eng);
	return  min(max(new_i, 0), WIDTH - 1);
}


// start all samples using a reference image
void initBackground(Mat src)
{
	//Mat float_bg(src.size(), CV_32FC3);
	//src.convertTo(float_bg, CV_32F, 1 / 255.0);
	for (int i = 0; i < N; i++)
	{
		//float_bg.copyTo(samples[i]);
		src.copyTo(samples[i]);
	}
}

// Vibe arguments:
// 1. Source image in grayscale
// 2. Mask to be altered


void vibe(Mat src, Mat& dst)
{
	// for each pixel
	for (int j = 0; j < HEIGHT; ++j)
	{
		for (int i = 0; i < WIDTH; ++i)
		{
			// compare pixel to background model
			int count = 0, index = 0;
			float dist = 0;

			while ((count < close) && (index < N))
			{
				// Euclidean dist
				float d0 = (src.at<Vec3f>(j, i)[0] - samples[index].at<Vec3f>(j, i)[0]);
				float d1 = (src.at<Vec3f>(j, i)[1] - samples[index].at<Vec3f>(j, i)[1]);
				float d2 = (src.at<Vec3f>(j, i)[2] - samples[index].at<Vec3f>(j, i)[2]);
				dist = MIN(1.0f, sqrtf(d0*d0 + d1*d1 + d2*d2) / 1.732051f);

				//cout << d0 << " : " << d1 << " : " << d2 << ":" << dist << endl;
				//cout << dist << endl;

				if (dist < floatR)
					count++;

				index++;

				// ckassify pixel and update model
				if (count >= close)
				{
					// is background
					segMap.at<uchar>(j, i) = background;

					// update pixel model
					int rand = distribution(eng);
					if (rand == 0)
					{
						// replace random sample
						rand = rand_sample(eng);
						samples[rand].at<Vec3f>(j, i) = src.at<Vec3f>(j, i);
					}

					rand = distribution(eng);
					if (rand == 0)
					{
						// get random neighbour pixel
						jr = getRandNeighbourJ(j);
						ir = getRandNeighbourI(i);
						// 
						rand = rand_sample(eng);
						samples[rand].at<Vec3f>(jr, ir) = src.at<Vec3f>(j, i);
					}

				}
				else // count < close
				{
					segMap.at<uchar>(j, i) = foreground;
				}
			}
		}
	}

	segMap.copyTo(dst);
}