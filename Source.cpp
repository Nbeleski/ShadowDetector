#include "Source.h"

// Generic declarations ----------------------------------------------/

Mat src;
Mat img_8u3c;

Mat diff;
Mat mask_8u(Size(WIDTH, HEIGHT), CV_8U);
Mat filtered_mask_8u(Size(WIDTH, HEIGHT), CV_8U);

Mat img_lab_8u3c(Size(WIDTH, HEIGHT), CV_8U);
Mat bg_lab_8u3c(Size(WIDTH, HEIGHT), CV_8U);

Mat bg_32fc3(Size(WIDTH, HEIGHT), CV_32FC3);
Mat img_32fc3(Size(WIDTH, HEIGHT), CV_32FC3);

// componentes conexos
vector <Rect> components;

// First background model generation ---------------------------------/

array<Mat, SAMPLES> samples;
array<Vec3b, SAMPLES> pixel_list;
int cont_t = 0;
int cont_n = 0;
Mat bg_8u3c(Size(WIDTH, HEIGHT), CV_8UC3);

// Background gradients ----------------------------------------------/

Mat bg_dx_32f(Size(WIDTH, HEIGHT), CV_32FC1);
Mat bg_dy_32f(Size(WIDTH, HEIGHT), CV_32FC1);
Mat bg_mag_32f(Size(WIDTH, HEIGHT), CV_32FC1);
Mat bg_ori_32f(Size(WIDTH, HEIGHT), CV_32FC1);

Mat img_dx_32f(Size(WIDTH, HEIGHT), CV_32FC1);
Mat img_dy_32f(Size(WIDTH, HEIGHT), CV_32FC1);
Mat img_mag_32f(Size(WIDTH, HEIGHT), CV_32FC1);
Mat img_ori_32f(Size(WIDTH, HEIGHT), CV_32FC1);


int main()
{

	// Opening video and testing integrity ---------------------------/

	VideoCapture capture(FILENAME);

	if (!capture.isOpened())
	{
		cerr << "Nao conseguiu abrir o video.\n";
		return -1;
	}

	//----------------------------------------------------------------/

	for (int i = 0; i < samples.size(); i++)
	{
		samples[i] = Mat(Size(WIDTH, HEIGHT), CV_8UC3);
	}

	while (cont_n < SAMPLES)
	{

		bool bSuccess = capture.read(src);

		if (!bSuccess)
		{
			cout << "ERROR: could not read frame from file..." << endl;
			break;
		}

		img_8u3c = resizeFixed(src);

		if (cont_t % INTERVAL == 0)
		{
			for (int j = 0; j < HEIGHT; j++)
			{
				for (int i = 0; i < WIDTH; i++)
				{
					samples[cont_n].at<Vec3b>(j, i) = img_8u3c.at<Vec3b>(j, i);
				}
			}

			cont_n++;
		}
		cont_t++;
	}


	for (int j = 0; j < HEIGHT; j++)
	{
		for (int i = 0; i < WIDTH; i++)
		{
			for (int c = 0; c < SAMPLES; c++)
			{
				pixel_list[c] = samples[c].at<Vec3b>(j, i);
			}
			sort(begin(pixel_list), end(pixel_list), Compare_Vec3f);
			bg_8u3c.at<Vec3b>(j, i) = pixel_list[SAMPLES / 2 + 1];
		}
	}

	// At this point we have an aproximation of the background
	// This is used to start the samples in the Vibe algorithm
	// imshow("Bg gerado", bg_8u3c);

	cvtColor(bg_8u3c, bg_lab_8u3c, CV_BGR2Lab);

	// Background gradients will be used in the texture patch test
	bg_8u3c.convertTo(bg_32fc3, CV_32FC3, 1 / 255.0);
	calcGradients(bg_32fc3, bg_dx_32f, bg_dy_32f, bg_mag_32f, bg_ori_32f);

	// Start of the real-time analysis (after initial bg generation) --------------------------/
	while (true)
	{
		bool bSuccess = capture.read(src);

		if (!bSuccess)
		{
			cout << "ERROR: could not read frame from file..." << endl;
			return -1;
		}

		img_8u3c = resizeFixed(src);
		// In this point we have the Mat image holding
		// a smaller version of the actual frame.

		cvtColor(img_8u3c, img_lab_8u3c, CV_BGR2Lab);
		img_8u3c.convertTo(img_32fc3, CV_32FC3, 1 / 255.0);

		// Bloco para gerar mascara - usado no lugar do Vibe para debug -----------------------/
		Mat img_8u_gray, bg_8u_gray;
		cvtColor(img_8u3c, img_8u_gray, CV_BGR2GRAY);
		cvtColor(bg_8u3c, bg_8u_gray, CV_BGR2GRAY);
		absdiff(img_8u_gray, bg_8u_gray, diff);
		threshold(diff, mask_8u, 50, 10, CV_8U);
		morphologyEx(mask_8u, mask_8u, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)));
		//GaussianBlur(mask_8u, mask_8u, Size(3, 3), 0);

		filtered_mask_8u = media_binary(mask_8u, 3, 10);

		// ------------------------------------------------------------------------------------/

		findConnectedComponents(filtered_mask_8u, components);

		// For all connected components:
		for (int i = 0; i < components.size(); i++)
		{
			Rect roi = components[i];

			calcGradients(Mat(img_32fc3, roi), Mat(img_dx_32f, roi),
				Mat(img_dy_32f, roi), Mat(img_mag_32f, roi), Mat(img_ori_32f, roi));

			// Detect shadows
			detectShadows(Mat(img_lab_8u3c, roi), Mat(bg_lab_8u3c, roi), Mat(filtered_mask_8u, roi), 
				Mat(img_dx_32f, roi), Mat(img_dy_32f, roi), Mat(img_mag_32f, roi), Mat(img_ori_32f, roi), 
				Mat(bg_dx_32f, roi), Mat(bg_dy_32f, roi), Mat(bg_mag_32f, roi), Mat(img_ori_32f, roi));

		}
		components.clear();


		imshow("mask", filtered_mask_8u);
		//imshow("img", img_8u3c);

		switch (waitKey(1))	{
		case ESC_KEY:
			return 0;
		}
	}

	capture.release();
	return 0;

}