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
RotatedRect r;

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

		if (RESIZE)
			img_8u3c = resizeFixed(src);
		else
			src.copyTo(img_8u3c);

		// Name of images saved:
		// String save = "1.jpg";

		if (cont_t % INTERVAL == 0)
		{
			for (int j = 0; j < HEIGHT; j++)
			{
				for (int i = 0; i < WIDTH; i++)
				{
					samples[cont_n].at<Vec3b>(j, i) = img_8u3c.at<Vec3b>(j, i);
				}
			}

			// Saving the used frames here:
			/*save[0] = cont_n + 97;
			imwrite(save, img_8u3c);*/

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
	
	//imwrite("bg.jpg", bg_8u3c);
	//exit(1);

	cvtColor(bg_8u3c, bg_lab_8u3c, CV_BGR2Lab);

	// Background gradients will be used in the texture patch test
	bg_8u3c.convertTo(bg_32fc3, CV_32FC3, 1 / 255.0);
	calcGradients(bg_32fc3, bg_dx_32f, bg_dy_32f, bg_mag_32f, bg_ori_32f);

	initBackground(bg_32fc3);
	Mat vibe_mask(Size(WIDTH, HEIGHT), CV_8U);
	Mat vibe_filtered(Size(WIDTH, HEIGHT), CV_8U);

	// Start of the real-time analysis (after initial bg generation) --------------------------/
	while (true)
	{
		bool bSuccess = capture.read(src);

		if (!bSuccess)
		{
			cout << "ERROR: could not read frame from file..." << endl;
			return -1;
		}

		if (RESIZE)
			img_8u3c = resizeFixed(src);
		else
			src.copyTo(img_8u3c);

		//// TCC : print examples
		//Mat print_gray, print_lab, print_float;
		//cvtColor(img_8u3c, print_gray, CV_BGR2GRAY);
		//cvtColor(img_8u3c, print_lab, CV_BGR2Lab);
		//bg_8u3c.convertTo(print_float, CV_32FC3, 1 / 255.0);

		//imwrite("1original.jpg", src);
		//imwrite("1print_gray.jpg", print_gray);
		//imwrite("1print_lab.jpg", print_lab);
		//imwrite("1print_float.jpg", print_float);

		//exit(1);

		// In this point we (may)have the Mat image holding
		// a smaller version of the actual frame.

		cvtColor(img_8u3c, img_lab_8u3c, CV_BGR2Lab);
		img_8u3c.convertTo(img_32fc3, CV_32FC3, 1 / 255.0);

		// Bloco para gerar mascara - usado no lugar do Vibe para debug -----------------------/
		
		if (!VIBE)
		{

			Mat img_8u_gray, bg_8u_gray;
			cvtColor(img_8u3c, img_8u_gray, CV_BGR2GRAY);
			cvtColor(bg_8u3c, bg_8u_gray, CV_BGR2GRAY);
			absdiff(img_8u_gray, bg_8u_gray, diff);
			threshold(diff, mask_8u, 50, 10, CV_8U);
			morphologyEx(mask_8u, mask_8u, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
			//GaussianBlur(mask_8u, mask_8u, Size(3, 3), 0);

			filtered_mask_8u = media_binary(mask_8u, 3, 10);
		}
		else
		{
			// Vibe -------------------------------------------------------------------------------/
			vibe(img_32fc3, vibe_mask);
			morphologyEx(vibe_mask, vibe_mask, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
			filtered_mask_8u = media_binary(vibe_mask, 3, 10);
		}
		// ------------------------------------------------------------------------------------/

		//findConnectedComponents(filtered_mask_8u, components);
		//components.clear();

		findConnectedComponents(filtered_mask_8u, components);

		// For all connected components:
		for (int i = 0; i < components.size(); i++)
		{
			Rect roi = components[i];

			calcGradients(Mat(img_32fc3, roi), Mat(img_dx_32f, roi),
				Mat(img_dy_32f, roi), Mat(img_mag_32f, roi), Mat(img_ori_32f, roi));

			/*imshow("ori_bg", bg_ori_32f);
			imshow("ori_fg", img_ori_32f);
			Mat diff_ori_ffs;
			absdiff(img_ori_32f, bg_ori_32f, diff_ori_ffs);
			threshold(diff_ori_ffs, diff_ori_ffs, 0.75, 255, CV_8U);
			imshow("ori_diff", Mat(diff_ori_ffs, roi));
			waitKey(1000000);*/

			// Detect shadows
			r = detectShadows(Mat(img_8u3c, roi), Mat(img_lab_8u3c, roi), Mat(bg_lab_8u3c, roi), Mat(filtered_mask_8u, roi),
				Mat(img_dx_32f, roi), Mat(img_dy_32f, roi), Mat(img_mag_32f, roi), Mat(img_ori_32f, roi), 
				Mat(bg_dx_32f, roi), Mat(bg_dy_32f, roi), Mat(bg_mag_32f, roi), Mat(img_ori_32f, roi), roi);

			Point2f vtx[4];
			r.points(vtx);

			// Move to the correct area of the original image.
			vtx[0].x += roi.x; vtx[0].y += roi.y;
			vtx[1].x += roi.x; vtx[1].y += roi.y;
			vtx[2].x += roi.x; vtx[2].y += roi.y;
			vtx[3].x += roi.x; vtx[3].y += roi.y;

			r.center.x += roi.x;
			r.center.y += roi.y;

			for (int j = 0; j < 4; j++)
				line(img_8u3c, vtx[j], vtx[(j + 1) % 4], Scalar(0, 255, 255), 2);

			ellipse(img_8u3c, r, Scalar(0, 255, 0), 2);

		}
		components.clear();


		//imshow("mask", filtered_mask_8u);
		//imshow("vibe", vibe_filtered);

		//imshow("img", img_8u3c);

		imshow("final", img_8u3c);
		waitKey(10000000);

		switch (waitKey(1))	{
		case ESC_KEY:
			return 0;
		}
	}

	capture.release();
	return 0;

}