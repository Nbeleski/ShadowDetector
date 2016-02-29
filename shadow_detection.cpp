#include "shadow_detection.h"
#include "iostream"

#define WIDTH 960
#define HEIGHT 540

#define TEXTURE_TEST_THRESHOLD 0.5f

Mat deltaP(Size(WIDTH, HEIGHT), CV_32FC1);
Mat windowAvgDiff(Size(WIDTH, HEIGHT), CV_32FC1);

Mat blur_buffer1 = Mat(Size(WIDTH, HEIGHT), CV_8UC1);
Mat blur_buffer2 = Mat(Size(WIDTH, HEIGHT), CV_8UC1);

Mat non_zero_in_rows_buffer = Mat(Size(WIDTH, HEIGHT), CV_32FC1);
Mat non_zero_in_windows_buffer = Mat(Size(WIDTH, HEIGHT), CV_32FC1);

bool testLab(Vec3b img, Vec3b bg)
{
	if (img[0] > bg[0])
		return (false);

	// Muito claro.
	if (img[0] > 80)
		return (false);

	// Muito escuro ou claro.
	if (img[0] < 20 || bg[0] > 80)
		return (true);

	// Cor muito diferente.
	float diff_a = img[1] - bg[1];
	float diff_b = img[2] - bg[2];
	float diff = sqrtf(diff_a*diff_a + diff_b*diff_b);
	if (diff > 20)
		return (false);

	return (true);
}

bool testGradient(float img_mag, float img_ori, float bg_mag, float bg_ori)
{
	if (img_mag < 0.01f || bg_mag < 0.01f) // Magnitude muito baixa. Não testa mais.
		return (true);

	if (img_mag > 0.1f && img_mag > bg_mag) // A imagem tem gradientes mais ou menos fortes, mas o BG não tem.
		return (false);

	if (fabs(img_mag - bg_mag) > 0.5f) // Diferenças muito grandes de magnitude.
		return (false);

	if (img_mag > 0.05f && bg_mag > 0.05f)
	{
		/*
		float ori_diff = img_ori - bg_ori;
		if (ori_diff < 0)
			ori_diff += 6.28318531f;
		if (ori_diff >= 3.14159265f)
			ori_diff = 6.28318531f - ori_diff;

		if (ori_diff > 5.0f / 180.0f*3.14159265f)
			return (false);
		*/

		// TESTE COM FASTATAN2, ANGULOS EM GRAUS
		float ori_diff = fabs(img_ori - bg_ori);

		//cout << ori_diff << endl;

		if (ori_diff > 10/90.0)  //5.0f / 180.0f*3.14159265f)
			return (false);
	}

	return (true);
}

void computeDeltaP(Mat& img_dx, Mat& img_dy, Mat& img_mag, Mat& img_ori, Mat& bg_dx, Mat& bg_dy, Mat& bg_mag, Mat& bg_ori, Mat& out)
{
	Size s = img_dx.size();
	int rows = s.height;
	int cols = s.width;

	// for each pixel
	for (int j = 0; j < rows; ++j) //270
	{
		for (int i = 0; i < cols; ++i) //480
		{
			// If gradient magnitude is too low
			if (img_mag.at<float>(j, i) < 0.05f || bg_mag.at<float>(j, i) < 0.05f)
				out.at<float>(j, i) = 0;
			
			else {
				float ax = img_dx.at<float>(j, i);
				float ay = img_dy.at<float>(j, i);
				float bx = bg_dx.at<float>(j, i);
				float by = bg_dy.at<float>(j, i);

				float denom = sqrtf((ax*ax + ay*ay) * (bx*bx + by*by));
				out.at<float>(j, i) = (ax*bx + ay*by) / denom;

				//cout << img_ori.at<float>(j, i) << ":" << bg_ori.at<float>(j, i) << endl;

				//cout << out.at<float>(j, i) << " or " << (img_ori.at<float>(j, i) - bg_ori.at<float>(j, i)) << endl;
				//waitKey(200);

				out.at<float>(j, i) = (float)acos(min(max(out.at<float>(j, i), -1.0f), 1.0f)) / (float)CV_PI;
			}


		}
	}
}

void countNonZeroInRows(Mat& in, Mat& out, int width)
{
	int center_offset = width / 2;
	for (int row = 0; row < in.rows; ++row)
	{
		// fil left and right with 0s
		for (int col = 0; col < center_offset; ++col)
		{
			out.at<float>(row, col) = 0;
			out.at<float>(row, in.cols - 1 - col) = 0;
		}

		float count = 0;
		for (int col = 0; col < width; col++)
		{
			if (in.at<float>(row, col) > 0)
				count++;
			
		}
		out.at<float>(row, center_offset) = count;

		for (int col = center_offset + 1; col < in.cols - center_offset; col++)
		{
			if (in.at<float>(row, col - center_offset - 1) > 0)
				count--;
			if (in.at<float>(row, col + center_offset) > 0)
				count++;
			out.at<float>(row, col) = count;
		}
	}
}

void computeWindowAvgDiff(Mat& input, Mat& output)
{
	// Averages the value deltaP for each pixel looking at neighbour pixels

	Mat blur1(blur_buffer1, Rect(0, 0, input.cols, input.rows));
	Mat blur2(blur_buffer1, Rect(0, 0, input.cols, input.rows));
	input.convertTo(blur1, CV_8U, 255);
	boxFilter(blur1, blur2, CV_8U, Size(TEXTURE_WINDOW_WIDTH, TEXTURE_WINDOW_WIDTH));
	blur2.convertTo(output, CV_32F, 1.0 / 255.0*TEXTURE_WINDOW_WIDTH*TEXTURE_WINDOW_WIDTH);

	Mat non_zero_in_rows(non_zero_in_rows_buffer, Rect(0, 0, input.cols, input.rows));
	countNonZeroInRows(input, non_zero_in_rows, TEXTURE_WINDOW_WIDTH);

	// We can now use another box filter to add the values vertically.
	Mat non_zero_in_windows(non_zero_in_windows_buffer, Rect(0, 0, input.cols, input.rows));
	boxFilter(non_zero_in_rows, non_zero_in_windows, CV_32F, Size(1, TEXTURE_WINDOW_WIDTH), Point(-1, -1), false);

	divide(output, non_zero_in_windows, output);

}

bool myTestTexture(Mat& window)
{
	float diff = 0;
	float total_weight = 0;

	// for each pixel
	for (int j = 0; j < window.rows; ++j)
	{
		for (int i = 0; i < window.cols; ++i)
		{
			if (window.at<float>(j, i) > 0)
			{
				total_weight++;
				diff += window.at<float>(j, i);
			}
		}
	}

	// Evita divisoes por zero
	if (!total_weight)
		total_weight++;

	diff /= total_weight;

	if (diff > 0.5f)
		return false;
	return true;
}

void detectShadows(Mat src, Mat bg, Mat& mask, Mat& img_dx, Mat& img_dy, Mat& img_mag, Mat& img_ori, Mat& bg_dx, Mat& bg_dy, Mat& bg_mag, Mat& bg_ori, Rect roi)
{
	Size s = src.size();
	int rows = s.height;
	int cols = s.width;

	computeDeltaP(img_dx, img_dy, img_mag, img_ori, bg_dx, bg_dy, bg_mag, bg_ori, Mat(deltaP, roi));
	computeWindowAvgDiff(Mat(deltaP, roi), Mat(windowAvgDiff, roi));

	imshow("deltaP", deltaP);
	imshow("windowAvgDiff", windowAvgDiff); // ESTE RESULTADO ESTA CERTO???
	waitKey(100000);


	// for each pixel
	for (int j = 0; j < rows; ++j) //270
	{
		for (int i = 0; i < cols; ++i) //480
		{
			// Se pixel está na mascara
			if (mask.at<uchar>(j, i))// > 255)
			{
				// Teste L*ab, se falso == provavelmente eh carro
				if (testLab(src.at<Vec3b>(j, i), bg.at<Vec3b>(j, i)))
				{
					// Should be a magnitude test here
					//if (testGradient(img_mag.at<float>(j, i), img_ori.at<float>(j, i), bg_mag.at<float>(j, i), bg_ori.at<float>(j, i)))
					{
						

						// AQUI VAI O TESTE DA MEDIA DA JANELA
						// if (windowAvgDiff.at<float>(j, i) < 0.5f)
						//     ...


						
						mask.at<uchar>(j, i) = 150; // remove this later!!

					}
				}
			}
		}
	}
}

