#include "shadow_detection.h"
#include "iostream"

#define WIDTH 960
#define HEIGHT 540

#define TEXTURE_TEST_THRESHOLD 0.5f

#define FIT_ELLIPSE true
#define ELLIPSE_REFINE_ITERATIONS 4
#define MAGICAL_VALUE 2.2

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
	float diff_a = (float)(img[1] - bg[1]);
	float diff_b = (float)(img[2] - bg[2]);
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

bool testTextureAvgDiff(Mat& in, Mat& avgDiff)
{
	return true;
}

/*

Era bom fundir este metodo e o proximo

*/
Mat takeCandidateCoords(Mat& mask)
{
	Mat fg_points_buffer(2, mask.rows * mask.cols, CV_32F); // Start with a matrix large enough to keep all the pixels.
	int n_pixels = 0;
	float* ptr_points = (float*)fg_points_buffer.data; // Pointer to the first row.
	int ptr_points_y_offset = fg_points_buffer.step / sizeof (float); // Offset that takes us to the same column in the second row.

	for (int row = 0; row < mask.rows; row++)
	{
		unsigned char* ptr_mask = (unsigned char*)mask.data + row*mask.step;
		for (int col = 0; col < mask.cols; col++)
		{
			if (*ptr_mask == 150) // 255 == FG_VALUE
			{
				*ptr_points = col;
				*(ptr_points + ptr_points_y_offset) = row;
				ptr_points++;
				n_pixels++;
			}
			ptr_mask++;
			// Aqui esta marcando os pixels certos.
		}
	}

	return (Mat(fg_points_buffer, Rect(0, 0, n_pixels, 2)));
}

Mat takeForegroundCoords(Mat& mask)
{
	Mat fg_points_buffer(2, mask.rows * mask.cols, CV_32F); // Start with a matrix large enough to keep all the pixels.
	int n_pixels = 0;
	float* ptr_points = (float*)fg_points_buffer.data; // Pointer to the first row.
	int ptr_points_y_offset = fg_points_buffer.step / sizeof (float); // Offset that takes us to the same column in the second row.

	for (int row = 0; row < mask.rows; row++)
	{
		unsigned char* ptr_mask = (unsigned char*)mask.data + row*mask.step;
		for (int col = 0; col < mask.cols; col++)
		{
			if (*ptr_mask == 255) // 255 == FG_VALUE
			{
				*ptr_points = col;
				*(ptr_points + ptr_points_y_offset) = row;
				ptr_points++;
				n_pixels++;
			}
			ptr_mask++;
			// Aqui esta marcando os pixels certos.
		}
	}

	return (Mat(fg_points_buffer, Rect(0, 0, n_pixels, 2)));

}

RotatedRect refineEllipse(Mat& mask, Mat& initial_mask)
{

	//cout << "starting ellipse refinement\n" << endl;

	//imshow("rE_mask", mask);
	//imshow("rE_initial_mask", initial_mask);
	//waitKey(1000000);

	// We need the foreground pixels coordinates:
	Mat fg_points = takeForegroundCoords(mask);
	Mat initial_fg_points = takeForegroundCoords(initial_mask);
	// Here we have a matrix in size (2, N) with all the coordinates detected as an object (255)

	Mat candidate_points = takeCandidateCoords(mask);

	// TESTANDO SE FORAM SELECIONADOS OS PIXELS CORRETOS
	/*Mat testefdp(Size(WIDTH, HEIGHT), CV_8UC1); int min0 = 0; int min1 = 0;
	for (int i = 0; i < initial_fg_points.cols; i++)
	{
		if (initial_fg_points.at<float>(1, i) > min1)
			min1 = initial_fg_points.at<float>(1, i);
		
		if (initial_fg_points.at<float>(0, i) > min0)
			min0 = initial_fg_points.at<float>(0, i);

		testefdp.at<uchar>(initial_fg_points.at<float>(1, i), initial_fg_points.at<float>(0, i)) = 255;
	}
	cout << "min0: " << min0 << " - min1: " << min1 << endl;
	imshow("testefdp", testefdp);
	waitKey(1000000);*/

	// Other useful matrices
	Mat covar(2, 2, CV_32FC1);
	Mat inv_covar(2, 2, CV_32FC1);
	Mat mean(2, 1, CV_32FC1);
	Mat coordinate_mat(2, 1, CV_32F);

	int inside = 0; // number of points inside ellipse

	// fg_points.cols is the number of foreground pixels ; "for every foregroud pixel"
	for (int iteration = 0; iteration < ELLIPSE_REFINE_ITERATIONS && inside < fg_points.cols; iteration++)
	{ 
		// atualiza os pontos
		fg_points = takeForegroundCoords(mask);

		// Obtain the mean vector and covariance matrix that describes the foreground points as a 2D Gaussian distribution.
		calcCovarMatrix(fg_points, covar, mean, CV_COVAR_NORMAL | CV_COVAR_COLS | CV_COVAR_SCALE, CV_32F);
		invert(covar, inv_covar, DECOMP_SVD);

		// Identify least probable pixels
		inside = 0;
		for (int i = 0; i < fg_points.cols; i++)
		{
			// extract a single pair of coordinates
			coordinate_mat.at<float>(0, 0) = fg_points.at<float>(0, i);
			coordinate_mat.at<float>(0, 1) = fg_points.at<float>(1, i);

			// HOW TO TEST THIS???

			if (Mahalanobis(coordinate_mat, mean, inv_covar) < MAGICAL_VALUE)
			{
				inside++;
			}
			else
			{
				// remove this pixel from the mask
				mask.at<uchar>(coordinate_mat.at<float>(0, 1), coordinate_mat.at<float>(0, 0)) = 30; // = 0

				// replace it with another one, randomly selected
				bool found = false;
				Point2d pt(0, 0);
				float safe = 0;
				while (!found && candidate_points.cols > 0)
				{
					/*int selected = rand() % initial_fg_points.cols;
					pt.x = initial_fg_points.at<float>(0, selected);
					pt.y = initial_fg_points.at<float>(1, selected);*/

					int selected = rand() % candidate_points.cols;
					pt.x = candidate_points.at<float>(0, selected);
					pt.y = candidate_points.at<float>(1, selected);

					coordinate_mat.at<float>(0, 0) = pt.x;	
					coordinate_mat.at<float>(0, 1) = pt.y;

					// Select the pixel only if it fits very well
					if (Mahalanobis(coordinate_mat, mean, inv_covar) < (1.5 + safe))
						found = true;
					else
						safe += 0.01; // EXCEPTIONAL XUNXO

					//double chance = 1.0 - Mahalanobis(coordinate_mat, mean, inv_covar);
					//if (chance > 0)
					//{
					//	double x = rand() / (double)RAND_MAX; // The pixel has a chance of being selected as a foreground pixel.
					//	if (x <= chance)
					//		found = true;
					//}

				}

				mask.at<uchar>(coordinate_mat.at<float>(0, 1), coordinate_mat.at<float>(0, 0)) = 255;
			}
		}
	}

	// Run through the mask once again, looking for shadow pixels inside the foreground area.
	//for (int row = 0; row < mask.rows; row++)
	//{
	//	for (int col = 0; col < mask.cols; col++)
	//	{
	//		coordinate_mat.at<float>(0, 0) = col;
	//		coordinate_mat.at<float>(0, 1) = row;

	//		if (initial_mask.at<float>(col, row) != 0 && mask.at<float>(col, row) == 0) // Detected as shadow.
	//		{
	//			double chance = 1.0 - Mahalanobis(coordinate_mat, mean, inv_covar);
	//			if (chance > 0)
	//			{
	//				double x = rand() / (double)RAND_MAX; // The pixel has a chance of being selected as a foreground pixel.
	//				if (x <= chance)
	//					mask.at<float>(col, row) = 255; // FG_VALUE
	//			}
	//		}
	//	}
	//}

	// Find the ellipse size and rotation angle by eigen analysis.
	float trace = (float)(covar.at <float>(0, 0) + covar.at <float>(1, 1));
	float det = (float)(covar.at <float>(0, 0) * covar.at <float>(1, 1) - covar.at <float>(1, 0) * covar.at <float>(0, 1));
	float term2 = sqrtf(trace*trace / 4.0f - det);
	float eigenvalue1 = trace / 2.0f + term2;
	float angle = (float)((covar.at <float>(0, 1) != 0) ? atan2(eigenvalue1 - covar.at <float>(0, 0), covar.at <float>(0, 1)) : 0);
	float length1 = 4.0f*sqrtf(eigenvalue1); // 5 eh o tamanho do boundingRect
	float length2 = 4.0f*sqrtf(trace / 2.0f - term2);

	return (RotatedRect(Point2f((float)mean.at <float>(0, 0), (float)mean.at <float>(1, 0)),
		Size2f(length1, length2), (float)(angle / CV_PI * 180)));

}

Mat output(Size(WIDTH, HEIGHT), CV_8UC1);

void detectShadows(Mat original, Mat src, Mat bg, Mat& mask, Mat& img_dx, Mat& img_dy, Mat& img_mag,
	Mat& img_ori, Mat& bg_dx, Mat& bg_dy, Mat& bg_mag, Mat& bg_ori, Rect roi)
{
	Size s = src.size();
	int rows = s.height;
	int cols = s.width;

	computeDeltaP(img_dx, img_dy, img_mag, img_ori, bg_dx, bg_dy, bg_mag, bg_ori, Mat(deltaP, roi));
	computeWindowAvgDiff(Mat(deltaP, roi), Mat(windowAvgDiff, roi));

	// Saida temporaria
	//Mat output(Size(WIDTH, HEIGHT), CV_8UC1);
	//Mat output; // COMO ISSO DEVE SER INICIADO??

	//imshow("deltaP", deltaP);
	//imshow("windowAvgDiff", windowAvgDiff); // ESTE RESULTADO ESTA CERTO???
	//waitKey(100000);

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
						

						// Compare with the average orientantion difference
						if (windowAvgDiff.at<float>(j + roi.tl().y, i + roi.tl().x) < 0.05f)
							mask.at<uchar>(j, i) = 150;					

					}
				}
			}
		}
	}

	mask.copyTo(output(roi)); // copy to output at position roi

	imshow("before refine", mask);
	//waitKey(100000);
	
	// Refine using ellipse
	RotatedRect r;
	if (FIT_ELLIPSE)
	{
		r = refineEllipse(Mat(output, roi), mask); //r = refineEllipse(Mat(outpur, roi), mask)

		dilate(output, output, getStructuringElement(cv::MORPH_ELLIPSE, Size(5, 5)));

		Point2f vtx[4];
		r.points(vtx);
		for (int j = 0; j < 4; j++)
			line(original, vtx[j], vtx[(j + 1) % 4], Scalar(0, 255, 255), 2);

		ellipse(original, r, Scalar(0, 255, 0), 2);
	}

	imshow("after refine", Mat(output, roi));
	//waitKey(10000000);
}

