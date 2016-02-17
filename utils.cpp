#include "utils.h"



// Compare which Vec3f is larger
bool Compare_Vec3f(Vec3f v1, Vec3f v2)
{
	return ((v1[0] + v1[1] + v1[2]) / 3) < ((v2[0] + v2[1] + v2[2]) / 3);
}

// Resize Mat to 960x540
Mat resizeFixed(Mat src)
{
	Mat res(Size(960, 540), CV_8UC3);

	int rows = res.rows;
	int cols = res.cols;

	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			res.at<Vec3b>(Point(col, row)) = src.at<Vec3b>(Point(col * 2, row * 2));;
		}
	}

	return res;
}


void calcGradients(Mat& img, Mat& dx, Mat& dy, Mat& mag, Mat& ori)
{
	Size s = img.size();
	int rows = s.height;
	int cols = s.width;

	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			// dx
			if (col == 0 || col == img.cols - 1)
				dx.at<float>(row, col) = 0;
			else
			{
				Vec3f left = img.at<Vec3f>(row, col - 1);
				Vec3f right = img.at<Vec3f>(row, col + 1);
				dx.at<float>(row, col) = abs(left[0] - right[0]);
			}

			// dy
			if (row == 0 || row == img.rows - 1)
				dy.at<float>(row, col) = 0;
			else
			{
				Vec3f upper = img.at<Vec3f>(row - 1, col);
				Vec3f lower = img.at<Vec3f>(row + 1, col);
				dy.at<float>(row, col) = abs(upper[0] - lower[0]);
			}

			// mag
			mag.at<float>(row, col) = sqrtf(dy.at<float>(row, col)*dy.at<float>(row, col) + dx.at<float>(row, col)*dx.at<float>(row, col));

			// ori
			ori.at<float>(row, col) = atan2(dy.at<float>(row, col), dx.at<float>(row, col));
			if (ori.at<float>(row, col) < 0)
				ori.at<float>(row, col) += (float)CV_PI * 2;
		}
	}
}

void findConnectedComponents(Mat& img, vector <Rect>& out)
{
	for (int row = 0; row < img.rows; row++)
	{
		unsigned char* ptr = (unsigned char*)img.data + (row * img.step);
		for (int col = 0; col < img.cols; col++)
		{
			if (*ptr == CANDIDATE_FG_VALUE) // Encontrou um novo blob!
			{
				// Inunda a partir da posição atual.
				Rect out_rect;
				int n_painted = floodFill(img, Point(col, row), Scalar(FG_VALUE), &out_rect);

				// Testezinhos simples...
				if (n_painted >= MIN_PIXELS &&
					out_rect.width > MIN_W && out_rect.height > MIN_H &&
					out_rect.y >= BLOB_MARGIN && out_rect.y + out_rect.height < img.rows - BLOB_MARGIN)
					out.push_back(out_rect); // Guarda o retângulo envolvente deste blob.
				else
					floodFill(img, Point(col, row), Scalar(FAILED_FG_VALUE));
			}

			ptr++;
		}
	}
}

Mat media_binary(Mat src, int size, int value) {
	int k = size;
	if (k % 2 == 0)
		k++;

	Size s = src.size();
	int rows = s.height;
	int cols = s.width;

	Mat dst = Mat(Size(cols, rows), CV_8U);

	for (int col = k / 2; col < cols - k / 2; col++)
	{
		for (int row = k / 2; row < rows - k / 2; row++)
		{
			int sum = 0;
			for (int linha = -1 * k / 2; linha <= k / 2; linha++)
			{
				for (int coluna = -1 * k / 2; coluna <= k / 2; coluna++)
				{
					sum += src.at<uchar>(Point(col + coluna, row + linha));
				}
			}

			if (sum >= 5*value)
				dst.at<uchar>(Point(col, row)) = value;
			else
				dst.at<uchar>(Point(col, row)) = 0;
		}
	}

	return dst;
}