#include "Source.h"

#include "fstream"
#include "sstream"
#include "string"

// Generic declarations ----------------------------------------------/

int n_frame = 1;

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

// Declarações dos testes de eficácia --------------------------------/

class ReferenceValue
{
public:
	int frame;
	int mid_x;
	int mid_y;
	float dist_a1;
	float dist_a2;
	float angle;

	void set_values(int, float, float, float);
};

void ReferenceValue::set_values(int f, float a1, float a2, float ang)
{
	frame = f;
	dist_a1 = a1;
	dist_a2 = a2;
	angle = ang;
}

int n_frame_ref = 0;
vector<ReferenceValue> lista_elipses;

float calcAccuracy(RotatedRect r, Point center, float axis1, float axis2, float angle, Mat &resp)
{
	Mat imgMask = Mat::zeros(Size(WIDTH, HEIGHT), CV_8UC1);
	ellipse(imgMask, r, 255, CV_FILLED);

	Mat imgRoi = Mat::zeros(Size(WIDTH, HEIGHT), CV_8UC1);
	ellipse(imgRoi, center, Size(axis1, axis2), angle, 0, 360, 255, CV_FILLED);

	resp = Mat::zeros(Size(WIDTH, HEIGHT), CV_8UC1);
	bitwise_and(imgMask, imgRoi, resp);

	// ROI
	int dBase = countNonZero(imgRoi);

	// Mask = Referencia
	int dMask = countNonZero(imgMask);

	// Intersection
	int dIntersec = countNonZero(resp);

	// Union = (ROI + Mask) - (Intersection(ROI, Mask))
	int dUnion = (dBase + dMask) - dIntersec;

	if (dIntersec > 10)
	{
		/*cout << "referência: ";
		cout << dMask << " - ";
		cout << "calculado: ";
		cout << dBase << " - ";
		cout << "intesec: ";
		cout << dIntersec << " - ";
		cout << "uniao: ";
		cout << dUnion << " - ";
		cout << "intersec/uniao: ";
		cout << (float(dIntersec) / float(dUnion)) << endl;*/

		cout << n_frame << "\t" << (float(dIntersec) / float(dUnion)) << endl;
	}

	// TODO: Take care... Division by zero...
	return float(dIntersec) / float(dUnion);
}

// -------------------------------------------------------------------/

int partition(array<Mat, SAMPLES>& s, int p, int r, int j, int i)
{
	//int p = 0;
	//int r = 10;
	Vec3f pivot = s[r].at<Vec3f>(j, i);

	while (p < r)
	{
		while (Compare_Vec3f(s[p].at<Vec3f>(j, i), pivot))
			p++;

		while (Compare_Vec3f(pivot, s[r].at<Vec3f>(j, i)))
			r--;

		if (s[p].at<Vec3f>(j, i)[0] == s[r].at<Vec3f>(j, i)[0] &&
			s[p].at<Vec3f>(j, i)[1] == s[r].at<Vec3f>(j, i)[1] &&
			s[p].at<Vec3f>(j, i)[2] == s[r].at<Vec3f>(j, i)[2])
		{
			p++;
		}
		else if (p < r)
		{
			Vec3f tmp = s[p].at<Vec3f>(j, i);
			s[p].at<Vec3f>(j, i) = s[r].at<Vec3f>(j, i);
			s[r].at<Vec3f>(j, i) = tmp;
		}
	}
	return r;
}

Vec3f quickselect(array<Mat, SAMPLES>& s, int p, int r, int k, int j, int i)
{
	if (p == r) return s[p].at<Vec3f>(j, i);

	int jota = partition(s, p, r, j, i);
	int length = jota - p + 1;

	if (length == k) return s[k].at<Vec3f>(j, i);
	else if (k < length) return quickselect(s, p, jota - 1, k, j, i);
	else return quickselect(s, jota + 1, r, k - length, j, i);
}

// -------------------------------------------------------------------/

VideoWriter video("completo.avi", CV_FOURCC('M', 'J', 'P', 'G'), 15, Size(WIDTH, HEIGHT), true);

//--------------------------------------------------------------------/

int main()
{

	// File for results
	FILE* file;
	file = freopen("results.txt", "a", stdout);

	// print parameters
	cout << "Iteracoes Ellipse: " << 4 << " Mahalanobis: " << 2.2 << " Texture window size: " << 11 << endl;

	if (file == NULL)
	{
		exit(-1);
	}

	// Prepare the reference ellipses to compare ---------------------/
	if (COMPARE)
	{
		ifstream infile("frames.txt"); // ONDE PEGA OS FRAMES SALVOS PRA DESENHAR

		// GUARDA TODOS OS QUADROS QUE TEM QUE DESENHAR EM UM VECTOR
		//vector<ReferenceFrame> validation;
		string line;
		string buf;
		while (std::getline(infile, line))
		{
			//v[cont++] = stof(buf);
			//cout << v[cont++] << endl;

			//cout << "line: " << line << endl;
			stringstream ss(line); // Insert the string into a stream

			vector<string> tokens; // Create vector to hold our words
			while (ss >> buf) {
				//cout << cont++ << " : " << buf << endl;
				tokens.push_back(buf);
			}

			ReferenceValue temp;
			//temp.set_values((int)v[0], v[1], v[2], v[3]);

			temp.frame = stoi(tokens[0]);
			temp.mid_x = stoi(tokens[1]);
			temp.mid_y = stoi(tokens[2]);
			temp.dist_a1 = stof(tokens[3]);
			temp.dist_a2 = stof(tokens[4]);
			temp.angle = stof(tokens[5]);

			tokens.clear();

			lista_elipses.push_back(temp);
		}

	}
	// ---------------------------------------------------------------/


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

	//cout << "CHEGANDO AQ??" << endl;
	//for (int j = 0; j < HEIGHT; j++)
	//{
	//	for (int i = 0; i < WIDTH; i++)
	//	{
	//		cout << "done 1 time" << endl;
	//		bg_8u3c.at<Vec3b>(j, i) = quickselect(samples, 0, 10, 6, j, i);
	//	}
	//}

	//imshow("Bg gerado", bg_8u3c);
	//waitKey(100000000);
	//exit(1);

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
			threshold(diff, mask_8u, 50, 10, CV_8U); // 50, 10 default
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
		vector<RotatedRect> ellipses_to_test;


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

			//imshow("quadro", Mat(img_8u3c, roi));
			if (n_frame == 2231)
				imwrite("RENOMEAR.jpg", Mat(img_8u3c, roi));

			//for (int j = 0; j < 4; j++)
			//line(img_8u3c, vtx[j], vtx[(j + 1) % 4], Scalar(0, 255, 255), 2);

			ellipse(img_8u3c, r, Scalar(0, 255, 0), 2); //Gerando a elispe calculada

			ellipses_to_test.push_back(r);

		}
		components.clear();

		if (COMPARE)
		{
			float percent;
			while (lista_elipses[n_frame_ref].frame - 251 == n_frame) // Gerando a elipse de referencia
			{
				//cout << "frame: " << n_frame << " - values: " << lista_elipses[n_frame_ref].frame << " " << lista_elipses[n_frame_ref].dist_a1 << " " << lista_elipses[n_frame_ref].dist_a2 << " " << lista_elipses[n_frame_ref].angle << endl;
				//ellipse(img_8u3c, Point(lista_elipses[n_frame_ref].mid_x, lista_elipses[n_frame_ref].mid_y), Size(lista_elipses[n_frame_ref].dist_a1, lista_elipses[n_frame_ref].dist_a2), lista_elipses[n_frame_ref].angle, 0, 360, Scalar(0, 0, 255), 2);

				Mat intersec = Mat::zeros(Size(WIDTH, HEIGHT), CV_8UC1);

				for (int i = 0; i < ellipses_to_test.size(); i++)
				{
					percent = calcAccuracy(ellipses_to_test[i], Point(lista_elipses[n_frame_ref].mid_x, lista_elipses[n_frame_ref].mid_y), lista_elipses[n_frame_ref].dist_a1, lista_elipses[n_frame_ref].dist_a2, lista_elipses[n_frame_ref].angle, intersec);

					if (percent > 0.5)
					{
						int approx = (int)(percent * 100);
						//putText(img_8u3c, to_string(approx ) + "%", Point(lista_elipses[n_frame_ref].mid_x, lista_elipses[n_frame_ref].mid_y), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 0), 2);
						//putText(img_8u3c, to_string(approx ) + "%", Point(10,80), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 0), 2);

						//waitKey(100000);
					}

				}

				ellipses_to_test.clear();


				n_frame_ref++;
			}
		}

		//imshow("mask", filtered_mask_8u);
		//imshow("vibe", vibe_filtered);

		putText(img_8u3c, to_string(n_frame), Point(10, 40), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 0), 2);
		imshow("final", img_8u3c);
		//waitKey(10000000);

		//Gravacao
		if (n_frame > 80){

			if (VIDEO && n_frame < 850 || n_frame > 1550)
			{
				if (n_frame < 2820 || n_frame > 4300)
				{
					if (n_frame < 5400)
						video.write(img_8u3c);
				}
			}
		}


		if (VIBE && n_frame == 900)
		{
			showAllBackgrounds();
		}

		switch (waitKey(1))	{
		case ESC_KEY:
			return 0;
		}
		n_frame++;
	}

	capture.release();
	return 0;

}