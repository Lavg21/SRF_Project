// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

int classifyBayes(Mat img, Mat priors, Mat likelihood, int d, int TH, int C) {
	Mat T(1, d, CV_32FC1);
	T.setTo(0);

	int height = img.rows;
	int width = img.cols;

	// process image
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			uchar value = img.at<uchar>(i, j);
			T.at<float>(0, i * width + j) = value >= TH ? 255 : 0;
		}
	}

	Mat logs(C, 1, CV_64FC1);
	logs.setTo(0);


	// Calculam logaritmul probabilitatilor posterioare
	// pentru fiecare clasa
	for (int c = 0; c < C; c++) {
		double log_c = log(priors.at<double>(c));
		for (int j = 0; j < d; j++) {
			if (T.at<float>(0, j) == 255) {
				log_c += log(likelihood.at<double>(c, j));
			}
			else {
				log_c += log(1 - likelihood.at<double>(c, j));
			}
		}
		logs.at<double>(c) = log_c;
	}

	//std::cout << logs;

	// Alegerea clasei cu logaritmul probabilității maxime
	int clasa = 0;
	double max = logs.at<double>(0);
	for (int c = 1; c < C; c++) {
		if (logs.at<double>(c) > max) {
			max = logs.at<double>(c);
			clasa = c;
		}
	}

	return clasa;
};
/*
void lab9_Bayes() {
	const int C = 10; // number of classes
	int nrinst = 60000;
	const int d = 28 * 28;
	int TH = 128;

	Mat priors(C, 1, CV_64FC1);
	priors.setTo(0);
	Mat likelihood(C, d, CV_64FC1);
	likelihood.setTo(0);

	Mat X(nrinst, d, CV_32FC1);
	X.setTo(0);
	Mat y(nrinst, 1, CV_8UC1);
	y.setTo(0);

	int n_c[C] = { 0 };
	char fname[256];
	int index = 0, rowX = 0;
	for (int c = 0; c < C; c++) {
		index = 0;
		while (1) {
			sprintf(fname, "D:/Facultate/UTCN/An IV/Semestrul 1/Sisteme de recunoastere a formelor/Laborator/Proiect/lab9/train/%d/%06d.png", c, index);
			Mat img = imread(fname, 0);
			if (img.cols == 0)
				break;

			int height = img.rows;
			int width = img.cols;

			// process image
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					uchar value = img.at<uchar>(i, j);
					X.at<float>(rowX, i * width + j) = value >= TH ? 255 : 0;
				}
			}
			y.at<uchar>(rowX) = c;
			rowX++;
			index++;
			n_c[c]++;
		}
	}

	//imshow("Features", X);
	waitKey(0);

	// priors
	for (int c = 0; c < C; c++) {
		// fracţia instanţelor aparţinând clasei i
		// dintre toate instanţele de antrenare
		priors.at<double>(c) = n_c[c] / (double)nrinst;
	}

	printf("PRIORS\n");
	std::cout << priors;

	// likelihood
	for (int c = 0; c < C; c++) {
		for (int j = 0; j < d; j++) {
			// egala cu numărul de instanţe din clasa c care au trăsătura 𝑥𝑗
			// egală cu 255 împărţit la numărul total de instanţe din clasa c
			int count = 0;
			for (int k = 0; k < nrinst; k++) {
				if (X.at<float>(k, j) == 255 && y.at<uchar>(k) == c) {
					count++;
				}
			}
			// trebuie să ne asigurăm că verosimilitățile nu au valoarea 0
			// utilizand netezirea Laplace
			likelihood.at<double>(c, j) = (count + 1) / (double)(n_c[c] + C);
		}
	}

	printf("LIKELIHOOD\n");
	std::cout << likelihood;

	Mat confusion(C, C, CV_32FC1);
	confusion.setTo(0);
	int wrong = 0;
	int nrtest = 0;
	for (int c = 1; c < 2; c++) {
		index = 0;
		while (1) {
			sprintf(fname, "D:/Facultate/UTCN/An IV/Semestrul 1/Sisteme de recunoastere a formelor/Laborator/Proiect/lab9/test/%d/%06d.png", c, index);
			Mat img = imread(fname, 0);
			if (img.cols == 0) break;

			int max_c = classifyBayes(img, priors, likelihood, d, TH, C);

			printf("\n MAX %d", max_c);

			confusion.at<float>(max_c, c)++;

			if (c != max_c) {
				wrong++;
			}

			//printf("Real class is %d, determined class is %d\n", c, max_c);

			index++;
			nrtest++;
		}

		printf("Eroare de clasificare este: %.3f\n", wrong / (float)nrtest);

		double accuracy = 0.0f;
		for (int i = 0; i < C; i++) {
			accuracy += confusion.at<float>(i, i);
		}
		double term = 0.0f;
		for (int i = 0; i < C; i++) {
			for (int j = 0; j < C; j++) {
				term += confusion.at<float>(i, j);
			}
		}

		printf("MATRICEA DE CONFUZIE\n");
		std::cout << confusion;
		int val;
		scanf("%d", &val);

		//printf("\nAccuracy is %.3f\n", accuracy / term);
		//printf("Classification error is %.3f\n", wrong / (double)nrtest);
	}
}*/

std::vector<int> L;
Mat initKMeans() {
	char fname[MAX_PATH];
	std::vector<Point> points;

	while (openFileDlg(fname)) {
		Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				if (img(i, j) == 0) {
					points.push_back(Point(j, i));
				}
			}
		}

		Mat_<double> X = Mat(points.size(), 2, CV_64FC1);

		for (int i = 0; i < points.size(); i++) {
			X(i, 0) = points.at(i).y;
			X(i, 1) = points.at(i).x;
		}

		return X;
	}
}

std::vector<float> calcHist(Mat_<Vec3b> img, int m) {

	std::vector<float> hist;
	int binSize = 256 / m;

	for (int i = 0; i < m * 3; i++) {
		hist.push_back(0);
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			Vec3b pixel = img(i, j);
			for (int k = 0; k < m; k++) {
				for (int l = 0; l < 2; l++) {
					if (pixel[l] >= k * m && pixel[l] < m * (k + 1)) {
						hist.at(pixel[l] + m * l)++;
					}
				}
			}
		}
	}

	int area = img.rows * img.cols;
	for (int i = 0; i < m * 3; i++) {
		hist.at(i) /= area;
	}

	return hist;
}

int knnClassifier(std::vector<float> hist, Mat_<float> X, Mat_<uchar> Y, int k, int nbClasses) {

	std::vector<std::tuple<double, int>> v;
	std::vector<int> voting;

	for (int i = 0; i < nbClasses; i++) {
		voting.push_back(0);
	}

	for (int i = 0; i < X.rows; i++) {
		float distance = 0;
		for (int j = 0; j < X.cols; j++) {
			//distance += pow(hist.at(j) - X(i, j), 2);
			distance += abs(hist.at(j) - X(i, j));
		}
		//v.push_back({ sqrt(distance), Y(i) });
		v.push_back({ distance, Y(i) });
	}

	std::sort(v.begin(), v.end());

	for (int i = 0; i < k; i++) {
		int index = std::get<1>(v.at(i));
		voting.at(index)++;
	}

	int maxim = 0, predictedClass = 0;
	for (int i = 0; i < voting.size(); i++) {
		if (voting.at(i) > maxim) {
			maxim = voting.at(i);
			predictedClass = i;
		}
	}

	return predictedClass;

}

/*void knn() {
	const int nrclasses = 6;
	char classes[nrclasses][10] = { "beach", "city", "desert", "forest", "landscape", "snow" };

	Mat_<float> X(672, 256 * 3, CV_64FC1);
	Mat_<int> Y(672, 1, CV_8UC1);

	char fname[MAX_PATH];
	int rowX = 0;
	for (int c = 0; c < nrclasses; c++) {
		int fileNr = 0;
		while (1) {
			sprintf(fname, "lab8/train/%s/%06d.jpeg", classes[c], fileNr++);
			printf("%s\n", fname);
			Mat img = imread(fname);
			if (img.cols == 0) break;

			std::vector<float> hist = calcHist(img, 256);

			for (int d = 0; d < hist.size(); d++)
				X(rowX, d) = hist.at(d);
			Y(rowX) = c;
			rowX++;
		}
	}

	char fnameTest[MAX_PATH];
	bool go = true;
	int c = 0, fileNr = 0, totalPredicted = 0, correctlyPredicted = 0, k = 10;
	while (go) {
		c++;
		if (c >= nrclasses) {
			go = false;
			break;
		}
		fileNr = 0;

		while (1) {
			sprintf(fnameTest, "lab8/test/%s/%06d.jpeg", classes[c], fileNr++);
			printf("%s\n", fnameTest);
			Mat img = imread(fnameTest);
			if (img.cols == 0) break;

			if (img.cols == 0 && c >= nrclasses) {
				go = false;
				break;
			}

			std::vector<float> hist = calcHist(img, 256);

			int predictedClass = knnClassifier(hist, X, Y, k, nrclasses);
			totalPredicted++;
			if (predictedClass == c) {
				correctlyPredicted++;
			}
		}
	}

	float acc = (100 * correctlyPredicted) / totalPredicted;

	std::cout << acc << "%";
	while (1);
}*/

void knn() {
	const int nrclasses = 5; // Update with the number of car categories: coupe, pickup, sedan, suv, van
	char classes[nrclasses][10] = { "coupe", "pickup", "sedan", "suv", "van" };

	Mat_<float> X(500, 256 * 3, CV_64FC1);
	Mat_<int> Y(500, 1, CV_8UC1);

	char fname[MAX_PATH];
	int rowX = 0;
	for (int c = 0; c < nrclasses; c++) {
		int fileNr = 0;
		while (1) {
			sprintf(fname, "D:/Facultate/UTCN/An IV/Semestrul 1/Sisteme de recunoastere a formelor/Project/SRF_Project/train/%s/%06d.jpg", classes[c], fileNr++);
			printf("%s\n", fname);
			Mat img = imread(fname);
			if (img.cols == 0) break;

			std::vector<float> hist = calcHist(img, 256);

			for (int d = 0; d < hist.size(); d++)
				X(rowX, d) = hist.at(d);
			Y(rowX) = c;
			rowX++;
		}
	}

	char fnameTest[MAX_PATH];
	bool go = true;
	int c = 0, fileNr = 0, totalPredicted = 0, correctlyPredicted = 0, k = 10;
	while (go) {
		c++;
		if (c >= nrclasses) {
			go = false;
			break;
		}
		fileNr = 0;

		while (1) {
			sprintf(fnameTest, "D:/Facultate/UTCN/An IV/Semestrul 1/Sisteme de recunoastere a formelor/Project/SRF_Project/test/%s/%06d.jpg", classes[c], fileNr++);
			printf("%s\n", fnameTest);
			Mat img = imread(fnameTest);
			if (img.cols == 0) break;

			if (img.cols == 0 && c >= nrclasses) {
				go = false;
				break;
			}

			std::vector<float> hist = calcHist(img, 256);

			int predictedClass = knnClassifier(hist, X, Y, k, nrclasses);
			totalPredicted++;
			if (predictedClass == c) {
				correctlyPredicted++;
			}
		}
	}

	float acc = (100 * correctlyPredicted) / totalPredicted;

	std::cout << "Accuracy: " << acc << "%" << std::endl;

	while (1);
}


int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Bayes\n");
		printf(" 11 - KNN\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				//lab9_Bayes();
				break;
			case 11:
				knn();
				break;
		}
	}
	while (op!=0);
	return 0;
}