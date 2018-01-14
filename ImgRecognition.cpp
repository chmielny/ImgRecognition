#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <stack>

const float lowpass[3][3] = { { 1.0, 1.0, 1.0 }, { 1.0, 1.0, 1.0 }, { 1.0, 1.0, 1.0 } };
const float hipass[3][3] = { { -1.0, -1.0, -1.0 }, { -1.0, 9.0, -1.0 }, { -1.0, -1.0, -1.0 } };


// funkcja wykonuje splot macierzy 3x3 z obrazem
// w celu nalozenia fitra 
cv::Mat& doConvolution(cv::Mat& I, cv::Mat& Iaft, const float kernel[3][3]){
	CV_Assert(I.depth() != sizeof(uchar));
	cv::Mat_<cv::Vec3b> _Iorg = I; // oryginalny
	cv::Mat_<cv::Vec3b> _I = Iaft; // przerobiony
	float filterSum;
	filterSum = 0;
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			filterSum += kernel[a][b];
		}
	}
	for( int i = 1; i < (I.rows - 1); ++i)
		for (int j = 1; j < (I.cols - 1); ++j){
			cv::Mat_<cv::Vec3b> tmpImg = I(cv::Rect(j - 1, i - 1, 3, 3));			//operator lokalny 3x3
			_I(i, j)[0] = 0;
			_I(i, j)[1] = 0;
			_I(i, j)[2] = 0;							// zerowanie obrazka docelowego
			float tmp[3] = {0.0, 0.0, 0.0};				// zmienna pomocnicza float
			for (int a = 0; a < 3; ++a) {
				for (int b = 0; b < 3; ++b) {
					tmp[0] += tmpImg(a, b)[0] * kernel[a][b];
					tmp[1] += tmpImg(a, b)[1] * kernel[a][b];
					tmp[2] += tmpImg(a, b)[2] * kernel[a][b];
				}
			}
			for(int c = 0; c < 3; ++c) {
				tmp[c] = tmp[c] / filterSum;			// dzielenie przez sume filtra
				if(tmp[c] > 255)
					_I(i, j)[c] = 255;
				else if(tmp[c] < 0)
					_I(i,j)[c] = 0;
				else
					_I(i,j)[c] = tmp[c];
			}
		}
		Iaft = _I;
	return Iaft;
}

// funkcja wykonujaca progowanie dla skladowej czerwonej
// oraz dla barwy bialej (wynik jako skladowa zielona)
cv::Mat& treshold(cv::Mat& inpImg, cv::Mat& outImg) {
	CV_Assert(inpImg.depth() != sizeof(uchar));
	cv::Mat_<cv::Vec3b> _I = inpImg;			// oryginalny
	cv::Mat_<cv::Vec3b> Iout = outImg;			// przerobiony
    for( int i = 0; i < (inpImg.rows); ++i)
		for (int j = 0; j < (inpImg.cols); ++j){
			Iout(i, j)[0] = 0;
			Iout(i, j)[1] = 0;
			Iout(i, j)[2] = 0;
			if (_I(i, j)[2] > 90 && _I(i, j)[0] < 80 &&  _I(i, j)[1] < 80) {	// wykrywanie czerwonego
				Iout(i, j)[2] = 255;
			} 
			if (_I(i, j)[2] > 140 && _I(i, j)[0] > 140 &&  _I(i, j)[1] > 140) {	// wykrywanie bialego
				Iout(i, j)[1] = 255;
			} 
	}
	return outImg;
}

// funkcja przeprowadzajaca erozje na obrazie
cv::Mat& erosion(cv::Mat& inpImg, cv::Mat& outImg, int color) {
	CV_Assert(inpImg.depth() != sizeof(uchar));
	cv::Mat_<cv::Vec3b> _I = inpImg;			// oryginalny
	cv::Mat_<cv::Vec3b> Iout = outImg;			// przerobiony
    for( int i = 1; i < (inpImg.rows - 1); ++i)
		for (int j = 1; j < (inpImg.cols - 1); ++j){
			Iout(i, j)[0] = _I(i, j)[0];
			Iout(i, j)[1] = _I(i, j)[1];
			Iout(i, j)[2] = _I(i, j)[2];
			if (_I(i-1, j)[color] == 0 || _I(i+1, j)[color] == 0 ||  _I(i, j-1)[color] == 0 || _I(i, j+1)[color] == 0) 	
				Iout(i, j)[color] = 0;
	}
	return outImg;
}

// funkcja przeprowadzajaca dylatacje na obrazie 
cv::Mat& dilation(cv::Mat& inpImg, cv::Mat& outImg, int color) {
	CV_Assert(inpImg.depth() != sizeof(uchar));
	cv::Mat_<cv::Vec3b> _I = inpImg;			// oryginalny
	cv::Mat_<cv::Vec3b> Iout = outImg;			// przerobiony
    for( int i = 1; i < (inpImg.rows - 1); ++i)
		for (int j = 1; j < (inpImg.cols - 1); ++j){
			Iout(i, j)[0] = _I(i, j)[0];
			Iout(i, j)[1] = _I(i, j)[1];
			Iout(i, j)[2] = _I(i, j)[2];
			if (_I(i-1, j)[color] == 255 || _I(i+1, j)[color] == 255 ||  _I(i, j-1)[color] == 255 || _I(i, j+1)[color] == 255) 	
				Iout(i, j)[color] = 255;
	}
	return outImg;
}

// funkcja wyswietlajaca obraz
void displayImg(std::string windowName, cv::Mat image) {
	cv::namedWindow(windowName, cv::WINDOW_NORMAL);
	cv::resizeWindow(windowName, 1024, 768);
	cv::imshow(windowName,image);
	return;
} 

// funkcja zamieniajaca obraz binarny (wartosci 0 i 255) na obraz
// w ktorym kolejne obiekty maja rozne wartosci w obrebie danej
// skladowej RGB
int labeling(const cv::Mat& inpImg, cv::Mat& outLabels, int color) {
	CV_Assert(inpImg.depth() != sizeof(uchar));
	cv::Mat_<cv::Vec3b> img = inpImg;			// oryginalny
	cv::Mat_<cv::Vec3b> labels = outLabels;			// przerobiony

    int label = 0;
    int w = inpImg.cols;
    int h = inpImg.rows;
    int i;

    for (int y = 0; y<h; y++)
        for (int x = 0; x<w; x++) {
			labels(y, x)[color] = 0;
			if(color != 0)
				labels(y, x)[0] = img(y,x)[0];
			if(color != 1)
				labels(y, x)[1] = img(y,x)[1];
			if(color != 2)
				labels(y, x)[2] = img(y,x)[2];
		}

    cv::Point point;
    for (int y = 0; y<h; y++)
        for (int x = 0; x<w; x++) {
            if ((img(y, x)[color]) > 0) {	// jezeli badany piksel nie jest tlem
                std::stack<int, std::vector<int> > stack2;
                i = x + y*w;				// zeby dalo sie wyluskac wiersz i kolumne z int-a
                stack2.push(i);

                std::vector<cv::Point> comp;

                while (!stack2.empty()) {
                    i = stack2.top();
                    stack2.pop();

                    int x2 = i%w;
                    int y2 = i / w;			// wspomniane wczesniej wyluskanie wiersza i kolumny

                    img(y2, x2)[color] = 0;

                    point.x = x2;
                    point.y = y2;
                    comp.push_back(point);
					// osiem sasiadow piksela
                    if (x2 > 0 && (img(y2, x2 - 1)[color] != 0)) {
                        stack2.push(i - 1);
                        img(y2, x2 - 1)[color] = 0;
                    }
                    if (y2 > 0 && (img(y2 - 1, x2)[color] != 0)) {
                        stack2.push(i - w);
                        img(y2 - 1, x2)[color] = 0;
                    }
                    if (y2 < h - 1 && (img(y2 + 1, x2)[color] != 0)) {
                        stack2.push(i + w);
                        img(y2 + 1, x2)[color] = 0;
                    }
                    if (x2 < w - 1 && (img(y2, x2 + 1)[color] != 0)) {
                        stack2.push(i + 1);
                        img(y2, x2 + 1)[color] = 0;
                    }
                    if (x2 > 0 && y2 > 0 && (img(y2 - 1, x2 - 1)[color] != 0)) {
                        stack2.push(i - w - 1);
                        img(y2 - 1, x2 - 1)[color] = 0;
                    }
                    if (x2 > 0 && y2 < h - 1 && (img(y2 + 1, x2 - 1)[color] != 0)) {
                        stack2.push(i + w - 1);
                        img(y2 + 1, x2 - 1)[color] = 0;
                    }
                    if (x2 < w - 1 && y2>0 && (img(y2 - 1, x2 + 1)[color] != 0)) {
                        stack2.push(i - w + 1);
                        img(y2 - 1, x2 + 1)[color] = 0;
                    }
                    if (x2 < w - 1 && y2 < h - 1 && (img(y2 + 1, x2 + 1)[color] != 0)) {
                        stack2.push(i + w + 1);
                        img(y2 + 1, x2 + 1)[color] = 0;
                    }
				}

                ++label;
                for (int k = 0; k <comp.size(); ++k) {
                    labels(comp[k])[color] = label;
                }
            }
			// usuwanie ramki 1-pikselowej wokol obrazu
			if ( y == 0 || y == (h-1) || x == 0 || x == (w-1)) {
				labels(y, x)[0] = 0;
				labels(y, x)[1] = 0;
				labels(y, x)[2] = 0;
			}
        }
    return label;
}


int main(int argc, char * argv[]) {
	if (argc == 2 && (cv::imread(argv[1]).data != NULL)) {
		std::cout << "Start ..." << std::endl;
		cv::Mat image = cv::imread(argv[1]);
		cv::Mat imageTr1 = cv::imread(argv[1]);
		cv::Mat imageTr2 = cv::imread(argv[1]);
		doConvolution(image, imageTr1, lowpass);	// imageTr1 - obraz po filtrowaniu dolnoprzepustowym
		treshold(imageTr1, imageTr2);				// imageTr2 - obraz po filtriowaniu i progowaniu
		erosion(imageTr2, imageTr1, 1);				// imageTr1 - obraz po filtrowaniu, progowaniu i erozji 
		dilation(imageTr1, imageTr2, 1);			// imageTr2 - obraz po filtrowaniu, progowaniu, erozji i dylatacji
		dilation(imageTr2, imageTr1, 1);			// imageTr1 - obraz po filtrowaniu, progowaniu, erozji i dwoch dylatacjach
		dilation(imageTr1, imageTr2, 1);			// imageTr1 - obraz po filtrowaniu, progowaniu, erozji i trech dylatacjach
		erosion(imageTr2, imageTr1, 2);				// imageTr1 - obraz po filtrowaniu, progowaniu i erozji drugiej skladowej (czerwonego))
		dilation(imageTr1, imageTr2, 2);			// imageTr1 - obraz po filtrowaniu, progowaniu, erozji i dylatacji drugiej skladowej
		dilation(imageTr2, imageTr1, 2);			// imageTr1 - obraz po filtrowaniu, progowaniu, erozji i dwoch dylatacjach drugiej skladowej

		labeling(imageTr1, imageTr2, 2);			// nazywanie obiektow jednego koloru
		labeling(imageTr2, imageTr1, 1);			// nazywanie obiektor drugiego koloru
		displayImg("Obraz oryginalny", image);
		displayImg("Obraz przygotowany", imageTr1);
		imwrite( "./out.bmp", imageTr1 );
		std::cout << image.isContinuous() ;				//<< image2.isContinuous() << std::endl;
		cv::waitKey(-1);
		return 0;
	} else {
		std::cout << "Podaj prawidlowa nazwe pliku graficznego w argumencie wywołania!" << std::endl;
		std::cout << "Prawidłowe użycie: imgrec nazwa_pliku_graficznego" << std::endl;
		return -1;
	}
}
