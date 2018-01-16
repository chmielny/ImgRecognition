#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <stack>
#include <array>
#include <iterator>
#include <vector>
#include <string>
#include <cmath>

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

// funkcja oblicza momenty geometryczne, sprawdza czy sa podobne do
// szukanych, porownuje polozenia znalezionych elementow i uznaje
// czy to szukane logo
std::vector<std::array<int, 2> > findLogo(cv::Mat& inpImg, const int maxRedLabel, const int maxGreenLabel) {
	CV_Assert(inpImg.depth() != sizeof(uchar));
	cv::Mat_<cv::Vec3b> img = inpImg;			
	int color = 2;
	std::vector<std::array<double, 7> > redMoments(maxRedLabel + 1);		//m00,L,m10,m01,m20,m02,m11
	std::vector<std::array<double, 7> > greenMoments(maxGreenLabel + 1); 
	std::vector<std::array<double, 7> > redIvariants(maxRedLabel + 1);		//S, L, W3, M1, M7, centrum (i,j)
	std::vector<std::array<double, 7> > greenIvariants(maxGreenLabel + 1);		//S, L, W3, M1, M7, centrum (i,j)
	std::vector<std::array<int, 2> > potentialLogos;						//wspolzedne i,j trojek czerwonych trojkotow ktore moga byc logiem

    for ( int i = 1; i < (inpImg.rows - 1); ++i)
		for (int j = 1; j < (inpImg.cols - 1); ++j){
			if (img(i, j)[color] > 0) {
				int label = img(i, j)[color];
				redMoments[label][0]++;					//m00, dodajemy piksel do pola obiektu
				if (img(i-1, j)[color]!=label || img(i, j-1)[color]!=label || img(i+1, j)[color]!=label || img(i, j+1)[color]!=label) 
					redMoments[label][1]++;					//L, dodajemy piksel do obwodu obiektu
				redMoments[label][2] += i;					//m10 
				redMoments[label][3] += j;					//m01
				redMoments[label][4] += i*i;				//m20
				redMoments[label][5] += j*j;				//m02
				redMoments[label][6] += i*j;				//m11
			}
		}
		// obliczenie niezmiennikow momentowych dla obiektow
	for(int k = 0; k <= maxRedLabel; ++k) {
		if (redMoments[k][0] > 0) {
			double M02, M20, M11;
			redIvariants[k][0] = redMoments[k][0]; //S
			redIvariants[k][1] = redMoments[k][1]; //L
			redIvariants[k][2] = (redMoments[k][1] / (2 * sqrt(M_PI *  redMoments[k][0]))) -1 ; //W3
			M02 = redMoments[k][5] - (redMoments[k][3] * redMoments[k][3]) / redMoments[k][0];
			M20 = redMoments[k][4] - (redMoments[k][2] * redMoments[k][2]) / redMoments[k][0];
			redIvariants[k][3] = (M20 + M02) / (redMoments[k][0] * redMoments[k][0]) ; //M1
			M11 = redMoments[k][6] - redMoments[k][2] * redMoments[k][3] / redMoments[k][0]; 
			redIvariants[k][4] = (M20*M02 - M11*M11) / (redMoments[k][0] * redMoments[k][0] * redMoments[k][0] * redMoments[k][0]) ; //M7
			redIvariants[k][5] = (double)(int)(redMoments[k][2] / redMoments[k][0] );  //i
			redIvariants[k][6] = (double)(int)(redMoments[k][3] / redMoments[k][0] ); //j
		}	
	}
	
	// warstwa zielona (kolor bialyu na oryginalnym zdjeciu)
	color = 1;
    for ( int i = 1; i < (inpImg.rows - 1); ++i)
		for (int j = 1; j < (inpImg.cols - 1); ++j){
			if (img(i, j)[color] > 0) {
				int label = img(i, j)[color];
				greenMoments[label][0]++;					//m00, dodajemy piksel do pola obiektu
				if (img(i-1, j)[color]!=label || img(i, j-1)[color]!=label || img(i+1, j)[color]!=label || img(i, j+1)[color]!=label) 
					greenMoments[label][1]++;					//L, dodajemy piksel do obwodu obiektu
				greenMoments[label][2] += i;					//m10 
				greenMoments[label][3] += j;					//m01
				greenMoments[label][4] += i*i;				//m20
				greenMoments[label][5] += j*j;				//m02
				greenMoments[label][6] += i*j;				//m11
			}
		}
		// obliczenie niezmiennikow momentowych dla obiektow
	for(int k = 0; k <= maxGreenLabel; ++k) {
		if (greenMoments[k][0] > 0) {
			double M02, M20, M11;
			greenIvariants[k][0] = greenMoments[k][0]; //S
			greenIvariants[k][1] = greenMoments[k][1]; //L
			greenIvariants[k][2] = (greenMoments[k][1] / (2 * sqrt(M_PI *  greenMoments[k][0]))) -1 ; //W3
			M02 = greenMoments[k][5] - (greenMoments[k][3] * greenMoments[k][3]) / greenMoments[k][0];
			M20 = greenMoments[k][4] - (greenMoments[k][2] * greenMoments[k][2]) / greenMoments[k][0];
			greenIvariants[k][3] = (M20 + M02) / (greenMoments[k][0] * greenMoments[k][0]) ; //M1
			M11 = greenMoments[k][6] - greenMoments[k][2] * greenMoments[k][3] / greenMoments[k][0]; 
			greenIvariants[k][4] = (M20*M02 - M11*M11) / (greenMoments[k][0] * greenMoments[k][0] * greenMoments[k][0] * greenMoments[k][0]) ; //M7
			greenIvariants[k][5] = (double)(int)(greenMoments[k][2] / greenMoments[k][0] );  //i
			greenIvariants[k][6] = (double)(int)(greenMoments[k][3] / greenMoments[k][0] ); //j
		}	
	}


		for(int k = 0; k <= maxRedLabel; ++k) {
			int isLogo = 0;
			int i = 0;
			int j = 0;
			// jezeli znajdziesz element c
			if (std::abs(redIvariants[k][3] - 0.236) < 0.007 && std::abs(redIvariants[k][4] - 0.0078) < 0.0003  && std::abs(redIvariants[k][2] - 0.15) < 0.03) {
				isLogo++;
				i += redIvariants[k][5]; 
				j += redIvariants[k][6]; 
				for(int z = 0; z <= maxRedLabel; ++z) {
					// to szukaj elementu a
					if (std::abs(redIvariants[z][3] - 0.196) < 0.003 && std::abs(redIvariants[z][4] - 0.0084) < 0.0003  && std::abs(redIvariants[z][2] - 0.08) < 0.03 &&
					    redIvariants[z][6] < redIvariants[k][6] && std::abs(redIvariants[z][6] - redIvariants[k][6]) < 30 &&	 
						redIvariants[z][5] < redIvariants[k][5] && std::abs(redIvariants[z][5] - redIvariants[k][5]) < 30) {
							isLogo++;
							i += redIvariants[z][5]; 
							j += redIvariants[z][6]; 
						}
					// oraz elementu b
					if (std::abs(redIvariants[z][3] - 0.269) < 0.02 && std::abs(redIvariants[z][4] - 0.0082) < 0.0003  && std::abs(redIvariants[z][2] - 0.21) < 0.05 &&
					    redIvariants[z][6] < redIvariants[k][6] && std::abs(redIvariants[z][6] - redIvariants[k][6]) < 60 &&	 
						std::abs(redIvariants[z][6] - redIvariants[k][6]) > 40 &&  std::abs(redIvariants[z][5] - redIvariants[k][5]) < 25) {
							isLogo++;
							i += redIvariants[z][5]; 
							j += redIvariants[z][6]; 
						}
				}
				//czy biale V pasuje do reszty loga (czerwonych trojkatow)) 
				if (isLogo == 3) {
					for(int z = 0; z <= maxGreenLabel; ++z) {
						if (std::abs(greenIvariants[z][3] - 0.371) < 0.03 && std::abs(greenIvariants[z][4] - 0.033) < 0.005  && std::abs(greenIvariants[z][2] - 0.83) < 0.08 &&
							std::abs(greenIvariants[z][6] - (j / 3)) < 10 && std::abs(greenIvariants[z][5] - (i / 3)) < 10) {
								isLogo++;
								i += greenIvariants[z][5]; 
								j += greenIvariants[z][6]; 
						} 
					}	
				}	
				if (isLogo == 4) {
					potentialLogos.push_back( {i / 4, j / 4});
				}
				std::cout << "Is logo: " << isLogo << std::endl;

			}
		}
				
		
		for(int k=0; k<=maxRedLabel;++k)
			std::cout << "Moments: " << k << " " << redMoments[k][0]<< " " << redMoments[k][1] << " " 
				 << redMoments[k][2]<< " " << redMoments[k][3] << " "  << redMoments[k][4]<< " "
				 << redMoments[k][5]  << std::endl;
		for(int k=0; k<=maxRedLabel;++k)
			std::cout << "Ivariats: " << k << " " << redIvariants[k][0]<< " " << redIvariants[k][1] << " " 
				 << redIvariants[k][2]<< " " << redIvariants[k][3] << " "  << redIvariants[k][4]<< " "
				 << redIvariants[k][5] << " "  << redIvariants[k][6] << std::endl;
		for(int k=0; k<=maxGreenLabel;++k)
			std::cout << "Moments: " << k << " " << greenMoments[k][0]<< " " << greenMoments[k][1] << " " 
				 << greenMoments[k][2]<< " " << greenMoments[k][3] << " "  << greenMoments[k][4]<< " "
				 << greenMoments[k][5]  << std::endl;
		for(int k=0; k<=maxGreenLabel;++k)
			std::cout << "Ivariats: " << k << " " << greenIvariants[k][0]<< " " << greenIvariants[k][1] << " " 
				 << greenIvariants[k][2]<< " " << greenIvariants[k][3] << " "  << greenIvariants[k][4]<< " "
				 << greenIvariants[k][5] << " "  << greenIvariants[k][6] << std::endl;
		return potentialLogos;
}

void markLogos(cv::Mat& inpImg, std::vector<std::array<int, 2> > logoList) {
	CV_Assert(inpImg.depth() != sizeof(uchar));
	cv::Mat_<cv::Vec3b> img = inpImg;			
	
	for (auto &i : logoList) {
		for (int k= -50; k < 50; ++k) {
			img(i[0] + k, i[1] + 50)[0] = 5;
			img(i[0] + k, i[1] + 50)[1] = 255;
			img(i[0] + k, i[1] + 50)[2] = 5;
			img(i[0] + k, i[1] - 50)[0] = 5;
			img(i[0] + k, i[1] - 50)[1] = 255;
			img(i[0] + k, i[1] - 50)[2] = 5;
			img(i[0] + 50, i[1] + k)[0] = 5;
			img(i[0] + 50, i[1] + k)[1] = 255;
			img(i[0] + 50, i[1] + k)[2] = 5;
			img(i[0] - 50, i[1] + k)[0] = 5;
			img(i[0] - 50, i[1] + k)[1] = 255;
			img(i[0] - 50, i[1] + k)[2] = 5;
		}
	}
	
}

int main(int argc, char * argv[]) {
	if (argc == 2 && (cv::imread(argv[1]).data != NULL)) {
		int maxRedLabel, maxGreenLabel;
		std::vector<std::array<int, 2> > logoList;

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

		maxRedLabel = labeling(imageTr1, imageTr2, 2);			// nazywanie obiektow jednego koloru
		maxGreenLabel = labeling(imageTr2, imageTr1, 1);			// nazywanie obiektor drugiego koloru

		logoList = findLogo(imageTr1, maxRedLabel, maxGreenLabel);
		markLogos(image, logoList);

		std::cout << "MaxRedLabel: " << maxRedLabel << std::endl;
		std::cout << "MaxGreenabel: " << maxGreenLabel << std::endl;

		displayImg("Obraz oryginalny", image);
//		displayImg("Obraz przygotowany", imageTr1);
		imwrite( "./out.bmp", imageTr1 );
		imwrite( "./out2.bmp", image );
		std::cout << image.isContinuous() ;				//<< image2.isContinuous() << std::endl;
		cv::waitKey(-1);
		return 0;
	} else {
		std::cout << "Podaj prawidlowa nazwe pliku graficznego w argumencie wywołania!" << std::endl;
		std::cout << "Prawidłowe użycie: imgrec nazwa_pliku_graficznego" << std::endl;
		return -1;
	}
}
