#include "dip.h"
#include <Windows.h>
#include <QPushButton>
#include <qmessagebox.h>
#include <qfiledialog.h>
#include <qlabel.h>
#include <qpixmap.h>
#include <QFileDialog>
#include <QDebug>
#include <string>
#include <string.h>
#include <iostream>
#include <fileapi.h>
#include <chrono>

#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

QString xml_path;
QString img_path;
QString save_lot;
QString result_lot;

dip::dip(QWidget* parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
}

// button function to load image file
void dip::openButton() {
	QFileDialog dialog(this);
	dialog.setFileMode(QFileDialog::Directory);
	
	// get image file's name
	save_lot = QFileDialog::getOpenFileName(this, "C:\\", "", "File(*.*)");

	int w = ui.labelImage->width();
	int h = ui.labelImage->height();

	// get image file's path
	img_path = QFileInfo(save_lot).filePath();

	// display image file's path
	ui.label2->setText(img_path);
	
	// convert image to Qimage type and display
	QImage img(img_path);
	QPixmap buf = QPixmap::fromImage(img.scaled(w / 3, h / 3, Qt::KeepAspectRatio));
	ui.labelImage2->setPixmap(buf);
}

// button function to load xml file
void dip::selectButton() {
	// get xml file's name
	QString xml_lot = QFileDialog::getOpenFileName(this, "C:\\", "", "File(*.xml)");
	
	// get xml file's path
	xml_path = QFileInfo(xml_lot).filePath();

	// diplay xml file's path
	ui.label1->setText(xml_path);
}

// button function to start detection
void dip::detectButton() {
	// convert image and xml 's path to string type
	string img_dir = img_path.toStdString();
	string xml_dir = xml_path.toStdString();

	// address to store result image 
	string result_dir = "C:\\Users\\soddong\\face_train\\George DB\\result";
	wstring temp = wstring(result_dir.begin(), result_dir.end());
	
	// if result_dir is none, create directory
	CreateDirectory(temp.c_str(), NULL);

	// Do so that result image's name is same original image's name   
	result_dir += '/' + QFileInfo(save_lot).fileName().toStdString();

	// convert string type to Qstring type
	result_lot = result_lot.fromStdString(result_dir);
	
	// load image and store for mat type
	Mat img = imread(img_dir);

	// resize img to 1024 * 768 pixels 
	cv::resize(img, img, Size(1024, 768), 0, 0, 1);
	
	// declare vector of rect type to store detection box
	vector<Rect> faces;

	// load xml file (training model)
	CascadeClassifier face_cascade;
	face_cascade.load(xml_dir);
	
	// put image and faces vector at cascade classifier
	chrono::system_clock::time_point start = chrono::system_clock::now();
	face_cascade.detectMultiScale(img, faces, 1.3, 3, 0 | CASCADE_SCALE_IMAGE);
	chrono::system_clock::time_point end = chrono::system_clock::now();
	chrono::duration<double, std::milli> d1 = end - start;
	// measure detection time

	// draw detection box at img
	for (int i = 0; i < faces.size(); i++) {
		rectangle(img, faces[i], Scalar(255, 0, 0), 2, 1);
	}

	// store result image includiing detection box
	imwrite(result_dir, img);
	
	// display message
	QMessageBox msg;
	msg.setText("detect!");
	msg.exec();

	int w = ui.labelImage->width();
	int h = ui.labelImage->height();

	// convert result image to Qimage type and display result image includding detection box
	QImage img2(result_lot);
	QPixmap buf = QPixmap::fromImage(img2.scaled(w, h)); // , Qt::KeepAspectRatio
	ui.labelImage->setPixmap(buf);
	ui.labelImage->resize(buf.width(), buf.height());

	// display result values at list window
	ui.listWidget->clear();
	QString s;
	s.append(QString::fromStdString("File name : "));  // image's name
	s.append(QFileInfo(save_lot).fileName()); 
	ui.listWidget->addItem(s);
	s.clear();
	ui.listWidget->addItem(QString::fromStdString("Image size : 1024 * 768"));  // image's size (pixel)
	s.append(QString::fromStdString("Total face number : "));  // detection face or car's number
	s.append(QString::number(faces.size()));
	ui.listWidget->addItem(s);
	s.clear();
	ui.listWidget->addItem(QString::fromStdString("Face size : ")); // detection face or car's size
	for (int i = 0; i < faces.size(); i++) {
		s.append(QString::number(i + 1));
		s.append(QString::fromStdString(") "));
		s.append(QString::number(faces[i].width));
		s.append(QString::fromStdString(" * "));
		s.append(QString::number(faces[i].height));
		ui.listWidget->addItem(s);
		s.clear();
	}
	s.append(QString::fromStdString("Image Time : "));  // detection time (per image)
	s.append(QString::number(d1.count() ));
	s.append(QString::fromStdString(" ms"));
	ui.listWidget->addItem(s);
	s.clear();
	s.append(QString::fromStdString("Face Time : "));  // detection time (per object)
	s.append(QString::number(d1.count() / faces.size()));
	s.append(QString::fromStdString(" ms"));
	ui.listWidget->addItem(s);
}

