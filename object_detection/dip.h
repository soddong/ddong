#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_dip.h"

#include <QPushButton>
#include <qmessagebox.h>
#include <qfiledialog.h>
#include <qlabel.h>
#include <qpixmap.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>


class dip : public QMainWindow
{
    Q_OBJECT

public:
    dip(QWidget *parent = Q_NULLPTR);

private:
    Ui::dipClass ui;
    private slots:
    void openButton();
    void selectButton();
    void detectButton();

};
