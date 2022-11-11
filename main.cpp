#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>
#include <ncurses.h>
#include <random>

#include <omp.h>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#define WORLD_MAX_X 1200
#define WORLD_MAX_Y 800

#define SOFTENING 2
#define NUM_PARTICLES 1000

#define DT 0.1


using namespace std;
using namespace cv;

typedef struct {
    double x, y, mass;
    double vx,vy;
    Scalar color;
} Body;

void randomlyAllocate(Body *data, int n) {
    double lower_bound = 0;
    std::uniform_real_distribution<double> unifx(lower_bound, WORLD_MAX_X);
    std::uniform_real_distribution<double> unify(lower_bound, WORLD_MAX_X);
    std::default_random_engine re;
    
    for (int i = 0; i < n; i++) {
        data[i].x = unifx(re);
        data[i].y = unify(re);
        data[i].mass = rand() % 4 + 1;
        data[i].color = Scalar(rand() % 255, rand() % 255, rand() % 255);
    }
}

void calcForce(Body *p, float dt, int numBodies) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numBodies; i++){
        double fx = 0.0f;
        double fy = 0.0f;

        for (int j = 0; j < numBodies; j++) {
            double dx = (p[j].x - p[i].x) / 100;
            double dy = (p[j].y - p[i].y) / 100;
            double sqdist = dx*dx + dy*dy + SOFTENING;
            double InvDist = 1.0f / sqrtf(sqdist);
            double InvDist3 = InvDist * InvDist * InvDist;
            fx += (dx * InvDist3 * p[j].mass) / p[i].mass ;
            fy += (dy * InvDist3 * p[j].mass) / p[i].mass ;
        }

        p[i].vx += dt*fx;
        p[i].vy += dt*fy;

    }
}

int main() {
    double scale_x = 100;
    double scale_y = 100;

    Mat disp(Mat(WORLD_MAX_Y, WORLD_MAX_X, CV_8UC3));
    disp = Scalar(128, 128, 128);
    int nBodies = NUM_PARTICLES;
    double dt = 0.1;

    Body *p = (Body*)malloc(nBodies * sizeof(Body));

    randomlyAllocate(p, nBodies);

    imshow("N-Particles Gravity Simulation", disp);

    int step = 0;

    for (int iter = 0; ; iter++) {
        disp = Scalar(0, 0, 0);
        dt = abs(dt);
        calcForce(p, dt, nBodies);

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nBodies; i++) {
            p[i].x += (int)p[i].vx * dt;
            p[i].y += (int)p[i].vy * dt;
            circle( disp, Point(round(p[i].x), round(p[i].y)), p[i].mass, p[i].color, -1);
        }

        putText(disp, "Epoch: " + to_string(iter), Point(10, 30), FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 255, 255), 1, LINE_8, false);
        imshow("N-Particles Gravity Simulation", disp);

        input:
        char c = (char)waitKey(30);
        if (c == 'q') {
            return 0;
        } else if (step || c == 'p') {
            step = 0;

            while (1) {
                c = (char)waitKey(0);
                if (c == 'p')
                    break;
                else if (c == 'q')
                    return 0;
                else if (c == 's') {
                    string name = "Epoch_" + to_string(iter) + ".png";
                    imwrite(name, disp);
                } else if (c == 'n') {
                    step = 1;
                    break;
                }
            }
        }
    }
}
