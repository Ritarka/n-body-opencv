#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>
#include <ncurses.h>
#include <chrono>
#include <random>
#include <fstream>
#include <immintrin.h>

#include <omp.h>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

//Screen Dimensions
#define WORLD_MAX_X 1200
#define WORLD_MAX_Y 800

#define SOFTENING 2
#define NUM_PARTICLES 1000

#define DT 0.1

//Can be changed by user
#define NUM_STEPS 5    //Number of calculations made per epoch
#define EPOCH_STEP 5   //Number of epochs skipped when fast-forwarding
#define INTEREST 3     //Body index of interest
#define DRAW_PATH 0    //Draw path of interest body
#define READ_FILE 0    //0 - Generate random initial conditions, 1 - Read initial conditions from file


using namespace std;
using namespace cv;

class Body {
public:
    double x, y, mass; //x-position , y-position, mass
    double vx, vy;     //x-velocity, y-velocity
    double ax, ay;     //x-acceleration, y-acceleration
    Scalar color;
    // vector<pair<int, int>> path;
};

void randomlyAllocate(Body *data, int n) {
    srand(time(NULL));
    std::uniform_real_distribution<double> unifx(0, WORLD_MAX_X);
    std::uniform_real_distribution<double> unify(0, WORLD_MAX_Y);
    std::default_random_engine re;
    
    for (int i = 0; i < n; i++) {
        data[i].x = unifx(re);
        data[i].y = unify(re);

        // data[i].path.push_back(make_pair(data[i].x, data[i].y));

        data[i].mass = rand() % 4 + 1;
        data[i].color = Scalar(rand() % 255, rand() % 255, rand() % 255);
    }
}

void readInput(Body* p) {
    std::ifstream inFile;
    std::string infilename = "n-body.csv";
    inFile.open(infilename);
    
    if (inFile.is_open())
    {
        string bodyline;
        int bodynum = 0;
        while (getline(inFile, bodyline))
        {
            std::vector<std::string> values;
            std::string::size_type beg = 0, end;
            do {
                end = bodyline.find(',', beg);
                if (end == std::string::npos) {
                    end = bodyline.size();
                }
                values.emplace_back(bodyline.substr(beg, end - beg));
                beg = end + 1;
            } while (beg < bodyline.size());
            
            p[bodynum].x = stod(values[0]);
            p[bodynum].y = stod(values[1]);
            p[bodynum].mass = stod(values[2]);
            p[bodynum].vx = stod(values[3]);
            p[bodynum].vy = stod(values[4]);
            p[bodynum].color = Scalar(stod(values[5]), stod(values[6]), stod(values[7]));
            bodynum++;
        }
    }
}


inline double exact_sqrt_asm(double x) {
    double result;
    asm (
        "sqrtsd %1, %0"
        : "=x" (result)  // Output: any XMM register
        : "x" (x)        // Input: any XMM register
    );
    return result;
}


void calcForce(Body *p, float dt, int numBodies) {
#pragma omp parallel for num_threads(2)
    for (int i = 0; i < numBodies; i++) {
        double fx = 0.0f;
        double fy = 0.0f;
        double invm = 1.0 / p[i].mass;

        for (int j = 0; j < numBodies; j++) {
            double dx = (p[j].x - p[i].x) / 100;
            double dy = (p[j].y - p[i].y) / 100;
            double sqdist = dx*dx + dy*dy + SOFTENING;
            double result;
            asm (
                    "RSQRTSS %1, %0"
                    : "=x" (result)  // Output: any XMM register
                    : "x" (sqdist)        // Input: any XMM register
                );
            double InvDist = result;
            double InvDist3 = InvDist * InvDist * InvDist;
            fx += dx * InvDist3 * p[j].mass * invm;
            fy += dy * InvDist3 * p[j].mass * invm;
        }

        p[i].vx += dt*fx;
        p[i].vy += dt*fy;

        p[i].ax = fx;
        p[i].ay = fy;
    }
}

#define DISPLAY 0

int main() {

    Mat disp(Mat(WORLD_MAX_Y, WORLD_MAX_X, CV_8UC3));
    if (DISPLAY) {
        disp = Scalar(128, 128, 128);
        imshow("N-Particles Gravity Simulation", disp);
    }

    int nBodies = NUM_PARTICLES;
    double dt = DT / NUM_STEPS;

    Body *p = (Body*)malloc(nBodies * sizeof(Body));

    if (READ_FILE) {
        readInput(p);
    } else {
        randomlyAllocate(p, nBodies);
    }



    int pauses = 0;
    int stats = 1; 

    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();

    // for (int iter = 0; std::chrono::duration_cast<std::chrono::seconds>(end - start).count() < 5; iter++) {
    for (int iter = 0; iter < 100; iter++) {
        if (iter % 10 == 0) {
            printf("Done with : %d\n", iter);
        }
        end = std::chrono::steady_clock::now();
        if (pauses <=1) {
            disp = Scalar(0, 0, 0);
        }

        for (int steps = 0; steps < NUM_STEPS; steps++) {

            calcForce(p, dt, nBodies);
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < nBodies; i++) {
                p[i].x += (int)p[i].vx * dt;
                p[i].y += (int)p[i].vy * dt;

                // if (DRAW_PATH && i == INTEREST) {
                //     p[i].path.push_back(make_pair(p[i].x, p[i].y));
                // }

                // if (DISPLAY && steps == NUM_STEPS - 1 && pauses <= 1)
                // {
                //     circle(disp, Point(round(p[i].x), round(p[i].y)), p[i].mass, p[i].color, -1);

                //     for (int j = 0; DRAW_PATH && i == INTEREST && j < p[1].path.size() - 1; j++)
                //     {
                //         line(disp, Point(p[i].path[j].first, p[i].path[j].second), Point(p[i].path[j + 1].first, p[i].path[j + 1].second), Scalar(200, 200, 200), 2);
                //     }
                // }
            }
        }

        if (pauses <= 1 && DISPLAY) {
            putText(disp, "Epoch: " + to_string(iter), Point(10, 30), FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 255, 255), 1, LINE_8, false);
            if (stats) {
                putText(disp, "Position: " + to_string((int)p[INTEREST].x) + ", " + to_string((int)p[INTEREST].y), Point(10, 50), FONT_HERSHEY_COMPLEX, 0.4, Scalar(255, 255, 255), 1, LINE_8, false);
                putText(disp, "Velocity: " + to_string((int)p[INTEREST].vx) + ", " + to_string((int)p[INTEREST].vy), Point(10, 65), FONT_HERSHEY_COMPLEX, 0.4, Scalar(255, 255, 255), 1, LINE_8, false);
                putText(disp, "Acceleration: " + to_string((int)p[INTEREST].ax) + ", " + to_string((int)p[INTEREST].ay), Point(10, 80), FONT_HERSHEY_COMPLEX, 0.4, Scalar(255, 255, 255), 1, LINE_8, false);
            }
            imshow("N-Particles Gravity Simulation", disp);
        }

        char c = (char)waitKey(30);
        if (c == 'q') {
            return 0;
        } else if (pauses || c == 'p') {
            if (pauses) {
                if (--pauses) {
                    continue;
                }
            }

            while (1) {
                c = (char)waitKey(0);
                if (c == 'p')
                    break;
                else if (c == 'q')
                    return 0;
                else if (c == 's') {
                    string name = "Epoch_" + to_string(iter) + ".png";
                    // if (DISPLAY) {
                    //     imwrite(name, disp);
                    // }
                } else if (c == 'n') {
                    pauses = EPOCH_STEP;
                    break;
                } else if (c == 'd') {
                    stats = !stats;
                } else if (c == 'e') {
                    ofstream outFile;
                    std::string filename = "n-body.csv";
                    outFile.open(filename, ofstream::out | ofstream::trunc);

                    if (outFile.is_open())
                    {
                        
                        for (int i = 0; i < nBodies; i++)
                        {
                            outFile << p[i].x << ",";
                            outFile << p[i].y << ",";
                            outFile << p[i].mass << ",";
                            outFile << p[i].vx << ",";
                            outFile << p[i].vy << ",";
                            outFile << p[i].color[0] << ",";
                            outFile << p[i].color[1] << ",";
                            outFile << p[i].color[2] << ",";
                            outFile << "\n";
                        }
                        
                        outFile.close();
                    }
                }
            }
        } else if (c == 'd') {
            stats = !stats;
        }
    }
}