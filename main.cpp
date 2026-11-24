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
#define NUM_PARTICLES 6192

#define DT 0.1

//Can be changed by user
#define NUM_STEPS 5    //Number of calculations made per epoch
#define EPOCH_STEP 5   //Number of epochs skipped when fast-forwarding
#define INTEREST 3     //Body index of interest
#define DRAW_PATH 0    //Draw path of interest body
#define READ_FILE 0    //0 - Generate random initial conditions, 1 - Read initial conditions from file

static_assert(NUM_PARTICLES % 8 == 0);

using namespace std;
using namespace cv;

// vector<double> body_x;
// vector<double> body_y;
// vector<double> masses;
// vector<double> inv_masses;


double* body_x;
double* body_y;
double* masses;
double* inv_masses;


class Body {
public:
    double vx, vy;     //x-velocity, y-velocity
    double ax, ay;     //x-acceleration, y-acceleration
    Scalar color;
    // vector<pair<int, int>> path;
};



void randomlyAllocate(Body *data, int n) {
    srand(0);
    std::uniform_real_distribution<double> unifx(0, WORLD_MAX_X);
    std::uniform_real_distribution<double> unify(0, WORLD_MAX_Y);
    std::default_random_engine re;
    
    for (int i = 0; i < n; i++) {
        body_x[i] = unifx(re);
        body_y[i] = unify(re);

        // data[i].path.push_back(make_pair(data[i].x, data[i].y));

        masses[i] = rand() % 4 + 1;
        inv_masses[i] = 1/ masses[i];
        data[i].color = Scalar(rand() % 255, rand() % 255, rand() % 255);
    }
}

// void readInput(Body* p) {
//     std::ifstream inFile;
//     std::string infilename = "n-body.csv";
//     inFile.open(infilename);
    
//     if (inFile.is_open())
//     {
//         string bodyline;
//         int bodynum = 0;
//         while (getline(inFile, bodyline))
//         {
//             std::vector<std::string> values;
//             std::string::size_type beg = 0, end;
//             do {
//                 end = bodyline.find(',', beg);
//                 if (end == std::string::npos) {
//                     end = bodyline.size();
//                 }
//                 values.emplace_back(bodyline.substr(beg, end - beg));
//                 beg = end + 1;
//             } while (beg < bodyline.size());
            
//             p[bodynum].x = stod(values[0]);
//             p[bodynum].y = stod(values[1]);
//             p[bodynum].mass = stod(values[2]);
//             p[bodynum].vx = stod(values[3]);
//             p[bodynum].vy = stod(values[4]);
//             p[bodynum].color = Scalar(stod(values[5]), stod(values[6]), stod(values[7]));
//             bodynum++;
//         }
//     }
// }


#define DISPLAY 0

void calcForce2(Body *p, float dt, int numBodies) {
#pragma omp parallel for num_threads(2)
    for (int i = 0; i < numBodies; i++) {
        double fx = 0.0f;
        double fy = 0.0f;
        double invm = 1.0 / masses[i];

        for (int j = 0; j < numBodies; j++) {
            double dx = (body_x[j] - body_x[i]) / 100;
            double dy = (body_y[j] - body_y[i]) / 100;
            double sqdist = dx*dx + dy*dy + SOFTENING;
            double result = 1 / sqrt(sqdist);
            double InvDist = result;
            double InvDist3 = InvDist * InvDist * InvDist;
            fx += dx * InvDist3 * masses[j] * invm;
            fy += dy * InvDist3 * masses[j] * invm;
        }

        p[i].vx += dt*fx;
        p[i].vy += dt*fy;

        p[i].ax = fx;
        p[i].ay = fy;
    }
}



double horizontal_sum_m256d(__m256d v) {
    // 1. Extract the high 128 bits and low 128 bits
    __m128d low  = _mm256_castpd256_pd128(v);
    __m128d high = _mm256_extractf128_pd(v, 1); // 1 selects the upper 128 bits

    // 2. Add the high and low 128-bit halves: [v0+v2, v1+v3]
    __m128d sum128 = _mm_add_pd(low, high);

    // 3. Horizontally add the two doubles in the 128-bit register
    // _mm_unpackhi_pd moves the second element to the lower position
    __m128d high64 = _mm_unpackhi_pd(sum128, sum128);
    __m128d result = _mm_add_sd(sum128, high64);

    // 4. Extract the scalar double value
    return _mm_cvtsd_f64(result);
}


void calcForce(Body *p, float dt, int numBodies) {
    asm volatile("# === BEGIN CRITICAL LOOP ===");
    __m256d softening = _mm256_set1_pd(SOFTENING);
    __m256d scale = _mm256_set1_pd(0.01);
    
    #pragma omp parallel for num_threads(2)
    for (int i = 0; i < numBodies; i++) {

        // Broadcast values belonging to body i
        __m256d xi = _mm256_set1_pd(body_x[i]);
        __m256d yi = _mm256_set1_pd(body_y[i]);
        __m256d invm_v = _mm256_set1_pd(inv_masses[i]);

        // Vector accumulators
        __m256d fx_v = _mm256_setzero_pd();
        __m256d fy_v = _mm256_setzero_pd();

        __m256d fx_v2 = _mm256_setzero_pd();
        __m256d fy_v2 = _mm256_setzero_pd();


        int j = 0;

        // Process 8 bodies per iteration
        for (; j < numBodies; j += 8) {
            asm volatile("# === BEGIN INNER LOOP ===");

            {
                // Load 4 positions
                __m256d xj = _mm256_load_pd(&body_x[j]);
                __m256d yj = _mm256_load_pd(&body_y[j]);
                __m256d mj = _mm256_load_pd(&masses[j]);

                // Calculate displacement
                __m256d dx = _mm256_sub_pd(xj, xi);
                __m256d dy = _mm256_sub_pd(yj, yi);

                dx = _mm256_mul_pd(dx, scale);
                dy = _mm256_mul_pd(dy, scale);

                // Calculate squared distance
                __m256d dx2 = _mm256_mul_pd(dx, dx);
                __m256d dy2 = _mm256_mul_pd(dy, dy);

                __m256d sqdist = _mm256_add_pd(dx2, dy2);
                sqdist = _mm256_add_pd(sqdist, softening);

                // Approximate reciprocal square root
                __m256d invDist = _mm256_rsqrt14_pd(sqdist);

                // Calculate inverse distance cubed
                __m256d invDist2 = _mm256_mul_pd(invDist, invDist);
                __m256d invDist3 = _mm256_mul_pd(invDist2, invDist);

                // Calculate force contributions
                __m256d factor = _mm256_mul_pd(invDist3, mj);
                factor = _mm256_mul_pd(factor, invm_v);

                fx_v = _mm256_fmadd_pd(dx, factor, fx_v);
                fy_v = _mm256_fmadd_pd(dy, factor, fy_v);
            }

            {
                // Load 4 positions
                __m256d xj = _mm256_load_pd(&body_x[j+4]);
                __m256d yj = _mm256_load_pd(&body_y[j+4]);
                __m256d mj = _mm256_load_pd(&masses[j+4]);

                // Calculate displacement
                __m256d dx = _mm256_sub_pd(xj, xi);
                __m256d dy = _mm256_sub_pd(yj, yi);

                dx = _mm256_mul_pd(dx, scale);
                dy = _mm256_mul_pd(dy, scale);

                // Calculate squared distance
                __m256d dx2 = _mm256_mul_pd(dx, dx);
                __m256d dy2 = _mm256_mul_pd(dy, dy);

                __m256d sqdist = _mm256_add_pd(dx2, dy2);
                sqdist = _mm256_add_pd(sqdist, softening);

                // Approximate reciprocal square root
                __m256d invDist = _mm256_rsqrt14_pd(sqdist);

                // Calculate inverse distance cubed
                __m256d invDist2 = _mm256_mul_pd(invDist, invDist);
                __m256d invDist3 = _mm256_mul_pd(invDist2, invDist);

                // Calculate force contributions
                __m256d factor = _mm256_mul_pd(invDist3, mj);
                factor = _mm256_mul_pd(factor, invm_v);

                fx_v2 = _mm256_fmadd_pd(dx, factor, fx_v2);
                fy_v2 = _mm256_fmadd_pd(dy, factor, fy_v2);
            }




            asm volatile("# === END INNER LOOP ===");
        }

        // Horizontal reduction
        double fx1 = horizontal_sum_m256d(fx_v);
        double fy1 = horizontal_sum_m256d(fy_v);

        double fx2 = horizontal_sum_m256d(fx_v2);
        double fy2 = horizontal_sum_m256d(fy_v2);

        double fx = fx1 + fx2;
        double fy = fy1 + fy2;

        asm volatile("# === END CRITICAL LOOP ===");

        p[i].vx += dt * fx;
        p[i].vy += dt * fy;

        p[i].ax = fx;
        p[i].ay = fy;
    }

    
}


int main() {

    Mat disp(Mat(WORLD_MAX_Y, WORLD_MAX_X, CV_8UC3));
    if (DISPLAY) {
        disp = Scalar(128, 128, 128);
        imshow("N-Particles Gravity Simulation", disp);
    }

    int nBodies = NUM_PARTICLES;
    double dt = DT / NUM_STEPS;


    size_t bytes = nBodies * sizeof(double);

    // Round allocation size up to a multiple of 64.
    size_t aligned_bytes = (bytes + 63) & ~((size_t)63);

    body_x     = (double*)aligned_alloc(64, aligned_bytes);
    body_y     = (double*)aligned_alloc(64, aligned_bytes);
    masses     = (double*)aligned_alloc(64, aligned_bytes);
    inv_masses = (double*)aligned_alloc(64, aligned_bytes);


    Body *p = (Body*)malloc(nBodies * sizeof(Body));

    if (READ_FILE) {
        // readInput(p);
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
                body_x[i] += (int)p[i].vx * dt;
                body_y[i] += (int)p[i].vy * dt;

                // if (DRAW_PATH && i == INTEREST) {
                //     p[i].path.push_back(make_pair(p[i].x, p[i].y));
                // }

                if (DISPLAY && steps == NUM_STEPS - 1 && pauses <= 1)
                {
                    circle(disp, Point(round(body_x[i]), round(body_y[i])), masses[i], p[i].color, -1);

                    // for (int j = 0; DRAW_PATH && i == INTEREST && j < p[1].path.size() - 1; j++)
                    // {
                    //     line(disp, Point(p[i].path[j].first, p[i].path[j].second), Point(p[i].path[j + 1].first, p[i].path[j + 1].second), Scalar(200, 200, 200), 2);
                    // }
                }
            }
        }

        if (pauses <= 1 && DISPLAY) {
            putText(disp, "Epoch: " + to_string(iter), Point(10, 30), FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 255, 255), 1, LINE_8, false);
            // if (stats) {
            //     putText(disp, "Position: " + to_string((int)p[INTEREST].x) + ", " + to_string((int)p[INTEREST].y), Point(10, 50), FONT_HERSHEY_COMPLEX, 0.4, Scalar(255, 255, 255), 1, LINE_8, false);
            //     putText(disp, "Velocity: " + to_string((int)p[INTEREST].vx) + ", " + to_string((int)p[INTEREST].vy), Point(10, 65), FONT_HERSHEY_COMPLEX, 0.4, Scalar(255, 255, 255), 1, LINE_8, false);
            //     putText(disp, "Acceleration: " + to_string((int)p[INTEREST].ax) + ", " + to_string((int)p[INTEREST].ay), Point(10, 80), FONT_HERSHEY_COMPLEX, 0.4, Scalar(255, 255, 255), 1, LINE_8, false);
            // }
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
                    // ofstream outFile;
                    // std::string filename = "n-body.csv";
                    // outFile.open(filename, ofstream::out | ofstream::trunc);

                    // if (outFile.is_open())
                    // {
                        
                    //     for (int i = 0; i < nBodies; i++)
                    //     {
                    //         outFile << p[i].x << ",";
                    //         outFile << p[i].y << ",";
                    //         outFile << p[i].mass << ",";
                    //         outFile << p[i].vx << ",";
                    //         outFile << p[i].vy << ",";
                    //         outFile << p[i].color[0] << ",";
                    //         outFile << p[i].color[1] << ",";
                    //         outFile << p[i].color[2] << ",";
                    //         outFile << "\n";
                    //     }
                        
                    //     outFile.close();
                    // }
                }
            }
        } else if (c == 'd') {
            stats = !stats;
        }
    }

    free(body_x);
    free(body_y);
    free(masses);
    free(inv_masses);


}