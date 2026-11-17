build:  main.cpp
	g++ -I/usr/include/opencv4 -L/usr/lib -fopenmp -g -march=native -o n_body_opencv main.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs 
	g++ -S -I/usr/include/opencv4 -L/usr/lib -fopenmp -g -march=native -o n_body_opencv.S main.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -masm=intel
	@[ -f n_body_opencv ] && echo "Compilation successful!" || echo "Compilation failed"

test:
	g++ -I/usr/include/opencv4 -L/usr/lib -fopenmp -O3 -march=native -o n_body_opencv main.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
	g++ -S -I/usr/include/opencv4 -L/usr/lib -fopenmp -O3 -march=native -o n_body_opencv_optimized.S main.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -masm=intel
	perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./n_body_opencv

clean: 
	rm -f n_body_opencv
	rm -f *.png
	rm -f *.txt
