build:  main.cpp
	g++ -I/usr/include/opencv4 -L/usr/lib -fopenmp -g -o n_body_opencv main.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
	@[ -f n_body_opencv ] && echo "Compilation successful!" || echo "Compilation failed"
	./n_body_opencv

clean: 
	rm -f n_body_opencv
	rm -f *.png
