#include <iostream>
#include <limits>

using namespace std;
inline int add(int a,int b) { return a+b; }


inline double sqrt_newton(double y){
	if(y<0){
		std::cerr << " input error"<<std::endl;
	}
	double eplon=1e-6;
	double x0=2;
	double x=x0-(x0*x0-y)/(2*x0);
	while(abs(x-x0)>eplon){
		x0=x;
		x=x0-(x0*x0-y)/(2*x0);
	}
	return x;
}
int main()
{
	std::cout << sqrt_newton(4.1)<<std::endl;
}