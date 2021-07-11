//Libraries
#include <stdlib.h>
#include <iostream>
#include "multivar.h"

using namespace std;

double summer(double* x){
    return x[0]+x[1]+x[2];
}

double squarer(double* x){
	return x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
}

int main(int argc, char **argv){
//    multivariateFunc function(summer,3);
	MULTIVAR_FUNC_T funcs[]  = {summer,squarer};
	multivariateFuncVector funcset(funcs,3,2);
    multivariateDomain domain;
    if(!domain.readSmart(argc,argv)) funcset.print(domain);
    return 0;
}