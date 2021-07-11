//Sebastián Luciano Gallardo --- Feb 2021
//Requires floquet.h

#define TAU 6.283185307179586 //=2pi

//Libraries
#include <cstdlib>
#include <math.h>
#include <complex>
#include <armadillo> 
#include <fftw3.h>
#include "../lib/floquet.h"

typedef double (*BATH_FUNC)(double);

#define C1 (cx_double){1,0}
#define CI (cx_double){0,1}
#define C0 (cx_double){0,0}

using namespace std;
using namespace arma;

//Pauli Matrices
#ifndef PAULI_SIGMA_MATRICES
#define PAULI_SIGMA_MATRICES
const cx_mat ID = {{C1,C0},{C0,C1}};
const cx_mat SX = {{C0,C1},{C1,C0}};
const cx_mat SY = {{C0,-CI},{CI,C0}};
const cx_mat SZ = {{C1,C0},{C0,-C1}};
const cx_mat SP = {{C0,C0},{C1,C0}};
const cx_mat SM = {{C0,C1},{C0,C0}};
#endif

class floquet_born_markov : public floquet{
    public:
    //input
    double T, gamma; //reciprocal temperature, system-bath coupling strength
    int nbaths;
    cx_mat *Ac; //system coupling operator to each bath
    BATH_FUNC *J; //spectral density of each bath

    //output
    cx_mat FBM_VL; //left Lambda eigenvectors
    cx_mat FBM_VR; //right Lambda eigenvectors
    cx_vec FBM_E; //Lambda eigenvalues

    //output yielding functions
    void calcFBM();
    cx_mat statAvgRho();
    cx_mat statRho(int t);
	cx_mat statRhoQEbasis();
	cx_mat rho(cx_mat rho_i, double t0, double t);
	cx_mat rhoQEbasis(cx_mat rho_i, double t0, double t);

	//maybe delete later
	cx_mat fourierSysQ(int q);


    floquet_born_markov(cx_mat(*H0_funcI)(double*), vec VI, double(*fI)(double), cx_mat *AcI, BATH_FUNC *JI, int nbathsI, int dimI, int nparI):
        floquet(H0_funcI,VI,fI,dimI,nparI),
        nbaths(nbathsI),
        J(JI),
        Ac(AcI)
        {}

};



void floquet_born_markov::calcFBM(){
    //Assumes H0 and floquet eigensystem have already been calculated.
	#define Q_M (NK/2)

    cx_mat id_dim = eye<cx_mat>(dim,dim);
    vec qe_diff(dim*dim,fill::none); //qe_diff(dim*a+b) = qe(a)-qe(b)
    cx_mat Rate(dim*dim,dim*dim,fill::zeros);

    for(int a=0;a<dim;a++)
    	for(int b=0;b<dim;b++)
    		qe_diff(dim*a+b)=qenergy(a)-qenergy(b);


    //for each bath
    for(int bath=0;bath<nbaths;bath++){
	    // First we wil calculate the Fourier transformed floquet operator components of A, i.e. :
    	cx_mat flAc[2*Q_M+1];
    	for(int q=-Q_M;q<=Q_M;q++) flAc[Q_M+q] = zeros<cx_mat>(dim,dim);
		for(int a=0;a<dim;a++)
			for(int b=0;b<dim;b++){
				cx_vec A_t = zeros<cx_vec>(NK);
				cx_vec A_k = zeros<cx_vec>(NK);
    			fftw_plan plan = fftw_plan_dft_1d(NK, (double(*)[2])&A_t(0), (double(*)[2])&A_k(0), FFTW_FORWARD, FFTW_MEASURE);
				for(int t=0;t<NK;t++){
					cx_double aux = cdot(flst_t[t].col(a),Ac[bath]*flst_t[t].col(b));
					A_t(t) = aux;
				}
            	fftw_execute(plan);
            	for(int q=-Q_M;q<=Q_M;q++){
            		flAc[Q_M+q](a,b) = A_k((NK+q)%NK)/((cx_double)NK);
            	}
            	fftw_destroy_plan(plan);
			}

	    //now we calculate the correlation matrix
	    cx_mat g[2*Q_M+1];
	    for(int q=-Q_M;q<=Q_M;q++){
	        g[Q_M+q]=zeros<cx_mat>(dim,dim);
	        for(int a=0;a<dim;a++)
	            for(int b=0;b<dim;b++){
	                double waux = qe_diff(dim*a+b)+q*w;
	                g[Q_M+q](a,b) = (waux? J[bath](waux)/(exp(waux/T)-1) : T*J[bath](0)); //J[bath](0) actually returns the slope at w=0
	            }
	    }

	    //approximate method for fast driving compared to dissipation ratesW
	    //now we calculate the rate tensors, here as (dim*dim)x(dim*dim) matrices
	    for(int q=-Q_M;q<=Q_M;q++)
	    	for(int a=0;a<dim;a++)
	    		for(int b=0;b<dim;b++)
	    			for(int ap=0;ap<dim;ap++)
	    				for(int bp=0;bp<dim;bp++)
                            Rate(dim*a+b,dim*ap+bp) += gamma*g[Q_M+q](a,ap)*flAc[Q_M+q](a,ap)*flAc[Q_M-q](bp,b);
		    				// Rate(dim*a+b,dim*ap+bp) += gamma*sqrt(g[Q_M+q](a,ap)*g[Q_M+q](b,bp))*flAc[Q_M+q](a,ap)*flAc[Q_M-q](bp,b);

	}

    //Auxilary matrix
    cx_mat Rate_aux(dim,dim,fill::none);
    for(int a=0;a<dim;a++)
        for(int b=0;b<dim;b++){
            cx_double cumsum = C0;
            for(int c=0;c<dim;c++) cumsum += Rate(dim*c+c,dim*a+b);
            Rate_aux(a,b)=cumsum;
        }

    //Now we calculate the dissipation tensor L
    cx_mat L(dim*dim,dim*dim,fill::none);
    for(int a=0;a<dim;a++)
    	for(int b=0;b<dim;b++)
    		for(int ap=0;ap<dim;ap++)
    			for(int bp=0;bp<dim;bp++)
    				L(dim*a+b,dim*ap+bp)=Rate(dim*a+b,dim*ap+bp)+conj(Rate(dim*b+a,dim*bp+ap))-id_dim(b,bp)*Rate_aux(ap,a)-id_dim(a,ap)*conj(Rate_aux(bp,b));

    eig_gen(FBM_E,FBM_VR,L+diagmat(-CI*qe_diff));
    FBM_VL = inv(FBM_VR).t();



    // cout << '\n';
    // cout << "max dissipation rate: " << (-real(FBM_E)).max() << '\n';
    // double min_dissipation_rate;
    // //find the index whose eigenvalue is closest to zero
    // int low_i=0;
    // double aux=real(FBM_E(0));
    // for(int i=1;i<dim*dim;i++) if(real(FBM_E(i))>aux){
    //     low_i=i;
    //     aux=real(FBM_E(i));
    // }
    // cout << "zero dissipation rate: " << -aux << '\n';
    // // find the second one
    // if(low_i != 0) aux=real(FBM_E(0));
    // else aux = real(FBM_E(1));
    // for(int i=0;i<dim*dim;i++){
    //     if(i==low_i) continue; 
    //     if(real(FBM_E(i))>aux){
    //     aux=real(FBM_E(i));
    //     }
    // }
    // cout << "min dissipation rate: " << -aux << '\n';

    // cout << "L:\n" << L+diagmat(-CI*qe_diff) << '\n';

    // for(int i=0;i<dim*dim;i++) cout << "LAMBDA(" << i << ") = " << FBM_E(i) << '\n';

}

cx_mat floquet_born_markov::fourierSysQ(int q){
    cx_mat flAc = zeros<cx_mat>(dim,dim);
    for(int a=0;a<dim;a++)
        for(int b=0;b<dim;b++)
            for(int k=0;k<NK;k++)
	            for(int k=-Q_M;k<=Q_M;k++)
	                flAc(a,b)+=cdot(flst_k[(NK+k)%NK].col(a),Ac[0]*(flst_k[(2*NK+k+q)%NK].col(b)));
    return flAc;
}


cx_mat floquet_born_markov::statAvgRho(){
    cx_mat rho_QE = statRhoQEbasis();
    cx_mat rho_op(dim,dim,fill::zeros);
    for(int t=0;t<NK;t++)
    	for(int a=0;a<dim;a++)
    		for(int b=0;b<dim;b++)
    			rho_op += rho_QE(a,b)*(flst_t[t].col(a))*(flst_t[t].col(b)).t();

    //normalize dividing by trace
    return symmatu((1/real(trace(rho_op)))*rho_op); //trace should be real, rho should be hermitian.
}

cx_mat floquet_born_markov::statRho(int t){
    //find the index whose eigenvalue is closest to zero
    int low_i=0;
    double low_d=real(FBM_E(0));
    for(int i=1;i<dim*dim;i++) if(real(FBM_E(i))>low_d){
        low_i=i;
        low_d=real(FBM_E(i));
    }


    cx_mat rho_op(dim,dim,fill::zeros);
    for(int a=0;a<dim;a++)
        for(int b=0;b<dim;b++)
            rho_op += FBM_VR(dim*a+b,low_i)*(flst_t[t].col(a))*(flst_t[t].col(b)).t();

    return symmatu((1/real(trace(rho_op)))*rho_op); //trace should be real, rho should be hermitian.
}


cx_mat floquet_born_markov::statRhoQEbasis(){
    int low_i=0;
    double low_d=real(FBM_E(0));
    for(int i=1;i<dim*dim;i++) if(real(FBM_E(i))>low_d){
        low_i=i;
        low_d=real(FBM_E(i));
    }

    // for(int i=0;i<dim*dim;i++) cout << "LAMBDA(" << i <<") = " << FBM_E(i) << '\n';

    cx_mat rho_op(dim,dim,fill::none);
    for(int a=0;a<dim;a++)
    	for(int b=a;b<dim;b++)
    		rho_op(a,b) = FBM_VR(dim*a+b,low_i);


    return symmatu((rho_op*(1/real(trace(rho_op)))));
}

cx_mat floquet_born_markov::rho(cx_mat rho_i, double t0, double t){
    int it = NK*t;
    it = it%NK;
    int it0 = NK*t0;
    it0 = it0%NK;
    cx_mat rho_i_floquet = flst_t[it0].t()*rho_i*flst_t[it0];
    cx_mat rho_op = flst_t[it]*rhoQEbasis(rho_i_floquet,t0,t)*flst_t[it].t();
    return (1/real(trace(rho_op)))*symmatu(rho_op);
}

cx_mat floquet_born_markov::rhoQEbasis(cx_mat rho_i, double t0, double t){
    //rho_i must be expressed in floquet basis
	cx_vec vrho_i(dim*dim,fill::none);
	cx_vec vrho_t(dim*dim,fill::zeros);
	cx_mat rho_t(dim,dim,fill::none);

	for(int a=0;a<dim;a++)
		for(int b=0;b<dim;b++)
			vrho_i(dim*a+b) = rho_i(a,b);
	
	for(int i=0;i<dim*dim;i++){
		vrho_t += exp((TAU/w)*FBM_E(i)*(t-t0))*cdot(FBM_VL.col(i),vrho_i)*FBM_VR.col(i);
		// cout << "weight " << i << ": " << exp(FBM_E(i)*(t-t0)) << '\n';
		// cout << "Projection " << i << ": " << cdot(FBM_VL.col(i),vrho_i) << '\n';
	}

	for(int a=0;a<dim;a++)
		for(int b=0;b<dim;b++)
			rho_t(a,b) = vrho_t(dim*a+b);

	// cout << "rho_QE:\n" << (1/real(trace(rho_t)))*rho_t << '\n'; 
	// cout << "stat rho_QE:\n" << statRhoQEbasis() << '\n';

	return symmatu(rho_t*(1/real(trace(rho_t))));
}