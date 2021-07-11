//Sebastián Luciano Gallardo --- Feb 2021
//Requires floquet.h

#define FBM_ZEROTH_FOURIER_MODE_APPROXIMATION
#define TAU 6.283185307179586 //=2pi

#define Q_M (NK/2)
#define Q_N (2*Q_M+1)
#define Q_RHO_M 50
#define Q_RHO_N (2*Q_RHO_M+1)

//Libraries
#include <cstdlib>
#include <math.h>
#include <complex>
#include <armadillo> 
#include <fftw3.h>
#include "../lib/floquet.h"

typedef double (*BATH_FUNC)(double*,double);

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

class floquet_born_markov_2 : public floquet{
    public:
    //input
    double T, gamma; //reciprocal temperature, system-bath coupling strength
    int nbaths;
    cx_mat *Ac; //system coupling operator to each bath
    BATH_FUNC *J; //spectral density of each bath

    //output
    cx_cube rho_stat;

    //output yielding functions
    void calcFBM();
    cx_mat statAvgRho();
    cx_mat statRho(int t);
    cx_mat statAvgRhoFloquet();
    cx_mat statRhoFloquet(int t);

    floquet_born_markov_2(cx_mat(*H0_funcI)(double*), vec VI, double(*fI)(double), cx_mat *AcI, BATH_FUNC *JI, int nbathsI, int dimI, int nparI):
        floquet(H0_funcI,VI,fI,dimI,nparI),
        nbaths(nbathsI),
        J(JI),
        Ac(AcI)
        {}

};



void floquet_born_markov_2::calcFBM(){
    //Assumes H0 and floquet eigensystem have already been calculated.
    cx_mat id_dim = eye<cx_mat>(dim,dim);
    vec qe_diff(dim*dim,fill::none); //qe_diff(dim*a+b) = qe(a)-qe(b)
    cx_mat L[Q_RHO_N];
    for(int q=-Q_RHO_M;q<=Q_RHO_M;q++) L[Q_RHO_M+q] = zeros<cx_mat>(dim*dim,dim*dim);

    for(int a=0;a<dim;a++)
    	for(int b=0;b<dim;b++)
    		qe_diff(dim*a+b)=qenergy(a)-qenergy(b);


    //for each bath
    for(int bath=0;bath<nbaths;bath++){
	    // First we wil calculate the Fourier transformed floquet operator components of A, i.e. :
    	cx_mat flAc[Q_N];
    	for(int q=-Q_M;q<=Q_M;q++) flAc[Q_M+q] = zeros<cx_mat>(dim,dim);
		for(int a=0;a<dim;a++)
			for(int b=0;b<dim;b++){
				cx_vec A_t = zeros<cx_vec>(NK);
				cx_vec A_k = zeros<cx_vec>(NK);
    			fftw_plan plan = fftw_plan_dft_1d(NK, (double(*)[2])&A_t(0), (double(*)[2])&A_k(0), FFTW_FORWARD, FFTW_MEASURE);
				for(int t=0;t<NK;t++)
                    A_t(t) = cdot(flst_t[t].col(a),Ac[bath]*flst_t[t].col(b));
            	fftw_execute(plan);
            	for(int q=-Q_M;q<=Q_M;q++)
            		flAc[Q_M+q](a,b) = A_k((NK+q)%NK)/((cx_double)NK);
            	fftw_destroy_plan(plan);
			}

	    //now we calculate the correlation matrix
	    cx_mat g[Q_N];
	    for(int q=-Q_M;q<=Q_M;q++){
	        g[Q_M+q]=zeros<cx_mat>(dim,dim);
	        for(int a=0;a<dim;a++)
	            for(int b=0;b<dim;b++){
	                double waux = qe_diff(dim*a+b)+q*w;
	                g[Q_M+q](a,b) = (waux? J[bath](par,waux)/(exp(waux/T)-1) : T);
	            }
	    }

	    //approximate method for sufficiently fast driving
	    //now we calculate the rate tensors, here as (dim*dim)x(dim*dim) matrices
        for(int k=-Q_RHO_M;k<=Q_RHO_M;k++)
	    	for(int a=0;a<dim;a++)
	    		for(int b=0;b<dim;b++)
	    			for(int ap=0;ap<dim;ap++)
	    				for(int bp=0;bp<dim;bp++){
                            cx_double cumsum=0;
                            for(int kp=-Q_M;kp<=Q_M;kp++){
                                cumsum += g[Q_M+kp](a,ap)*flAc[Q_M+kp](a,ap)*flAc[(Q_M-kp-k+Q_N)%Q_N](bp,b);
                                cumsum += g[Q_M+kp](b,bp)*flAc[(Q_M+kp-k+Q_N)%Q_N](a,ap)*flAc[Q_M-kp](bp,b);
                                if(b==bp)
                                    for(int app=0;app<dim;app++)
                                        cumsum -= g[Q_M+kp](app,ap)*flAc[(Q_M-kp-k+Q_N)%Q_N](a,app)*flAc[Q_M+kp](app,ap);
                                if(a==ap)
                                    for(int bpp=0;bpp<dim;bpp++)
                                        cumsum -= g[Q_M+kp](bpp,bp)*flAc[Q_M-kp](bp,bpp)*flAc[(Q_M+kp-k+Q_N)%Q_N](bpp,b);
                                }
                            L[Q_RHO_M+k](dim*a+b,dim*ap+bp) = gamma*cumsum;
                            }
	}

    //add unitary evolution
    for(int a=0;a<dim;a++)
        for(int b=0;b<dim;b++)
            L[Q_RHO_M](dim*a+b,dim*a+b) += -CI*(qenergy(a)-qenergy(b));

    // for(int q=-Q_RHO_M;q<=Q_RHO_M;q++)
    //     cout << "L(" << q << ")=\n" << L[Q_RHO_M+q] << '\n';

    // cout << "Hermiticity conserving?\n";
    // for(int a=0;a<dim;a++)
    //     for(int b=0;b<dim;b++)
    //         for(int ap=0;ap<dim;ap++)
    //             for(int bp=0;bp<dim;bp++)
    //                 for(int q=-Q_RHO_M;q<=Q_RHO_M;q++)
    //                     cout << abs(L[Q_RHO_M+q](dim*a+b,dim*ap+bp)-conj(L[Q_RHO_M-q](dim*b+a,dim*bp+ap))) << '\n';
    // cout << "Trace conserving?\n";
    // for(int q=-Q_RHO_M;q<=Q_RHO_M;q++)
    //     for(int ap=0;ap<dim;ap++)
    //         for(int bp=0;bp<dim;bp++){
    //             cx_double cumsum=0;
    //             for(int a=0;a<dim;a++)
    //                 cumsum += L[Q_RHO_M+q](dim*a+a,dim*ap+bp);
    //             cout << q << ',' << ap << ',' << bp << '=' << abs(cumsum) << '\n';
    //         }


    //Now let's make the LL matrix
    #define LL_DIM ((Q_RHO_N)*dim*dim)
    cx_mat LL(LL_DIM,LL_DIM,fill::zeros);
    cx_mat LL_V(LL_DIM,LL_DIM,fill::none);
    cx_vec LL_E(LL_DIM,fill::none);

    for(int k=-Q_RHO_M;k<=Q_RHO_M;k++)
        for(int a=0;a<dim;a++)
            for(int b=0;b<dim;b++)
                for(int kp=-Q_RHO_M;kp<=Q_RHO_M;kp++)
                    for(int ap=0;ap<dim;ap++)
                        for(int bp=0;bp<dim;bp++){
                            int dk = k-kp;
                            LL(dim*dim*(Q_RHO_M+k)+dim*a+b, dim*dim*(Q_RHO_M+kp)+dim*ap+bp) = (abs(dk)<=Q_RHO_M ? L[Q_RHO_M+dk](dim*a+b,dim*ap+bp) : 0);
                        }

    cx_mat ikw = zeros<cx_mat>(LL_DIM,LL_DIM);
    for(int k=-Q_RHO_M;k<=Q_RHO_M;k++)
        for(int a=0;a<dim;a++)
            for(int b=0;b<dim;b++){
                int u = dim*dim*(Q_RHO_M+k)+dim*a+b;
                ikw(u,u) = CI*(k*w);
            }



    eig_gen(LL_E,LL_V,LL-ikw);
    // cout << "LL = \n" << LL << '\n';
    cout << "\nmax dissipation rate: " << max(-real(LL_E)) << '\n';
    LL_V = LL_V.t(); //???

    //Find zero eigenvec
    int rho_idx = 0;
    double aux = norm(LL_E(0));
    // cout << "\nLL_E(0) = " << LL_E(0) << '\n';

    for(int i=1;i<LL_DIM;i++){
        if(norm(LL_E(i))<aux){
            rho_idx=i;
            aux = norm(LL_E(i));
        }
    }


    //eigenvectors defined up to a complex multiple.
    //Multiply by a complex number such that rho_stat[q=0] is hermitian with rho_stat[q=0](h,h) >=0.
    //where we choose h to be the state with the highest population
    int h_idx = 0;
    aux = norm(LL_V(rho_idx,dim*dim*Q_RHO_M));
    for(int h=0;h<dim;h++) if(norm(LL_V(rho_idx,dim*dim*Q_RHO_M+dim*h+h))>aux){
        h_idx = h;
        aux = norm(LL_V(rho_idx,dim*dim*Q_RHO_M+dim*h+h));
    }
    cx_double normfact = conj(LL_V(rho_idx,dim*dim*Q_RHO_M))/abs(LL_V(rho_idx,dim*dim*Q_RHO_M+dim*h_idx+h_idx));
    double tracehh = 0;
    for(int a=0;a<dim;a++)
        tracehh += real(normfact*LL_V(rho_idx,dim*dim*Q_RHO_M+dim*a+a));

    //save result
    rho_stat = zeros<cx_cube>(dim,dim,Q_RHO_N);
    for(int k=-Q_RHO_M;k<=Q_RHO_M;k++)
        for(int a=0;a<dim;a++)
            for(int b=0;b<dim;b++)
                rho_stat(a,b,Q_RHO_M+k) = (normfact/tracehh)*LL_V(rho_idx,dim*dim*(Q_RHO_M+k)+dim*a+b);//LL_V(rho_idx,dim*dim*(Q_RHO_M+k)+dim*a+b);

    // cout << "Stat rho, rho_idx = " << rho_idx << "\n";
    for(int q=-Q_RHO_M;q<=Q_RHO_M;q++)
        cout << "rho(" << q << ") ~ " << accu(abs(rho_stat.slice(Q_RHO_M+q))) << '\n'; //<< rho_stat.slice(Q_RHO_M+q) << '\n';
        // cout << "rho(" << q << ") = \n" << rho_stat.slice(Q_RHO_M+q) << '\n';


}


cx_mat floquet_born_markov_2::statAvgRho(){
    cx_mat rho_op(dim,dim,fill::zeros);
    for(int t=0;t<NK;t++)
        rho_op += statRho(t);
    return symmatu((1/real(trace(rho_op)))*rho_op); //trace should be real, rho should be hermitian.
}

cx_mat floquet_born_markov_2::statRho(int t){
    //find the index whose eigenvalue is closest to zero
    cx_mat rho_op(dim,dim,fill::zeros);
    for(int a=0;a<dim;a++)
        for(int b=0;b<dim;b++)
            for(int k=-Q_RHO_M;k<=Q_RHO_M;k++)
                rho_op += exp(CI*(k*t*TAU/NK))*rho_stat(a,b,Q_RHO_M+k)*(flst_t[t%NK].col(a))*(flst_t[t%NK].col(b)).t();
    return symmatu((1/real(trace(rho_op)))*rho_op); //trace should be real, rho should be hermitian.
}

cx_mat floquet_born_markov_2::statAvgRhoFloquet(){
    cx_mat rho_op(dim,dim,fill::zeros);
    for(int t=0;t<NK;t++)
        rho_op += statRhoFloquet(t);
    return symmatu((1/real(trace(rho_op)))*rho_op); //trace should be real, rho should be hermitian.
}

cx_mat floquet_born_markov_2::statRhoFloquet(int t){
    //find the index whose eigenvalue is closest to zero
    cx_mat rho_op(dim,dim,fill::zeros);
    for(int a=0;a<dim;a++)
        for(int b=0;b<dim;b++)
            for(int k=-Q_RHO_M;k<=Q_RHO_M;k++)
                rho_op += exp(CI*(k*t*TAU/NK))*rho_stat.slice(Q_RHO_M+k);
    return symmatu((1/real(trace(rho_op)))*rho_op); //trace should be real, rho should be hermitian.

}