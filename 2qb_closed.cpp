#include <iostream>
#include <fstream>
#include <cstdlib>
#include <complex>
#include <armadillo>
#include <math.h>
#include <functional>
#include "../lib/floquet.h"
#include "../lib/multivar.h"
#include "../lib/entanglement.h"

#define FL_FUNC_VOID function<void(floquet*,double*)>
#define FL_FUNC_DOUBLE function<double(floquet*,double*)>

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
const cx_mat SP = {{C1,C0},{C0,C0}};
const cx_mat SM = {{C0,C0},{C0,C1}};
#endif


//Hamiltonian definition
#define DIM 4
#define H0_PARS 7
#define DRIVING_FUNC std::sin
#define CONC_TIME_SAMPLING 16
#define MIXMAT_I 0
#define MIXMAT_EPSILON .0001

//basis : {|00>,|01>,|10>,|11>} ; SZ|0>=+|0>, SZ|1>=-|1>
// cx_mat H0(double *par){
// 	return -par[0]*kron(SZ,ID)/2 - par[1]*kron(SX,ID)/2 - par[2]*kron(ID,SZ)/2 - par[3]*kron(ID,SX)/2 - par[4]*kron(SZ,SZ)/2 -par[5]*((1-par[6])*kron(SX,SX)+par[6]*kron(SY,SY))/2;
// }

cx_vec delta[DIM]; //auxiliary, defined in main


cx_mat H0(double *par){
	#define EPS par[0]
	#define Q_EPS par[1]
	// #define EPS2 par[1]
	#define DEL par[2]
	#define Q_DEL par[3]
	#define JZ par[4]
	#define JC par[5]
	#define P par[6]
	return EPS*.5*(kron(SZ,ID) + Q_EPS*kron(ID,SZ)) + .5*DEL*(kron(SX,ID) + Q_DEL*kron(ID,SX)) - .5*JZ*kron(SZ,SZ) -.5*JC*((1-P)*kron(SX,SX)+P*kron(SY,SY));
	// return EPS*.5*kron(SZ,ID) + EPS2*.5*kron(ID,SZ) + .5*DEL*(kron(SX,ID) + Q_DEL*kron(ID,SX)) - .5*JZ*kron(SZ,SZ) -.5*JC*((1-P)*kron(SX,SX)+P*kron(SY,SY));

}


// cx_mat H0(double *par){
// 	#define EPS par[0]
// 	#define Q_EPS par[1]
// 	// #define EPS2 par[1]
// 	#define g par[2]
// 	#define Q_g par[3]
// 	#define N par[4]
// 	double del1 = g*(sqrt(N)+sqrt(N-1));
// 	double del2 = g*Q_g*(sqrt(N)+sqrt(N-1))/2;
// 	double diffsq = (sqrt(N)-sqrt(N-1))/2;
// 	return EPS*.5*(kron(SZ,ID) + Q_EPS*kron(ID,SZ)) + .5*del1*kron(SX,ID) + .5*del2*kron(ID,SX) + diffsq*(g*kron(SX,SZ)+Q_g*g*kron(SZ,SX));
// 	// return EPS*.5*kron(SZ,ID) + EPS2*.5*kron(ID,SZ) + .5*DEL*(kron(SX,ID) + Q_DEL*kron(ID,SX)) - .5*JZ*kron(SZ,SZ) -.5*JC*((1-P)*kron(SX,SX)+P*kron(SY,SY));

// }




//setter function
void setPars(floquet *F, double *x){
	int npar = F->npar;
	bool H0_changed = false, V_changed = false;
	
	for(int i=0;i<npar;i++) if(F->par[i]!=x[i]){
		H0_changed = true;			
		F->par[i] = x[i];
	}

	if(F->A!=x[npar]){
		V_changed = true;
		F->A = x[npar];
	}

	if(F->w!=x[npar+1]){
		V_changed = true;
		F->w = x[npar+1];
	}
	
	//if H0 changed, recalculate everything
	if(H0_changed){
		F->calcH0();
		F->calcFloquet();
		return;
	}

	//if only V changed, just recalculate the Floquet stuff.
	if(V_changed) F->calcFloquet();

};


int main(){
	srand(102030);
	char input;

	//auxiliary variables
	int aux_i;
	for(int i=0;i<DIM;i++){
		delta[i] = zeros<cx_vec>(DIM);
		delta[i](i) = 1;
	}


	//driving matrix definition
	vec V={1,0,0,-1};

	//driving function definition
	double (*f)(double) = std::cos;

	

	cout << "NK=" << NK << "; CONC_TIME_SAMPLING=" << CONC_TIME_SAMPLING << "\n\n";
	// cout << "H(t) = -EPS_1*SZID/2-DEL_1*SXID/2-EPS_2*IDSZ/2-DEL_2*IDSX/2-JZ/2*SZSZ-JC/2*((1-P)*SXSX+P*SYSY)+(A1*SZID+A2*IDSZ)f(t)/2\n\n";
	cout << "H(t) = EPS*(SZID+Q_EPS*IDSZ)/2+DEL*(SXID+Q_DEL*IDSX)/2-JZ/2*SZSZ-JC/2*((1-P)*SXSX+P*SYSY)+A*(SZID+IDSZ)f(t)/2\n\n";
	floquet double_qubit(H0, V, DRIVING_FUNC, DIM, H0_PARS);


	cout << "Time dependent functions? (y/n): ";
	cin >> input;

	// cout << "x0: EPS_1\n";
	// cout << "x1: DEL_1\n";
	// cout << "x2: EPS_2\n";
	// cout << "x3: DEL_2\n";
	cout << "x0: EPS\n";
	cout << "x1: Q_EPS\n";
	cout << "x2: DEL\n";
	cout << "x3: Q_DEL\n";
	cout << "x4: JZ\n";
	cout << "x5: JC\n";
	cout << "x6: P\n";
	cout << "x7: A\n";
	cout << "x8: w\n";
	// cout << "x0: DEL\n";
	// cout << "x1: Q_DEL\n";
	// cout << "x2: g\n";
	// cout << "x3: Q_g\n";
	// cout << "x4: N\n";
	// cout << "x5: A\n";
	// cout << "x6: w\n";




	#define RSQRT2 (1/sqrt(2))
	cx_vec state2qb[] = {{0,0,0,1},{1,0,0,0},{0,RSQRT2,RSQRT2,0},{0,RSQRT2,-RSQRT2,0}};

	switch(input){
		case 'n':
		case 'N':
			{
			cout << '\n';
			//functions of interest definition:
			//Here I initialize functions from <functional> with lambdas. The functions take the floquet problem and a set of parameters given by the user.

			//time independent
			#define NFUNCS ((4*DIM) + (DIM*(DIM-1))/2 + 2)
			FL_FUNC_DOUBLE funcList[NFUNCS];
			int nfunc = 0;

			//H0 functions
			for(int i=0;i<DIM;i++){
				funcList[nfunc] = [i](floquet *F, double *x){setPars(F,x); return F->energy[i];};
				cout << 'f' << nfunc << ": " << i << "th energy\n";
				nfunc++;
			}

			for(int i=0;i<DIM;i++){
				funcList[nfunc] = [i](floquet *F, double *x){setPars(F,x); return F->qenergy[i];};
				cout << 'f' << nfunc << ": " << i << "th quasienergy\n";
				nfunc++;
			}

			for(int i=0;i<DIM;i++)
				for(int j=i+1;j<DIM;j++){
					funcList[nfunc] = [i,j](floquet *F, double *x){setPars(F,x); return abs(cdot(F->eigst.col(i),F->mixingMatrix(MIXMAT_I,MIXMAT_EPSILON)*F->eigst.col(j)));};
					cout << 'f' << nfunc << ": mixing matrix over x" << MIXMAT_I << ", element " << i << ',' << j << '\n';
					nfunc++;
				}


			//trans prob functions
			for(int i=0;i<DIM;i++){
				funcList[nfunc] = [i](floquet *F, double *x){setPars(F,x); return F->transProbAvg(F->eigst.col(0),delta[i]);};
				cout << 'f' << nfunc << ": double avg transition probability 0->" << i << " (comp. basis)\n";
				nfunc++;
			}

			for(int i=0;i<DIM;i++){
				funcList[nfunc] = [i](floquet *F, double *x){setPars(F,x); return F->transProbAvg(F->eigst.col(0),F->eigst.col(i));};
				cout << 'f' << nfunc << ": double avg transition probability 0->" << i << " (H0 basis)\n";
				nfunc++;
			}

			//average concurrence
			funcList[nfunc] = [](floquet *F, double *x){
				setPars(F,x);
				double aux;
				cx_vec psi_i = F->eigst.col(0);

			    //the largest timescale is given by the smallest difference between quasienergies
			    double mindq = F->w+F->qenergy[0]-F->qenergy[3];
			    for(int i=1;i<4;i++) if((aux=(F->qenergy[i]-F->qenergy[i-1]))<mindq) mindq=aux;	
			    // if(mindq < 1e-9) mindq=1e-9;
			    long nt_step = (long)(TAU/mindq*NK);
				if(nt_step>1e10) nt_step=(long)1e10;

				aux=0;			    
			    for(long t0=0;t0<NK;t0++){
			    	long t = t0;
			    	for(int i=0;i<CONC_TIME_SAMPLING;i++){
			    		t+=nt_step+rand()%nt_step; //antialiasing
			    		aux+=concurrence4d_pure(F->propagate_discrete(psi_i,t0,t));
			    	}
			    }
			    return aux/NK/CONC_TIME_SAMPLING;
			};
			cout << 'f' << nfunc << ": double avg concurrence\n";
			nfunc++;

			funcList[nfunc] = [](floquet *F, double *x){
				setPars(F,x);
				double aux;
				cx_vec psi_i = F->eigst.col(0);
				cx_vec psi_t;

			    //the largest timescale is given by the smallest difference between quasienergies
			    double mindq = F->w+F->qenergy[0]-F->qenergy[3];
			    for(int i=1;i<4;i++) if((aux=(F->qenergy[i]-F->qenergy[i-1]))<mindq) mindq=aux;	
			    // if(mindq < 1e-9) mindq=1e-9;
			    long nt_step = (long)(TAU/mindq*NK);
				if(nt_step>1e10) nt_step=(long)1e10;

				aux=0;			    
			    for(long t0=0;t0<NK;t0++){
			    	long t = t0;
			    	for(int i=0;i<CONC_TIME_SAMPLING;i++){
			    		t+=nt_step+rand()%nt_step; //antialiasing
			    		psi_t = F->propagate_discrete(psi_i,t0,t);
			    		aux += concurrence4d_rho(psi_t*psi_t.t());
			    	}
			    }
			    return aux/NK/CONC_TIME_SAMPLING;
			};
			cout << 'f' << nfunc << ": double avg concurrence (rho)\n";
			nfunc++;

			mv_iterator<floquet,double> iter(double_qubit,H0_PARS+2,funcList,NFUNCS);
			iter.go();
			break;

			}
		case 'y':
		case 'Y':
			{
			cout << "x9: t\n";
			cout << "x10: t0\n\n";

			#define NFUNCS_T 5
			FL_FUNC_DOUBLE funcList_t[NFUNCS_T];
			int nfunc = 0;

			for(int i=0;i<4;i++){
				funcList_t[nfunc] = [state2qb,i](floquet *F, double *x){
					setPars(F,x);
					int npar = F->npar;
					double t = x[npar+2];
					double t0 = x[npar+3];
					// return F->transProb(state2qb[0],state2qb[i],t0,t);
					return norm(cdot(state2qb[i],F->propagate(state2qb[0],t0,t)));
				};
				cout << 'f' << nfunc << ": trans prob (t0,t) 00->state2qb(" << i << ")\n";
				nfunc++;
			}

			funcList_t[nfunc] = [ state2qb](floquet *F, double *x){
				setPars(F,x);
				int npar = F->npar;
				double t = x[npar+2];
				double t0 = x[npar+3];
				return concurrence4d_pure(F->propagate(state2qb[0],t0,t));
			};
			cout << 'f' << nfunc << ": concurrence (t0,t) from 00\n";
			nfunc++;

			cout << '\n';
			mv_iterator<floquet,double> iter(double_qubit,H0_PARS+4,funcList_t,NFUNCS_T);
			iter.go();

			}
	}
    return 0;
}

