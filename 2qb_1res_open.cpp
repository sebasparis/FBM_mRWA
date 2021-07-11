//Sebastian Luciano Gallardo --- May 17
//This program solves the dynamics of a system consisting of two qubits coupled to a EM field mode,
//with periodic driving of some system operator and Born-Markov coupling to dissipative bath(s).

#include <iostream>
#include <fstream>
#include <complex>
#include <armadillo>
#include <math.h>
#include <string>
#include <functional>
#include "../lib/floquet_born_markov.h"
#include "../lib/multivar.h"
#include "../lib/entanglement.h"

using namespace std;
using namespace arma;

//dimensions of the problem
#define MAX_PHOTONS 3
#define DIM (4*(MAX_PHOTONS+1))

//useful definitions
#define FBM_FUNC_DOUBLE function<double(floquet_born_markov*,double*)>

#define C1 (cx_double){1,0}
#define CI (cx_double){0,1}
#define C0 (cx_double){0,0}

#define STRING(X) #X
#define STRING2(X) STRING(X) //to print compiler macro definitions

#define TAU 6.283185307179586
#define SQRT2 1.4142135623730951

////////////////////////////
// global variables
////////////////////////////

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

//Bell states
#ifndef BELL_STATES
#define BELL_STATES
const cx_vec PHI_PLUS = {1/SQRT2,0,0,1/SQRT2};
const cx_vec PHI_MINUS = {1/SQRT2,0,0,-1/SQRT2};
const cx_vec PSI_PLUS = {0,1/SQRT2,1/SQRT2,0};
const cx_vec PSI_MINUS = {0,1/SQRT2,-1/SQRT2,0};
#endif

//delta vector
cx_mat delta[DIM]; //defined in main
cx_mat delta_qb[4];
cx_mat delta_ph[MAX_PHOTONS+1];

//EM field operators
const cx_mat ID_PH = eye<cx_mat>(MAX_PHOTONS+1,MAX_PHOTONS+1);
cx_mat a(MAX_PHOTONS+1,MAX_PHOTONS+1,fill::zeros); //defined in main
cx_mat aT(MAX_PHOTONS+1,MAX_PHOTONS+1,fill::zeros); //defined in main


/////////////////////////
// driving functions
/////////////////////////

//Must be TAU-periodic

double square(double x){
	double u = x/TAU;
	u -= floor(u);
	return u<.5 ? 1:-1;
}

double sawtooth(double x){
	double u = x/TAU+.5;
	u -= floor(u);
	return 2*u-1;
}

double triangle(double x){
	double u = x/TAU+.25;
	u -= floor(u);
	return u<.5 ? 4*u-1 : 3-4*u; 
}


////////////////////////////
// coupling densities
////////////////////////////

//IMPORTANT: To handle transitions between degenerate energy levels, if w=0, J(0) returns the SLOPE of J(w) at w=0!

double J_ohm(double w){
	return w!=0 ? w : 1;
}


////////////////////////////
// miscellaneous functions
////////////////////////////

//triple kronecker product
cx_mat kron(cx_mat a1, cx_mat a2, cx_mat a3){
	return kron(a1,kron(a2,a3));
}
//vector kronecker product
cx_vec kron_v(cx_vec v1, cx_vec v2){
	int d1 = v1.n_rows;
	int d2 = v2.n_rows;
	cx_vec v1v2(d1*d2,fill::none);
	for(int i1=0;i1<d1;i1++)
		for(int i2=0;i2<d2;i2++)
			v1v2(d2*i1+i2) = v1(i1)*v2(i2);
	return v1v2;
}

//setter function
void setPars(floquet_born_markov *F, double *x);

//trace out EM field
cx_mat reduced(cx_mat rho_total){ 
	cx_mat rho_reduced(4,4,fill::zeros);
	for(int i=0;i<4;i++)
		for(int j=0;j<4;j++)
			for(int q=0;q<=MAX_PHOTONS;q++)
				rho_reduced(i,j) += rho_total(4*q+i,4*q+j);
	return rho_reduced;
}


/////////////////////////////
// problem definition
/////////////////////////////

//All hardcoded parameters of the problem can be set here

//H0 definition
#define H0_PARS 8
string H_PARS_NAMES[] = {"EPS","Q_EPS","DEL","Q_DEL","WR","G","Q_G","JZ","gamma","T","A","1/w","t","t0"};

#define EPS par[0]
#define Q_EPS par[1]
#define DEL par[2]
#define Q_DEL par[3]
#define WR par[4]
#define G par[5]
#define Q_G par[6]
#define JZ par[7]

#define S_GAP SY
#define S_COUP SX

cx_mat H0(double *par){
	return 
		.5*EPS*(kron(ID_PH,SZ,ID) + Q_EPS*(kron(ID_PH,ID,SZ))) +
		.5*DEL*(kron(ID_PH,S_GAP,ID) + Q_DEL*kron(ID_PH,ID,S_GAP)) +
		WR*kron(aT*a,ID,ID) +
		.5*G*(kron(a+aT,S_COUP,ID)+Q_G*kron(a+aT,ID,S_COUP)) +
		JZ*kron(ID_PH,SZ,SZ);
}

//driving
#define DRIVING_FUNC sin
#define DRIVING_OP real(diagvec(.5*(kron(ID_PH,SZ,ID)+1.001*kron(ID_PH,ID,SZ)))) //must be diagonal in computational basis, w/o loss of generality

//bath coupling
#define BATH_NUMBER 1
#define BATH_DENSITIES {J_ohm}//,J_ohm,J_ohm}
#define BATH_OPS {kron(a+aT,ID,ID)}//,.005*kron(ID_PH,SX+SZ,ID),.005*kron(ID_PH,ID,SX+SZ)}

//initial state
#define PSI_I kron_v(delta_ph[0],delta_qb[3]) //F->eigst.col(0) // //F is the floquet_born_markov problem, see main()


////////////////////////
// printers
////////////////////////

void printDimensions(void){
	cout << "H0_PARS = " << H0_PARS << '\n';
	cout << "MAX_PHOTONS = " << MAX_PHOTONS << '\n';
	cout << "DIM = " << DIM << '\n';
	cout << "NK = " << NK << '\n';
}

void printH0(void){
	cout << "H0 = .5*EPS*(SZ1 + Q_EPS*SZ2) + .5*DEL*(" STRING2(S_GAP) "1 + Q_DEL*" STRING2(S_GAP) "2) +\n";
	cout << "     WR*aT*a + .5*G*(a+aT)*(" STRING2(S_COUP) "1+Q_G*" STRING2(S_COUP) "2)\n";
}

void printHvars(bool time_dependent){
	for(int i=0;i<H0_PARS+4+2*time_dependent;i++) 
		cout << 'x' << i << ": " << H_PARS_NAMES[i] << '\n';
}

void printDriving(void){
	cout << "Driving operator: " STRING2(DRIVING_OP) "\n";
	cout << "Driving function: A*" STRING2(DRIVING_FUNC) "(wt)\n";
}

void printBath(void){
	//not working for multiple baths
	// cout << "Bath coup. operators: " STRING2(BATH_OPS) "\n";
	// cout << "Bath coup. densities: " STRING2(BATH_DENSITIES) "\n";
}

void printInitialState(void){
	cout << "Initial state: " STRING2(PSI_I) "\n";
}

//qubit states
string qbName[4] = {"00","PSI+","PSI-","11"};
cx_vec qbState[4];

int main(){
	char input;

	//calculate global variables
	for(int i=0;i<DIM;i++){
		delta[i] = zeros<cx_vec>(DIM);
		delta[i](i) = 1;
	}
	for(int i=0;i<4;i++){
		delta_qb[i] = zeros<cx_mat>(4);
		delta_qb[i](i) = 1;
	}
	for(int i=0;i<=MAX_PHOTONS;i++){
		delta_ph[i] = zeros<cx_mat>(MAX_PHOTONS+1);
		delta_ph[i](i) = 1;
	}

	for(int i=0;i<MAX_PHOTONS;i++) a(i,i+1) = sqrt(i+1);
	aT = a.t();

	qbState[0] = delta_qb[0];
	qbState[1] = PSI_PLUS;
	qbState[2] = PSI_MINUS;
	qbState[3] = delta_qb[3];

	//set up the problem
	cx_mat bath_ops[] = BATH_OPS;
	BATH_FUNC bath_densities[] = BATH_DENSITIES;
	floquet_born_markov double_qubit(H0, DRIVING_OP, DRIVING_FUNC, bath_ops, bath_densities, BATH_NUMBER, DIM, H0_PARS);

	//print problem variables
	printDimensions();
	cout << '\n';
	printH0();
	cout << '\n';
	printBath();
	cout << '\n';
	printDriving();
	cout << '\n';
	printInitialState();
	cout << '\n';

	//two cases:
	cout << "Time dependent functions? (y/n): ";
	cin >> input;
	cout << '\n';

	switch(input){
		case 'n':
		case 'N':
			{
			printHvars(false);
			cout << '\n';

			//functions of interest definition:
			//Here I initialize functions from <functional> with lambdas. The functions take the floquet_born_markov problem and a set of parameters given by the user.

			//time independent
			#define NFUNCS (2*DIM+4+2+2*(MAX_PHOTONS+1)+DIM+5+DIM*DIM+DIM)

			FBM_FUNC_DOUBLE funcList[NFUNCS];
			string funcNames[NFUNCS];
			int nfunc = 0;

			//H0 functions
			for(int i=0;i<DIM;i++){
				funcNames[nfunc] = to_string(i) + "th energy";
				funcList[nfunc] = [i](floquet_born_markov *F, double *x){setPars(F,x); return F->energy[i];};
				nfunc++;
			}

			for(int i=0;i<DIM;i++){
				funcNames[nfunc] = to_string(i) + "th quasienergy";
				funcList[nfunc] = [i](floquet_born_markov *F, double *x){setPars(F,x); return F->qenergy[i];};
				nfunc++;
			}

			//pop
			//comp basis -- qubit states
			for(int i=0;i<2;i++)
				for(int j=0;j<2;j++){
					funcNames[nfunc] = "avg pop " + to_string(i) + to_string(j) + " comp basis";
					funcList[nfunc] = [i,j](floquet_born_markov *F, double *x){
						setPars(F,x);
						double cumsum = 0;
						if(F->gamma){
							cx_mat rho = F->statAvgRho();
							for(int q=0;q<=MAX_PHOTONS;q++) cumsum += real(rho(4*q+2*i+j,4*q+2*i+j));
						}
						else
							for(int q=0;q<=MAX_PHOTONS;q++) cumsum += F->transProbAvg(PSI_I,delta[4*q+2*i+j]);
						return cumsum;
						
					};
					nfunc++;
				}

			//Bell states
			//psi+
			funcNames[nfunc] = "avg pop PSI+";
			funcList[nfunc] = [](floquet_born_markov *F, double *x){
				setPars(F,x);
				double cumsum = 0;
				if(F->gamma){
					cx_mat rho = F->statAvgRho();
					return real(cdot(PSI_PLUS,reduced(rho)*PSI_PLUS));
				}
				else
					for(int q=0;q<=MAX_PHOTONS;q++)
						cumsum += F->transProbAvg(PSI_I,kron_v(delta_ph[q],PSI_PLUS));
				return cumsum;
			};
			nfunc++;

			//psi-
			funcNames[nfunc] = "avg pop PSI-";
			funcList[nfunc] = [](floquet_born_markov *F, double *x){
				setPars(F,x);
				double cumsum = 0;
				if(F->gamma){
					cx_mat rho = F->statAvgRho();
					return real(cdot(PSI_MINUS,reduced(rho)*PSI_MINUS));
				}
				else
					for(int q=0;q<=MAX_PHOTONS;q++)
						cumsum += F->transProbAvg(PSI_I,kron_v(delta_ph[q],PSI_MINUS));
				return cumsum;
			};
			nfunc++;


			//comp basis -- photon states
			for(int q=0;q<=MAX_PHOTONS;q++){
				funcNames[nfunc] = "avg pop " + to_string(q) + " photons";
				funcList[nfunc] = [q](floquet_born_markov *F, double *x){
					setPars(F,x);
					double cumsum = 0;
					if(F->gamma){
						cx_mat rho = F->statAvgRho();
						for(int i=0;i<4;i++) cumsum += real(rho(4*q+i,4*q+i));
					}
					else
						for(int i=0;i<4;i++) cumsum += F->transProbAvg(PSI_I,delta[4*q+i]);
					return cumsum;
					
				};
				nfunc++;
			}

			//comp basis -- all states psi-
			for(int q=0;q<=MAX_PHOTONS;q++){
				funcNames[nfunc] = "avg pop " + to_string(q) + " ph PSI-";
				funcList[nfunc] = [q](floquet_born_markov *F, double *x){
					setPars(F,x);
					cx_vec psi_f = kron_v(delta_ph[q],PSI_MINUS);
					if(F->gamma){
						cx_mat rho = F->statAvgRho();
						return real(cdot(psi_f,rho*psi_f));
					}else
						return F->transProbAvg(PSI_I,psi_f);
				};
				nfunc++;
			}

			//energy basis
			for(int i=0;i<DIM;i++){
				funcNames[nfunc] = "avg pop " + to_string(i) + " H0 basis";
				funcList[nfunc] = [i](floquet_born_markov *F, double *x){
					setPars(F,x);
					double cumsum = 0;
					cx_vec eigst_i = F->eigst.col(i);
					if(F->gamma)
						return real(cdot(eigst_i,F->statAvgRho()*eigst_i));
					else
						return F->transProbAvg(PSI_I,eigst_i);
				};
				nfunc++;
			}

			//concurrence
			funcNames[nfunc] = "stat conc";
			funcList[nfunc] = [](floquet_born_markov *F, double *x){
				setPars(F,x);
				double cumsum = 0;
				if(F->gamma){
					for(int t=0;t<NK;t++) cumsum += concurrence4d_rho(reduced(F->statRho(t)));
					cumsum /= NK;
				}
				else{
					#define CONC_TIME_SAMPLING 12
					#define CONC_EPSILON 1e-6
					cx_vec psi_i = PSI_I;
				    //the largest timescale is given by the smallest difference between quasienergies
				    double aux;
				    double mindq = F->w+F->qenergy[0]-F->qenergy[3];
				    for(int i=1;i<4;i++) if((aux=(F->qenergy[i]-F->qenergy[i-1]))<mindq) mindq=aux;
				   	//within limits: if there are degeneracies we don't wanna divide by zero. Thus we introduce an epsilon
				    if(mindq<CONC_EPSILON) mindq = CONC_EPSILON;
					//with this we can define the integration step
				    int nt_step = (int)(TAU/mindq*NK);
					//also, we'd like nt_step and NK to be coprime to avoid aliasing

					while(__gcd(nt_step,NK)!=1) nt_step++;
				    for(int t0=0;t0<NK;t0+=NK/4){
						double cumsum_2 = 0;
				    	int t = t0;
				    	for(int i=0;i<CONC_TIME_SAMPLING;i++){
				    		t+=nt_step+110*t0;
				    		if(t<0) t = t0+102*NK; 
				    		cx_vec psi_t = F->propagate_discrete(psi_i,t0,t);
				    		cumsum_2+=concurrence4d_rho(symmatu(reduced(psi_t*psi_t.t())));
				    	}
				 		cumsum += cumsum_2;
				    }
				    cumsum /= NK*CONC_TIME_SAMPLING;
				}
				return cumsum;
			};
			nfunc++;

			//max concurrence
			#define CONC_MAX_PERIOD_SAMPLING 15
			funcNames[nfunc] = "max conc first " + to_string(CONC_MAX_PERIOD_SAMPLING) + " periods";
			funcList[nfunc] = [](floquet_born_markov *F, double *x){
				setPars(F,x);
				double aux, maxconc=0;
				cx_vec psi_i = PSI_I;
				if(F->gamma){
					for(int t=0;t<NK*CONC_MAX_PERIOD_SAMPLING;t++){
			    		aux = concurrence4d_rho(reduced(F->rho(psi_i,0,t)));
						if(aux>maxconc) maxconc = aux;
					}
				}
				else{
					for(int t=0;t<NK*CONC_MAX_PERIOD_SAMPLING;t++){
			    		cx_vec psi_t = F->propagate_discrete(psi_i,0,t);
			    		aux = concurrence4d_rho(symmatu(reduced(psi_t*psi_t.t())));
						if(aux>maxconc) maxconc = aux;
					}
				}
				return maxconc;
			};
			nfunc++;

			//concurrence of initial state
			funcNames[nfunc] = "initial state conc";
			funcList[nfunc] = [](floquet_born_markov *F, double *x){
				setPars(F,x);
				cx_vec psi_i = PSI_I;
				return concurrence4d_rho(reduced(psi_i*psi_i.t()));
			};
			nfunc++;

			funcNames[nfunc] = "min dissipation rate";
			funcList[nfunc] = [](floquet_born_markov *F, double *x){
				setPars(F,x);
				int dim = F->dim;
			    //find the index whose eigenvalue is closest to zero
			    int low_i=0;
			    double aux=real(F->FBM_E(0));
			    for(int i=1;i<dim*dim;i++) if(real(F->FBM_E(i))>aux){
			        low_i=i;
			        aux=real(F->FBM_E(i));
			    }
			    // find the second one
			    if(low_i != 0) aux=real(F->FBM_E(0));
			    else aux = real(F->FBM_E(1));
			    for(int i=0;i<dim*dim;i++){
			        if(i==low_i) continue; 
			        if(real(F->FBM_E(i))>aux){
			        aux=real(F->FBM_E(i));
			        }
			    }
			    return -aux;
			};
			nfunc++;

			funcNames[nfunc] = "max dissipation rate";
			funcList[nfunc] = [](floquet_born_markov *F, double *x){
				setPars(F,x);
			    return (-real(F->FBM_E)).max();
			};
			nfunc++;

			//mixing matrix elements
			#define MIXMAT_I 0 //parameter number over which to calculate mixing
			#define MIXMAT_EPS .000001 //epsilon to numerically calculate derivative
			for(int k=0;k<=MAX_PHOTONS;k++)
				for(int q=0;q<=MAX_PHOTONS;q++)
					for(int i=0;i<4;i++)
						for(int j=0;j<4;j++){
							funcNames[nfunc] = "mixing of (" + to_string(k) + ")" + qbName[i] + " and (" + to_string(q) + ")" + qbName[j] + " along " + H_PARS_NAMES[MIXMAT_I];
							funcList[nfunc] = [k,i,q,j](floquet_born_markov *F, double *x){
								setPars(F,x);
								cx_vec psi_1 = kron_v(delta_ph[k],qbState[i]);
								cx_vec psi_2 = kron_v(delta_ph[q],qbState[j]);
								return abs(cdot(psi_1,F->mixingMatrix(MIXMAT_I,MIXMAT_EPS)*psi_2));
							};
							nfunc++;
						}

			// for(int q=0;q<2;q++){
			// 	//0psi- 000
			// 	funcNames[nfunc] = "mixing between (0)PSI- and (0)00";
			// 	funcList[nfunc] = [q](floquet_born_markov *F, double *x){
			// 		setPars(F,x);
			// 		return abs(cdot(kron_v(delta_ph[q],PSI_MINUS),F->mixingMatrix(0,.00001)*kron_v(delta_ph[q],delta_qb[0])));
			// 	};
			// 	nfunc++;

			// 	//0psi- 0psi+
			// 	funcNames[nfunc] = "mixing between (" + to_string(q) + ")PSI- and (" + to_string(q)+")PSI+";
			// 	funcList[nfunc] = [q](floquet_born_markov *F, double *x){
			// 		setPars(F,x);
			// 		return abs(cdot(kron_v(delta_ph[q],PSI_MINUS),F->mixingMatrix(0,.00001)*kron_v(delta_ph[q],PSI_PLUS)));
			// 	};
			// 	nfunc++;

			// 	//0psi- 011
			// 	funcNames[nfunc] = "mixing between (" + to_string(q)+")PSI- and (" + to_string(q)+")11";
			// 	funcList[nfunc] = [q](floquet_born_markov *F, double *x){
			// 		setPars(F,x);
			// 		return abs(cdot(kron_v(delta_ph[q],PSI_MINUS),F->mixingMatrix(0,.00001)*kron_v(delta_ph[q],delta_qb[3])));
			// 	};
			// 	nfunc++;

			// 	funcNames[nfunc] = "mixing between (" + to_string(q)+")00- and (" + to_string(q)+")11";
			// 	funcList[nfunc] = [q](floquet_born_markov *F, double *x){
			// 		setPars(F,x);
			// 		return abs(cdot(kron_v(delta_ph[q],delta_qb[0]),F->mixingMatrix(0,.00001)*kron_v(delta_ph[q],delta_qb[3])));
			// 	};
			// 	nfunc++;

			// 	funcNames[nfunc] = "mixing between (" + to_string(q)+")PSI+ and (" + to_string(q)+")11";
			// 	funcList[nfunc] = [q](floquet_born_markov *F, double *x){
			// 		setPars(F,x);
			// 		return abs(cdot(kron_v(delta_ph[q],PSI_PLUS),F->mixingMatrix(0,.00001)*kron_v(delta_ph[q],delta_qb[3])));
			// 	};
			// 	nfunc++;

			// 	//Tavis Cummings amputated
			// 	funcNames[nfunc] = "mixing between (" + to_string(q) + ")psi- and (" + to_string(q+1) + ")11";
			// 		funcList[nfunc] = [q](floquet_born_markov *F, double *x){
			// 		setPars(F,x);
			// 		return abs(cdot(kron_v(delta_ph[q],PSI_MINUS),F->mixingMatrix(0,.00001)*kron_v(delta_ph[q],delta_qb[3])));
			// 	};
			// 	nfunc++;
			
			// 	funcNames[nfunc] = "mixing between (" + to_string(q) + ")psi+ and (" + to_string(q+1) + ")11";
			// 		funcList[nfunc] = [q](floquet_born_markov *F, double *x){
			// 		setPars(F,x);
			// 		return abs(cdot(kron_v(delta_ph[q],PSI_PLUS),F->mixingMatrix(0,.00001)*kron_v(delta_ph[q],delta_qb[3])));
			// 	};
			// 	nfunc++;
			
			// }

			//PSI- component of H0 eigenstates

			for(int i=0;i<DIM;i++){
				funcNames[nfunc] = "PSI- comp. of " + to_string(i) + "th eigenstate";
				funcList[nfunc] = [i](floquet_born_markov *F, double *x){
					setPars(F,x);
					double cumsum = 0;
					for(int q=0;q<=MAX_PHOTONS;q++) cumsum += norm(cdot(kron_v(delta_ph[q],PSI_MINUS),F->eigst.col(i)));
					return cumsum;
				};
				nfunc++;
			}

			

			//print function names
			for(int i=0;i<NFUNCS;i++)
				cout << 'f' << i << ": " << funcNames[i] << '\n';
			cout << '\n';

			//launch the iterator (it will ask the user to set parameters and choose functions)
			mv_iterator<floquet_born_markov,double> iter(double_qubit,H0_PARS+4,funcList,NFUNCS,H_PARS_NAMES,funcNames);
			iter.go();
			break;
			}
		case 'y':
		case 'Y':
			//time is measured on periods of the driving
			//NOTE: I still have to redefine functions as to comply with the above
			printHvars(true);			
			cout << '\n';

			#define NFUNCS_T (4 + 2 + DIM + 4*(MAX_PHOTONS+1) + 1 /*+DIM*DIM*/ +1)
			FBM_FUNC_DOUBLE funcList[NFUNCS_T];
			string funcNames[NFUNCS];
			int nfunc = 0;

			for(int i=0;i<2;i++)
				for(int j=0;j<2;j++){
					funcNames[nfunc] = to_string(i) + to_string(j) + " pop comp basis";
					funcList[nfunc] = [i,j](floquet_born_markov *F, double *x){
						setPars(F,x);
						double t = x[H0_PARS+4];
						double t0 = x[H0_PARS+5];
						cx_vec psi_i = PSI_I;
						double cumsum = 0;
						if(F->gamma){
							cx_mat rho_t = F->rho(psi_i*psi_i.t(),t0,t);
							for(int q=0;q<=MAX_PHOTONS;q++) cumsum += real(rho_t(4*q+2*i+j,4*q+2*i+j));
						}
						else{
							cx_vec psi_t = F->propagate_discrete(psi_i,t0,t);
							for(int q=0;q<=MAX_PHOTONS;q++) cumsum += norm(psi_t(4*q+2*i+j));
						}
						return cumsum;
						
					};
					cout << 'f' << nfunc << ": " << funcNames[nfunc] << '\n';
					nfunc++;
				}

				funcNames[nfunc] = "psi+ pop comp basis";
				funcList[nfunc] = [](floquet_born_markov *F, double *x){
					setPars(F,x);
					double t = x[H0_PARS+4];
					double t0 = x[H0_PARS+5];
					cx_vec psi_i = PSI_I;
					double cumsum = 0;
					if(F->gamma){
						cx_mat rho_t = F->rho(psi_i*psi_i.t(),t0,t);
						for(int q=0;q<=MAX_PHOTONS;q++){
							cx_vec psi_f = kron_v(delta_ph[q],PSI_PLUS);
							cumsum += real(cdot(psi_f,rho_t*psi_f));
						}
					}
					else{
						cx_vec psi_t = F->propagate_discrete(psi_i,t0,t);
						for(int q=0;q<=MAX_PHOTONS;q++){
							cx_vec psi_f = kron_v(delta_ph[q],PSI_PLUS);
							cumsum += norm(cdot(psi_f,psi_t));
						}
					}
					return cumsum;
					
				};
				cout << 'f' << nfunc << ": " << funcNames[nfunc] << '\n';
				nfunc++;

				funcNames[nfunc] = "psi- pop comp basis";
				funcList[nfunc] = [](floquet_born_markov *F, double *x){
					setPars(F,x);
					double t = x[H0_PARS+4];
					double t0 = x[H0_PARS+5];
					cx_vec psi_i = PSI_I;
					double cumsum = 0;
					if(F->gamma){
						cx_mat rho_t = F->rho(psi_i*psi_i.t(),t0,t);
						for(int q=0;q<=MAX_PHOTONS;q++){
							cx_vec psi_f = kron_v(delta_ph[q],PSI_MINUS);
							cumsum += real(cdot(psi_f,rho_t*psi_f));
						}
					}
					else{
						cx_vec psi_t = F->propagate_discrete(psi_i,t0,t);
						for(int q=0;q<=MAX_PHOTONS;q++){
							cx_vec psi_f = kron_v(delta_ph[q],PSI_MINUS);
							cumsum += norm(cdot(psi_f,psi_t));
						}
					}
					return cumsum;
					
				};
				cout << 'f' << nfunc << ": " << funcNames[nfunc] << '\n';
				nfunc++;


			//pop H0 basis
			for(int i=0;i<DIM;i++){
				funcNames[nfunc] = to_string(i) + "th pop H0 basis";
				funcList[nfunc] = [i](floquet_born_markov *F, double *x){
					setPars(F,x);
					double t = x[H0_PARS+4];
					double t0 = x[H0_PARS+5];
					cx_vec psi_i = PSI_I;
					cx_vec eigst_f = F->eigst.col(i);
					if(F->gamma)
						return real(cdot(eigst_f,F->rho(psi_i*psi_i.t(),t0,t)*eigst_f));
					else{
						cx_vec psi_t = F->propagate_discrete(psi_i,t0,t);
						return norm(psi_t(i));
					}
				};
				cout << 'f' << nfunc << ": " << funcNames[nfunc] << '\n';
				nfunc++;
			}

			//comp/fock basis pop
			for(int q=0;q<=MAX_PHOTONS;q++)
				for(int i=0;i<4;i++){
				funcNames[nfunc] = to_string(q) + " ph " + qbName[i] + " pop";
				funcList[nfunc] = [q,i](floquet_born_markov *F, double *x){
					setPars(F,x);
					double t = x[H0_PARS+4];
					double t0 = x[H0_PARS+5];
					cx_vec psi_i = PSI_I;
					cx_vec psi_f = kron_v(delta_ph[q],qbState[i]);
					if(F->gamma)
						return real(cdot(psi_f,F->rho(psi_i*psi_i.t(),t0,t)*psi_f));
					else{
						cx_vec psi_t = F->propagate_discrete(psi_i,t0,t);
						return norm(cdot(psi_f,psi_t));
					}
				};
				cout << 'f' << nfunc << ": " << funcNames[nfunc] << '\n';
				nfunc++;
			}

			//concurrence
			funcNames[nfunc] = "conc";
			funcList[nfunc] = [](floquet_born_markov *F, double *x){
				setPars(F,x);
				double t = x[H0_PARS+4];
				double t0 = x[H0_PARS+5];
				cx_vec psi_i = PSI_I;
				if(F->gamma)
					return concurrence4d_rho(reduced(F->rho(psi_i*psi_i.t(),t0,t)));
				else{
					cx_vec psi_t = F->propagate_discrete(psi_i,t0,t);
					return concurrence4d_rho(reduced(psi_t*psi_t.t()));
				}
			};
			cout << 'f' << nfunc << ": " << funcNames[nfunc] << '\n';
			nfunc++; 

			// for(int a=0;a<DIM;a++){
			// 	for(int n=0;n<DIM;n++){
			// 		funcList[nfunc] = [a,n](floquet_born_markov *F, double *x){
			// 			setPars(F,x);
			// 			int t = x[H0_PARS+4];
			// 			return norm(F->flst_t[t%NK](n,a));
			// 		};
			// 		cout << 'f' << nfunc << ": norm of" << n << "th component of " << a << "th floquet state\n";
			// 		nfunc++;
			// 	}
			// }

			//driving function
			funcNames[nfunc] = "driving func";
			funcList[nfunc] = [](floquet_born_markov *F, double *x){
				double t = x[H0_PARS+4];
				double t0 = x[H0_PARS+5];
				return F->f(TAU*(t-t0));
			};
			cout << 'f' << nfunc << ": " << funcNames[nfunc] << '\n';
			nfunc++;

			cout << '\n';

			mv_iterator<floquet_born_markov,double> iter(double_qubit,H0_PARS+6,funcList,NFUNCS_T,H_PARS_NAMES,funcNames);
			iter.go();
			break;
	}

    return 0;
}


void setPars(floquet_born_markov *F, double *x){//(set parameters, calculate solution performing minimal calculations)
	int npar = F->npar;
	bool H0_changed = false;
	bool V_changed = false;
	bool bath_changed = false;

	//H0 variables
	for(int i=0;i<npar;i++) if(F->par[i]!=x[i]){
		H0_changed = true;			
		F->par[i] = x[i];
	}

	//bath variables
	if(x[npar]!=F->gamma){
		bath_changed = true;
		F->gamma = x[npar];
	}

	if(x[npar+1]!=F->T){
		bath_changed = true;
		F->T = x[npar+1];
	}

	//driving variables
	if(F->A!=x[npar+2]){
		V_changed = true;
		F->A = x[npar+2];
	}

	if(F->w!=1/x[npar+3]){
		V_changed = true;
		F->w = 1/x[npar+3];
	}

	if(F->gamma){
		//dissipative
		//if H0 changed, recalculate everything
		if(H0_changed){
			F->calcH0();
			F->calcFloquet();
			F->calcFBM();
			return;
		}

		//if V changed, recalculate starting from the Floquet stuff.
		if(V_changed){
			F->calcFloquet();
			F->calcFBM();
			return;
		}

		//if only bath variables changed, just recalculate the FBM stuff.
		if(bath_changed){
			F->calcFBM();
			return;
		}

	}else{
		//non-dissipative	
		//if H0 changed, recalculate everything
		if(H0_changed){
			F->calcH0();
			F->calcFloquet();
			return;
		}

		//if only V changed, just recalculate the Floquet stuff.
		if(V_changed){
			F->calcFloquet();
			return;
		}	
	}
}



