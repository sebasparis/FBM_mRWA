cx_double reSqrt(cx_double x){ //returns the real part of the square root of a complex number
    return real(sqrt(x));
}

cx_mat funcDiagMat(cx_double(*f)(const cx_double&),cx_mat m) { //returns a function evaluated for a square diagonalizable matrix M, i.e. f(M)
    int dim=m.n_rows;
    if(dim!=m.n_cols) throw "Matrix must be square!";
    cx_mat fDiag(dim,dim,fill::zeros), eigvec(dim,dim,fill::none);
    cx_vec eigval(dim,fill::none);
    eig_gen(eigval,eigvec,m);
    for(int i=0;i<dim;i++) fDiag(i,i)=f(eigval(i));
    cx_mat inveigvec(dim,dim,fill::none);

    if(inv(inveigvec,eigvec)) return eigvec*fDiag*inveigvec;
    else{cout << "\nCouldn't invert matrix!! Setting to identity!\n";
         return eye<cx_mat>(dim,dim);
        }
}

double concurrence4d_rho(cx_mat rho){ //returns the concurrence (an entanglement measure) of a given 4x4 density matrix
    if(rho.n_rows!=4||rho.n_cols!=4) throw "Matrix must be 4x4!";
    cx_mat aux = funcDiagMat(std::sqrt,rho);
    vec eigval(4,fill::none);
    eig_sym(eigval,symmatu(aux*kron(SY,SY)*conj(rho)*kron(SY,SY)*aux)); //symmatu is to avoid non-hermiticity caused by rounding errors.
    double c = sqrt(abs(eigval(3)))-sqrt(abs(eigval(2)))-sqrt(abs(eigval(1)))-sqrt(abs(eigval(0)));
    return c>0? c:0;
}

double concurrence4d_rho_inefficient(cx_mat rho){ //returns the concurrence (an entanglement measure) of a given 4x4 density matrix
    if(rho.n_rows!=4||rho.n_cols!=4) throw "Matrix must be 4x4!";
    cx_mat aux = funcDiagMat(std::sqrt,rho);
    vec eigval(4,fill::none);
    aux = funcDiagMat(std::sqrt,symmatu(aux*kron(SY,SY)*conj(rho)*kron(SY,SY)*aux));
    eig_sym(eigval,aux); //symmatu is to avoid non-hermiticity caused by rounding errors.
    double c = real(eigval(3))-real(eigval(2))-real(eigval(1))-real(eigval(0));
    return c>0? c:0;
}

double concurrence4d_pure(cx_vec psi){ //returns the concurrence (enganglement measure) of a given state vector
    return abs(dot(psi,kron(SY,SY)*psi));
}


