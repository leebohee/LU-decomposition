#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include "mpi.h"

MPI_Status status;

void inv(double**A, double** invA, int n){
	int i, j, k;
    double temp;
    double** I;
    I = new double* [n];
    for (i=0; i<n; i++) { 
    	I[i] = new double [2*n];
    	for(j=0; j<n; j++) I[i][j] = A[i][j];
        for (; j<2*n; j++) {
           	if (j == (i + n)) I[i][j] = 1; 
            else I[i][j] = 0;
        } 
    } 
  
    for (i=n-1; i>0; i--) { 
        if (fabs(I[i-1][0]) < fabs(I[i][0])) { 
            double* temp_ptr = I[i]; 
            I[i] = I[i - 1]; 
            I[i - 1] = temp_ptr; 
        } 
    } 
    for (i=0; i<n; i++) { 
        for (j=0; j<n; j++) { 
            if (i != j) { 
                temp = I[j][i] / I[i][i]; 
                for (k=0; k<2*n; k++) { 
                    I[j][k] -= I[i][k] * temp; 
                } 
            } 
        } 
    } 
    for (i=0;i<n; i++) { 
        temp = I[i][i]; 
        for (j=0; j<2*n; j++) { 
            I[i][j] = I[i][j] / temp; 
        } 
    } 
    for(i=0; i<n; i++){
		for(j=n; j<2*n; j++){
			invA[i][j-n] = I[i][j];
		}
		delete [] I[i];
	}
	delete [] I;
	return;
}

/*************************************
          COMPILE COMMAND
mpic++ LU.cpp -o LU -lm -O3

         EXECUTION COMMAND
time mpiexec --mca btl self --mca btl_openib_cpc_include rdmacm --machinefile ./hosts.txt -n 64 --map-by node ./LU 1000 1
**************************************/
int main(int argc, char* argv[]){
	int n, s, i, j, k, m, itr, r, c, rr, cc, p, ori_n, offset = 0;
	double **A, **L, **U, **E, **subA, **subL, **subU, **subL_inv, **subU_inv;
	double local_sum = 0.0, sum = 0.0, norm, temp;
	int my_rank, proc_num, offset_r, offset_c, rank;

	clock_t start = clock();

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	if(argc != 3){
		printf("ERROR: wrong arguments\n");
		return -1;
	}
	ori_n = n = atoi(argv[1]);
	s = atoi(argv[2]); // seed
	p = sqrt(proc_num); // # of processes
	srand(s);
	if(n % p > 0){
		n = n + (p - (n%p));
	}
	m = n / p;
	A = new double* [n];
	L = new double* [n];
	U = new double* [n];
	E = new double* [n];
	subA = new double* [m];
	subL = new double* [m];
	subU = new double* [m];
	subL_inv = new double* [m];
	subU_inv = new double* [m];
	for(i=0; i<n; i++){
		A[i] = new double [n];
		L[i] = new double [n];
		U[i] = new double [n];
		E[i] = new double [n];
		memset(L[i], 0.0, sizeof(double)*n);
		memset(U[i], 0.0, sizeof(double)*n);
		for(j=0; j<n; j++){
			if(i < ori_n && j < ori_n){
				A[i][j] = (int)(rand()/10000)+1;
			}
			else if(i == j){
				A[i][j] = 1;
			}
			else{
				A[i][j] = 0;
			}
		}
	}
	for(i=0; i<m; i++){
		subA[i] = new double [m];
		subL[i] = new double [m];
		subU[i] = new double [m];
		subL_inv[i] = new double [m];
		subU_inv[i] = new double [m];
		memset(subL[i], 0.0, sizeof(double)*m);
		memset(subU[i], 0.0, sizeof(double)*m);
	}

	// master process
	if(my_rank == 0){
		for(offset_r=0; offset_r<p; offset_r++){
			for(offset_c=0; offset_c<p; offset_c++){
				if(offset_r == offset_c && offset_r == 0) continue;
				rank = (offset_r*p+offset_c);
				MPI_Send(&offset_r, 1, MPI_INT, rank, 1, MPI_COMM_WORLD);
				MPI_Send(&offset_c, 1, MPI_INT, rank, 2, MPI_COMM_WORLD);
			}
		}
		offset_r = offset_c = 0;
	}
	// worker process
	else{
		MPI_Recv(&offset_r, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
		MPI_Recv(&offset_c, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
	}
	
	// block partitioning
	rr = offset_r*m;
	cc = offset_c*m;
	for(i=rr; i<rr+m; i++){
		for(j=cc; j<cc+m; j++){
			subA[i-rr][j-cc] = A[i][j];
		}
	}

	start = clock();
	// LU decomposition
	for(itr=0; itr<p; itr++){
		if(offset_r == offset_c && offset_r==itr){
			// LU decomposition of subA
			for(i=0; i<m; i++){
				memset(subL[i], 0.0, sizeof(double)*m);
				memset(subU[i], 0.0, sizeof(double)*m);
			}
			for(k=0; k<m; k++){
				subU[k][k] = subA[k][k];
				for(i=k+1; i<m; i++){
					subA[i][k] = subA[i][k] / subA[k][k];
				}
				for(i=k+1; i<m; i++){
					for(j=k+1; j<m; j++){
						subA[i][j] = subA[i][j] - subA[i][k]*subA[k][j];
					}
				}
			}	
			for(i=0; i<m; i++){
				for(j=0; j<i; j++){
					subL[i][j] = subA[i][j];
				}
				subL[i][j] = 1.0;
				for(; j<m; j++){
					subU[i][j] = subA[i][j];
				}
			}

			// send subL and subU to master thread
			for(i=0; i<m; i++){
				MPI_Send(subL[i], m, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD);
				MPI_Send(subU[i], m, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD);
			}
			
			// compute inverse of subL and subU
			inv(subL, subL_inv, m);
			inv(subU, subU_inv, m);
		
			// send inverse of subU
			for(r=itr+1; r<p; r++){
				rank = r*p + offset_c;
				for(i=0; i<m; i++){
					MPI_Send(subU_inv[i], m, MPI_DOUBLE, rank, 4, MPI_COMM_WORLD);
				}
			}
			// send inverse of subL
			for(c=itr+1; c<p; c++){
				rank = offset_r*p + c;
				for(i=0; i<m; i++){
					MPI_Send(subL_inv[i], m, MPI_DOUBLE, rank, 4, MPI_COMM_WORLD);
				}
			}
		}
		else if(offset_c == itr && offset_r > offset_c){
			// receive inverse of subU
			rank = itr*p+itr;
			for(i=0; i<m; i++){
				MPI_Recv(subU_inv[i], m, MPI_DOUBLE, rank, 4, MPI_COMM_WORLD, &status);
			}
					
			// compute subL by subA*subU_inv
			for(i=0; i<m; i++){
				for(j=0; j<m; j++){
					subL[i][j] = 0.0;
					for(k=0; k<m; k++){
						subL[i][j] += subA[i][k]*subU_inv[k][j];
					}
				}
			}
			// send subL to master process and blocks of same row
			for(i=0; i<m; i++){
				MPI_Send(subL[i], m, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
				for(j=offset_c+1; j<p; j++){
					rank = offset_r*p + j;
					MPI_Send(subL[i], m, MPI_DOUBLE, rank, 6, MPI_COMM_WORLD);
				}
			}
		}
		else if(offset_r == itr && offset_r < offset_c){ // offset_r < offset_c
			rank = itr*p+itr;
			for(i=0; i<m; i++){
				MPI_Recv(subL_inv[i], m, MPI_DOUBLE, rank, 4, MPI_COMM_WORLD, &status);
			}
					
			// compute subU
			for(i=0; i<m; i++){
				for(j=0; j<m; j++){
					subU[i][j] = 0.0;
					for(k=0; k<m; k++){
						subU[i][j] += subL_inv[i][k]*subA[k][j];
					}
				}
			}
			// send subU to master process and blocks of same column
			for(i=0; i<m; i++){
				MPI_Send(subU[i], m, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
				for(j=offset_r+1; j<p; j++){
					rank = j*p + offset_c;
					MPI_Send(subU[i], m, MPI_DOUBLE, rank, 7, MPI_COMM_WORLD);
				}
			}
		}
		else if(offset_r > itr && offset_c > itr){
			// receive subL and subU from previous iteration
			rank = offset_r*p+itr;
			for(i=0; i<m; i++){
				MPI_Recv(subL[i], m, MPI_DOUBLE, rank, 6, MPI_COMM_WORLD, &status);
			}
			rank = itr*p+offset_c;
			for(i=0; i<m; i++){
				MPI_Recv(subU[i], m, MPI_DOUBLE, rank, 7, MPI_COMM_WORLD, &status);
			}
			// update subA
			for(i=0; i<m; i++){
				for(j=0; j<m; j++){
					temp = 0;
					for(k=0; k<m; k++){
						temp += subL[i][k]*subU[k][j];
					}
					subA[i][j] -= temp;
				}
			}
		}
		if(my_rank == 0){
			// receive subL and subU (master thread)
			rank = itr*p + itr;
			for(i=0; i<m; i++){
				MPI_Recv(subL[i], m, MPI_DOUBLE, rank, 10, MPI_COMM_WORLD, &status);
				MPI_Recv(subU[i], m, MPI_DOUBLE, rank, 11, MPI_COMM_WORLD, &status);
			}
			rr = itr*m;
			cc = itr*m;
			for(i=rr; i<rr+m; i++){
				for(j=cc; j<cc+m; j++){
					L[i][j] = subL[i-rr][j-cc];
					U[i][j] = subU[i-rr][j-cc];
				}
			}
			for(r=itr+1; r<p; r++){
				rank = r*p + itr;
				for(i=0; i<m; i++){
					MPI_Recv(subL[i], m, MPI_DOUBLE, rank, 5, MPI_COMM_WORLD, &status);
				}
				rr = r*m;
				cc = itr*m;
				for(i=rr; i<rr+m; i++){
					for(j=cc; j<cc+m; j++){
						L[i][j] = subL[i-rr][j-cc];
					}
				}
			}
			for(c=itr+1; c<p; c++){
				rank = itr*p + c;
				for(i=0; i<m; i++){
					MPI_Recv(subU[i], m, MPI_DOUBLE, rank, 5, MPI_COMM_WORLD, &status);
				}
				rr = itr*m;
				cc = c*m;
				for(i=rr; i<rr+m; i++){
					for(j=cc; j<cc+m; j++){
						U[i][j] = subU[i-rr][j-cc];
					}
				}
			}
		}
	}
	//printf("Runtime 3 = %.4fs => LU\n", (float)(clock()-start)/CLOCKS_PER_SEC);
	for(i=0; i<n; i++){
		MPI_Bcast(L[i], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
	m = n / (proc_num-1);
	int left = n % (proc_num-1);
		
	if(my_rank == 0){	
		/*
		printf("======== A =========\n");
		for(i=0; i<ori_n; i++){
			for(j=0; j<ori_n; j++){
				printf("%10lf\n", A[i][j]);
			}
			printf("\n");
		}
		printf("======== L =========\n");
		for(i=0; i<ori_n; i++){
			for(j=0; j<ori_n; j++){
				printf("%10lf\n", L[i][j]);
			}
			printf("\n");
		}	
		printf("======== U =========\n");
		for(i=0; i<ori_n; i++){
			for(j=0; j<ori_n; j++){
				printf("%10lf\n", U[i][j]);
			}
			printf("\n");
		}*/
		for(rank=1, offset=m; rank<proc_num-1; rank++, offset+=m){
			MPI_Send(&offset, 1, MPI_INT, rank, 19, MPI_COMM_WORLD);
			for(i=0; i<n; i++){
				MPI_Send(&U[i][offset], m, MPI_DOUBLE, rank, 20, MPI_COMM_WORLD);
			}
		}
		if(left == 0) left = m;
		MPI_Send(&offset, 1, MPI_INT, rank, 19, MPI_COMM_WORLD);
		for(i=0; i<n; i++){
			MPI_Send(&U[i][offset], left, MPI_DOUBLE, rank, 20, MPI_COMM_WORLD);
		}
		offset = 0;
	}
	else{
		if(my_rank == proc_num-1 && left > 0) m = left;
		MPI_Recv(&offset, 1, MPI_INT, 0, 19, MPI_COMM_WORLD, &status);
		for(i=0; i<n; i++){
			MPI_Recv(&U[i][offset], m, MPI_DOUBLE, 0, 20, MPI_COMM_WORLD, &status);
		}
	}

	for(i=0; i<ori_n; i++){
		for(j=offset; j<offset+m; j++){
			temp = 0.0;
			for(k=0; k<ori_n; k++){
				temp += L[i][k]*U[k][j];
			}
			E[i][j] = A[i][j] - temp;
		}
	}
	for(j=offset; j<offset+m; j++){
		norm = 0.0;
		for(i=0; i<ori_n; i++){
			norm += E[i][j]*E[i][j];
		}
		local_sum += sqrt(norm);
	}
	MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if(my_rank == 0){
		printf("sum = %lf\n", sum);
	}

	for(i=0; i<n; i++){
		delete [] A[i];
		delete [] L[i];
		delete [] U[i];
		delete [] E[i];
	}
	for(i=0; i<m; i++){
		delete [] subA[i];	
		delete [] subL[i];
		delete [] subU[i];	
		delete [] subL_inv[i];
		delete [] subU_inv[i];
	}
	delete [] A;
	delete [] L;
	delete [] U;
	delete [] E;
	delete [] subA;
	delete [] subL;
	delete [] subU;
	delete [] subL_inv;
	delete [] subU_inv;	
		
	MPI_Finalize();
	return 0;
}
