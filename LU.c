#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void print(double** A, int n){
	int i, j;
	for(i=0; i<n; i++){
		for(j=0; j<n; j++){
			printf("%10lf ", A[i][j]);
		}
		printf("\n");
	}
}
int main(int argc, char* argv[]){
	int n, s, t, i, j, k;
	double *A, *L, *U, *E, *oriA;
	double sum = 0.0, norm, temp;

	if(argc != 4){
		printf("ERROR: wrong arguments\n");
		return -1;
	}
	n = atoi(argv[1]);
	s = atoi(argv[2]);
	t = atoi(argv[3]);

	srand(s);
	A = (double*)malloc(sizeof(double)*n*n);
	L = (double*)malloc(sizeof(double)*n*n);
	U = (double*)malloc(sizeof(double)*n*n);
	E = (double*)malloc(sizeof(double)*n*n);
	oriA = (double*)malloc(sizeof(double)*n*n);
	for(i=0; i<n; i++){
		for(j=0; j<n; j++){
			oriA[i*n+j] = A[i*n+j] = (double)rand();
		}
	}
	memset(L, 0.0, sizeof(double)*n*n);
	memset(U, 0.0, sizeof(double)*n*n);
  
	// LU decomposition
	for(k=0; k<n; k++){
		U[k*n+k] = A[k*n+k];
		for(i=k+1; i<n; i++){
			A[i*n+k] = A[i*n+k] / A[k*n+k];
		}
		for(i=k+1; i<n; i++){
			for(j=k+1; j<n; j++){
				A[i*n+j] = A[i*n+j] - A[i*n+k]*A[k*n+j];
			}
		}
	}	
	for(i=0; i<n; i++){
		for(j=0; j<i; j++){
			L[i*n+j] = A[i*n+j];
		}
		L[i*n+j] = 1.0;
		for(; j<n; j++){
			U[i*n+j] = A[i*n+j];
		}
	}
	// E = A - LU
	for(i=0; i<n; i++){
		for(j=0; j<n; j++){
			temp = 0.0;
			for(k=0; k<n; k++){
				temp += L[i*n+k]*U[k*n+j];
			}
			E[i*n+j] = oriA[i*n+j] - temp;
		}
	}

	// sum of Euclidean norm
	for(i=0; i<n; i++){
		norm = 0.0;
		for(j=0; j<n; j++){
			norm += E[i*n+j]*E[i*n+j];
		}
		sum += sqrt(norm);
	}
	printf("sum = %lf\n", sum);
	return 0;
}
