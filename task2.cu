/*
 * Purpose: Times Vector addition in CPU and GPU for array size n (integer)
 *
 * Author: Gurpal Singh
 * Date: 3/22/2017
 * To Compile: nvcc task2.cu -arch=sm_30 -o task2.exe
 * To Run: ./task2.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/resource.h>
#include "timer.h"

//GPU Kernel
__global__ void vector_add (int n, float *a, float *b, float *c){
	int tid = blockIdx.x*blockDim.x + threadIdx.x; //Term for Indexing
	if (tid <  n)
		c[tid] = a[tid] + b[tid]; //Vector Sum
}

//CPU Function
void CPU_vector_add(int n, float *a, float *b, float *c){
	for (int i = 0; i < n; i++){
		c[i] = a[i] + b[i];
	}
}

int main(void){
	
	//Scan for array size n (integer)	
	int n;
    	printf("Enter the integer value for n: ");
    	scanf("%d", &n);

	//Initilize Pointers
    	float *x, *y, *z;

	//Allocate Memory
    	cudaMallocManaged(  &x, n * sizeof (float));
    	cudaMallocManaged(  &y, n * sizeof (float));
    	cudaMallocManaged(  &z, n * sizeof (float));

    	//Initialize vector values
	for (int i = 0; i < n; i++){
        	x[i] = 3.5;
        	y[i] = 1.5;
    	}
	
	//Timing the CPU Function	
     	StartTimer();	
     	CPU_vector_add(n, x, y, z);
     	float CPU_time = GetTimer(); //Get the time elapsed CPU addition
     	CPU_time = CPU_time*1000; //Converting seconds to ms		
     	printf("CPU z[100] = %.2f\n", z[100]);	
     	printf("elapsed wall time (CPU): %.2f ms\n", CPU_time);
	
	//Timing the GPU Kernel
     	cudaEvent_t timeStart, timeStop; //WARNING!!! use events only to time the device
     	cudaEventCreate(&timeStart);
     	cudaEventCreate(&timeStop);
     	float elapsedTime; // make sure it is of type float, precision is milliseconds (ms) !!!

    	int blockSize = 256;
    	int nBlocks   = (n + blockSize -1) / blockSize; //round up if n is not a multiple of blocksize
    	
	cudaEventRecord(timeStart, 0); //don't worry for the 2nd argument zero, it is about cuda streams
    	
	vector_add <<< nBlocks, blockSize >>> (n, x, y, z);
    	cudaDeviceSynchronize();

    	printf("GPU z[100] = %4.2f\n",z[100]);

    	cudaEventRecord(timeStop, 0);
    	cudaEventSynchronize(timeStop);
		
    	//WARNING!!! do not simply print (timeStop-timeStart)!!

    	cudaEventElapsedTime(&elapsedTime, timeStart, timeStop);

    	printf("elapsed wall time (GPU) = %.2f ms\n", elapsedTime);

    	cudaEventDestroy(timeStart);
    	cudaEventDestroy(timeStop);
	
	//Verify the results are correct
	int i;
	for(i = 0; i < n ; ++i){
		if (z[i] != 5){
		printf("Element y[%d] == %d != 5\n", i, z[i]);
		break;
		}	
	}
	if (i == n){
		printf("SUCCESS!\n");
	}
	//Storing the results in a file
    	FILE *fptr = fopen("Task2_Result.txt", "a+"); 
		if (fptr == NULL) {
        	printf("Error!");
        	exit(1);
    	}

    	fprintf(fptr, "\n");
    	fprintf(fptr, "Vector Size: %d\n", n);
    	fprintf(fptr, "elapsed wall time (CPU) = %.2f ms\n", CPU_time);
    	fprintf(fptr, "elapsed wall time (GPU) = %.2f ms\n", elapsedTime);
    	fclose(fptr);
    	cudaFree(x);
    	cudaFree(y);
    	cudaFree(z);

    	return EXIT_SUCCESS;
}
