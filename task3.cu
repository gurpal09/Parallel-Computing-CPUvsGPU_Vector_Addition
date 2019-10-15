/*
 * Purpose: Times 1-d Stencil on CPU/GPU for array size n (integer)
 *
 * Author: Gurpal Singh
 * Date: 3/22/2017
 * To Compile: nvcc task3.cu -arch=sm_30 -o task3.exe
 * To Run: ./task3.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/resource.h>
#include "timer.h"

//Defining the Radius adn BlockSize
#define RADIUS 3
#define BLOCK_SIZE 256

//GPU Kernel
__global__ void stencil_GPU(float *in, float *out) {
	__shared__ int temp[BLOCK_SIZE + 2 * RADIUS]; //Shared Variable
	int gindex = threadIdx.x + blockIdx.x * blockDim.x; //Creating index term
	int lindex = threadIdx.x + RADIUS; //Creating index term for each element
	
	//Read input elements into shared memory
	temp[lindex] = in[gindex];
	if(threadIdx.x < RADIUS){
		temp[lindex - RADIUS] = in[gindex - RADIUS];
		temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
	}

	//Synchronize to ensure all data is available
        __syncthreads();
	
	//Apply the Stencil
	int result = 0;
	for(int offset = -RADIUS; offset <= RADIUS; offset++)
		result += temp[lindex+offset];

	//Store the Result
	out[gindex] = result;
}

//CPU Function
void stencil_CPU (int n, float *a, float *b){
	int i,j;

	//Stencil 1-d Computation
	for (i = RADIUS; i < n - RADIUS; i++){
		float stencil_sum = 0.f;
		for(j = -RADIUS; j <= RADIUS; j++){
			stencil_sum += a[i+j];
		}
		b[i] = stencil_sum;
	}
}

int main(void){

	//Scan for array size
	int n;
    	printf("Enter the integer value for n: ");
    	scanf("%d", &n);
 
	//Initialize pointers
    	float *x, *y;

	//Allocate Memory
    	cudaMallocManaged(  &x, n * sizeof (float) + (2*RADIUS));
    	cudaMallocManaged(  &y, n * sizeof (float) + (2*RADIUS));
	
	//Initialize Array Values
    	for (int i = 0; i < n; i++){
        	x[i] = 1;
        	y[i] = 0;
   	}
	
	//Timing the CPU Function	
     	StartTimer();	
     	stencil_CPU(n, x, y);
     	float CPU_time = GetTimer(); //Get the time elapsed CPU addition
     	CPU_time = CPU_time*1000; //Converting seconds to ms		
     	printf("CPU y[100] = %.2f\n", y[100]);	
     	printf("elapsed wall time (CPU): %.2f ms\n", CPU_time);

	//Timing the GPU Kernel
     	cudaEvent_t timeStart, timeStop; //WARNING!!! use events only to time the device
     	cudaEventCreate(&timeStart);
     	cudaEventCreate(&timeStop);
     	float elapsedTime; // make sure it is of type float, precision is milliseconds (ms) !!!

    	int blockSize = 256;
    	int nBlocks   = (n + blockSize -1) / blockSize; //round up if n is not a multiple of blocksize
    	cudaEventRecord(timeStart, 0); //don't worry for the 2nd argument zero, it is about cuda streams
    	stencil_GPU <<< nBlocks, blockSize >>> (x + RADIUS, y + RADIUS);
    	cudaDeviceSynchronize();

    	printf("GPU y[100] = %.2f\n",y[100]);

    	cudaEventRecord(timeStop, 0);
    	cudaEventSynchronize(timeStop);

    	//WARNING!!! do not simply print (timeStop-timeStart)!!

    	cudaEventElapsedTime(&elapsedTime, timeStart, timeStop);

    	printf("elapsed wall time (GPU) = %.2f ms\n", elapsedTime);

    	cudaEventDestroy(timeStart);
    	cudaEventDestroy(timeStop);
        
	//Verify the results are correct
	int i;
	for(i = 3; i < n-3 ; ++i){
		if (y[i] != 7){
		printf("Element y[%d] == %d != 7\n", i, y[i]);
		break;
		}	
	}
	if (i == n-3){
		printf("SUCCESS!\n");
	}
	//Writing the Results to a File
    	FILE *fptr = fopen("Task3_Result.txt", "a+");
		if (fptr == NULL) {
        		printf("Error!");
        		exit(1);
    		}
    	fprintf(fptr, "\n");
    	fprintf(fptr, "Vector Size: %d\n", n);
    	fprintf(fptr, "elapsed wall time (CPU) = %.2f ms\n", CPU_time);
    	fprintf(fptr, "elapsed wall time (GPU) = %.2f ms\n", elapsedTime);
    	fclose(fptr);

	//Cleaning Up
    	cudaFree(x);
    	cudaFree(y);

	return EXIT_SUCCESS;
}
