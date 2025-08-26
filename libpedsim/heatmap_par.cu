#include "ped_model.h"
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
//Heatmap functions
__global__ void fadeHeatmap(int *flatHeatmap) {
    __shared__ int sharedHeatmap[SIZE]; // Allocate shared memory for a block

    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < SIZE * SIZE) {
        sharedHeatmap[threadIdx.x] = flatHeatmap[x]; // Load from global memory to shared memory
        __syncthreads(); // Ensure all threads have loaded data

        sharedHeatmap[threadIdx.x] = (int)round(sharedHeatmap[threadIdx.x] * 0.80);

        __syncthreads(); // Ensure all computations are done before writing back

        flatHeatmap[x] = sharedHeatmap[threadIdx.x]; // Store back to global memory
    }
}


__global__ void intensifyHeat(int *flatHeatmap, int *c_desiredx, int *c_desiredy, int numAgents) {
    //printf("intensify heat");
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numAgents) {
        int x = c_desiredx[i];
        int y = c_desiredy[i];
        int index = y * SIZE + x;
        if(x>0 && x<SIZE && y>0 && y<SIZE)
        {
            atomicAdd(&flatHeatmap[index], 40);
            /*if (flatHeatmap[index] != 0)
            {
                printf("intesify heatmap %d", flatHeatmap[index]);
            }*/
            
        }
        
    }
}

__global__ void capAt255(int *flatHeatmap) {
    //printf("cap at 255");
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    //int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < SIZE*SIZE) {
        //int index = y * SIZE + x;
        flatHeatmap[x] = atomicMin(&flatHeatmap[x], 255);
        /*if ( flatHeatmap[x] != 0)
        {
            printf("cap heatmap %d", flatHeatmap[x]);
        }*/
        
    }
}

__global__ void scaleHeatmap(int *flatHeatmap, int *flatScaledHeatmap) {
    __shared__ int sharedHeatmap[SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < SIZE * SIZE) {
        sharedHeatmap[threadIdx.x] = flatHeatmap[i]; 
        __syncthreads();

        int x = threadIdx.x;
        int y = blockIdx.x;
        int value = sharedHeatmap[x];

        for (int cellY = 0; cellY < CELLSIZE; cellY++) {
            for (int cellX = 0; cellX < CELLSIZE; cellX++) {
                int index = cellY + x * CELLSIZE + cellX;
                if (index < SCALED_SIZE) {
                    int scaledIndex = (y * CELLSIZE + cellY) * SCALED_SIZE + (x * CELLSIZE + cellX);
                    flatScaledHeatmap[scaledIndex] = value;
                }
            }
        }
    }
}


__global__ void blurHeatmap(int *flatScaledHeatmap, int *blurredHeatmap) {

    // Weigths for blur
	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
		{ 4, 16, 26, 16, 4 },
		{ 7, 26, 41, 26, 7 },
		{ 4, 16, 26, 16, 4 },
		{ 1, 4, 7, 4, 1 }
	};
#define WEIGHTSUM 273
    //printf("blurring");
    int x = threadIdx.x;
	int y = blockIdx.x;
	if(x >= 2 && x <= SCALED_SIZE - 2 && y >= 2 && y <= SCALED_SIZE - 2){
		int sum = 0;
		for (int k = -2; k < 3; k++)
		{
			for (int l = -2; l < 3; l++)
			{
				int weight = w[2 + k][2 + l];
				int index = (x + l) + (y + k) * SCALED_SIZE;
				sum += weight * flatScaledHeatmap[index];
			}
		}
		int value = sum / WEIGHTSUM;
        /*if (value != 0 )
        {
            printf("blurred value parallel heatmap %d", value); 
        }*/
        
		blurredHeatmap[y*SCALED_SIZE + x] = 0x00FF0000 | value << 24;
	}
}

//Heatmap helper functions
// CUDA Setup Function
void Ped::Model::setup() {
    // Allocate host memory
    numAgents = agents.size(); 
    printf("numagents found");
    //heatmappp
    // Allocate GPU memory
    cudaMalloc((void **)&flatHeatmap, SIZE * SIZE * sizeof(int));
    cudaMalloc((void **)&flatScaledHeatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int));
    cudaMalloc((void **)&flatBlurredHeatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int));
    
    // Copy data to GPU
    cudaMemcpy(flatHeatmap, heatmap[0], SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(flatBlurredHeatmap, blurred_heatmap[0], SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(flatScaledHeatmap, scaled_heatmap[0], SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);

    //next desired position
    cudaMalloc((void **)&c_desiredX, numAgents * sizeof(int));
    cudaMalloc((void **)&c_desiredY, numAgents * sizeof(int));

    //cudaMemcpy(c_desiredX, desiredX, numAgents*sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(c_desiredY, desiredY, numAgents*sizeof(int), cudaMemcpyHostToDevice);

    //everything for calc
    cudaMalloc((void **)&c_cudaWaypoints, waypointsSum *sizeof(cudaWaypoint));
    cudaMalloc((void **)&c_waypointsIndex, numAgents * sizeof(int));
    cudaMalloc((void **)&c_xPos, numAgents * sizeof(int));
    cudaMalloc((void **)&c_yPos, numAgents * sizeof(int));
    cudaMalloc((void **)&c_destX, numAgents * sizeof(int));
    cudaMalloc((void **)&c_destY, numAgents * sizeof(int));
    cudaMalloc((void **)&c_destR, numAgents * sizeof(int));

    cudaMemcpy(c_cudaWaypoints, cudaWaypoints.data(), waypointsSum * sizeof(cudaWaypoint), cudaMemcpyHostToDevice);
    cudaMemcpy(c_waypointsIndex, waypointsIndex, numAgents * sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(c_xPos, cpuPosX, numAgents * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c_yPos, cpuPosY, numAgents * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
}

void::Ped::Model::runCudaHeatmap() {


    cudaEvent_t start, stop;
    float time_fade = 0, time_intensify = 0, time_cap = 0, time_scale = 0, time_blur = 0, total_time = 0;

    // Create events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start total timer
    cudaEventRecord(start);

    cudaMemcpyAsync(c_desiredX, desiredX, numAgents*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(c_desiredY, desiredY, numAgents*sizeof(int), cudaMemcpyHostToDevice);


    // 1. Fade Heatmap
    cudaEventRecord(start);
       
    fadeHeatmap<<<SIZE, SIZE>>>(flatHeatmap);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_fade, start, stop);

   
    // 2. Intensify Heatmap
    cudaEventRecord(start);

    int threads = 256;
    int blocks = (numAgents + threads - 1) / threads;
    intensifyHeat<<<SIZE, blocks>>>(flatHeatmap, c_desiredX, c_desiredY, numAgents);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_intensify, start, stop);

    // 3. Cap Heatmap at 255
    cudaEventRecord(start);

    capAt255<<<SIZE, SIZE>>>(flatHeatmap);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_cap, start, stop);

    // 4. Scale Heatmap
    cudaEventRecord(start);

    scaleHeatmap<<<SIZE, SIZE>>>(flatHeatmap, flatScaledHeatmap);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_scale, start, stop);

    cudaEventRecord(start);

    blurHeatmap<<<SIZE, SIZE>>>(flatScaledHeatmap, flatBlurredHeatmap);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_blur, start, stop);

    cudaMemcpy(blurred_heatmap[0], flatBlurredHeatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    //int *blur_heatmap = blurred_heatmap[0]; 
    //printf("blurre blurredheatmap %d",  flatBlurredHeatmap[0]);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Stop total timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&total_time, start, stop);

    // Print times
    printf("CUDA Timing Results:\n");
    printf("Fade Heatmap Time: %f ms\n", time_fade);
    printf("Intensify Heat Time: %f ms\n", time_intensify);
    printf("Cap at 255 Time: %f ms\n", time_cap);
    printf("Scale Heatmap Time: %f ms\n", time_scale);
    printf("Blur Heatmap Time: %f ms\n", time_blur);
    printf("Total GPU Execution Time: %f ms\n", total_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

//bonus 

__global__ void nextDesiredPosition( cudaWaypoint* c_cudaWaypoints, int* c_cudawaypointsIndex, int *c_destR, int *c_xPos, int *c_yPos, int *c_destX, int *c_destY, int* c_desiredX, int *c_desiredY, int numAgents){
    //printf("inside next desired");
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < numAgents){
        int tdestX = (int)c_destX[i];
        int tdestY = (int)c_destY[i];
        int txPos = (int)c_xPos[i];
        int tyPos = (int)c_yPos[i];
        
        int diffX = tdestX - txPos; 
        int diffY = tdestY - tyPos; 
        int len = __fsqrt_rn((int)(diffX * diffX + diffY * diffY));
        int agentReached = len < c_destR[i]; 
        int waypointsIndex = i*2;
        //printf("waypointIndex %d", waypointsIndex);
        //simulate next desired position
        if (agentReached || c_cudaWaypoints[waypointsIndex].flag < 0)
        {
            //printf("destx 1 %d destx 2 %d desty 1 %d desty %d", c_cudaWaypoints[waypointsIndex].destX, c_cudaWaypoints[waypointsIndex+1].destX, c_cudaWaypoints[waypointsIndex].destY, c_cudaWaypoints[waypointsIndex+1].destY);
            c_cudaWaypoints[waypointsIndex].flag += 1;
            if(c_cudaWaypoints[waypointsIndex].flag <c_cudawaypointsIndex[i])
            {
                int dX = c_cudaWaypoints[waypointsIndex+c_cudaWaypoints[waypointsIndex].flag].destX;
                int dY = c_cudaWaypoints[waypointsIndex+c_cudaWaypoints[waypointsIndex].flag].destY;
                int dR = c_cudaWaypoints[waypointsIndex+c_cudaWaypoints[waypointsIndex].flag].radius;
                atomicExch(&c_destX[i], dX);
                atomicExch(&c_destY[i], dY);
                atomicExch(&c_destR[i], dR);
            }
            else{
                c_cudaWaypoints[waypointsIndex].flag = -1; 
            }
        }

        tdestX = (int)c_destX[i];
        tdestY = (int)c_destY[i];

        diffX = tdestX - txPos; 
        diffY = tdestY - tyPos; 
        len = __fsqrt_rn(diffX * diffX + diffY * diffY);  // Fast sqrt approximation
        
        int newX = (int)round(txPos + diffX / (float)len);  
        int newY = (int)round(tyPos + diffY / (float)len);

        atomicExch(&c_desiredX[i], newX);
        atomicExch(&c_desiredY[i], newY);   
    }   
}

void Ped::Model::cudaNextDesiredPosition(){
    cudaMemcpyAsync(c_xPos, cpuPosX, numAgents * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(c_yPos, cpuPosY, numAgents * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    int threads = 256;
    int blocks = (numAgents + threads - 1) / threads;
    
    //printf("trying to run next desired");
    if (!c_cudaWaypoints || !c_destR || !c_xPos || !c_yPos || !c_destX || !c_destY) {
        printf("Error: One or more device pointers are NULL!\n");
        return;
    }
    nextDesiredPosition<<<blocks, threads>>>(c_cudaWaypoints, c_waypointsIndex, c_destR, c_xPos, c_yPos, c_destX, c_destY, c_desiredX, c_desiredY, numAgents);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(desiredX, c_desiredX, numAgents * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(desiredY, c_desiredY, numAgents * sizeof(int), cudaMemcpyDeviceToHost);

}

void Ped::Model::cleanupCuda() {
    cudaFree(flatHeatmap);
    cudaFree(flatScaledHeatmap);
    cudaFree(flatBlurredHeatmap);

    cudaFree(c_desiredX);
    cudaFree(c_desiredY);

    cudaFree(c_cudaWaypoints);
    cudaFree(c_waypointsIndex);
    cudaFree(c_xPos);
    cudaFree(c_yPos);
    cudaFree(c_destX);
    cudaFree(c_destY);
    cudaFree(c_destR);

    cudaDeviceSynchronize();
}