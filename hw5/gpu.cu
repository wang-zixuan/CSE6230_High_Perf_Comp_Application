#include "common.h"
#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <iostream>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;

int blks_bin;

int* bins_gpu;
int* particle_id_gpu;
int* bins_start;

int num_threads_forces;

int rows, cols;
int bin_length;
int bin_size;
double block_size = cutoff;

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int* bins_gpu, int* particle_id_gpu, int bin_length, int cols, int rows) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= bin_length)
        return;

    // find the begin index and end index in particle_id_gpu
    int part_begin_idx = bins_gpu[tid];
    int part_end_idx = (tid == bin_length - 1) ? num_parts : bins_gpu[tid + 1];

    int block_idx_y = tid / cols;
    int block_idx_x = tid % cols;

    for (int i = part_begin_idx; i < part_end_idx; i++) {
        int idx = particle_id_gpu[i];
        particles[idx].ax = 0;
        particles[idx].ay = 0;
        // neighbour bins
        for (int j = -1; j <= 1; j++) {
            for (int k = -1; k <= 1; k++) {
                int block_new_idx_y = block_idx_y + j;
                int block_new_idx_x = block_idx_x + k;
                
                // boundary check
                if (block_new_idx_y < 0 || block_new_idx_y >= rows || block_new_idx_x < 0 || block_new_idx_x >= cols) {
                    continue;
                }
                
                int block_new_idx = block_new_idx_x + block_new_idx_y * cols;
                int neighbor_part_idx_begin = bins_gpu[block_new_idx];
                int neighbor_part_idx_end = (block_new_idx == bin_length - 1) ? num_parts : bins_gpu[block_new_idx + 1];

                for (int m = neighbor_part_idx_begin; m < neighbor_part_idx_end; m++) {
                    apply_force_gpu(particles[idx], particles[particle_id_gpu[m]]);
                }
            }
        }
    }
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

__global__ void compute_bin_counts_gpu(int* bins_gpu, particle_t* parts, int num_parts, double block_size, int cols) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;
    
    int block_idx_x = (int) (parts[tid].x / block_size);
    int block_idx_y = (int) (parts[tid].y / block_size);

    int block_idx = block_idx_x + block_idx_y * cols;
    // increment the count in bins_gpu
    atomicAdd(&bins_gpu[block_idx], 1);
}

__global__ void separate_array(int* bins_start, particle_t* parts, int num_parts, int* particle_id_gpu, double block_size, int cols) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;
    
    int block_idx_x = (int) (parts[tid].x / block_size);
    int block_idx_y = (int) (parts[tid].y / block_size);

    int block_idx = block_idx_x + block_idx_y * cols;
    int pos = atomicAdd(&bins_start[block_idx], 1);
    // store id into particle_id_gpu
    particle_id_gpu[pos] = tid;
}

void prefix_sum_bin_counts(int *bins_gpu, particle_t* parts, int num_parts, int bin_length) {
    // count number of particles per bin
    compute_bin_counts_gpu<<<blks, NUM_THREADS>>>(bins_gpu, parts, num_parts, block_size, cols);
    
    // prefix sum the bin counts using thrust, store the result in bins_gpu
    thrust::device_ptr<int> device_ptr(bins_gpu);
    thrust::exclusive_scan(device_ptr, device_ptr + bin_length, device_ptr);
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    int max_grid_x = (int) (size / block_size);
    rows = max_grid_x + 1;
    cols = rows;

    // flattened length of bins
    bin_length = rows * cols;
    bin_size = bin_length * sizeof(int);

    blks_bin = (bin_length + NUM_THREADS - 1) / NUM_THREADS;

    // initialize array on GPU
    cudaMalloc((void **)&bins_gpu, bin_size);
    cudaMalloc((void **)&bins_start, bin_size);
    cudaMalloc((void **)&particle_id_gpu, num_parts * sizeof(int));
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    // reset bins_gpu array because it's a new simulation
    cudaMemset(bins_gpu, 0, bin_size);
    // compute prefix sum on bin_gpu
    prefix_sum_bin_counts(bins_gpu, parts, num_parts, bin_length);
    // copy data from bins_gpu to bins_start only on GPU
    cudaMemcpy(bins_start, bins_gpu, bin_size, cudaMemcpyDeviceToDevice);s
    // separate particles and store its id into particle_id_gpu
    separate_array<<<blks, NUM_THREADS>>>(bins_start, parts, num_parts, particle_id_gpu, block_size, cols);
    // compute forces between bins, with numBlocks of current kernel function being blks_bin
    compute_forces_gpu<<<blks_bin, NUM_THREADS>>>(parts, num_parts, bins_gpu, particle_id_gpu, bin_length, cols, rows);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}

void clear_simulation() {
    cudaFree(bins_gpu);
    cudaFree(particle_id_gpu);
    cudaFree(bins_start);
}
