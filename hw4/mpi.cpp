#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <set>
#include <map>
#include <numeric>
#include <iostream>

// Put any static global variables here that you will use throughout the simulation.
typedef struct {
    std::set<particle_t*> particles;
} block;

std::vector<block> grid;

double block_size = cutoff;

int rows, cols, grid_size, proc_row, proc_col, num_rows_per_proc, num_cols_per_proc;

int row_begin = -1, row_end = -1, col_begin = -1, col_end = -1;
int proc_in_use;

int run_times = 0;

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}


int find_number_factorization(int num) {
    if (num == 1 || num == 2) return 1;
    for (int i = sqrt(num); i >= 2; i--) {
        if (num % i == 0) {
            return i;
        }
    }

    return -1;
}

std::vector<particle_t> collect_particles(const std::vector<block>& blocks) {
    std::vector<particle_t> res;

    for (auto blk : blocks) {
        for (auto p : blk.particles) {
            res.push_back(*p);
        }
    }

    return res;
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here

    proc_in_use = num_procs;

    // check if the number of processors is prime
    int factorization_num = find_number_factorization(proc_in_use);
    if (factorization_num == -1) {
        // if it's a prime number, then leave the last processor idle
        proc_in_use--;
        factorization_num = find_number_factorization(proc_in_use);
    }

    if (rank < proc_in_use) {
        // compute # of rows and columns of the grid
        int max_grid_x = (int) (size / block_size);
        rows = max_grid_x + 1;
        cols = rows;

        // find how to partition the grid based on the processors
        proc_row = factorization_num;
        proc_col = proc_in_use / factorization_num;
        grid_size = rows * cols;
        grid.resize(grid_size);

        // compute # of rows and columns that a processor can process
        num_rows_per_proc = rows / proc_row;
        num_cols_per_proc = cols / proc_col;

        // compute row_begin and row_end based on rank
        row_begin = (rank / proc_col) * num_rows_per_proc;
        row_end = row_begin + num_rows_per_proc;

        if (rows % proc_row != 0) {
            // last row takes the reaminder
            if (rank / proc_col == proc_row - 1) {
                row_end = rows;
            }
        }

        // compute col_begin and col_end based on rank
        col_begin = (rank % proc_col) * num_cols_per_proc;
        col_end = col_begin + num_cols_per_proc;

        if (cols % proc_col != 0) {
            // last column takes the reaminder
            if ((rank + 1) % proc_col == 0) {
                col_end = cols;
            }
        }

        // all processors initialize the grid vector
        for (int i = 0; i < num_parts; i++) {
            int block_idx_x = (int) (parts[i].x / block_size);
            int block_idx_y = (int) (parts[i].y / block_size);
            // insert particle into corresponding block in the grid
            if (block_idx_y < row_begin || block_idx_y >= row_end || block_idx_x < col_begin || block_idx_x >= col_end) continue;
            block& blk = grid[block_idx_x + block_idx_y * cols];
            blk.particles.insert(&parts[i]);
        }
    }
}

std::vector<int> find_neighbours(int rank, std::map<int, std::vector<block>>& neighbour_blocks_map) {
    std::vector<int> neighbours;
    int row = rank / proc_col;
    int col = rank % proc_col;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;
            int new_row = row + i;
            int new_col = col + j;
            // we will find valid neighbours in 8 directions and add related blocks into the map
            // we only add 1 block in upper left / upper right / lower left / lower right direction
            // add 1 row or 1 column in other directions
            // since we only need to take these blocks into consideration
            if (new_row >= 0 && new_row < proc_row && new_col >= 0 && new_col < proc_col) {
                // index of neighbour
                int new_idx = new_row * proc_col + new_col;
                neighbours.push_back(new_idx);
                // upper left corner
                if (i == -1 && j == -1) neighbour_blocks_map[new_idx].push_back(grid[row_begin * cols + col_begin]);
                // upper right corner
                else if (i == -1 && j == 1) neighbour_blocks_map[new_idx].push_back(grid[row_begin * cols + col_end - 1]);
                // lower left corner
                else if (i == 1 && j == -1) neighbour_blocks_map[new_idx].push_back(grid[(row_end - 1) * cols + col_begin]);
                // lower right corner
                else if (i == 1 && j == 1) neighbour_blocks_map[new_idx].push_back(grid[(row_end - 1) * cols + col_end - 1]);

                else {
                    // up
                    if (i == -1 && j == 0) {
                        for (int jj = col_begin; jj < col_end; jj++) {
                            neighbour_blocks_map[new_idx].push_back(grid[row_begin * cols + jj]);
                        }
                    } else if (i == 0 && j == -1) { // left
                        for (int ii = row_begin; ii < row_end; ii++) {
                            neighbour_blocks_map[new_idx].push_back(grid[ii * cols + col_begin]);
                        }
                    } else if (i == 0 && j == 1) { // right
                        for (int ii = row_begin; ii < row_end; ii++) {
                            neighbour_blocks_map[new_idx].push_back(grid[ii * cols + col_end - 1]);
                        }
                    } else if (i == 1 && j == 0) { // down
                        for (int jj = col_begin; jj < col_end; jj++) {
                            neighbour_blocks_map[new_idx].push_back(grid[(row_end - 1) * cols + jj]);
                        }
                    } 
                }
            }
        }
    }

    return neighbours;
}

// apply force from received particles into particles in block ranging [r_begin, r_end) and [c_begin, c_end)
void apply_force_nearby(int r_begin, int r_end, int c_begin, int c_end, std::vector<particle_t>& recv) {
    for (int i = r_begin; i < r_end; i++) {
        for (int j = c_begin; j < c_end; j++) {
            block& blk = grid[i * cols + j];
            for (auto p : blk.particles) {
                for (auto new_p : recv) {
                    apply_force(*p, new_p);
                }
            }
        }
    }
}

int get_target_processor(int row_idx, int col_idx) {
    int max_row_idx = proc_row * num_rows_per_proc - 1;
    int max_col_idx = proc_col * num_cols_per_proc - 1;

    // when # of blocks in a row or a column is not divisible by # of procs in a row or a column
    // we should truncate it
    int calc_row_idx = std::min(max_row_idx, row_idx);
    int calc_col_idx = std::min(max_col_idx, col_idx);

    return calc_row_idx / num_rows_per_proc * proc_col + calc_col_idx / num_cols_per_proc;
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    if (rank < proc_in_use) {
        for (int i = row_begin; i < row_end; i++) {
            for (int j = col_begin; j < col_end; j++) {

                int block_idx = i * cols + j;
                std::set<particle_t*>& particles = grid[block_idx].particles;

                for (auto p : particles) {
                    p->ax = p->ay = 0;

                    // 8 neighbours
                    for (int k = -1; k <= 1; k++) {
                        for (int m = -1; m <= 1; m++) {

                            int block_new_idx_i = i + k;
                            int block_new_idx_j = j + m;
                            // not in the current processor
                            if (block_new_idx_i < row_begin || block_new_idx_i >= row_end || block_new_idx_j < col_begin || block_new_idx_j >= col_end) {
                                continue;
                            }

                            int block_new_idx = block_new_idx_i * cols + block_new_idx_j;
                            // if the new block doesn't have any particle, then just continue
                            if (grid[block_new_idx].particles.size() == 0) {
                                continue;
                            }

                            std::set<particle_t*>& new_particles = grid[block_new_idx].particles;
                            // apply force between two blocks
                            for (auto new_p : new_particles) {
                                apply_force(*p, *new_p);
                            }
                        }
                    }   
                }
            }
        }

        std::map<int, std::vector<block>> neighbour_blocks_map;
        // find neighbour processors based on current rank, and store the blocks into map
        std::vector<int> neighbours = find_neighbours(rank, neighbour_blocks_map);

        int num_neighbours = neighbours.size();
        std::vector<int> particle_count(proc_in_use);
        std::vector<std::vector<particle_t>> particles_to_send(proc_in_use);

        // for each neighbour, store count and particles into send buffer
        for (auto& entry : neighbour_blocks_map) {
            int neighbour_rank = entry.first;
            auto& blocks = entry.second;
            auto tmp = collect_particles(blocks);
            particle_count[neighbour_rank] = tmp.size();
            particles_to_send[neighbour_rank] = std::move(tmp);
        }

        std::vector<MPI_Request> reqs(num_neighbours * 2);
        std::vector<int> recv_counts(proc_in_use);

        // communicate count first
        for (int i = 0; i < neighbours.size(); i++) {
            int neighbour_rank = neighbours[i];
            MPI_Isend(&particle_count[neighbour_rank], 1, MPI_INT, neighbour_rank, 0, MPI_COMM_WORLD, &reqs[i]);
            MPI_Irecv(&recv_counts[neighbour_rank], 1, MPI_INT, neighbour_rank, 0, MPI_COMM_WORLD, &reqs[num_neighbours + i]);
        }

        // wait for all the sending and receiving to finish
        MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

        std::vector<std::vector<particle_t>> particles_to_recv(proc_in_use); 
        // resize receive buffer
        for (int i = 0; i < proc_in_use; i++) {
            particles_to_recv[i].resize(recv_counts[i]);
        }

        // send real data
        std::vector<MPI_Request> data_reqs(num_neighbours * 2);
        for (int i = 0; i < num_neighbours; i++) {
            int neighbour_rank = neighbours[i];
            auto send_buffer = particles_to_send[neighbour_rank];
            MPI_Isend(send_buffer.data(), send_buffer.size(), PARTICLE, neighbour_rank, 1, MPI_COMM_WORLD, &data_reqs[i]);
            auto& recv_buffer = particles_to_recv[neighbour_rank];
            MPI_Irecv(recv_buffer.data(), recv_buffer.size(), PARTICLE, neighbour_rank, 1, MPI_COMM_WORLD, &data_reqs[num_neighbours + i]);
        }
        
        // wait for all the sending and receiving to finish
        MPI_Waitall(data_reqs.size(), data_reqs.data(), MPI_STATUSES_IGNORE);

        // apply force from received particles from neighbour rank
        for (int i = 0; i < num_neighbours; i++) {
            int neighbour_rank = neighbours[i];
            std::vector<particle_t> recv = particles_to_recv[neighbour_rank];

            if (recv.size() == 0) continue;

            int rank_row = rank / proc_col;
            int rank_col = rank % proc_col;

            int neighbour_rank_row = neighbour_rank / proc_col;
            int neighbour_rank_col = neighbour_rank % proc_col;

            // relative position between current processor and neighbour processor
            int dx = neighbour_rank_row - rank_row, dy = neighbour_rank_col - rank_col;

            // upper left
            if (dx == -1 && dy == -1) {
                apply_force_nearby(row_begin, row_begin + 1, col_begin, col_begin + 1, recv);
            } else if (dx == -1 && dy == 0) { // up
                apply_force_nearby(row_begin, row_begin + 1, col_begin, col_end, recv);
            } else if (dx == -1 && dy == 1) { // upper right
                apply_force_nearby(row_begin, row_begin + 1, col_end - 1, col_end, recv);
            } else if (dx == 0 && dy == -1) { // left
                apply_force_nearby(row_begin, row_end, col_begin, col_begin + 1, recv);
            } else if (dx == 0 && dy == 1) { // right
                apply_force_nearby(row_begin, row_end, col_end - 1, col_end, recv);
            } else if (dx == 1 && dy == -1) { // lower left
                apply_force_nearby(row_end - 1, row_end, col_begin, col_begin + 1, recv);
            } else if (dx == 1 && dy == 0) { // down
                apply_force_nearby(row_end - 1, row_end, col_begin, col_end, recv);
            } else if (dx == 1 && dy == 1) { // lower right
                apply_force_nearby(row_end - 1, row_end, col_end - 1, col_end, recv);
            }
        }

        // after applying force, move particles
        for (int i = row_begin; i < row_end; i++) {
            for (int j = col_begin; j < col_end; j++) {
                block& blk = grid[i * cols + j];
                for (auto p : blk.particles) {
                    move(*p, size);
                }
            }
        }
    }

    std::vector<std::vector<particle_t>> particles_to_send_alltoall(proc_in_use);
    
    // find out which particles needs to be sent to other processors after computing new x and y position of a particle
    if (rank < proc_in_use) {
        for (int i = row_begin; i < row_end; i++) {
            for (int j = col_begin; j < col_end; j++) {
                int old_idx = i * cols + j;
                block& blk = grid[old_idx];
                std::set<particle_t*>& particles = blk.particles;

                for (auto it = particles.begin(); it != particles.end(); ) {
                    particle_t* p = *it;

                    int block_idx_x = (int) (p->x / block_size);
                    int block_idx_y = (int) (p->y / block_size);

                    int new_idx = block_idx_x + block_idx_y * cols;
                    int new_idx_row = new_idx / cols;
                    int new_idx_col = new_idx % cols;

                    if (new_idx != old_idx) {
                        // remains in the same processor
                        if (new_idx_row >= row_begin && new_idx_row < row_end && new_idx_col >= col_begin && new_idx_col < col_end) {
                            // move to new index
                            block* new_block = &grid[new_idx_row * cols + new_idx_col];
                            new_block->particles.insert(p);
                        } else {
                            // compute target processor based on new index
                            int target_processor = get_target_processor(new_idx_row, new_idx_col);
                            // add it to send buffer
                            particles_to_send_alltoall[target_processor].push_back(*p);
                        }
                        
                        // remove from current block
                        it = particles.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
        }
    }
    
    std::vector<particle_t> flat_send_buffer;
    std::vector<int> send_counts_alltoall(num_procs, 0);

    // put count and particles into send buffer
    for (int i = 0; i < proc_in_use; i++) {
        send_counts_alltoall[i] = particles_to_send_alltoall[i].size();
        flat_send_buffer.insert(flat_send_buffer.end(), particles_to_send_alltoall[i].begin(), particles_to_send_alltoall[i].end());
    }

    // communicate count first
    std::vector<int> recv_counts_alltoall(num_procs);
    MPI_Alltoall(send_counts_alltoall.data(), 1, MPI_INT, recv_counts_alltoall.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> sdispls(num_procs, 0), rdispls(num_procs, 0);
    std::partial_sum(send_counts_alltoall.begin(), send_counts_alltoall.end() - 1, sdispls.begin() + 1);
    std::partial_sum(recv_counts_alltoall.begin(), recv_counts_alltoall.end() - 1, rdispls.begin() + 1);

    std::vector<particle_t> flat_recv_buffer(std::accumulate(recv_counts_alltoall.begin(), recv_counts_alltoall.end(), 0));

    particle_t dummy_send_particle;
    particle_t dummy_recv_particle;

    // if send buffer and receive buffer is null, replace it with dummy buffer
    void* send_buf = flat_send_buffer.size() == 0 ? &dummy_send_particle : flat_send_buffer.data();
    void* recv_buf = flat_recv_buffer.size() == 0 ? &dummy_recv_particle : flat_recv_buffer.data();

    // communicate real data
    MPI_Alltoallv(send_buf, send_counts_alltoall.data(), sdispls.data(), PARTICLE, recv_buf, recv_counts_alltoall.data(), rdispls.data(), PARTICLE, MPI_COMM_WORLD);

    // add received particles into blocks
    if (rank < proc_in_use) {
        for (int i = 0; i < flat_recv_buffer.size(); i++) {
            particle_t* p = new particle_t(flat_recv_buffer[i]);
            int block_idx_x = (int) (p->x / block_size);
            int block_idx_y = (int) (p->y / block_size);
            block& blk = grid[block_idx_x + block_idx_y * cols];
            blk.particles.insert(p);
        }
    }
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
    int local_count = 0;

    std::vector<particle_t> local_particle_data;
 
    if (rank < proc_in_use) {
        for (int i = row_begin; i < row_end; i++) {
            for (int j = col_begin; j < col_end; j++) {
                block& blk = grid[i * cols + j];
                local_count += blk.particles.size();
                for (auto p : blk.particles) {
                    local_particle_data.push_back(*p);
                }
            }
        }
    }

    // communicate counts first
    std::vector<int> all_counts(num_procs);
    MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<particle_t> all_data;
    std::vector<int> displacements(proc_in_use, 0);

    if (rank == 0) {
        int total_count = 0;
        for (int i = 0; i < proc_in_use; ++i) {
            total_count += all_counts[i];
            if (i > 0) {
                displacements[i] = displacements[i - 1] + all_counts[i - 1];
            }
        }
        all_data.resize(total_count);
    }

    // gather real data
    MPI_Gatherv(local_particle_data.data(), local_count, PARTICLE, all_data.data(), all_counts.data(), displacements.data(), PARTICLE, 0, MPI_COMM_WORLD);

    // update parts array based on id
    if (rank == 0) {
        for (int i = 0; i < num_parts; i++) {
            particle_t p = all_data[i];
            int id = p.id;
            parts[id - 1].ax = p.ax;
            parts[id - 1].ay = p.ay;
            parts[id - 1].x = p.x;
            parts[id - 1].y = p.y;
            parts[id - 1].vx = p.vx;
            parts[id - 1].vy = p.vy;
        }
    }
}
