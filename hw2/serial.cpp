#include "common.h"
#include <cmath>
#include <vector>
#include <set>
#include <iostream>

std::vector<std::set<particle_t*>> grid;
int cols;
int rows;
double block_size = 0.01;

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

void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here

    // calculate the size of the grid
    int max_grid_x = (int) (ceil(sqrt(density * num_parts) / block_size));
    int max_grid_y = max_grid_x;

    rows = max_grid_y + 1;
    cols = max_grid_x + 1;
    grid.resize(rows * cols);

    std::cout << rows << std::endl;

    for (int i = 0; i < num_parts; i++) {
        // calculate the particle belongs to which block
        int block_idx_x = (int) (parts[i].x / block_size);
        int block_idx_y = (int) (parts[i].y / block_size);
        // insert particle into corresponding block in the grid
        grid[block_idx_x + block_idx_y * cols].insert(&parts[i]);
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    for (int i = 0; i < grid.size(); i++) {
        std::set<particle_t*>& particles = grid[i];

        for (auto it = particles.begin(); it != particles.end(); ) {
            particle_t* p = *it;

            int block_idx_x = (int) (p->x / block_size);
            int block_idx_y = (int) (p->y / block_size);
            // calculate the new block index
            int new_idx = block_idx_x + block_idx_y * cols;

            // if it needs to be moved, then delete it from current set and add it to the new set
            if (new_idx != i) {
                std::cout << "new: " << new_idx << ", old: " << i << std::endl;
                it = particles.erase(it);
                grid[new_idx].insert(p);
            } else {
                ++it;
            }
        }
    }

    for (int i = 0; i < grid.size(); i++) {
        std::set<particle_t*>& particles = grid[i];
        int block_idx_y = i / cols;
        int block_idx_x = i % cols;

        for (auto p : particles) {
            p->ax = p->ay = 0;
            // traverse through the 9 blocks (itself and neighbours)
            for (int j = -1; j <= 1; j++) {
                for (int i = -1; i <= 1; i++) {
                    int block_new_idx_y = block_idx_y + j;
                    int block_new_idx_x = block_idx_x + i;
                    // boundary check
                    if (block_new_idx_y < 0 || block_new_idx_y >= rows || block_new_idx_x < 0 || block_new_idx_x >= cols) {
                        continue;
                    }

                    int block_new_idx = block_new_idx_x + block_new_idx_y * cols;
                    // if the new block doesn't have any particle, then just continue
                    if (grid[block_new_idx].size() == 0) {
                        continue;
                    }

                    std::set<particle_t*>& new_particles = grid[block_new_idx];
                    // apply force between two blocks
                    for (auto new_p : new_particles) {
                        apply_force(*p, *new_p);
                    }
                }
            }
        }
    }

    // Move Particles
    for (int i = 0; i < num_parts; ++i) {
        std::cout << "before:" << parts[i].x << std::endl;
        move(parts[i], size);
        std::cout << "after:" << parts[i].x << std::endl;
    }
}
