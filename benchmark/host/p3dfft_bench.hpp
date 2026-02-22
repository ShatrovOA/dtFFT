#pragma once

#include "config.hpp"

#include "p3dfft.h"
#include <math.h>
#include <stdio.h>
#include <cstring>
#include <cmath>


double run_p3dfft(const std::vector<int>& dims)
{
    int comm_rank, comm_size;
    int gdims[3], gdims2[3], gsdims[3];
    int mem_order1[3];
    int pdims[3];

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    if (comm_rank == 0)
    {
        printf("#####################################\n");
        printf("P3DFFT++ benchmark\n");
        printf("#####################################\n");
    }


    // Establish 2D processor grid decomposition, either by reading from file 'dims' or by an MPI default

    // Set up work structures for P3DFFT
    p3dfft::setup();
    // Set up transform types for 3D transform: real-to-complex and complex-to-real
    int type_ids1[3] = {p3dfft::CFFT_FORWARD_D, p3dfft::CFFT_FORWARD_D, p3dfft::CFFT_FORWARD_D};
    int type_ids2[3] = {p3dfft::CFFT_BACKWARD_D, p3dfft::CFFT_BACKWARD_D, p3dfft::CFFT_BACKWARD_D};

    // Define the transform types for these two transforms
    p3dfft::trans_type3D type_forward(type_ids1);
    p3dfft::trans_type3D type_backward(type_ids2);

    // Set up global dimensions of the grid and processor and memory ordering.

    gdims[0] = dims[0];
    gdims[1] = dims[1];
    gdims[2] = dims[2];
    for (int i = 0; i < 3; i++)
    {
        mem_order1[i] = i; // The simplest case of sequential ordering
    }

    int dmap1[] = {0, 1, 2}; // Mapping data dimension X onto processor dimension X, and so on -
                             // this is an X pencil, since Px =1

    int mem_order2[] = {1, 2, 0};

    // Initialize final grid configuration

    int dmap2[] = {1, 2, 0}; // Mapping data dimension X onto processor dimension Y,
                             // Y onto Z and Z onto X
                             // this is a Z-pencil, since Px =1 - or at least one way to define it
                             // (the other would be (2,1,0))
    pdims[0] = 1;
    int p, pmin = -1;
    int sdims1[3], glob_start1[3];
    double tmin = 10000000.;

    double best_time = INFINITY;
    int best_grid[3];

    for(int p=1; p <= comm_size; p++)
    {
        if ( comm_size % p ) continue;
        if ( dims[1] < p || dims[2] < comm_size / p ) continue;

        pdims[0] = 1;
        pdims[1] = p;
        pdims[2] = comm_size / p;

        p3dfft::ProcGrid pgrid(pdims, MPI_COMM_WORLD);
        if (comm_rank == 0)
            printf("Using processor grid %d x %d\n", pdims[1], pdims[2]);

        // Initialize the initial grid

        p3dfft::DataGrid Xpencil(gdims, -1, &pgrid, dmap1, mem_order1);
        p3dfft::DataGrid Zpencil(gdims, -1, &pgrid, dmap2, mem_order2);


        for (int i = 0; i < 3; i++)
        {
            glob_start1[mem_order1[i]] = Xpencil.GlobStart[i];
            sdims1[mem_order1[i]] = Xpencil.Ldims[i];
        }

        long int size1 = sdims1[0] * sdims1[1] * sdims1[2];

        // printf("sdims1 = %dx%dx%d\n", sdims1[0], sdims1[1], sdims1[2]);


        // init_wave(IN,gdims,sdims1,glob_start1);

        // std::memset()

        // Determine local array dimensions and allocate fourier space, complex-valued out array

        int sdims2[3], glob_start2[3];
        for (int i = 0; i < 3; i++)
        {
            glob_start2[mem_order2[i]] = Zpencil.GlobStart[i];
            sdims2[mem_order2[i]] = Zpencil.Ldims[i];
            gsdims[mem_order2[i]] = gdims[i];
        }

        long int size2 = sdims2[0] * sdims2[1] * sdims2[2];
        long int alloc_size = std::max(size1, size2);


        // Allocate input/outpu array for in-place tansform, using the larger size of the two
        p3dfft::complex_double *IN = new p3dfft::complex_double[alloc_size];
        p3dfft::complex_double *OUT = new p3dfft::complex_double[alloc_size];

        double create_time = -MPI_Wtime();

        // Set up 3D transforms, including stages and plans, for forward trans.
        auto *trans_f = new p3dfft::transform3D<p3dfft::complex_double, p3dfft::complex_double>(Xpencil, Zpencil, &type_forward);
        // Set up 3D transforms, including stages and plans, for backward trans.
        auto *trans_b = new p3dfft::transform3D<p3dfft::complex_double, p3dfft::complex_double>(Zpencil, Xpencil, &type_backward);

        create_time += MPI_Wtime();

        // if(comm_rank == 0) {
        //     printf("Plans created successfully, time spent: %f\n", create_time);
        // }

        memset(IN, 0, alloc_size * sizeof(p3dfft::complex_double));

        for (int counter = 0; counter < WARMUP_ITERATIONS; counter++) {
            trans_f->exec(IN, OUT);
            trans_b->exec(OUT, IN);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        double start_time = MPI_Wtime();
        for (int counter = 0; counter < TEST_ITERATIONS; counter++) {
            trans_f->exec(IN, OUT);
            trans_b->exec(OUT, IN);
        }
        double end_time = MPI_Wtime();
        double elapsed_time = (end_time - start_time);

        double min_s, max_s, avg_s;
        MPI_Allreduce(&elapsed_time, &min_s, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&elapsed_time, &max_s, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&elapsed_time, &avg_s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        avg_s /= (double)comm_size;

        // if(comm_rank == 0) {
        //     printf("min time: %f [s]\n", min_s);
        //     printf("max time: %f [s]\n", max_s);
        //     printf("avg time: %f [s]\n", avg_s);
        //     printf("avg time per iteration: %f [s]\n", avg_s / (double)TEST_ITERATIONS);
        //     printf("----------------------------------------\n");
        // }

        if ( max_s < best_time ) {
            best_time = max_s;
            best_grid[0] = pdims[0];
            best_grid[1] = pdims[1];
            best_grid[2] = pdims[2];
        }


        delete[] IN, OUT;
        delete trans_f, trans_b;
        // Clean up P3DFFT++ structures
    }
    if (comm_rank == 0) {
        printf("P3DFFT++ Results\n");
        printf("----------------------------------------\n");
        printf("Fastest execution time: %f\n", best_time);
        printf("Fastest grid: %ix%ix%i\n", best_grid[0], best_grid[1], best_grid[2]);
        printf("----------------------------------------\n");
    }

    return best_time;
}
