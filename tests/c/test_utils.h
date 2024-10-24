


#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void print_timers(double time)
{
  double t_sum, t_min, t_max;
  int comm_size, comm_rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);


  MPI_Reduce(&time, &t_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&time, &t_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&time, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if(comm_rank == 0)
  {
    printf("AVG = %f\n", t_sum / (double) comm_size);
    printf("MIN = %f\n", t_min);
    printf("MAX = %f\n", t_max);
    printf("----------------------------------------\n");
  }
}

void report_execution_time(double time_forward, double time_backward)
{
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  if(comm_rank == 0)
  {
    printf("|          Forward execution           |\n");
    printf("----------------------------------------\n");
  }
  print_timers(time_forward);
  if(comm_rank == 0)
  {
    printf("|         Backward execution           |\n");
    printf("----------------------------------------\n");
  }
  print_timers(time_backward);
}

#endif

