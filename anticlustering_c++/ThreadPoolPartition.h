#ifndef CLUSTERING_THREADPOOLPARTITION_H
#define CLUSTERING_THREADPOOLPARTITION_H

#include "sdp_branch_and_bound.h"
#include "util.h"
#include "matlab_util.h"
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

typedef struct PartitionJob {

    int ac_id;
    double ub_mssc;
    arma::mat init_sol;
    arma::mat ac_data;
    UserConstraints ac_constraints;

} PartitionJob;


typedef struct SharedDataPartition {

    // Between workers and main
    std::condition_variable mainConditionVariable;
    std::vector<bool> threadStates;

    // Queue of requests waiting to be processed
    std::deque<PartitionJob *> queue;
    // This condition variable is used for the threads to wait until there is work to do
    std::condition_variable queueConditionVariable;
    // Mutex to protect queue
    std::mutex queueMutex;

    std::vector<double> lb_ac; // used to store the lower bound extracted from each sub-problem
    std::vector<double> lbplus_ac; // used to store the upper bound extracted from each sub-problem
    std::vector<double> gap_ac; // final gap of each sub-problem
    std::vector<arma::mat> sol_ac; // used to store the partial solutions
    bool print; // used to print solution on file
    bool worker_failed = false;
    std::string worker_error;

} SharedDataPartition;

class ThreadPoolPartition {

private:

    SharedDataPartition  *shared_data;

    // We store the threads in a vector, so we can later stop them gracefully
    std::vector<std::thread> threads;

    // This will be set to true when the thread pool is shutting down. This tells
    // the threads to stop looping and finish
    bool done;
    int max_concurrent_solves;
    int active_solves;
    std::mutex solveMutex;
    std::condition_variable solveConditionVariable;

    void doWork(int id);


public:

    ThreadPoolPartition(SharedDataPartition *shared_data, int n_thread);
    void quitPool();
    void addJob(PartitionJob *partitionJob);

};


#endif //CLUSTERING_THREADPOOLPARTITION_H
