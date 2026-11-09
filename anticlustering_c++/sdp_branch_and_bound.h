#ifndef CLUSTERING_SDP_BRANCH_AND_BOUND_H
#define CLUSTERING_SDP_BRANCH_AND_BOUND_H

#include <armadillo>
#include <ostream>
#include <string>
#include "MatlabEngine.hpp"
#include "MatlabDataArray.hpp"
#include "JobQueue.h"

typedef struct UserConstraints {

    double gamma, delta;
    std::vector<std::pair<int,int>> ml_pairs, cl_pairs;
    std::vector<std::vector<int>> cl_per_point;
    std::vector<std::vector<int>> ml_per_point;
    std::vector<int> cl_weights;

} UserConstraints;

typedef struct MatlabStruct {

    std::unique_ptr<matlab::engine::MATLABEngine> matlabPtr;
    matlab::data::ArrayFactory factory;

} MatlabStruct;


typedef struct SharedData {

    // Between workers and main
    std::condition_variable mainConditionVariable;
    std::vector<bool> threadStates;

    // Queue of requests waiting to be processed
    JobAbstractQueue *queue;
    // This condition variable is used for the threads to wait until there is work to do
    std::condition_variable queueConditionVariable;
    // Mutex to protect queue
    std::mutex queueMutex;

    double root_lb;
    double best_lb;
    double global_ub;
    double gap;
    arma::sp_mat global_X;
    arma::mat global_centroids;
    int n_nodes;
    double sum_ineq;
    double sum_cp_iter;
    bool print;
    std::ostream *log_stream = nullptr;
    bool worker_failed = false;
    std::string worker_error;

} SharedData;

typedef struct InputData {

    arma::mat Ws;
    int k;
    double C_trace;
	arma::mat distances;

} InputData;


double sdp_branch_and_bound(int k, arma::mat &Ws, double &ub, UserConstraints &constraints, arma::mat &sol, bool print, std::ostream *log_stream = nullptr, double *out_gap = nullptr, double *out_root_gap = nullptr);
double sdp_branch_and_bound(int k, arma::mat &Ws, double &ub, UserConstraints &constraints, arma::mat &sol, double init_ub, bool print, std::ostream *log_stream = nullptr, double *out_gap = nullptr, double *out_root_gap = nullptr);
double sdp_branch_and_bound(int k, arma::mat &Ws, double &ub, UserConstraints &constraints, arma::mat &sol, double init_ub, const arma::mat &init_sol, bool print, std::ostream *log_stream = nullptr, double *out_gap = nullptr, double *out_root_gap = nullptr);
std::pair<JobData *, JobData *> build_root_problem(MatlabStruct *matlab_struct, InputData *input_data, SharedData *shared_data, UserConstraints &constraints);
std::pair<JobData *, JobData *> build_cl_problem(MatlabStruct *matlab_struct, NodeData *job_data, InputData *input_data, SharedData *shared_data);
std::pair<JobData *, JobData *> build_ml_problem(MatlabStruct *matlab_struct, NodeData *job_data, InputData *input_data, SharedData *shared_data);

bool is_thread_pool_working(std::vector<bool> &thread_state);
#endif //CLUSTERING_SDP_BRANCH_AND_BOUND_H
