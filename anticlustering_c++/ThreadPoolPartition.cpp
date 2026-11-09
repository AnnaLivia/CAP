#include <thread>
#include <condition_variable>
#include <vector>
#include <sstream>
#include <gurobi_c++.h>
#include "ThreadPoolPartition.h"
#include "config_params.h"
#include "cap.h"

// Constructor initializes the thread pool with a given number of threads
ThreadPoolPartition::ThreadPoolPartition(SharedDataPartition* shared_data, int n_thread)
    : shared_data(shared_data),
      done(false),
      max_concurrent_solves(std::max(1, std::min(n_thread, sdp_solver_max_concurrent_solves))),
      active_solves(0)
{
    const int numberOfThreads = n_thread;
    threads.reserve(numberOfThreads);
    for (int i = 0; i < numberOfThreads; ++i) {
        threads.emplace_back(&ThreadPoolPartition::doWork, this, i);
    }
}

// The destructor joins all the threads so the program can exit gracefully
void ThreadPoolPartition::quitPool() {
    {
        std::lock_guard<std::mutex> l(shared_data->queueMutex);
        done = true;
    }
    shared_data->queueConditionVariable.notify_all();
    solveConditionVariable.notify_all();
    for (auto& th : threads) th.join();
}

// Function to add a job to the thread pool's queue
void ThreadPoolPartition::addJob(PartitionJob* job) {
    {
        std::lock_guard<std::mutex> l(shared_data->queueMutex);
        shared_data->queue.push_back(job);
    }
    shared_data->queueConditionVariable.notify_one();
}

// Function used by the threads to grab work from the queue
void ThreadPoolPartition::doWork(int id) {
    // Reuse big buffers per thread to avoid repeated heap allocs
    thread_local arma::mat sol;
    thread_local arma::mat cls;

    for (;;) {
        PartitionJob* job = nullptr;

        // ---- Grab one job or exit ----
        {
            std::unique_lock<std::mutex> l(shared_data->queueMutex);
            shared_data->queueConditionVariable.wait(
                l, [&] { return done || !shared_data->queue.empty(); }
            );
            // Exit only when we're done AND nothing left to do
            if (done && shared_data->queue.empty()) break;

            job = shared_data->queue.front();
            shared_data->queue.pop_front();
            shared_data->threadStates[id] = true;
        }

        bool solve_slot_acquired = false;
        auto release_solve_slot = [&]() {
            std::lock_guard<std::mutex> sl(solveMutex);
            active_solves--;
            solveConditionVariable.notify_one();
        };

        try {
            // ---- Process job (no locks) ----
            const int np = static_cast<int>(job->ac_data.n_rows);
            const int n_constraints = static_cast<int>(
                job->ac_constraints.ml_pairs.size() + job->ac_constraints.cl_pairs.size());
            std::ostringstream sdp_log;
            double ub = 0;
            double gap = 0.0;
            {
                std::unique_lock<std::mutex> sl(solveMutex);
                solveConditionVariable.wait(sl, [&] {
                    return done || active_solves < max_concurrent_solves;
                });
                if (done) {
                    std::lock_guard<std::mutex> l(shared_data->queueMutex);
                    shared_data->threadStates[id] = false;
                    delete job;
                    shared_data->mainConditionVariable.notify_all();
                    break;
                }
                active_solves++;
                solve_slot_acquired = true;
            }
            const double lb_mssc = sdp_branch_and_bound(
                k, job->ac_data, ub, job->ac_constraints, sol, job->ub_mssc, job->init_sol, shared_data->print, &sdp_log, &gap
            );
            release_solve_slot();
            solve_slot_acquired = false;

            if (sol.n_rows != static_cast<arma::uword>(np) ||
                sol.n_cols != static_cast<arma::uword>(k)) {
                std::ostringstream oss;
                oss << "AC " << job->ac_id
                    << ": sdp_branch_and_bound returned solution of size "
                    << sol.n_rows << "x" << sol.n_cols
                    << " but expected " << np << "x" << k
                    << ". This typically means the local constrained clustering "
                    << "subproblem is infeasible (e.g., ML/CL conflict after restriction, "
                    << "or effective components < k after ML shrinking).";
                throw std::runtime_error(oss.str());
            }

            arma::uvec labels = arma::index_max(sol, 1);
            if (labels.n_elem != static_cast<arma::uword>(np)) {
                std::ostringstream oss;
                oss << "AC " << job->ac_id
                    << ": invalid label vector size after SDP solve (labels="
                    << labels.n_elem << ", expected " << np
                    << ", sol=" << sol.n_rows << "x" << sol.n_cols << ").";
                throw std::runtime_error(oss.str());
            }

            cls.set_size(np, d + 1);
            cls.cols(0, d - 1) = job->ac_data;
            cls.col(d) = arma::conv_to<arma::colvec>::from(labels);

            if (shared_data->print) {
                std::lock_guard<std::mutex> log_lock(file_mutex);
                log_file << "\n===== PC-SOS-SDP subproblem AC " << job->ac_id << " closed =====\n";
                log_file << sdp_log.str();
                log_file << "POINTS: " << np << "\n";
                log_file << "N_CONSTRAINTS: " << n_constraints << "\n";
                log_file << "===== END PC-SOS-SDP subproblem AC " << job->ac_id << " =====\n";
            }

            // ---- Publish results (very short lock) ----
            {
                std::lock_guard<std::mutex> l(shared_data->queueMutex);
                shared_data->lb_ac[job->ac_id]     = lb_mssc;
                shared_data->lbplus_ac[job->ac_id] = ub;
                shared_data->gap_ac[job->ac_id] = gap;
                shared_data->sol_ac[job->ac_id]    = cls;
                shared_data->threadStates[id]      = false;
            }

            shared_data->mainConditionVariable.notify_one();
            delete job;
        } catch (const GRBException &e) {
            if (solve_slot_acquired)
                release_solve_slot();
            std::lock_guard<std::mutex> l(shared_data->queueMutex);
            if (!shared_data->worker_failed) {
                shared_data->worker_failed = true;
                shared_data->worker_error = "Gurobi error in partition worker (AC " +
                    std::to_string(job->ac_id) + ", code " + std::to_string(e.getErrorCode()) +
                    "): " + e.getMessage();
            }
            shared_data->threadStates[id] = false;
            done = true;
            delete job;
            solveConditionVariable.notify_all();
            while (!shared_data->queue.empty()) {
                auto *pending = shared_data->queue.front();
                shared_data->queue.pop_front();
                delete pending;
            }
            shared_data->mainConditionVariable.notify_all();
            shared_data->queueConditionVariable.notify_all();
        } catch (const std::exception &e) {
            if (solve_slot_acquired)
                release_solve_slot();
            std::lock_guard<std::mutex> l(shared_data->queueMutex);
            if (!shared_data->worker_failed) {
                shared_data->worker_failed = true;
                shared_data->worker_error = "Exception in partition worker (AC " +
                    std::to_string(job->ac_id) + "): " + std::string(e.what());
            }
            shared_data->threadStates[id] = false;
            done = true;
            delete job;
            solveConditionVariable.notify_all();
            while (!shared_data->queue.empty()) {
                auto *pending = shared_data->queue.front();
                shared_data->queue.pop_front();
                delete pending;
            }
            shared_data->mainConditionVariable.notify_all();
            shared_data->queueConditionVariable.notify_all();
        } catch (...) {
            if (solve_slot_acquired)
                release_solve_slot();
            std::lock_guard<std::mutex> l(shared_data->queueMutex);
            if (!shared_data->worker_failed) {
                shared_data->worker_failed = true;
                shared_data->worker_error = "Unknown exception in partition worker.";
            }
            shared_data->threadStates[id] = false;
            done = true;
            delete job;
            solveConditionVariable.notify_all();
            while (!shared_data->queue.empty()) {
                auto *pending = shared_data->queue.front();
                shared_data->queue.pop_front();
                delete pending;
            }
            shared_data->mainConditionVariable.notify_all();
            shared_data->queueConditionVariable.notify_all();
        }
    }
}
