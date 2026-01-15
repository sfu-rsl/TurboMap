// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, W. Burgard
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Modified Raul Mur Artal (2014)
// - Stop criterium (solve function)

#include "optimization_algorithm_levenberg.h"

#include <iostream>

#include "../stuff/timeutil.h"
#include "../../../../include/LoopClosureDetector.h"
#include "base_edge.h"

#include "sparse_optimizer.h"
#include "solver.h"
#include "batch_stats.h"
#include "chrono"
using namespace std;

namespace g2o {

  OptimizationAlgorithmLevenberg::OptimizationAlgorithmLevenberg(Solver* solver) :
    OptimizationAlgorithmWithHessian(solver)
  {
    _currentLambda = -1.;
    _tau = 1e-5; // Carlos: originally 1e-5
    _goodStepUpperScale = 2./3.;
    _goodStepLowerScale = 1./3.;
    _userLambdaInit = _properties.makeProperty<Property<double> >("initialLambda", 0.);
    _maxTrialsAfterFailure = _properties.makeProperty<Property<int> >("maxTrialsAfterFailure", 10); // Carlos: Originally 10 iterations
    _ni=2.;
    _levenbergIterations = 0;
    _nBad = 0;
  }

  OptimizationAlgorithmLevenberg::~OptimizationAlgorithmLevenberg()
  {
  }

  OptimizationAlgorithm::SolverResult OptimizationAlgorithmLevenberg::solve(int iteration, bool online)
  {
    chrono::steady_clock::time_point buildStructureStart = chrono::steady_clock::now();
    assert(_optimizer && "_optimizer not set");
    assert(_solver->optimizer() == _optimizer && "underlying linear solver operates on different graph");

    if (iteration == 0 && !online) { // built up the CCS structure, here due to easy time measure
      bool ok = _solver->buildStructure();
      if (! ok) {
        cerr << __PRETTY_FUNCTION__ << ": Failure while building CCS structure" << endl;
        return OptimizationAlgorithm::Fail;
      }
    }

    chrono::steady_clock::time_point buildStructureEnd = chrono::steady_clock::now();
    if(LoopClosureDetector::instance().isLoopClosureDetected() || _gbaStats)
    {
      cout << "Levenberg [Build Structure]: " << chrono::duration_cast<chrono::microseconds>(buildStructureEnd - buildStructureStart).count() << " ms" << endl;
    }

    double t = get_monotonic_time();
    _optimizer->computeActiveErrors();
    G2OBatchStatistics* globalStats = G2OBatchStatistics::globalStats();
    if (globalStats) {
      globalStats->timeResiduals = get_monotonic_time()-t;
      t=get_monotonic_time();
    }

    chrono::steady_clock::time_point computeActiveErrors = chrono::steady_clock::now();

    if(LoopClosureDetector::instance().isLoopClosureDetected() || _gbaStats)
    {
      cout << "Levenberg [ComputeActiveErrors]: " << chrono::duration_cast<chrono::microseconds>(computeActiveErrors - buildStructureEnd).count() << " ms" << endl;
    }

    double currentChi = _optimizer->activeRobustChi2();
    double tempChi=currentChi;

    double iniChi = currentChi;

    _solver->buildSystem(iteration);

    if (globalStats) {
      globalStats->timeQuadraticForm = get_monotonic_time()-t;
    }

    chrono::steady_clock::time_point buildSystemEnd = chrono::steady_clock::now();

    if(LoopClosureDetector::instance().isLoopClosureDetected() || _gbaStats)
    {
      cout << "Levenberg [Build System]: " << chrono::duration_cast<chrono::microseconds>(buildSystemEnd - computeActiveErrors).count() << " ms" << endl;
    }

    // core part of the Levenbarg algorithm
    if (iteration == 0) {       
      _currentLambda = computeLambdaInit();
      _ni = 2;
      _nBad = 0;
    }

    chrono::steady_clock::time_point computeLambdaInit = chrono::steady_clock::now();

    if(LoopClosureDetector::instance().isLoopClosureDetected() || LoopClosureDetector::instance().isMergeDetected() || _gbaStats)
    {
      cout << "Levenberg [computeLambdaInit]: " << chrono::duration_cast<chrono::microseconds>(computeLambdaInit - buildSystemEnd).count() << " ms" << endl;
    }

    std::chrono::steady_clock::time_point start, end, t1, t2, t3;
    std::chrono::duration<double> updateSolution = std::chrono::duration<double>::zero();
    std::chrono::duration<double> linearSolution = std::chrono::duration<double>::zero();
    std::chrono::duration<double> restSolution = std::chrono::duration<double>::zero();

    double rho=0;
    int& qmax = _levenbergIterations;
    qmax = 0;
    do {
      _optimizer->push();
      if (globalStats) {
        globalStats->levenbergIterations++;
        t=get_monotonic_time();
      }
      start = std::chrono::steady_clock::now();
      // update the diagonal of the system matrix
      t1 = std::chrono::steady_clock::now();
      _solver->setLambda(_currentLambda, true);
      t2 = std::chrono::steady_clock::now();
      bool ok2 = _solver->solve();
      t3 = std::chrono::steady_clock::now();
      if (globalStats) {
        globalStats->timeLinearSolution+=get_monotonic_time()-t;
        t=get_monotonic_time();
      }
      end = std::chrono::steady_clock::now();
      linearSolution += end - start;
      start = std::chrono::steady_clock::now();
      if(_optimizer->getUseGPU())
      {
        _optimizer->graphInterface()->update(_solver->x());
        _optimizer->update(_solver->x());
      } else {
        _optimizer->update(_solver->x());
      }
      if (globalStats) {
        globalStats->timeUpdate = get_monotonic_time()-t;
      }

      end = std::chrono::steady_clock::now();
      updateSolution = end - start;
      start = std::chrono::steady_clock::now();
      // restore the diagonal
      _solver->restoreDiagonal();

      _optimizer->computeActiveErrors();
      tempChi = _optimizer->activeRobustChi2();
      // cout << "tempChi: " << tempChi << endl;
      if (! ok2)
        tempChi=std::numeric_limits<double>::max();

      rho = (currentChi-tempChi);
      double scale = computeScale();
      scale += 1e-3; // make sure it's non-zero :)
      rho /=  scale;

      if (rho>0 && g2o_isfinite(tempChi)){ // last step was good
        double alpha = 1.-pow((2*rho-1),3);
        // crop lambda between minimum and maximum factors
        alpha = (std::min)(alpha, _goodStepUpperScale);
        double scaleFactor = (std::max)(_goodStepLowerScale, alpha);
        _currentLambda *= scaleFactor;
        _ni = 2;
        currentChi=tempChi;
        _optimizer->discardTop();
      } else {
        _currentLambda*=_ni;
        _ni*=2;
        _optimizer->pop(); // restore the last state before trying to optimize
      }
      qmax++;
      end = std::chrono::steady_clock::now();
      restSolution += end - start;
    } while (rho<0 && qmax < _maxTrialsAfterFailure->value() && ! _optimizer->terminate());

    if (qmax == _maxTrialsAfterFailure->value() || rho==0)
    {
      // cout << "qmax = " << qmax << "             rho = " << rho << endl;
      return Terminate;
    }

    //Stop criterium (Raul)
    if((iniChi-currentChi)*1e3<iniChi)
        _nBad++;
    else
        _nBad=0;

    if(_nBad>=3)
    {
        return Terminate;
    }

    chrono::steady_clock::time_point mainAlgorithmEnd = chrono::steady_clock::now();

    if(LoopClosureDetector::instance().isLoopClosureDetected() || _gbaStats)
    {
      cout << "Update: " << chrono::duration_cast<chrono::microseconds>(updateSolution).count() << " ms" << endl;
      cout << "Linear: " << chrono::duration_cast<chrono::microseconds>(linearSolution).count() << " [ " <<
      chrono::duration_cast<chrono::microseconds>(t2 - t1).count() << ", " << chrono::duration_cast<chrono::microseconds>(t3 - t2).count() << "] ms" << endl;
      cout << "Rest: " << chrono::duration_cast<chrono::microseconds>(restSolution).count() << " ms" << endl;
      cout << "Levenberg [Main Algorithm]: " << chrono::duration_cast<chrono::microseconds>(mainAlgorithmEnd - computeLambdaInit).count() << " ms" << endl;
      // cout << "Iteration : " << iteration << endl;
      // cout << "Edges: " << _optimizer->activeEdges().size() << endl;
      // cout << "Vertices: " << _optimizer->activeVertices().size() << endl;
      // std::string filename = "opt_" + std::to_string(iteration) + ".txt";
      // _optimizer->save(filename.c_str());
    }

    return OK;
  }

  double OptimizationAlgorithmLevenberg::computeLambdaInit() const
  {
    if (_userLambdaInit->value() > 0)
      return _userLambdaInit->value();
    double maxDiagonal=0.;
    for (size_t k = 0; k < _optimizer->indexMapping().size(); k++) {
      OptimizableGraph::Vertex* v = _optimizer->indexMapping()[k];
      assert(v);
      int dim = v->dimension();
      for (int j = 0; j < dim; ++j){
        maxDiagonal = std::max(fabs(v->hessian(j,j)),maxDiagonal);
      }
    }
    return _tau*maxDiagonal;
  }

  double OptimizationAlgorithmLevenberg::computeScale() const
  {
    double scale = 0.;
    for (size_t j=0; j < _solver->vectorSize(); j++){
      scale += _solver->x()[j] * (_currentLambda * _solver->x()[j] + _solver->b()[j]);
    }
    return scale;
  }

  void OptimizationAlgorithmLevenberg::setMaxTrialsAfterFailure(int max_trials)
  {
    _maxTrialsAfterFailure->setValue(max_trials);
  }

  void OptimizationAlgorithmLevenberg::setUserLambdaInit(double lambda)
  {
    _userLambdaInit->setValue(lambda);
  }

  void OptimizationAlgorithmLevenberg::printVerbose(std::ostream& os) const
  {
    os
      << "\t schur= " << _solver->schur()
      << "\t lambda= " << FIXED(_currentLambda)
      << "\t levenbergIter= " << _levenbergIterations;
  }

} // end namespace
