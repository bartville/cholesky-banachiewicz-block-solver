#pragma once

#include "bovector.h"
#include "bomatrix.h"
#include "botriangular_mat.h"
#include <iostream>

namespace bo_mary {
  
  template<typename VectorBlock, typename MatrixBlock>
  class CholSolver {
  public:    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    struct Timings {
      Timings() { 
        cholesky = 0.0;
        forward = 0.0;
        backward = 0.0;
      }
      double cholesky;
      double forward;
      double backward;
    };

    CholSolver();    
    ~CholSolver() {
      std::cerr << "[CholSolver]: deleting" << std::endl;
      delete _L;
      delete _y;
      std::cerr << "[CholSolver]: deleted!" << std::endl;
    }
    
    inline void allocate(const int n_blocks);

    inline void solve(BoVector<VectorBlock>& dx,
               BoMatrix<MatrixBlock>& H,
               BoVector<VectorBlock>& b,
               Timings& time_stats);
    
  protected:
    inline void backwardSubstitution(BoVector<VectorBlock>& X, 
                              BoTriangMatrix<MatrixBlock>& L, 
                              BoVector<VectorBlock>& B);
    inline void forwardSubstitution(BoVector<VectorBlock>& X, 
                             BoTriangMatrix<MatrixBlock>& L, 
                             BoVector<VectorBlock>& B);    
    
  private:    
    BoTriangMatrix<MatrixBlock>* _L; // cholesky
    BoVector<VectorBlock>* _y;       // tmp vector for back-forw substitution

    
  };
  
}

#include "chol_solver.hpp"