#include <chrono>

namespace bo_mary {
    
  template<typename VectorBlock, typename MatrixBlock>
  CholSolver<VectorBlock, MatrixBlock>::CholSolver() {
    _L = new BoTriangMatrix<MatrixBlock>();
    _y = new BoVector<VectorBlock>(); 
  }
  

  
  template<typename VectorBlock, typename MatrixBlock>
  void CholSolver<VectorBlock, MatrixBlock>::allocate(const int n) {
    _L->allocate(n);
    _y->allocate(n);
  }
  
  template<typename VectorBlock, typename MatrixBlock>
  void CholSolver<VectorBlock, MatrixBlock>::forwardSubstitution(BoVector<VectorBlock>& X, 
                                                                 BoTriangMatrix<MatrixBlock>& L, 
                                                                 BoVector<VectorBlock>& B) {
    const int rows = X.size();
    if(rows != L.rows() || rows != B.size()) {
      throw std::runtime_error("[BoMatrix]: Forward Substitution not allowed for matrix with different sizes!");
    }
    
    for(int r = 0; r < rows; ++r) {
      X.at(r) = B.at(r);
      for(int c = 0; c < r; ++c)
        X.at(r) -= L.at(r,c) * X.at(c);
      
      X.at(r) = L.at(r,r).inverse() * X.at(r);
    }
    
  }

  template<typename VectorBlock, typename MatrixBlock>
  void CholSolver<VectorBlock, MatrixBlock>::backwardSubstitution(BoVector<VectorBlock>& X, 
                                                                  BoTriangMatrix<MatrixBlock>& L, 
                                                                  BoVector<VectorBlock>& B) {
    const int rows = X.size();
    if(rows != L.rows() || rows != B.size()) {
      throw std::runtime_error("[BoMatrix]: Backward Substitution not allowed for matrix with different sizes!");
    }
    
    for(int r = rows-1; r >= 0; --r) {
      X.at(r) = B.at(r);
      for(int c = r+1; c < rows; c++)
        X.at(r) -= L.at(c,r).transpose() * X.at(c);

      X.at(r) = L.at(r,r).transpose().inverse() * X.at(r);
    }

  }
  
  
  template<typename VectorBlock, typename MatrixBlock>
  void CholSolver<VectorBlock, MatrixBlock>::solve(BoVector<VectorBlock>& dx, 
                                                   BoMatrix<MatrixBlock>& H,
                                                   BoVector<VectorBlock>& b,
                                                   Timings& time_stats) {

    auto cholesky_start = std::chrono::high_resolution_clock::now();
    H.computeCholesky(*_L);
    auto cholesky_end = std::chrono::high_resolution_clock::now();
    time_stats.cholesky += std::chrono::duration_cast<std::chrono::microseconds>(cholesky_end- cholesky_start).count();

    auto forward_start = std::chrono::high_resolution_clock::now();
    forwardSubstitution(*_y, *_L, b);
    auto forward_end = std::chrono::high_resolution_clock::now();
    time_stats.forward += std::chrono::duration_cast<std::chrono::microseconds>(forward_end- forward_start).count();


    auto backward_start = std::chrono::high_resolution_clock::now();
    backwardSubstitution(dx, *_L, *_y);
    auto backward_end = std::chrono::high_resolution_clock::now();
    time_stats.backward += std::chrono::duration_cast<std::chrono::microseconds>(backward_end- backward_start).count();

  }
  
}