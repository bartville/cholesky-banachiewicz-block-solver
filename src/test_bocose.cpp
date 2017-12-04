#include <Eigen/Core>
#include <Eigen/Geometry>
#include "bovector.h"
#include "bomatrix.h"
#include "botriangular_mat.h"

#include <iostream>

using namespace bo_mary;

Eigen::Matrix<float, 6, 6> cholBart(const Eigen::Matrix<float, 6, 6>& src) {
  
  Eigen::Matrix<float, 6, 6> L;
  L.setZero();
  
  for(int r = 0; r < 6; ++r) {
    for(int c = 0; c < r+1; ++c) {
      
      float accumulator = src(r,c);
      for(int k = 0; k < c; ++k)
        accumulator -= L(r,k) * L(c,k);
      if(r == c) {
        L(r,c) = sqrt(accumulator);        
      }
      else
        L(r,c) =  1./L(c,c) * (accumulator);
    }    
  }
  return L;

}


template<typename MatBlock, typename VecBlock>
void forwardSubstitution(BoVector<VecBlock>& X, BoTriangMatrix<MatBlock>& L, BoVector<VecBlock>& B) {
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

template<typename MatBlock, typename VecBlock>
void backwardSubstitution(BoVector<VecBlock>& X, BoTriangMatrix<MatBlock>& L, BoVector<VecBlock>& B) {
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



int main(){

  const int block_size = 3;
  const int elem_size = 3;
  typedef Eigen::Matrix<float, elem_size, 1> VectorBlock;
  typedef Eigen::Matrix<float, elem_size, elem_size> MatrixBlock;
 
  BoVector<VectorBlock> vector3block;
  vector3block.allocate(100);
  std::cerr << "current vector size: " << vector3block.size() << std::endl;
  
  Eigen::Vector3f& ith_block = vector3block.at(10);
  ith_block = Eigen::Vector3f(10, 2, 5.1);
  std::cerr << vector3block.at(10).transpose() << std::endl;
  

  BoMatrix<Eigen::Matrix3f> matrix3block;
  matrix3block.allocate(100,100);
  std::cerr << "current mat size: " << matrix3block.size() << std::endl;

  Eigen::Matrix3f& rcth_block = matrix3block.at(3,90);
  rcth_block << 5, 7, 8, 0, -1, 1.2, 5.4, 6, 7;
  std::cerr << matrix3block.at(3,90) << std::endl;

  BoTriangMatrix<Eigen::Matrix3f> triangular_mat3;
  triangular_mat3.allocate(100);
  triangular_mat3.at(10,10) = matrix3block.at(3,90)*3;
  std::cerr << triangular_mat3.at(10,10) << std::endl; 

  // This istruction has to fail!
  // std::cerr << triangular_mat3.at(10,11) << std::endl;

  BoMatrix<Eigen::Matrix2f> m;
  m.allocate(3,3);
  
  m.at(0,0) << 2.3168,   1.3925, 1.3925,   1.3940;
  m.at(0,1) << 2.1461,   1.9458, 1.1110,   1.2274;
  m.at(0,2) << 2.1218,   2.7295, 1.3216,   1.7726;
  m.at(1,0) << 2.1461,   1.1110, 1.9458,   1.2274; 
  m.at(1,1) << 2.3324,   1.9385, 1.9385,   1.8288;
  m.at(1,2) << 2.0305,   2.5147, 1.7442,   2.4410;
  m.at(2,0) << 2.1218,   1.3216, 2.7295,   1.7726;
  m.at(2,1) << 2.0305,   1.7442, 2.5147,   2.4410;
  m.at(2,2) << 2.3935,   2.9153, 2.9153,   4.0859;

  BoTriangMatrix<Eigen::Matrix2f> L;
  L.allocate(3);

  m.computeCholesky(L);

  Eigen::Matrix<float, 6, 6> src4f;
  src4f << 2.3168,   1.3925,   2.1461,   1.9458,   2.1218,   2.7295,
           1.3925,   1.3940,   1.1110,   1.2274,   1.3216,   1.7726,
           2.1461,   1.1110,   2.3324,   1.9385,   2.0305,   2.5147,
           1.9458,   1.2274,   1.9385,   1.8288,   1.7442,   2.4410,
           2.1218,   1.3216,   2.0305,   1.7442,   2.3935,   2.9153,
           2.7295,   1.7726,   2.5147,   2.4410,   2.9153,   4.0859;
          
  // octave code
  // A = 0.94255, 0.83920, 0.32534, 0.95912, 0.83920, 0.32534;  0.83920, 1.91262, 0.82886, 1.33141, 0.82886, 0.38143;   0.32534, 0.82886, 0.38143, 0.55118, 0.83920, 1.91262; 0.95912, 1.33141, 0.55118, 1.35574, 0.32534, 0.95912; 0.83920, 0.82886, 0.83920, 0.32534, 1.91262, 0.82886; 0.32534, 0.83920, 1.91262, 0.95912, 0.82886, 0.12345];
          
  std::cerr << "\n\n Det src4f: " << src4f.determinant() << std::endl;
          
//  Eigen::Matrix4f bartL = cholBart(src4f);
  Eigen::Matrix<float, 6, 6> eigenL = src4f.llt().matrixL();
  
  std::cerr << "eigenL   :\n" << eigenL << std::endl;
  std::cerr << "cholBartL:\n" << cholBart(src4f) << std::endl;
  std::cerr << "boL      :\n" << L << std::endl;
  
  Eigen::Matrix<float, 6,1> b;
  b << 0.1 ,0.2 ,0.3 ,0.4, 0.5, 0.6;

  
  Eigen::Matrix<float, 6, 1> eigen_dx = src4f.llt().solve(-b);
  std::cerr << "Eigen sol : " << eigen_dx.transpose() << std::endl;
  
  BoVector<Eigen::Vector2f> bo_b;
  bo_b.allocate(3);
  bo_b.at(0) << -0.1, -0.2;
  bo_b.at(1) << -0.3, -0.4;
  bo_b.at(2) << -0.5, -0.6;

  BoVector<Eigen::Vector2f> bo_dy, bo_dx;
  bo_dy.allocate(3);
  bo_dx.allocate(3);
  
  forwardSubstitution(bo_dy, L, bo_b);
  backwardSubstitution(bo_dx, L, bo_dy);
  
  std::cerr << "BoMary sol: ";
  for(int i=0; i < 3; ++i)
    std::cerr << bo_dx.at(i).transpose() << " ";
  std::cerr << std::endl;
  
  return 0;
}
