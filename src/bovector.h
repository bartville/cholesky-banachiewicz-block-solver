#pragma once
#include <iostream>
#include <map>

namespace bo_mary {
  template<typename Block> class BoVector {

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    BoVector() {
      _size = 0;
      _allocated = 0;
    }
    ~BoVector() {
      for(auto& block_ref : _bovector){
        delete block_ref.second;
      }
      std::cerr << "[BoVector]: deleted!" << std::endl;
    }
    
    inline void allocate(const int N);

    inline Block& at(const int i) {
      if(i >= _size)
        throw std::runtime_error("accessing index out of bovector size!");
      return *(_bovector.at(i));
    }

    inline const Block& at(const int i) const{
      Block& b = at(i);
      return b;
    }

    inline const int size() const {
      return _size;
    }

    inline const int allocated() const {
      return _allocated;
    }
    
    friend std::ostream& operator<<(std::ostream& stream, BoVector<Block>& vec) {
      const int rows = vec.size();
      for(int r = 0; r < rows; ++r) {
            stream << vec.at(r).transpose() << " ";
      }      
      return stream;
    }

  private:
    int _size;
    int _allocated;

    std::map<int, Block*, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Block*> > > _bovector;
  
  };

}
 
#include "bovector.hpp"
