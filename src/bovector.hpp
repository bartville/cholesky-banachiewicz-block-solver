
namespace bo_mary {

  
  template<typename Block>
  void BoVector<Block>::allocate(const int N) {
    if(_allocated >= N){
      _size = N;
      return;
    }
    
    for(int i = _allocated; i < N; ++i) {
      Block* block = new Block();
      _bovector.insert(std::pair<int, Block*>(i, block));
    }
    _allocated = N;
    _size = N;
    
  }

  
}
