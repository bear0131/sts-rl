// Included inside the simulator shim namespace in place of its fixed vector.
// Common piles stay inline; generated cards can grow without truncation.
template<class T>struct vector {
 static constexpr int CAP=vector_capacity<T>::value;
 alignas(T) unsigned char storage_[CAP*sizeof(T)];
 T*heap_;int n,capacity_;
 vector():heap_(nullptr),n(0),capacity_(CAP){}
 explicit vector(size_t z):vector(){resize(z);}
 vector(size_t z,const T&v):vector(){resize(z,v);}
 vector(initializer_list<T>v):vector(){for(const auto&x:v)emplace_back(x);}
 template<class I,class=decltype(*static_cast<I*>(nullptr)[0])>vector(I b,I e):vector(){assign(b,e);}
 vector(const vector&v):vector(){for(const auto&x:v)emplace_back(x);}
 vector(vector&&v):vector(){for(auto&x:v)emplace_back(move(x));v.clear();}
 ~vector(){clear();if(heap_)::free(heap_);}
 vector&operator=(const vector&v){
  if(this==&v)return *this;
  int common=n<v.n?n:v.n;
  for(int i=0;i<common;++i)data()[i]=v.data()[i];
  while(n>v.n)pop_back();
  for(int i=common;i<v.n;++i)emplace_back(v.data()[i]);
  return *this;
 }
 vector&operator=(vector&&v){
  if(this==&v)return *this;
  int common=n<v.n?n:v.n;
  for(int i=0;i<common;++i)data()[i]=move(v.data()[i]);
  while(n>v.n)pop_back();
  for(int i=common;i<v.n;++i)emplace_back(move(v.data()[i]));
  v.clear();return *this;
 }
 vector&operator=(initializer_list<T>v){assign(v.begin(),v.end());return *this;}
 T*data(){return heap_?heap_:reinterpret_cast<T*>(storage_);}
 const T*data()const{return heap_?heap_:reinterpret_cast<const T*>(storage_);}
 T*begin(){return data();}const T*begin()const{return data();}
 T*end(){return data()+n;}const T*end()const{return data()+n;}
 size_t size()const{return n;}size_t capacity()const{return capacity_;}bool empty()const{return n==0;}
 T&operator[](size_t i){assert(i<(size_t)n);return data()[i];}
 const T&operator[](size_t i)const{assert(i<(size_t)n);return data()[i];}
 T&at(size_t i){return (*this)[i];}const T&at(size_t i)const{return (*this)[i];}
 T&front(){return (*this)[0];}const T&front()const{return (*this)[0];}
 T&back(){return (*this)[n-1];}const T&back()const{return (*this)[n-1];}
 void reserve(size_t z){
  if(z<=(size_t)capacity_)return;
  assert(z<=1073741823);
  int grown=capacity_;while((size_t)grown<z){assert(grown<=1073741823);grown*=2;}
  T*next=static_cast<T*>(::malloc(sizeof(T)*(size_t)grown));assert(next);
  T*old=data();for(int i=0;i<n;++i){new(next+i)T(move(old[i]));old[i].~T();}
  if(heap_)::free(heap_);heap_=next;capacity_=grown;
 }
 void clear(){while(n)pop_back();}
 void resize(size_t z){reserve(z);while(n>(int)z)pop_back();while(n<(int)z)emplace_back();}
 void resize(size_t z,const T&v){T copy(v);reserve(z);while(n>(int)z)pop_back();while(n<(int)z)emplace_back(copy);}
 template<class...A>T&emplace_back(A&&...args){
  if(n==capacity_){
   // Arguments may refer to this vector, e.g. pile.push_back(pile.front()).
   T copy(forward<A>(args)...);reserve((size_t)n+1);
   T*p=new(data()+n)T(move(copy));++n;return *p;
  }
  T*p=new(data()+n)T(forward<A>(args)...);++n;return *p;
 }
 void push_back(const T&v){emplace_back(v);}void push_back(T&&v){emplace_back(move(v));}
 void pop_back(){assert(n>0);data()[--n].~T();}
 template<class V>T*insert_one(const T*at,V&&value){
  int j=at-data();assert(j>=0&&j<=n);T copy(forward<V>(value));
  if(j==n)emplace_back(move(copy));
  else {int last=n;emplace_back(move(back()));for(int i=last-1;i>j;--i)data()[i]=move(data()[i-1]);data()[j]=move(copy);}
  return data()+j;
 }
 T*insert(const T*at,const T&v){return insert_one(at,v);}
 T*insert(const T*at,T&&v){return insert_one(at,move(v));}
 template<class I,class=decltype(*static_cast<I*>(nullptr)[0])>T*insert(const T*at,I b,I e){
  int first=at-data(),j=first;assert(j>=0&&j<=n);for(;b!=e;++b)insert(data()+j++,*b);return data()+first;
 }
 T*erase(const T*b,const T*e){
  int x=b-data(),y=e-data();assert(x>=0&&x<=y&&y<=n);int d=y-x;
  if(d){for(int i=y;i<n;++i)data()[i-d]=move(data()[i]);resize(n-d);}return data()+x;
 }
 T*erase(const T*p){return erase(p,p+1);}
 template<class I,class=decltype(*static_cast<I*>(nullptr)[0])>void assign(I b,I e){clear();for(;b!=e;++b)emplace_back(*b);}
 void assign(size_t z,const T&v){T copy(v);clear();reserve(z);for(size_t i=0;i<z;++i)emplace_back(copy);}
};
