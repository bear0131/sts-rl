#include <cassert>
#include <cstdlib>
#include <initializer_list>
#include <new>
#include <string>
#include <utility>
#include <vector>
namespace candidate {
using std::size_t;using std::initializer_list;using std::move;using std::forward;
template<class T>struct vector_capacity{static constexpr int value=128;};
#include "env/search/cuda_vector.h"
}
struct Counted {
 static int alive;
 std::string s;
 Counted():s("initial"){++alive;}
 explicit Counted(int n):s(std::to_string(n)){++alive;}
 Counted(const Counted&v):s(v.s){++alive;}
 Counted(Counted&&v):s(std::move(v.s)){++alive;}
 Counted&operator=(const Counted&)=default;
 Counted&operator=(Counted&&)=default;
 ~Counted(){--alive;}
};
int Counted::alive=0;
int main(){
 {
  candidate::vector<Counted> a;
  for(int i=0;i<128;++i)a.emplace_back(i);
  a.push_back(a.front());assert(a.size()==129&&a.back().s=="0");
  for(int i=129;i<1025;++i)a.emplace_back(i);
  auto b=a;auto c=a;b[0].s="branch";assert(a[0].s=="0"&&c[0].s=="0");
  c.insert(c.begin()+3,c.back());assert(c[3].s=="1024"&&c[4].s=="3");
  c.erase(c.begin()+3,c.begin()+500);assert(c[3].s=="499");
  b.resize(4097,b.front());assert(b.back().s=="branch");
  auto d=std::move(c);assert(c.empty());c=a;assert(c.size()==a.size());
  d=a;d.clear();d.assign(513,a[7]);assert(d.size()==513&&d.back().s=="7");
  a=a;assert(a.size()==1025);
 }
 assert(Counted::alive==0);
}
