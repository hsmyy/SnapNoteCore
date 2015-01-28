#ifndef DISJOINT_SET
#define DISJOINT_SET

// disjoint-set forests using union-by-rank and path compression (sort of).

typedef struct {
  int rank;
  int parent;//parent idx
  int size;
} disjointElement;

class DisjointSet {
public:
  DisjointSet(int elements);
  ~DisjointSet();
  int find(int idx);  
  void join(int idxA, int idxB);
  int size(int idx) const { return elts[idx].size; }
  int num_sets() const { return num; }

private:
  disjointElement *elts;
  int num;
};

DisjointSet::DisjointSet(int elements) {
  elts = new disjointElement[elements];
  num = elements;
  for (int i = 0; i < elements; i++) {
    elts[i].rank = 0;
    elts[i].size = 1;
    elts[i].parent = i;
  }
}
  
DisjointSet::~DisjointSet() {
  delete [] elts;
}

int DisjointSet::find(int idx) {
  int y = idx;
  while (y != elts[y].parent)
    y = elts[y].parent;
  elts[idx].parent = y;
  return y;
}

void DisjointSet::join(int idxA, int idxB) {
  if (elts[idxA].rank > elts[idxB].rank) {
    elts[idxB].parent = idxA;
    elts[idxA].size += elts[idxB].size;
  } else {
    elts[idxA].parent = idxB;
    elts[idxB].size += elts[idxA].size;
    if (elts[idxA].rank == elts[idxB].rank)
      elts[idxB].rank++;
  }
  num--;
}

#endif
