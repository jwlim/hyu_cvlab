//
// voctree.h -
// Copyright (C) 2010 Honda Research Institute USA Inc.
// All rights reserved.
//
// MANIFEST:
//
#ifndef _VOCTREE_H_
#define _VOCTREE_H_

#include <math.h>
#include <vector>
#include <map>
#include <algorithm>
#include <stdio.h>
#include <iostream>
using namespace std;
//-----------------------------------------------------------------------------

template <int K, int L, int D>
class voctree_t
{
public:
  voctree_t()   { _int_node=NULL, _leaf_node=NULL; }
  ~voctree_t()  { free(); }

  bool is_valid() const  { return _int_node!=NULL && _leaf_node!=NULL; }

  bool load(const char *voctree_path);
  void free();

  size_t find_leaf(const float feat[D]) const;
  int cnt_inserted_doc = 0;
  bool insert_doc(int doc_id, const std::vector<float*> &feat);
  //void update_voctree();  // computes weights : NO WEIGHTING

  void query_doc(const std::vector<float*> feat, std::vector< std::pair<float,int> > &doc_score) const;

protected:
  struct int_node_t { float c[D*K]; } *_int_node;
  struct leaf_node_t { std::map<int,float> doc; } *_leaf_node;
  size_t _num_int, _num_leaf;

  static size_t child_idx(size_t i, int j=0)  { return i*K+j+1; }
  static size_t parent_idx(size_t i)  { return (i+K-1)/K-1; }
  size_t leaf_idx(size_t i) const  { return i-_num_int; }

  static float dist_func(const float f0[D], const float f1[D])
 // { float r=0; for (int i=0; i<D; ++i) r+=fabs(f0[i]-f1[i]); return r; }

  { 
    int dist = 0;
    for (int i=0; i<D; ++i) {
     int a = (int) f0[i];
     int b = (int) f1[i]; 
     unsigned int v = (a) ^ (b);
     v = v - ((v >> 1) & 0x55555555 );
     v = (v & 0x33333333) + ((v >> 2 ) & 0x33333333);
     dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }
    return (float) dist;
   } 

};

//-----------------------------------------------------------------------------

//template <int L> size_t _num_node(int K) {  return _num_node<L-1>(K)*K + 1; }
//template <> size_t _num_node<0>(int K)  { return 1; }

template <int K, int L, int D>
bool voctree_t<K,L,D>::load(const char *voctree_path)
{
  FILE *fp = fopen(voctree_path, "rb");
  if (fp == NULL)
    return false;
  free();
/*
  size_t num_node = 1, num_node_level = 1;
  for (int l=1; l<L; ++l)  num_node += (num_node_level *= K);
  _num_int = num_node;  //_num_node<L-1>(K);
  _num_leaf = (num_node += (num_node_level *= K)) - _num_int;
  cout<< num_node<<","<<_num_leaf<<","<<_num_int<<endl;
*/
  size_t num_node = (pow(K,L+1)-1)/(K-1);
   _num_leaf = pow(K,L);
   _num_int = num_node-_num_leaf;
  cout<< num_node<<","<<_num_leaf<<","<<_num_int<<endl;


  _int_node = new int_node_t[_num_int];
  _leaf_node = new leaf_node_t[_num_leaf];

  for (size_t i=0; i<_num_int; ++i)
    if (fread(_int_node[i].c, sizeof(float), D*K, fp) < D*K)
      { fclose(fp); return false; }
  fclose(fp);
//HMSG("done");
//  size_t ni = child_idx(child_idx(0,49), 49);
//  for (int i=0, ii=0; i<6; ++i) {
//    int_node_t &node = _int_node[ni];
//    HMSG_("### \t");
//    for (int j=0; j<D; ++j, ++ii)
//      HMSG_(" %.4g", node.c[ii]);
//    HMSG(" ");
//  }
  return true;
}


template <int K, int L, int D>
void voctree_t<K,L,D>::free()
{
  if (_int_node)  delete[] _int_node;
  if (_leaf_node)  delete[] _leaf_node;
}


template <int K, int L, int D>
size_t voctree_t<K,L,D>::find_leaf(const float feat[D]) const
{
  size_t idx = 0;  // root node
  for (int lvl=0; lvl < L; ++lvl) {
    int_node_t &node = _int_node[idx];
    float dist, mindist = dist_func(node.c, feat);
    int minidx = 0;
    for (int i=1; i < K; ++i)
      if ((dist = dist_func(&node.c[D*i], feat)) < mindist)
        mindist = dist, minidx = i;
    idx = child_idx(idx, minidx);
  }
  return idx;
}


template <int K, int L, int D>
bool voctree_t<K,L,D>::insert_doc(int doc_id, const std::vector<float*> &feat)
{   
  cnt_inserted_doc++;
 // cout<<"insertion "<<cnt_inserted_doc<<endl;

  int featcnt = 0;
  for (size_t i=0; i<feat.size(); ++i)
    if (feat[i] != NULL)
      ++featcnt;

  if (featcnt <= 0)
    return false;

  const float w = 1.f/featcnt;

  for (size_t i=0; i<feat.size(); ++i) {
    if (feat[i] == NULL)
      continue;
 // cout<<"here"<<endl;
    size_t idx = find_leaf(feat[i]);
    leaf_node_t &leaf = _leaf_node[leaf_idx(idx)];

 // cout<<"here"<<endl;
    std::map<int,float>::iterator it = leaf.doc.find(doc_id);
    if (it == leaf.doc.end()){
      leaf.doc.insert(std::pair<int,float>(doc_id,w));
   //   cout<<"here"<<endl;
    } else
      it->second += w;
//  HMSG_(" [%d,%.5f]", (int)idx, leaf.doc[doc_id]);
  }
//HMSG(": %d", feat.size());
  return true;
}


template <int K, int L, int D>
void voctree_t<K,L,D>::query_doc(const std::vector<float*> feat, std::vector< std::pair<float,int> > &doc_score) const
{
  std::map<size_t,int> q;
  int featcnt = 0;
  for (size_t i=0; i<feat.size(); ++i) {
    if (feat[i] == NULL)
      continue;
    size_t idx = find_leaf(feat[i]);
    const leaf_node_t &leaf = _leaf_node[leaf_idx(idx)];
    if (leaf.doc.size() > 0) {
      std::map<size_t,int>::iterator it = q.find(idx);
      if (it == q.end())
        q.insert(std::pair<size_t,int>(idx,1));
      else
        ++it->second;
//  HMSG_(" %d", (int)idx);
    }
    ++featcnt;
  }
// cout<< "query_doc"<<endl;
//for (std::map<size_t,int>::iterator i=q.begin(); i!=q.end(); ++i)
//  HMSG_(" %d,%d", (int)i->first, i->second);
//HMSG(": %d", q.size());
  std::map<size_t,int>::const_iterator q_it;
  std::map<int,float> score;  // doc_id, score
  for (q_it = q.begin(); q_it != q.end(); ++q_it) {
    const leaf_node_t &leaf = _leaf_node[leaf_idx(q_it->first)];
  //query_doc's feat count (term frequency)
		float tf = log(q_it->second+1);
    float n = q_it->second;  
	 
    std::map<int,float>::const_iterator d_it;
    std::map<int,float>::iterator j;

		int feat_frequency_cnt = 0;
	  for (d_it = leaf.doc.begin(); d_it != leaf.doc.end(); ++d_it) 
			if( d_it->second != 0 ) feat_frequency_cnt ++;
 	 
  //inverse document frequency
    
  	double idf 
     = feat_frequency_cnt == cnt_inserted_doc?
       0:log((cnt_inserted_doc)/(float)(1+feat_frequency_cnt));
   
//    cout<< cnt_inserted_doc<<","<<feat_frequency_cnt<<endl;	
    for (d_it = leaf.doc.begin(); d_it != leaf.doc.end(); ++d_it) {
      int doc_id = d_it->first;
      float m = d_it->second;
      float tmp = fabs(n-m)-n-m;
      float tmp1 = tf*idf*m;
//    cout<< tf<<","<<idf<<","<<m<<endl;
//HMSG("\t%d, %.5f / %d, %.5f", q_it->first, n, doc_id, m);
      if ((j = score.find(d_it->first)) == score.end())
        score.insert(std::pair<int,float>(doc_id, tmp1));
      else
        j->second += tmp1;
    }
  }
  doc_score.clear();
  doc_score.reserve(score.size());
  for (std::map<int,float>::iterator j=score.begin(); j != score.end(); ++j)
    doc_score.push_back(std::pair<float,int>(-1*j->second, j->first));
  std::sort(doc_score.begin(), doc_score.end());
}


//-----------------------------------------------------------------------------
#endif //_VOCTREE_H_

