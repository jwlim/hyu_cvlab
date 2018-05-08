// thread.h
//
// Author : Jongwoo Lim <jongwoo.lim@gmail.com>

#ifndef _RVSLAM_THREAD_H_
#define _RVSLAM_THREAD_H_

#include <queue>

#include <pthread.h>
#include <sys/errno.h>  // for ETIMEDOUT
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>

namespace rvslam {

//-----------------------------------------------------------------------------
// Mutex and ConditionalVariable

struct Mutex {
  Mutex()       { pthread_mutex_init(&mutex, NULL); }
  void Lock()   { pthread_mutex_lock(&mutex); }
  void Unlock() { pthread_mutex_unlock(&mutex); }

  pthread_mutex_t mutex;
};

struct ScopedMutexLock {
  ScopedMutexLock(Mutex* mutex) {
    if ((mutex_ = mutex) != NULL) mutex_->Lock();
  }
  ~ScopedMutexLock() { if (mutex_ != NULL) mutex_->Unlock(); }

  Mutex* mutex_;
};

struct ConditionalVariable {
  ConditionalVariable()   { pthread_cond_init(&cond, NULL); }
  void Signal()           { pthread_cond_signal(&cond); }
  void Broadcast()        { pthread_cond_broadcast(&cond); }
  void Wait(Mutex* mutex) { pthread_cond_wait(&cond, &mutex->mutex); }
  void Wait(Mutex* mutex, int to) {  // timeout_usec)
    struct timeval tv;
    gettimeofday(&tv, NULL);
    unsigned int sec = tv.tv_sec, usec = tv.tv_usec;
    const unsigned int M = 1000000;
    sec += to / M, usec += to % M;
    if (usec > M) usec -= M, sec += 1;
    timespec ts;
    ts.tv_sec = sec, ts.tv_nsec = usec * 1000;
    pthread_cond_timedwait(&cond, &mutex->mutex, &ts);
  }
  bool TimedWait(Mutex* mutex, double t) {  // abstime)
    timespec ts;
    ts.tv_sec = (int)t, ts.tv_nsec = (int)((t - (int)t) * 1000000000);
    return pthread_cond_timedwait(&cond, &mutex->mutex, &ts) != ETIMEDOUT;
  }
  pthread_cond_t cond;
};

struct ReadWriteLock {
  ReadWriteLock() { pthread_rwlock_init(&lock, NULL); }
  ~ReadWriteLock() { pthread_rwlock_destroy(&lock); }
  void ReadLock() { pthread_rwlock_rdlock(&lock); }
  void TryReadLock() { pthread_rwlock_tryrdlock(&lock); }
  void WriteLock() { pthread_rwlock_wrlock(&lock); }
  void TryWriteLock() { pthread_rwlock_trywrlock(&lock); }
  void Unlock() { pthread_rwlock_unlock(&lock); }

  pthread_rwlock_t lock;
  // Check http://stackoverflow.com/questions/244316/reader-writer-locks-in-c
};

struct ScopedReadLock {
  ScopedReadLock(ReadWriteLock* rwlock) {
    VLOG(2) << 
        "------------------------read_lock.lock-----------------------------------";
    if ((rwlock_ = rwlock) != NULL) rwlock_->ReadLock();
  }
  ~ScopedReadLock() {
    VLOG(2) << 
        "------------------------read_lock.unlock---------------------------------";
    if (rwlock_ != NULL) rwlock_->Unlock();
  }
  ReadWriteLock* rwlock_;
};

struct ScopedWriteLock {
  ScopedWriteLock(ReadWriteLock* rwlock) {
    VLOG(2) <<
        "----------------------write_lock.lock-----------------------------------";
    if ((rwlock_ = rwlock) != NULL) rwlock_->WriteLock();
  }
  ~ScopedWriteLock() {
    VLOG(2) <<
        "----------------------write_lock.unlock---------------------------------";
    if (rwlock_ != NULL) rwlock_->Unlock();
  }
  ReadWriteLock* rwlock_;
};

//-----------------------------------------------------------------------------
// Thread-safe MessageQueue

template <class Message>
class MessageQueue {
 public:
  MessageQueue(int max_size = -1) { max_size_ = max_size, valid_ = true; }
  ~MessageQueue() { Close(); }
/*
  bool Push(msg_t *msg, int timeout_usec=-1);
    // push timeout : when full, -1 waits indefinitely
    //                -2 erases front, >=0 timeouts (may return false)
  msg_t* Pop(int timeout_usec);
  msg_t* Pop_last(int timeout_usec);
    // pop timeout : when empty, -1 waits indefinitely
    //               -1 waits indefinitely, >=0 timeouts (may return NULL)
  void close();
*/
  bool IsValid() const { return valid_; }

  size_t Size() {
    mutex_.Lock();
    size_t sz = q_.size();
    mutex_.Unlock();
    return sz;
  }

  bool Push(const Message& msg, int timeout_usec = -1) {
    mutex_.Lock();
    if (max_size_ > 0 && q_.size() >= max_size_) {
      if (timeout_usec >= 0) cond_push_.Wait(&mutex_, timeout_usec);
      else if (timeout_usec == -1) cond_push_.Wait(&mutex_);
      else q_.pop();
    }
    bool ret = valid_ && (max_size_ <= 0 || q_.size() < max_size_);
    if (ret) {
      q_.push(msg);
      cond_pop_.Signal();
    }
    mutex_.Unlock();
    return ret;
  }

  bool Pop(Message* msg, int timeout_usec = -1) {
    mutex_.Lock();
    if (timeout_usec < 0) {
      // if queue is not empty or cond_pop_ signal occurr, break the loop.
      while (valid_ && q_.empty()) 
        cond_pop_.Wait(&mutex_);
    } else if (timeout_usec > 0) {
      double t = CurrentTime() + timeout_usec / 1e6;
      while (valid_ && q_.empty()) {
        if (cond_pop_.TimedWait(&mutex_, t) == false) 
          break;
      }
    }
    bool ret = (valid_ && q_.empty() == false);
    if (ret) {
      *msg = q_.front();
      q_.pop();
      cond_push_.Signal();
    }
    mutex_.Unlock();
    return ret;
  }

  bool PopLast(Message* msg, int timeout_usec = -1) {
    mutex_.Lock();
    if (timeout_usec < 0) {
      while (valid_ && q_.empty()) cond_pop_.Wait(&mutex_);
    } else if (timeout_usec > 0) {
      double t = CurrentTime() + timeout_usec / 1e6;
      while (valid_ && q_.empty()) {
        if (cond_pop_.TimedWait(&mutex_, t) == false) break;
      }
    }
    bool ret = (valid_ && q_.empty() == false);
    if (ret) {
      *msg = q_.back();
      q_.clear();
      cond_push_.Signal();
    }
    mutex_.Unlock();
    return ret;
  }

  void Close() {
    valid_ = false;
    cond_push_.Signal();
    cond_pop_.Signal();
  }

 private:
  static double CurrentTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
  }

  std::queue<Message> q_;
  ConditionalVariable cond_push_, cond_pop_;
  Mutex mutex_;
  int max_size_;  // maximum queue size: no limit if <=0 [def:-1]
  bool valid_;  // Valid flag - if set false, all waiting threads will resume.
};

}  // namespace rvslam
#endif  // _RVSLAM_THREAD_H_
