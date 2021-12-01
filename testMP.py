import multiprocessing as mp 
from itertools import repeat
from time import sleep
import os
from tqdm import tqdm

adict = {}

def change_dict(x):
    print("hello, in dict funct")
    print("i am " + x)
    print('the dict is now' + str(adict))
    adict[x] = x
    sleep(int(x))
    print(f'I am {x}, the dict is now' + str(adict))
    return None

def main():
  pool = mp.Pool(processes = 3)
  print("hello, pool setup done")
  result = list(pool.map(change_dict, ['10','8','2', '6', '3']))

from collections import Counter

  
def main(params_mlist):
  params, mlist,lock= params_mlist
  
  for i in range(1000):
    lock.acquire()
    mlist.append((os.getpid(), params))
    lock.release()


if __name__ == '__main__':
  
  
  manager = mp.Manager()
  lock = manager.Lock()
  mlist = manager.list()
  
  
  
  
  pool = mp.Pool(processes = 3)
  
  list(tqdm(pool.imap(main, zip(range(10), repeat(mlist), repeat(lock))), total=10))
  # print(mlist)
  
  mycounter = Counter(mlist)
  print(mycounter)
  
  

      
  
    # I am 6, the dict is now{'2': '2', '6': '6'}
    # I am 10, the dict is now{'10': '10'}
    # I am 3, the dict is now{'8': '8', '3': '3'}