"""
Ray First time Tips

0. ray.remote, ray.get, ray.put
1. delay for ray.get
2. avoid tiny task
3. avoid passing same object repeatedly to remote tasks
4. Pipeline data processing
"""

import time

import ray

ray.init(num_cpus = 4)


@ray.remote
def do_some_work(x):
    return x


start = time.time()
num_calls = 1000
[ray.get(do_some_work.remote(x)) for x in range(num_calls)]
print('invocation overhead =', (time.time() - start)*1000/num_calls)

