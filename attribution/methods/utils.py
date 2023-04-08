import time

def get_time(fun_name):
    def warpper(fun):
        def inner(*arg, **kwarg):
            s_time = time.time()
            res = fun(*arg, **kwarg)
            e_time = time.time()
            print(f"{fun_name}: {e_time - s_time} s")
            return res
        return inner
    return warpper