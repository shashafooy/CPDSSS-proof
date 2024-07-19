"""
Functions for general utility such as updaing filenames or threading
"""
from datetime import timedelta
import os
from threading import Thread
import time
import re
from theano.printing import pydotprint
import theano.d3viz as d3v



def update_filename(path,old_name,iter,rename=True):
    reg_pattern = r"\(\d{1,3}_iter\)"
    iter_name = "({}_iter)".format(iter)
    match = re.search(reg_pattern,old_name)
    # Replace iteration number, append if it doesn't exist
    if match:
        new_name = re.sub(reg_pattern,iter_name,old_name)    
    else:
        new_name = old_name + iter_name

    # Attach unique pid to filename
    if str(os.getpid()) not in new_name:
        new_name = new_name + "_" + str(os.getpid())


    # WHILE LOOP SHOULD NOT RUN
    unique_name=new_name
    #Check if name already exists, append number to end until we obtain new name
    i=0
    while os.path.isfile(os.path.join(path,unique_name + '.pkl')):
        unique_name = new_name + '_' + str(i)        
        i=i+1
    #create file if it doesn't exists.
    #Sometimes had race condition of two programs used the same name because 
    #   new name file wasn't used for a few more clock cycles
    #   creating the file right after getting unique name should prevent this
    os.makedirs(path,exist_ok=True)
    open(os.path.join(path,unique_name + '.pkl'),'a').close()
    new_name=unique_name 
    if(rename):
        os.rename(os.path.join(path,old_name + '.pkl'),os.path.join(path,new_name + '.pkl'))
    return new_name

def print_border(msg):
    print("-"*len(msg) + "\n" + msg + "\n" + "-"*len(msg))


def time_exec(func,print_time=True):
    """Time how long the given function takes

    Args:
        func (Lambda): Lambda function with the given code that will run. e.g. lambda: myfunc(x,y)
        print_time (Bool): Set True to print the total time
    """
    start_time = time.time()
    result = func()
    end_time = time.time()
    tot_time = end_time - start_time
    print(f"Elapsed Time: {tot_time:.4f} sec")
    return result, tot_time
    

class BackgroundThread(Thread):    
    def __init__(self, group = None, target = None, name = None, args = (), kwargs = {}):
        """Wrapper for the Thread class allowing for return values. 
        Typical usage: BackgroundThread(target=func, args=(para1,))

        Args:
            group (_type_, optional): should be None; reserved for future extension when a ThreadGroup class is implemented. Defaults to None.
            target (_type_, optional): the callable object to be invoked by the run() method. Defaults to None, meaning nothing is called. Defaults to None.
            name (_type_, optional): the thread name. By default, a unique name is constructed of the form "Thread-N" where N is a small decimal number.
            args (tuple, optional): list or tuple of arguments for the target invocation. A single argument uses the form (arg1,). Defaults to ().
            kwargs (dict, optional): dictionary of keyword arguments for the target invocation. Defaults to {}.
            
        Returns:
            _type_: Thread for the user to start and terminate
        """
        Thread.__init__(self,group, target, name, args, kwargs)
        self._return = None
        self.result_read = False
    # def __init__(self,func,args=()):
    #     super().__init__()
    #     self.func=func
    #     self.args=args
    #     self.result=None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,**self._kwargs)
        # self.result = self.func(*self.args)

    def get_result(self,print_time=True):
        waiting=False
        if self.is_alive():
            waiting=True
            start_time = time.time()
        Thread.join(self)
        if waiting and print_time:
            end_time = time.time()
            print(f"thread wait time: {str(timedelta(seconds=int(end_time-start_time)))}")
        self.result_read=True
        return self._return
    
    def used_result(self):
        """Returns true if thread is done and the result has previously been obtained.
            Usedful as to not duplicate results if it has already been read
        """        
        return self.result_read

    
def print_theano_graph(obj,path):
    """Print a theano graph such as a gradient and save it in the given path

    Args:
        obj (): theano output object to comput the graph for
        path (_type_): path including filename to save the graph. Do not include filetype
    """
    pydotprint(obj,path+".png")
    d3v.d3viz(obj,path+".html")
    
