"""# 
Doc::
All related to distributed compute and atomic read/write
   Thread Safe
   Process Safe
   Lock Mechanism
   
"""
import os, sys, socket, platform, time, gc,logging, random

###############################################################################################
from utilmy.utilmy import log, log2

def help():
    """function help.
    Doc::
            
            Args:
            Returns:
                
    """
    from utilmy import help_create
    ss  = help_create("utilmy.distributed", prefixs= [ 'test'])  #### Merge test code
    ss += HELP
    print(ss)


#########################################################################################################
####### Atomic File Index  read/writing #################################################################
class IndexLock(object):
    """Keep a Global Index of processed files.
      INDEX = IndexLock(findex)

      flist = index.save_isok(flist)  ## Filter out files in index and return available files
      ### only process correct files
      

    """
    ### Manage Invemtory Index with Atomic Write/Read
    def __init__(self, findex, file_lock=None, min_size=5, skip_comment=True, ntry=20):
        """ IndexLock:__init__.
        Doc::
                
                    Args:
                        findex:     
                        file_lock:     
                        min_size:     
                        skip_comment:     
                        ntry:     
                    Returns:
                       
        """
        self.findex= findex
        os.makedirs(os.path.dirname( os.path.abspath(self.findex)), exist_ok=True)

        if file_lock is None:
            file_lock = os.path.dirname(findex) +"/"+ findex.split("/")[-1].replace(".", "_lock.")
        self.plock = file_lock

        ### Initiate the file
        if not os.path.isfile(self.findex):
            with open(self.findex, mode='a') as fp:
                fp.write("")

        self.min_size=min_size
        self.skip_comment=True
        self.ntry =ntry


    def read(self,):
        """ IndexLock:read.
        Doc::
                
                    Args:
                        :     
                    Returns:
                       
        """
        return self.get()


    def save_isok(self, flist:list):
        """ IndexLock:save_isok.
        Doc::
                
                    Args:
                        flist (function["arg_type"][i]) :     
                    Returns:
                       
        """
        return put(self, val)

    def save_filter(self, val:list=None):
        """ IndexLock:save_filter.
        Doc::
                
                    Args:
                        val (function["arg_type"][i]) :     
                    Returns:
                       
        """
        return put(self, val)


    ######################################################################
    def get(self, **kw):
        """ IndexLock:get.
        Doc::
                
                    Args:
                        **kw:     
                    Returns:
                       
        """
        ## return the list of files
        with open(self.findex, mode='r') as fp:
            flist = fp.readlines()

        if len(flist) < 1 : return []

        flist2 = []
        for t  in flist :
            if len(t) < self.min_size: continue
            if self.skip_comment and t[0] == "#"  : continue
            flist2.append( t.strip() )
        return flist2


    def put(self, val:list=None):
        """ Read, check if the insert values are there, and save the files.
        Doc::
                
                      flist = index.check_filter(flist)   ### Remove already processed files
                      if  len(flist) < 1 : continue   ### Dont process flist
            
                      ### Need locking mechanism Common File to check for Check + Write locking.
            
        """
        import random, time
        if val is None : return True

        if isinstance(val, str):
            val = [val]

        i = 1
        while i < self.ntry :
            try :
                lock_fd = os_lock_acquireLock(self.plock)

                ### Check if files exist  #####################
                fall =  self.read()
                val2 = [] ; isok= True
                for fi in val:
                    if fi in fall :
                        print('exist in Index, skipping', fi)
                        isok =False
                    else :
                        val2.append(fi)

                if len(val2) < 1 : return []

                #### Write the list of files on Index: Wont be able to use by other processes
                ss = ""
                for fi in val2 :
                  x  = str(fi)
                  ss = ss + x.strip() + "\n"

                with open(self.findex, mode='a') as fp:
                    fp.write( ss )

                os_lock_releaseLock(lock_fd)
                return val2

            except Exception as e:
                log2(f"file lock waiting {i}s")
                time.sleep( random.random() * i )
                i += 1






if __name__ == '__main__':
    import fire
    fire.Fire()
