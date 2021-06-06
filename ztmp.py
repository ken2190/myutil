

import sys

from loguru import logger

config = {
    "handlers": [
        {
            "sink": sys.stdout,
            "format": "<level>{level: <8}</level>|<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        }
    ]
}

config = {
    "handlers": [
        { "sink": sys.stdout,
          "format": "<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        },

        {
            "sink": sys.stderr,
            "format": "<level>{level: <8}</level>|<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        }


    ]
}
logger.configure(**config)


def myfun():
  logger.info(  f'myinfo', 'ok')a
  logger.error( f' some error')

  ...: logger.debug("debug message"    ) 
   ...: logger.info("info level message") 
   ...: logger.warning("warning level message") 
   ...: logger.critical("critical level message")

myfun()


log  = logger.info
log2 = logger.debug
logw = logger.warning
loge = logger.error
logc = logger.critical



x = 7

def myfun2():
   log(  f'  {x}' )
   log2(  f'  {x}' )
   logw(  f'  {x}' )
   loge(  f'  {x}' )
   logc(  f'  {x}' )





x = 7

def myfun2():
   log(  f'  {x}' )
   log3(  f'  {x}' )

myfun2()



logger.add(log_file_name,level=level,format=format, 
        rotation="30 days", filter=None, colorize=None, serialize=False, backtrace=True, enqueue=False, catch=True)



person = {'name': 'Alice', 'age': 12}
logger.info(f"info: {person}")



  ...: logger.debug("debug message"    ) 
   ...: logger.info("info level message") 
   ...: logger.warning("warning level message") 
   ...: logger.critical("critical level message")




import os
import sys
 
from loguru import logger
 
logger.add(os.path.expanduser("~/Desktop/exception_log.log"), backtrace=True, diagnose=True)
 
def func(a, b):
    return a / b
 
def nested(c):
    try:
        func(5, c)
    except ZeroDivisionError:
        logger.exception("What?!")
 
if __name__ == "__main__":
    nested(0)





https://qxf2.com/blog/replace-python-standard-logging-with-loguru/

