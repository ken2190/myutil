




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
  logger.info('myinfo', 'ok')
  logger.error(' some error')


myfun()


log  = logger.info
log2 = logger.warning
log3 = logger.error




x = 7

def myfun2():
   log(  f'  {x}' )
   log3(  f'  {x}' )

myfun2()



logger.add(log_file_name,level=level,format=format, 
        rotation="30 days", filter=None, colorize=None, serialize=False, backtrace=True, enqueue=False, catch=True)



person = {'name': 'Alice', 'age': 12}
logger.info(f"info: {person}")



