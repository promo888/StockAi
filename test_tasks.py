import pprint
import socket
import sys, time

import pykka
#https://pykka.readthedocs.io/en/stable/examples/


# class Resolver(pykka.ThreadingActor):
#     def resolve(self, ip):
#         try:
#             info = socket.gethostbyaddr(ip)
#             print(f"Finished resolving {ip}")
#             return info[0]
#         except Exception:
#             print(f"Failed resolving {ip}")
#             return None
#
#
# def run(pool_size, *ips):
#
#     # Start resolvers
#     resolvers = [Resolver.start().proxy() for _ in range(pool_size)]
#
#     # Distribute work by mapping IPs to resolvers (not blocking)
#     hosts = []
#     for i, ip in enumerate(ips):
#         hosts.append(resolvers[i % len(resolvers)].resolve(ip))
#
#     # Gather results (blocking)
#     ip_to_host = zip(ips, pykka.get_all(hosts))
#     pprint.pprint(list(ip_to_host))
#
#     # Clean up
#     pykka.ActorRegistry.stop_all()
#
#
# if __name__ == "__main__":
#     start_time = time.time()
#     pool_size = 10 #100
#     print(f"Pool worker size: {pool_size}")
#     if len(sys.argv[1:]) >= 2:
#         run(int(sys.argv[1]), *sys.argv[2:])
#     else:
#         ips = [f"193.35.52.{i}" for i in range(1, 100)]
#         run(pool_size, *ips)
#
#     print(f"{pool_size} actors took {(time.time() - start_time):.2f} secs")

from predictor import StockPredictor
class Resolver(pykka.ThreadingActor):
    def resolve(self, ticker):
        try:
            #s =
            StockPredictor(ticker=ticker, predict_field="isUp_close",  \
                           shift_n_periods=-1, displayChart=False, debug=False)
            print(f"Finished resolving {ticker}")
            #return info[0]
        except Exception as ex:
            print(f"\nFailed resolving {ticker}\nException:{ex}")
            #return None


def run(pool_size, *tickers):

    # Start resolvers
    resolvers = [Resolver.start().proxy() for _ in range(pool_size)]

    explorations = []
    for i, ticker in enumerate(tickers):
        explorations.append(resolvers[i % len(resolvers)].resolve(ticker))

    # Gather results (blocking)
    # pass

    # Clean up
    pykka.ActorRegistry.stop_all()


if __name__ == "__main__":
    start_time = time.time()
    pool_size = 10 #100
    print(f"Pool worker size: {pool_size}")
    if len(sys.argv[1:]) >= 2:
        run(int(sys.argv[1]), *sys.argv[2:])
    else:
        tickers = ["AI", "FSLY", "TDOC", "JPM", "KGC", "GDX", "NET", "^GSPC", "^VIX", "^DJI"] #, "GLD"]
        print(f"Checking {len(tickers)} tickers")
        run(pool_size, *tickers)

    print(f"{pool_size} actors took {(time.time() - start_time):.2f} secs")




import time
from celery import Celery
#
# #app = Celery('tasks', broker='pyamqp://guest@localhost//')
# app = Celery(
#   name='tasks',
#   #broker='redis://localhost:6379', #redis://localhost
#   broker='pyamqp://guest@localhost//',
#   backend='db+sqlite:///db.sqlite3')

#docker run -d -p 6379:6379 redis
#docker run -d -p 5672:5672 rabbitmq
#celery -A test_tasks worker -l info -Q celery --concurrency 5 #test_tasks - filename
#celery -A test_tasks worker -l info -Q celery --concurrency 5

# @app.task
# def absoluteSub(a, b):
#     return abs(a-b)
#
#
# if __name__ == "__main__":
#     from test_tasks import absoluteSub
#
#     start_time  = time.time()
#
#     for i in range(1000):
#         result = absoluteSub.delay(10, 20)
#         result.status
#         #print(result.get())
#15sec ;-/
#    print(f"1000 iterations took {time.time() - start_time} secs")


