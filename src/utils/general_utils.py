import psutil
import threading
from pprint import pformat


class CPUUsageMonitor(threading.Thread):
    def __init__(self, interval=1):
        super().__init__()
        self.interval = interval
        self.running = True
        self.usage_sum = 0.0
        self.count = 0

    def run(self):
        while self.running:
            # psutil.cpu_percent(interval=1) returns a percentage over the interval for all cores
            usage = psutil.cpu_percent(interval=self.interval)
            self.usage_sum += usage
            self.count += 1

    def stop(self):
        self.running = False

    def get_average_usage(self):
        return self.usage_sum / self.count if self.count else 0.0


def log_args(args):
    args_dict = vars(args)
    print("Script Arguments:\n" + pformat(args_dict))